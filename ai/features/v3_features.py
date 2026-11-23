"""GridLeak AI v3 — Sıfır Veri Sızıntılı Zaman Serisi Özellik Mühendisliği Motoru.

Bu modül, SGCC akıllı sayaç günlük elektrik tüketim verisinden (2014–2016, 1.034 gün)
toplam 581 aday özellik üretir (49 temel v2 özelliği + 190 2016 Dalga 2 özelliği +
198 2014–2016 geçmiş/YoY özelliği + 144 kat-içi bağlam/peer/SVD özelliği).
Eğitim ve tahminde bu havuzdan OOF TreeSHAP + Permutation AP ile seçilmiş en iyi
160 özellik kullanılır.

Sıfır veri sızıntısı sözleşmesi:
- Abone bazlı tüm özellikler yalnızca o abonenin kendi zaman serisinden satır bazında hesaplanır.
- Aboneler arası bağlam özellikleri (şebeke profili, ortak kesinti günleri, tüketim segmenti
  peer z-score ve TruncatedSVD gömmeleri) `FoldContextTransformer` içinde toplanır ve yalnızca
  CV eğitim katında (veya final eğitim kümesinde) `fit` edilir; hedef etiket (`FLAG`) asla okunmaz.
"""
from __future__ import annotations

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

ROOT = Path(__file__).resolve().parents[2]


def load_raw_for_ids(target_ids: list[str] | pd.Index, raw_path: Path | None = None):
    """Yalnızca istenen abone kimlikleri için tarih sıralı (N x 1034, float32) matris yükler; FLAG okumaz."""
    raw_path = raw_path or (ROOT / "data/raw/data.csv")
    id_list = [str(x) for x in target_ids]
    id_set = set(id_list)
    hdr = pd.read_csv(raw_path, nrows=0).columns.tolist()
    date_cols = [c for c in hdr if c not in ("CONS_NO", "FLAG")]
    dates = pd.to_datetime(date_cols, format="%Y/%m/%d")
    order = np.argsort(dates.to_numpy())
    ordered_cols = [date_cols[i] for i in order]
    ordered_dates = dates[order]

    chunks = {}
    for ch in pd.read_csv(raw_path, usecols=["CONS_NO"] + ordered_cols, dtype={"CONS_NO": str}, chunksize=4000):
        sub = ch[ch["CONS_NO"].isin(id_set)]
        if len(sub) == 0:
            continue
        vals = sub[ordered_cols].to_numpy(dtype=np.float32)
        for cid, row in zip(sub["CONS_NO"].tolist(), vals):
            chunks[cid] = row

    if len(chunks) != len(id_set):
        missing = len(id_set) - len(chunks)
        raise ValueError(f"Raw CSV missing {missing} requested subscribers.")

    arr = np.vstack([chunks[cid] for cid in id_list]).astype(np.float32)
    return arr, ordered_dates


def _max_run_length(mask: np.ndarray) -> np.ndarray:
    """2B mantıksal (boolean) dizide her satır için en uzun ardışık True serisini hesaplar."""

    n, t = mask.shape
    if t == 0:
        return np.zeros(n, dtype=np.float32)
    cur = np.zeros(n, dtype=np.int32)
    best = np.zeros(n, dtype=np.int32)
    for j in range(t):
        m = mask[:, j]
        cur = np.where(m, cur + 1, 0)
        best = np.maximum(best, cur)
    return best.astype(np.float32)


def _count_episodes(mask: np.ndarray) -> np.ndarray:
    """Her satırdaki birbirinden ayrı ardışık True bloklarının (olayların) sayısını hesaplar."""

    if mask.shape[1] == 0:
        return np.zeros(mask.shape[0], dtype=np.float32)
    padded = np.pad(mask.astype(np.int8), ((0, 0), (1, 0)), constant_values=0)
    starts = (padded[:, 1:] == 1) & (padded[:, :-1] == 0)
    return starts.sum(axis=1).astype(np.float32)


def _safe_div(num: np.ndarray, den: np.ndarray, eps: float = 1.0) -> np.ndarray:
    return num / (np.abs(den) + eps)


def _row_autocorr(filled: np.ndarray, lag: int) -> np.ndarray:
    if filled.shape[1] <= lag:
        return np.full(filled.shape[0], np.nan, dtype=np.float32)
    left = filled[:, :-lag]
    right = filled[:, lag:]
    lc = left - left.mean(axis=1, keepdims=True)
    rc = right - right.mean(axis=1, keepdims=True)
    num = (lc * rc).mean(axis=1)
    den = np.sqrt((lc * lc).mean(axis=1) * (rc * rc).mean(axis=1)) + 1e-8
    return (num / den).astype(np.float32)


def _row_corr_with_vector(mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """NaN değerleri yoksayarak her satır ile 1B şebeke referans eğrisi arasındaki Pearson korelasyonunu hesaplar."""

    valid = ~np.isnan(mat)
    cnt = valid.sum(axis=1)
    v = vec[None, :]
    xm = np.where(valid, mat, 0.0)
    vm = np.where(valid, v, 0.0)
    safe_cnt = np.maximum(cnt, 1)
    mean_x = xm.sum(axis=1, keepdims=True) / safe_cnt[:, None]
    mean_v = vm.sum(axis=1, keepdims=True) / safe_cnt[:, None]
    dx = np.where(valid, mat - mean_x, 0.0)
    dv = np.where(valid, v - mean_v, 0.0)
    cov = (dx * dv).sum(axis=1)
    std_x = np.sqrt((dx * dx).sum(axis=1))
    std_v = np.sqrt((dv * dv).sum(axis=1))
    corr = cov / (std_x * std_v + 1e-8)
    corr = np.where(cnt >= 10, corr, np.nan)
    return corr.astype(np.float32)


def _linear_slope(mat: np.ndarray) -> np.ndarray:
    """Geçerli günler üzerinden her abonenin normalleştirilmiş doğrusal tüketim eğimini (trendini) hesaplar."""

    t = mat.shape[1]
    x = np.linspace(-1.0, 1.0, t, dtype=np.float32)[None, :]
    valid = ~np.isnan(mat)
    cnt = valid.sum(axis=1)
    safe_cnt = np.maximum(cnt, 1)
    xm = np.where(valid, x, 0.0).sum(axis=1, keepdims=True) / safe_cnt[:, None]
    ym = np.where(valid, mat, 0.0).sum(axis=1, keepdims=True) / safe_cnt[:, None]
    dx = np.where(valid, x - xm, 0.0)
    dy = np.where(valid, mat - ym, 0.0)
    slope = (dx * dy).sum(axis=1) / ((dx * dx).sum(axis=1) + 1e-8)
    norm = np.abs(ym[:, 0]) + 1.0
    out = slope / norm
    return np.where(cnt >= 5, out, np.nan).astype(np.float32)


def _cusum_and_step(mat: np.ndarray, step_stride: int = 15, min_side: int = 20):
    """Normalleştirilmiş CUSUM kırılma şiddetini/konumunu ve en sert basamak düşüş/sıçrama oranlarını hesaplar."""

    n, t = mat.shape
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(mat, axis=1)
    med_safe = np.where(np.isnan(med), 0.0, med)
    filled = np.where(np.isnan(mat), med_safe[:, None], mat)
    mean = filled.mean(axis=1, keepdims=True)
    std = filled.std(axis=1, keepdims=True) + 1e-6
    centered = (filled - mean) / std
    cs = np.cumsum(centered, axis=1) / np.sqrt(max(t, 1))
    abs_cs = np.abs(cs)
    cusum_max = abs_cs.max(axis=1)
    cusum_pos = abs_cs.argmax(axis=1) / max(t - 1, 1)
    cusum_signed = cs[np.arange(n), abs_cs.argmax(axis=1)]

    
    # Aday kırılma noktalarını tarayarak en keskin düşüş (sonra/önce) ve sıçrama oranlarını bul
    valid = ~np.isnan(mat)
    cs_val = np.cumsum(np.where(valid, mat, 0.0), axis=1)
    cs_cnt = np.cumsum(valid.astype(np.int32), axis=1)
    tot_val = cs_val[:, -1]
    tot_cnt = cs_cnt[:, -1]

    best_drop_ratio = np.ones(n, dtype=np.float32)
    best_drop_pos = np.full(n, 0.5, dtype=np.float32)
    best_jump_ratio = np.ones(n, dtype=np.float32)
    best_jump_pos = np.full(n, 0.5, dtype=np.float32)

    for s in range(min_side, t - min_side, step_stride):
        c1 = cs_cnt[:, s]
        c2 = tot_cnt - c1
        ok = (c1 >= 10) & (c2 >= 10)
        m1 = cs_val[:, s] / np.maximum(c1, 1)
        m2 = (tot_val - cs_val[:, s]) / np.maximum(c2, 1)
        ratio = (m2 + 0.5) / (m1 + 0.5)
        better_drop = ok & (ratio < best_drop_ratio)
        best_drop_ratio = np.where(better_drop, ratio, best_drop_ratio)
        best_drop_pos = np.where(better_drop, s / t, best_drop_pos)
        better_jump = ok & (ratio > best_jump_ratio)
        best_jump_ratio = np.where(better_jump, ratio, best_jump_ratio)
        best_jump_pos = np.where(better_jump, s / t, best_jump_pos)

    return (
        cusum_max.astype(np.float32),
        cusum_pos.astype(np.float32),
        cusum_signed.astype(np.float32),
        best_drop_ratio.astype(np.float32),
        best_drop_pos.astype(np.float32),
        best_jump_ratio.astype(np.float32),
        best_jump_pos.astype(np.float32),
    )


def _post_missing_drop(mat: np.ndarray, window: int = 7) -> np.ndarray:
    """Eksik (kesinti) bloklarından hemen sonraki tüketimin hemen önceki tüketime ortalama oranını hesaplar."""

    n, t = mat.shape
    isna = np.isnan(mat)
    # Kesinti öncesi (sonrasında NaN gelen) ve kesinti sonrası (öncesinde NaN bulunan) geçerli günleri kıyasla
    has_prev_nan = np.zeros_like(isna)
    has_next_nan = np.zeros_like(isna)
    for k in range(1, window + 1):
        has_prev_nan[:, k:] |= isna[:, :-k]
        has_next_nan[:, :-k] |= isna[:, k:]
    after_mask = (~isna) & has_prev_nan & (~has_next_nan)
    before_mask = (~isna) & has_next_nan & (~has_prev_nan)
    c_after = after_mask.sum(axis=1)
    c_before = before_mask.sum(axis=1)
    m_after = np.where(after_mask, mat, 0.0).sum(axis=1) / np.maximum(c_after, 1)
    m_before = np.where(before_mask, mat, 0.0).sum(axis=1) / np.maximum(c_before, 1)
    ratio = (m_after + 0.5) / (m_before + 0.5)
    return np.where((c_after >= 2) & (c_before >= 2), ratio, np.nan).astype(np.float32)


def _extract_2016_wave2(arr_all: np.ndarray, dates_all: pd.DatetimeIndex, base_df: pd.DataFrame) -> pd.DataFrame:
    """2016-01-01..2016-10-31 penceresinden 190 Dalga 2 ayrıştırıcı özellik üretir:
    tek seferlik zirve vs. kalıcı yük, Ekim sapması, CUSUM/basamak kırılması,
    haftalık/günlük düzenlilik, FFT spektral enerji, sabit okuma (flatline) ve 44 haftalık profil."""
    mask_2016 = (dates_all >= "2016-01-01") & (dates_all <= "2016-10-31")
    a = arr_all[:, mask_2016].astype(np.float64)  # (N, 304)
    d = dates_all[mask_2016]
    n, t = a.shape
    out = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(a, axis=1)
        med = np.nanmedian(a, axis=1)
        std = np.nanstd(a, axis=1)
        p10, p25, p75, p90, p95, p99 = np.nanpercentile(a, [10, 25, 75, 90, 95, 99], axis=1)
        vmax = np.nanmax(a, axis=1)
        vmin = np.nanmin(a, axis=1)

    med_safe = np.where(np.isnan(med), 0.0, med)
    mean_safe = np.where(np.isnan(mean), 0.0, mean)
    std_safe = np.where(np.isnan(std), 0.0, std)
    filled = np.where(np.isnan(a), med_safe[:, None], a)
    scale = np.abs(mean_safe) + 1.0

    
    # --- 1. Tek Seferlik Tüketim Zirvesi vs. Kalıcı Yüksek Yük Ayrımı ---
    out["w2_med_to_p95"] = _safe_div(med_safe, p95, 1.0)
    out["w2_p90_to_p99"] = _safe_div(p90, p99, 1.0)
    out["w2_p95_to_max"] = _safe_div(p95, vmax, 1.0)
    out["w2_peak_tail_ratio"] = _safe_div(vmax, p90, 1.0)
    out["w2_stable_load_ratio"] = _safe_div(mean_safe, std_safe, 1.0)
    out["w2_mad_to_median"] = _safe_div(np.nanmedian(np.abs(a - med_safe[:, None]), axis=1), med_safe, 1.0)
    out["w2_iqr_to_median"] = _safe_div(p75 - p25, med_safe, 1.0)

    # En yüksek k tüketim değeri (NaN değerler sıralamada en sona itilir)
    sort_a = np.sort(np.where(np.isnan(a), -1e12, a), axis=1)
    top1 = np.where(sort_a[:, -1] > -1e11, sort_a[:, -1], np.nan)
    top2 = np.where(sort_a[:, -2] > -1e11, sort_a[:, -2], np.nan)
    top3_mean = np.where(sort_a[:, -3] > -1e11, sort_a[:, -3:].mean(axis=1), np.nan)
    top7_mean = np.where(sort_a[:, -7] > -1e11, sort_a[:, -7:].mean(axis=1), np.nan)
    out["w2_top3_to_max"] = _safe_div(top3_mean, top1, 1.0)
    out["w2_top7_to_max"] = _safe_div(top7_mean, top1, 1.0)
    out["w2_max_gap_to_2nd"] = _safe_div(top1 - top2, std_safe, 1.0)
    out["w2_top7_to_median"] = _safe_div(top7_mean, med_safe, 1.0)

    # Uç değerlerden arındırılmış ortalama (%10 alt/üst kırpılmış) ile genel ortalama oranı
    k10 = max(1, int(0.10 * t))
    trimmed = np.sort(filled, axis=1)[:, k10:-k10].mean(axis=1)
    out["w2_trimmed_to_mean"] = _safe_div(trimmed, mean_safe, 1.0)

    # Zirve gün sayıları ile izole (tek günlük) ve kalıcı yüksek yük serileri
    valid = ~np.isnan(a)
    valid_cnt = np.maximum(valid.sum(axis=1), 1)
    hi_2std = valid & (a > (mean_safe + 2.0 * std_safe)[:, None])
    hi_3std = valid & (a > (mean_safe + 3.0 * std_safe)[:, None])
    hi_2med = valid & (a > np.maximum(2.0 * med_safe, med_safe + 1.0)[:, None])
    hi_3med = valid & (a > np.maximum(3.0 * med_safe, med_safe + 2.0)[:, None])
    hi_p90 = valid & (a > p90[:, None])

    out["w2_spike_cnt_2std"] = hi_2std.sum(axis=1) / valid_cnt
    out["w2_spike_cnt_3std"] = hi_3std.sum(axis=1) / valid_cnt
    out["w2_spike_cnt_2med"] = hi_2med.sum(axis=1) / valid_cnt
    out["w2_spike_cnt_3med"] = hi_3med.sum(axis=1) / valid_cnt
    out["w2_max_high_run_p90"] = _max_run_length(hi_p90)
    out["w2_max_high_run_2med"] = _max_run_length(hi_2med)
    out["w2_high_episodes_p90"] = _count_episodes(hi_p90)

    # İzole tek günlük zirveler: gün > 2*medyan iken önceki ve sonraki gün <= 1.5*medyan
    prev_ok = np.pad(filled[:, :-1] <= np.maximum(1.5 * med_safe, med_safe + 0.5)[:, None], ((0, 0), (1, 0)), constant_values=True)
    next_ok = np.pad(filled[:, 1:] <= np.maximum(1.5 * med_safe, med_safe + 0.5)[:, None], ((0, 0), (0, 1)), constant_values=True)
    iso_spikes = hi_2med & prev_ok & next_ok
    out["w2_isolated_spike_cnt"] = iso_spikes.sum(axis=1)
    out["w2_isolated_spike_ratio"] = iso_spikes.sum(axis=1) / (hi_2med.sum(axis=1) + 1.0)
    out["w2_sustained_high_ratio"] = out["w2_max_high_run_p90"] / (hi_p90.sum(axis=1) + 1.0)

    # Çarpıklık (skewness) ve basıklık (kurtosis)
    z_arr = (filled - mean_safe[:, None]) / (std_safe[:, None] + 1e-6)
    out["w2_skew_2016"] = np.clip((z_arr ** 3).mean(axis=1), -20, 20)
    out["w2_kurt_2016"] = np.clip((z_arr ** 4).mean(axis=1), 0, 100)

    
    # --- 2. Ekim Tüketiminin Tipik Aylara Oranı ve Trend / Değişim Noktaları ---
    m_cols = [f"month_{i}_mean" for i in range(1, 11)]
    m_mat = base_df[m_cols].to_numpy(dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        typ_med = np.nanmedian(m_mat[:, :9], axis=1)
        typ_mean = np.nanmean(m_mat[:, :9], axis=1)
        typ_std = np.nanstd(m_mat[:, :9], axis=1)
        all_m_med = np.nanmedian(m_mat, axis=1)
        all_m_mean = np.nanmean(m_mat, axis=1)
        all_m_std = np.nanstd(m_mat, axis=1)

    oct_m = m_mat[:, 9]
    sep_m = m_mat[:, 8]
    aug_m = m_mat[:, 7]
    out["w2_oct_vs_typical_med"] = _safe_div(oct_m, typ_med, 1.0)
    out["w2_oct_vs_typical_mean"] = _safe_div(oct_m, typ_mean, 1.0)
    out["w2_oct_vs_sep"] = _safe_div(oct_m, sep_m, 1.0)
    out["w2_oct_vs_aug"] = _safe_div(oct_m, aug_m, 1.0)
    out["w2_oct_zscore_months"] = (oct_m - typ_mean) / (typ_std + 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out["w2_sep_vs_typical_med"] = _safe_div(sep_m, np.nanmedian(m_mat[:, :8], axis=1), 1.0)
        out["w2_month_min_to_med"] = _safe_div(np.nanmin(m_mat, axis=1), all_m_med, 1.0)
        out["w2_month_max_to_med"] = _safe_div(np.nanmax(m_mat, axis=1), all_m_med, 1.0)
        out["w2_monthly_cv"] = _safe_div(all_m_std, all_m_mean, 1.0)
        out["w2_monthly_down_frac"] = (np.diff(m_mat, axis=1) < 0).sum(axis=1) / 9.0
        out["w2_summer_to_spring"] = _safe_div(np.nanmean(m_mat[:, 5:8], axis=1), np.nanmean(m_mat[:, 2:5], axis=1), 1.0)
        out["w2_autumn_to_summer"] = _safe_div(np.nanmean(m_mat[:, 8:10], axis=1), np.nanmean(m_mat[:, 5:8], axis=1), 1.0)
        out["w2_winter_to_spring"] = _safe_div(np.nanmean(m_mat[:, 0:2], axis=1), np.nanmean(m_mat[:, 2:5], axis=1), 1.0)

    # Ölçekten bağımsız aylık göreli tüketim profili (1–10. aylar)
    for m_idx in range(10):
        out[f"w2_m{m_idx+1}_rel"] = _safe_div(m_mat[:, m_idx], all_m_mean, 1.0)

    # 2016 doğrusal eğimleri, CUSUM ve basamak kırılma noktaları
    out["w2_slope_2016"] = _linear_slope(a)
    out["w2_slope_last90"] = _linear_slope(a[:, -90:])
    out["w2_slope_last60"] = _linear_slope(a[:, -60:])
    out["w2_slope_last30"] = _linear_slope(a[:, -30:])
    (
        out["w2_cusum_max_2016"],
        out["w2_cusum_pos_2016"],
        out["w2_cusum_signed_2016"],
        out["w2_step_drop_ratio_2016"],
        out["w2_step_drop_pos_2016"],
        out["w2_step_jump_ratio_2016"],
        out["w2_step_jump_pos_2016"],
    ) = _cusum_and_step(a, step_stride=15, min_side=21)

    # 30 günlük hareketli ortalama alt/üst sınırları ve zirve sonrası düşüş
    win30 = np.lib.stride_tricks.sliding_window_view(filled, 30, axis=1).mean(axis=2)  # (N, 275)
    max30_idx = win30.argmax(axis=1)
    max30_val = win30.max(axis=1)
    min30_val = win30.min(axis=1)
    out["w2_min30_to_max30"] = _safe_div(min30_val, max30_val, 1.0)
    # En yüksek 30 günlük dönemden SONRA görülen en düşük 30 günlük tüketim oranı
    idx_grid = np.arange(win30.shape[1])[None, :]
    after_peak = np.where(idx_grid >= max30_idx[:, None], win30, np.inf)
    min30_after = after_peak.min(axis=1)
    out["w2_min30_after_max30_ratio"] = _safe_div(min30_after, max30_val, 1.0)
    out["w2_last30_to_first30"] = _safe_div(filled[:, -30:].mean(axis=1), filled[:, :30].mean(axis=1), 1.0)
    out["w2_last60_to_prev60"] = _safe_div(filled[:, -60:].mean(axis=1), filled[:, -120:-60].mean(axis=1), 1.0)

    # Günlük farklar, ani düşüş/yükseliş oranları ve kesinti sonrası kırılma
    delta = np.diff(filled, axis=1)
    out["w2_large_down_05"] = (delta < -0.5 * scale[:, None]).mean(axis=1)
    out["w2_large_up_05"] = (delta > 0.5 * scale[:, None]).mean(axis=1)
    out["w2_large_down_10"] = (delta < -1.0 * scale[:, None]).mean(axis=1)
    out["w2_large_up_10"] = (delta > 1.0 * scale[:, None]).mean(axis=1)
    out["w2_jump_asymmetry"] = out["w2_large_up_05"] - out["w2_large_down_05"]
    out["w2_max_drop_rel"] = _safe_div(-delta.min(axis=1), scale, 0.0)
    out["w2_max_jump_rel"] = _safe_div(delta.max(axis=1), scale, 0.0)
    out["w2_daily_diff_rel"] = _safe_div(np.abs(delta).mean(axis=1), scale, 0.0)
    out["w2_post_missing_drop_2016"] = _post_missing_drop(a, window=7)

    
    # --- 3. Haftalık / Günlük Düzenlilik, Sıfır / Sabit Okuma ve Spektral (FFT) Özellikler ---
    dow = d.dayofweek.to_numpy()
    wknd_mask = dow >= 5
    wkdy_mask = dow < 5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        wknd_med = np.nanmedian(a[:, wknd_mask], axis=1)
        wkdy_med = np.nanmedian(a[:, wkdy_mask], axis=1)
    out["w2_weekend_med_ratio"] = _safe_div(wknd_med, wkdy_med, 1.0)

    dow_means = np.zeros((n, 7), dtype=np.float64)
    for k in range(7):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            dow_means[:, k] = np.nanmean(a[:, dow == k], axis=1)
    dow_rel = _safe_div(dow_means, mean_safe[:, None], 1.0)
    out["w2_dow_std_rel"] = np.nanstd(dow_rel, axis=1)
    out["w2_dow_min_to_max"] = _safe_div(np.nanmin(dow_means, axis=1), np.nanmax(dow_means, axis=1), 1.0)
    for k in range(7):
        out[f"w2_dow_{k}_rel"] = dow_rel[:, k]

    # Sıfır tüketim, sıfıra yakın düşük yük (kısmi baypas) ve sabit okuma (donmuş sayaç) serileri
    zero_mask = valid & (a == 0.0)
    near_zero_mask = valid & (a > 0.0) & (a < 0.10 * np.maximum(med_safe, 0.5)[:, None])
    low_load_mask = valid & (a < 0.25 * np.maximum(med_safe, 0.5)[:, None])
    out["w2_near_zero_ratio_2016"] = near_zero_mask.sum(axis=1) / valid_cnt
    out["w2_low_load_ratio_2016"] = low_load_mask.sum(axis=1) / valid_cnt
    out["w2_longest_near_zero_2016"] = _max_run_length(low_load_mask)
    out["w2_zero_episodes_2016"] = _count_episodes(zero_mask)
    out["w2_zero_late_minus_early"] = (
        zero_mask[:, -90:].mean(axis=1) - zero_mask[:, :90].mean(axis=1)
    )

    both_valid_nz = valid[:, 1:] & valid[:, :-1] & (a[:, 1:] > 0) & (a[:, :-1] > 0)
    flat_mask = both_valid_nz & (np.abs(a[:, 1:] - a[:, :-1]) < 1e-4)
    out["w2_flatline_ratio_2016"] = flat_mask.sum(axis=1) / np.maximum(both_valid_nz.sum(axis=1), 1)
    out["w2_longest_flatline_2016"] = _max_run_length(flat_mask)

    # 2016 eksiklik blok sayısı, en uzun kesinti ve Eylül kesinti yapısı
    isna_2016 = np.isnan(a)
    out["w2_missing_episodes_2016"] = _count_episodes(isna_2016)
    out["w2_longest_missing_2016"] = _max_run_length(isna_2016)
    out["w2_missing_late_minus_early"] = isna_2016[:, -90:].mean(axis=1) - isna_2016[:, :90].mean(axis=1)
    sep_mask = d.month.to_numpy() == 9
    out["w2_sep_missing_episodes"] = _count_episodes(isna_2016[:, sep_mask])
    out["w2_sep_longest_missing"] = _max_run_length(isna_2016[:, sep_mask])

    # Farklı gecikmelerde (1, 2, 7, 14, 28 gün) otokorelasyonlar
    for lag in [1, 2, 7, 14, 28]:
        out[f"w2_autocorr_{lag}"] = _row_autocorr(filled, lag)

    # Spektral (FFT) özellikler: haftalık periyodiklik, düşük/yüksek frekans gücü ve entropi
    centered = filled - filled.mean(axis=1, keepdims=True)
    fft_mag = np.abs(np.fft.rfft(centered, axis=1)) ** 2
    fft_mag[:, 0] = 0.0
    tot_pow = fft_mag.sum(axis=1) + 1e-8
    freqs = np.fft.rfftfreq(t, d=1.0)
    wk_band = (freqs >= 1.0 / 8.0) & (freqs <= 1.0 / 6.0)
    low_band = (freqs > 0) & (freqs < 1.0 / 20.0)
    high_band = freqs > 1.0 / 3.5
    out["w2_fft_weekly_ratio"] = fft_mag[:, wk_band].sum(axis=1) / tot_pow
    out["w2_fft_low_ratio"] = fft_mag[:, low_band].sum(axis=1) / tot_pow
    out["w2_fft_high_ratio"] = fft_mag[:, high_band].sum(axis=1) / tot_pow
    p_spec = fft_mag[:, 1:] / tot_pow[:, None]
    out["w2_spectral_entropy"] = -(p_spec * np.log(p_spec + 1e-12)).sum(axis=1) / np.log(p_spec.shape[1])

    # 2016 yılı 44 haftalık göreli ortalama ve sıfır tüketim oranları
    for k, start in enumerate(range(0, t, 7)):
        blk = a[:, start : start + 7]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            wm = np.nanmean(blk, axis=1)
        out[f"w2_wk{k}_rel"] = wm / scale
        out[f"w2_wk{k}_zero"] = (blk == 0).mean(axis=1)

    df_out = pd.DataFrame(out, index=base_df.index, dtype=np.float32)
    return df_out.replace([np.inf, -np.inf], np.nan)


def _extract_history_2014_2016(arr_all: np.ndarray, dates_all: pd.DatetimeIndex, index: pd.Index) -> pd.DataFrame:
    """2014–2016 döneminden 198 çok yıllı geçmiş, sayaç yaşam döngüsü ve yıldan yıla (YoY) özellik üretir:
    kurulum sonrası gerçek kesinti olayları, kesinti sonrası 7/14/30 gün tüketim düşüşü,
    2014–2015 istatistikleri, aynı ay/yıl YoY oranları ve 3 yıllık CUSUM/basamak kırılmaları."""
    a = arr_all.astype(np.float64)
    n, t = a.shape
    out = {}

    valid_all = ~np.isnan(a)
    valid_cnt_all = valid_all.sum(axis=1)
    out["h_valid_ratio_all"] = valid_cnt_all / t

    # İlk ve son geçerli okuma günü (sayaç kurulum zamanı ve aktif kullanım aralığı)
    has_any = valid_cnt_all > 0
    first_idx = np.where(has_any, valid_all.argmax(axis=1), t)
    last_idx = np.where(has_any, t - 1 - valid_all[:, ::-1].argmax(axis=1), 0)
    out["h_first_valid_day"] = first_idx / t
    out["h_last_valid_day"] = last_idx / t
    out["h_active_span"] = np.maximum(last_idx - first_idx + 1, 0) / t
    # Yalnızca kurulum (ilk okuma) SONRASI eksiklikler (gerçek sayaç kesintisi imzası)
    idx_grid = np.arange(t)[None, :]
    after_start = idx_grid >= first_idx[:, None]
    miss_after = (~valid_all) & after_start
    miss_after_cnt = miss_after.sum(axis=1)
    out["h_missing_after_start"] = miss_after_cnt / np.maximum(after_start.sum(axis=1), 1)
    out["h_missing_episodes_after_start"] = _count_episodes(miss_after)
    out["h_longest_missing_after_start"] = _max_run_length(miss_after)
    out["h_missing_mean_gap_len"] = miss_after_cnt / np.maximum(out["h_missing_episodes_after_start"], 1.0)
    out["h_missing_max_to_total"] = out["h_longest_missing_after_start"] / np.maximum(miss_after_cnt, 1.0)

    # Benzersiz tüketim değeri oranı (sabitlenmiş veya manipüle edilmiş sayaçları yakalar)
    a_sort = np.sort(np.where(valid_all, a, -1e9), axis=1)
    uniq_steps = ((a_sort[:, 1:] - a_sort[:, :-1]) > 1e-4) & (a_sort[:, :-1] > -1e8)
    out["h_unique_val_ratio_all"] = (uniq_steps.sum(axis=1) + 1.0) / np.maximum(valid_cnt_all, 1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        all_mean = np.nanmean(a, axis=1)
        all_med = np.nanmedian(a, axis=1)
        all_std = np.nanstd(a, axis=1)
        all_max = np.nanmax(a, axis=1)
        all_p90 = np.nanpercentile(a, 90, axis=1)
        all_p10 = np.nanpercentile(a, 10, axis=1)

    all_mean_safe = np.where(np.isnan(all_mean), 0.0, all_mean)
    all_med_safe = np.where(np.isnan(all_med), 0.0, all_med)
    out["h_all_mean"] = all_mean
    out["h_all_med"] = all_med
    out["h_all_std"] = all_std
    out["h_all_cv"] = _safe_div(all_std, all_mean_safe, 1.0)
    out["h_all_peak_ratio"] = _safe_div(all_max, all_mean_safe, 1.0)
    out["h_all_p90_to_med"] = _safe_div(all_p90, all_med_safe, 1.0)
    out["h_all_zero_ratio"] = (valid_all & (a == 0)).sum(axis=1) / np.maximum(valid_cnt_all, 1)
    out["h_all_longest_zero"] = _max_run_length(valid_all & (a == 0))
    out["h_all_zero_episodes"] = _count_episodes(valid_all & (a == 0))
    out["h_all_near_zero"] = (valid_all & (a > 0) & (a < 0.1 * np.maximum(all_med_safe, 0.5)[:, None])).sum(axis=1) / np.maximum(valid_cnt_all, 1)

    # 3 yıllık sabit okuma (flatline) ve 7/14/30 günlük kesinti sonrası tüketim düşüşü
    bv = valid_all[:, 1:] & valid_all[:, :-1] & (a[:, 1:] > 0) & (a[:, :-1] > 0)
    fl = bv & (np.abs(a[:, 1:] - a[:, :-1]) < 1e-4)
    out["h_all_flatline_ratio"] = fl.sum(axis=1) / np.maximum(bv.sum(axis=1), 1)
    out["h_all_longest_flatline"] = _max_run_length(fl)
    out["h_post_missing_drop_all"] = _post_missing_drop(a, window=7)
    out["h_post_missing_drop_14d"] = _post_missing_drop(a, window=14)
    out["h_post_missing_drop_30d"] = _post_missing_drop(a, window=30)

    # Kesinti bloklarından hemen sonra sıfır tüketim oranındaki artış (baypas / kesme imzası)
    isna_all = ~valid_all
    prev_nan_7 = np.zeros_like(isna_all)
    next_nan_7 = np.zeros_like(isna_all)
    for k in range(1, 8):
        prev_nan_7[:, k:] |= isna_all[:, :-k]
        next_nan_7[:, :-k] |= isna_all[:, k:]
    after_m = valid_all & prev_nan_7 & (~next_nan_7)
    before_m = valid_all & next_nan_7 & (~prev_nan_7)
    z_after = (after_m & (a == 0)).sum(axis=1) / np.maximum(after_m.sum(axis=1), 1)
    z_before = (before_m & (a == 0)).sum(axis=1) / np.maximum(before_m.sum(axis=1), 1)
    out["h_post_missing_zero_inc"] = np.where((after_m.sum(axis=1) >= 2) & (before_m.sum(axis=1) >= 2), z_after - z_before, np.nan)

    
    # 2014, 2015 ve 2016 yıllarına ait yıllık özet istatistikler
    years = dates_all.year.to_numpy()
    months = dates_all.month.to_numpy()
    dow = dates_all.dayofweek.to_numpy()
    yr_stats = {}
    for yr in [2014, 2015, 2016]:
        ym = years == yr
        sub = a[:, ym]
        sv = ~np.isnan(sub)
        scnt = sv.sum(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = np.nanmean(sub, axis=1)
            md = np.nanmedian(sub, axis=1)
            sd = np.nanstd(sub, axis=1)
            mx = np.nanmax(sub, axis=1)
            q90 = np.nanpercentile(sub, 90, axis=1)
            q10 = np.nanpercentile(sub, 10, axis=1)
        yr_stats[yr] = dict(mean=m, med=md, std=sd, max=mx, p90=q90, p10=q10, valid_ratio=scnt / sub.shape[1])
        zero_r = (sv & (sub == 0)).sum(axis=1) / np.maximum(scnt, 1)
        zero_r = np.where(scnt > 0, zero_r, np.nan)
        yr_stats[yr]["zero"] = zero_r
        if yr in (2014, 2015):
            out[f"h_{yr}_valid_ratio"] = scnt / sub.shape[1]
            out[f"h_{yr}_mean"] = m
            out[f"h_{yr}_med"] = md
            out[f"h_{yr}_std"] = sd
            out[f"h_{yr}_cv"] = _safe_div(sd, np.where(np.isnan(m), 0.0, m), 1.0)
            out[f"h_{yr}_max"] = mx
            out[f"h_{yr}_p90"] = q90
            out[f"h_{yr}_p10"] = q10
            out[f"h_{yr}_peak_ratio"] = _safe_div(mx, np.where(np.isnan(m), 0.0, m), 1.0)
            out[f"h_{yr}_zero_ratio"] = zero_r
            out[f"h_{yr}_longest_zero"] = _max_run_length(sv & (sub == 0))
            out[f"h_{yr}_longest_missing"] = _max_run_length(~sv)
            out[f"h_{yr}_missing_episodes"] = _count_episodes(~sv)
            md_s = np.where(np.isnan(md), 0.0, md)
            sub_f = np.where(np.isnan(sub), md_s[:, None], sub)
            out[f"h_{yr}_autocorr_1"] = np.where(scnt >= 30, _row_autocorr(sub_f, 1), np.nan)
            out[f"h_{yr}_autocorr_7"] = np.where(scnt >= 30, _row_autocorr(sub_f, 7), np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                wknd = np.nanmean(sub[:, dow[ym] >= 5], axis=1)
                wkdy = np.nanmean(sub[:, dow[ym] < 5], axis=1)
            out[f"h_{yr}_weekend_ratio"] = _safe_div(wknd, wkdy, 1.0)
            # Yıllık sabit okuma, sıfıra yakın yük, kesinti sonrası düşüş ve basamak değişimi
            bv_y = sv[:, 1:] & sv[:, :-1] & (sub[:, 1:] > 0) & (sub[:, :-1] > 0)
            fl_y = bv_y & (np.abs(sub[:, 1:] - sub[:, :-1]) < 1e-4)
            out[f"h_{yr}_flatline_ratio"] = np.where(bv_y.sum(axis=1) >= 5, fl_y.sum(axis=1) / np.maximum(bv_y.sum(axis=1), 1), np.nan)
            out[f"h_{yr}_near_zero"] = np.where(scnt >= 10, (sv & (sub > 0) & (sub < 0.1 * np.maximum(md_s, 0.5)[:, None])).sum(axis=1) / np.maximum(scnt, 1), np.nan)
            out[f"h_{yr}_post_missing_drop"] = _post_missing_drop(sub, window=7)
            _, _, _, sd_r, _, sj_r, _ = _cusum_and_step(sub, step_stride=20, min_side=25)
            out[f"h_{yr}_step_drop_ratio"] = np.where(scnt >= 60, sd_r, np.nan)
            out[f"h_{yr}_step_jump_ratio"] = np.where(scnt >= 60, sj_r, np.nan)

    # Tüm 1.034 gün boyunca ardışık 30 günlük bloklar arası en sert düşüş ve sıçrama
    blk30 = np.array([np.nanmean(a[:, s:s+30], axis=1) for s in range(0, t - 29, 30)]).T
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        blk30_ratio = (blk30[:, 1:] + 0.5) / (blk30[:, :-1] + 0.5)
        out["h_max_30d_block_drop_all"] = np.nanmin(blk30_ratio, axis=1)
        out["h_max_30d_block_jump_all"] = np.nanmax(blk30_ratio, axis=1)

    
    # Yıldan yıla (YoY) tüketim, değişkenlik ve sıfır günü karşılaştırmaları
    for y2, y1 in [(2016, 2015), (2015, 2014), (2016, 2014)]:
        out[f"h_yoy_{y2}_{y1}_mean"] = _safe_div(yr_stats[y2]["mean"], yr_stats[y1]["mean"], 1.0)
        out[f"h_yoy_{y2}_{y1}_med"] = _safe_div(yr_stats[y2]["med"], yr_stats[y1]["med"], 1.0)
        out[f"h_yoy_{y2}_{y1}_p90"] = _safe_div(yr_stats[y2]["p90"], yr_stats[y1]["p90"], 1.0)
        out[f"h_yoy_{y2}_{y1}_max"] = _safe_div(yr_stats[y2]["max"], yr_stats[y1]["max"], 1.0)
        out[f"h_yoy_{y2}_{y1}_std"] = _safe_div(yr_stats[y2]["std"], yr_stats[y1]["std"], 1.0)
        out[f"h_yoy_{y2}_{y1}_zero_diff"] = yr_stats[y2]["zero"] - yr_stats[y1]["zero"]
        out[f"h_yoy_{y2}_{y1}_valid_diff"] = yr_stats[y2]["valid_ratio"] - yr_stats[y1]["valid_ratio"]

    
    # 34 aylık (Ocak 2014 – Ekim 2016) aylık göreli tüketim, eksiklik ve sıfır matrisi
    ym_means = {}
    for yr in [2014, 2015, 2016]:
        max_m = 10 if yr == 2016 else 12
        for mo in range(1, max_m + 1):
            sel = (years == yr) & (months == mo)
            blk = a[:, sel]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mm = np.nanmean(blk, axis=1)
            ym_means[(yr, mo)] = mm
            if yr in (2014, 2015):
                out[f"h_m{yr}_{mo:02d}_rel"] = _safe_div(mm, all_mean_safe, 1.0)
                out[f"h_m{yr}_{mo:02d}_miss"] = np.isnan(blk).mean(axis=1)
                out[f"h_m{yr}_{mo:02d}_zero"] = (blk == 0).sum(axis=1) / np.maximum((~np.isnan(blk)).sum(axis=1), 1)

    # Aynı ay yıldan yıla (YoY) tüketim oranları (2016/2015 ve 2015/2014)
    yoy16_15_months = []
    for mo in range(1, 11):
        r = _safe_div(ym_means[(2016, mo)], ym_means[(2015, mo)], 1.0)
        out[f"h_yoy16_15_m{mo:02d}"] = r
        yoy16_15_months.append(r)
    yoy16_15_mat = np.column_stack(yoy16_15_months)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out["h_yoy16_15_min_month"] = np.nanmin(yoy16_15_mat, axis=1)
        out["h_yoy16_15_max_month"] = np.nanmax(yoy16_15_mat, axis=1)
        out["h_yoy16_15_std_month"] = np.nanstd(yoy16_15_mat, axis=1)
        out["h_yoy16_15_drop_months"] = (yoy16_15_mat < 0.6).sum(axis=1) / np.maximum((~np.isnan(yoy16_15_mat)).sum(axis=1), 1)

    for mo in [1, 4, 7, 8, 9, 10]:
        out[f"h_yoy15_14_m{mo:02d}"] = _safe_div(ym_means[(2015, mo)], ym_means[(2014, mo)], 1.0)

    # Aynı takvim günü yıldan yıla korelasyonu (artık yıl 29 Şubat kaymasını önleyerek eşleştirilir)
    md_2016 = [(dt.month, dt.day) for dt in dates_all[years == 2016]]
    md_2015 = {(dt.month, dt.day): i for i, dt in enumerate(dates_all) if dt.year == 2015}
    md_2014 = {(dt.month, dt.day): i for i, dt in enumerate(dates_all) if dt.year == 2014}
    idx16 = [i for i, dt in enumerate(dates_all) if dt.year == 2016 and (dt.month, dt.day) in md_2015]
    idx15_for16 = [md_2015[(dates_all[i].month, dates_all[i].day)] for i in idx16]
    idx15_for14 = [i for i, dt in enumerate(dates_all) if dt.year == 2015 and (dt.month, dt.day) in md_2014]
    idx14_for15 = [md_2014[(dates_all[i].month, dates_all[i].day)] for i in idx15_for14]

    def _pair_corr(m1, m2):
        ok = (~np.isnan(m1)) & (~np.isnan(m2))
        cnt = ok.sum(axis=1)
        sc = np.maximum(cnt, 1)[:, None]
        mu1 = np.where(ok, m1, 0.0).sum(axis=1, keepdims=True) / sc
        mu2 = np.where(ok, m2, 0.0).sum(axis=1, keepdims=True) / sc
        d1 = np.where(ok, m1 - mu1, 0.0)
        d2 = np.where(ok, m2 - mu2, 0.0)
        c = (d1 * d2).sum(axis=1) / (np.sqrt((d1 * d1).sum(axis=1) * (d2 * d2).sum(axis=1)) + 1e-8)
        return np.where(cnt >= 30, c, np.nan).astype(np.float32)

    out["h_yoy_day_corr_16_15"] = _pair_corr(a[:, idx16], a[:, idx15_for16])
    out["h_yoy_day_corr_15_14"] = _pair_corr(a[:, idx15_for14], a[:, idx14_for15])

    
    # Tüm 1.034 gün boyunca uzun dönemli eğim, CUSUM, basamak kırılması ve 90 günlük çöküş oranı
    out["h_slope_all"] = _linear_slope(a)
    (
        out["h_cusum_max_all"],
        out["h_cusum_pos_all"],
        out["h_cusum_signed_all"],
        out["h_step_drop_ratio_all"],
        out["h_step_drop_pos_all"],
        out["h_step_jump_ratio_all"],
        out["h_step_jump_pos_all"],
    ) = _cusum_and_step(a, step_stride=30, min_side=45)

    filled_all = np.where(np.isnan(a), all_med_safe[:, None], a)
    win90 = np.lib.stride_tricks.sliding_window_view(filled_all, 90, axis=1)[:, ::5, :].mean(axis=2)
    max90_idx = win90.argmax(axis=1)
    max90_val = win90.max(axis=1)
    min90_val = win90.min(axis=1)
    out["h_min90_to_max90_all"] = _safe_div(min90_val, max90_val, 1.0)
    ig = np.arange(win90.shape[1])[None, :]
    min90_after = np.where(ig >= max90_idx[:, None], win90, np.inf).min(axis=1)
    out["h_min90_after_max90_all"] = _safe_div(min90_after, max90_val, 1.0)
    out["h_oct2016_vs_all_med"] = _safe_div(ym_means[(2016, 10)], all_med_safe, 1.0)
    out["h_oct2016_vs_oct2015"] = _safe_div(ym_means[(2016, 10)], ym_means[(2015, 10)], 1.0)
    out["h_oct2016_vs_oct2014"] = _safe_div(ym_means[(2016, 10)], ym_means[(2014, 10)], 1.0)

    df_out = pd.DataFrame(out, index=index, dtype=np.float32)
    return df_out.replace([np.inf, -np.inf], np.nan)


def extract_subscriber_features(
    arr_all: np.ndarray,
    dates_all: pd.DatetimeIndex,
    base_49_df: pd.DataFrame,
    include_history: bool = True,
) -> pd.DataFrame:
    """Tüm satır bazlı (aboneye özgü, sıfır sızıntılı) özellikleri birleştirir."""
    w2 = _extract_2016_wave2(arr_all, dates_all, base_49_df)
    parts = [base_49_df.astype(np.float32), w2]
    if include_history:
        hist = _extract_history_2014_2016(arr_all, dates_all, base_49_df.index)
        parts.append(hist)
    return pd.concat(parts, axis=1)


class FoldContextTransformer:
    """Yalnızca eğitim katına fit edilen aboneler arası bağlam dönüştürücüsü (144 özellik):
    1) Ortak şebeke kesinti günlerini (özellikle Eylül 2016) ayıklayarak aboneye özel eksiklik
    2) Günlük şebeke medyan yük profili ile korelasyon ve şebeke kalıntıları
    3) 10 tüketim segmenti içi göreceli sapma (peer z-score)
    4) Normalleştirilmiş tüketim, kurulum sonrası eksiklik, sıfır ve kalıntı matrislerinin
       denetimsiz TruncatedSVD gömme bileşenleri
    """

    PEER_COLS_2016 = [
        "en_yuksek_tuketim",
        "peak_ratio",
        "w2_oct_vs_typical_med",
        "ctx_sep_specific_missing",
        "variation",
        "sifir_gun_orani",
        "w2_near_zero_ratio_2016",
        "late_early_ratio",
        "ctx_grid_corr_2016",
        "w2_step_drop_ratio_2016",
        "w2_cusum_max_2016",
        "month_10_mean",
    ]
    PEER_COLS_HIST = [
        "h_yoy_2016_2015_mean",
        "h_yoy_2016_2015_med",
        "ctx_grid_corr_all",
        "h_step_drop_ratio_all",
    ]

    def __init__(self, include_history: bool = True, n_bins: int = 10, random_state: int = 42):
        self.include_history = include_history
        self.n_bins = n_bins
        self.random_state = random_state

    def fit(self, arr_all: np.ndarray, dates_all: pd.DatetimeIndex, feat_df: pd.DataFrame) -> "FoldContextTransformer":
        self.dates_all_ = dates_all
        self.mask_2016_ = (dates_all >= "2016-01-01") & (dates_all <= "2016-10-31")
        self.mask_2015_ = dates_all.year == 2015
        self.mask_2014_ = dates_all.year == 2014
        self.mask_sep16_ = (dates_all.year == 2016) & (dates_all.month == 9)
        self.mask_oct16_ = (dates_all.year == 2016) & (dates_all.month == 10)
        self.mask_h1_16_ = (dates_all >= "2016-01-01") & (dates_all <= "2016-05-31")
        self.mask_h2_16_ = (dates_all >= "2016-06-01") & (dates_all <= "2016-10-31")

        
        # 1. Günlük şebeke geneli eksiklik oranı (yalnızca eğitim katından öğrenilir)
        isna = np.isnan(arr_all)
        self.day_miss_rate_ = isna.mean(axis=0).astype(np.float64)
        sep_miss = self.day_miss_rate_[self.mask_sep16_]
        # Eylül ayı ortak şebeke kesinti günleri (şebekede >%4 eksiklik görülen günler)
        self.sep_common_outage_ = sep_miss > 0.04
        self.all16_common_outage_ = self.day_miss_rate_[self.mask_2016_] > 0.04
        self.rare_miss_days_16_ = self.day_miss_rate_[self.mask_2016_] < 0.02
        self.rare_miss_days_all_ = self.day_miss_rate_ < 0.05

        
        # 2. Günlük şebeke tüketim profili (medyan ve ortalama eğrisi)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.grid_curve_ = np.nanmedian(arr_all, axis=0).astype(np.float64)
            self.grid_mean_curve_ = np.nanmean(arr_all, axis=0).astype(np.float64)
        self.grid_curve_norm_16_ = self.grid_curve_[self.mask_2016_] / (np.nanmean(self.grid_curve_[self.mask_2016_]) + 1e-8)
        self.grid_curve_norm_all_ = self.grid_curve_ / (np.nanmean(self.grid_curve_) + 1e-8)

        
        # 3. Normalleştirilmiş tüketim, eksiklik, sıfır ve kalıntı matrisleri için denetimsiz SVD gömmeleri
        self.svd_norm16_ = TruncatedSVD(n_components=10, random_state=self.random_state)
        self.svd_norm16_.fit(self._prep_norm_matrix(arr_all[:, self.mask_2016_]))

        self.svd_miss16_ = TruncatedSVD(n_components=12, random_state=self.random_state)
        self.svd_miss16_.fit(np.isnan(arr_all[:, self.mask_2016_]).astype(np.float32))

        if self.include_history:
            self.svd_norm_all_ = TruncatedSVD(n_components=20, random_state=self.random_state)
            self.svd_norm_all_.fit(self._prep_norm_matrix(arr_all))
            self.svd_miss_all_ = TruncatedSVD(n_components=24, random_state=self.random_state)
            self.svd_miss_all_.fit(np.isnan(arr_all).astype(np.float32))
            self.svd_miss_after_ = TruncatedSVD(n_components=16, random_state=self.random_state)
            self.svd_miss_after_.fit(self._prep_miss_after_start(arr_all))
            self.svd_zero_all_ = TruncatedSVD(n_components=10, random_state=self.random_state)
            self.svd_zero_all_.fit((arr_all == 0).astype(np.float32))
            self.svd_res_all_ = TruncatedSVD(n_components=10, random_state=self.random_state)
            self.svd_res_all_.fit(self._prep_res_matrix(arr_all))

        
        # 4. 10 tüketim segmenti sınırları ve segment içi ortalama/standart sapma (peer z-score)
        ctx_tr = self._compute_grid_and_svd(arr_all, feat_df.index)
        combined_tr = pd.concat([feat_df, ctx_tr], axis=1)
        seg_vals = combined_tr["ortanca_tuketim"].to_numpy(dtype=np.float64)
        quantiles = np.linspace(0, 1, self.n_bins + 1)
        edges = np.unique(np.nanquantile(seg_vals, quantiles))
        edges[0] = -np.inf
        edges[-1] = np.inf
        self.peer_edges_ = edges
        bins = np.clip(np.digitize(np.nan_to_num(seg_vals, nan=0.0), edges) - 1, 0, len(edges) - 2)

        peer_cols = list(self.PEER_COLS_2016) + (list(self.PEER_COLS_HIST) if self.include_history else [])
        self.peer_cols_ = [c for c in peer_cols if c in combined_tr.columns]
        self.peer_stats_ = {}
        for col in self.peer_cols_:
            vals = combined_tr[col].to_numpy(dtype=np.float64)
            g_mu = float(np.nanmean(vals)) if np.isfinite(vals).any() else 0.0
            g_sd = float(np.nanstd(vals)) if np.isfinite(vals).any() else 1.0
            g_sd = g_sd if g_sd > 1e-6 else 1.0
            b_stats = {}
            for b in range(len(edges) - 1):
                sub = vals[bins == b]
                sub = sub[np.isfinite(sub)]
                if len(sub) >= 15:
                    mu = float(sub.mean())
                    sd = float(sub.std())
                    b_stats[b] = (mu, sd if sd > 1e-6 else g_sd)
                else:
                    b_stats[b] = (g_mu, g_sd)
            self.peer_stats_[col] = (b_stats, g_mu, g_sd)
        return self

    @staticmethod
    def _prep_norm_matrix(mat: np.ndarray) -> np.ndarray:
        x = np.log1p(np.clip(np.where(np.isnan(mat), 0.0, mat), 0.0, None)).astype(np.float32)
        mu = x.mean(axis=1, keepdims=True)
        sd = x.std(axis=1, keepdims=True) + 1e-5
        return (x - mu) / sd

    @staticmethod
    def _prep_miss_after_start(arr_all: np.ndarray) -> np.ndarray:
        valid = ~np.isnan(arr_all)
        has_any = valid.any(axis=1)
        t = arr_all.shape[1]
        first_idx = np.where(has_any, valid.argmax(axis=1), t)
        after = np.arange(t)[None, :] >= first_idx[:, None]
        return ((~valid) & after).astype(np.float32)

    def _prep_res_matrix(self, arr_all: np.ndarray) -> np.ndarray:
        a = arr_all.astype(np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mall = np.nanmean(a, axis=1)
        mall_s = np.where(np.isnan(mall), 0.0, mall) + 1.0
        res = a / mall_s[:, None] - self.grid_curve_norm_all_[None, :]
        res = np.clip(np.where(np.isnan(res), 0.0, res), -5.0, 5.0)
        return res.astype(np.float32)

    def _compute_grid_and_svd(self, arr_all: np.ndarray, index: pd.Index) -> pd.DataFrame:
        out = {}
        isna_all = np.isnan(arr_all)
        isna_16 = isna_all[:, self.mask_2016_]
        isna_sep = isna_all[:, self.mask_sep16_]
        sep_rates = self.day_miss_rate_[self.mask_sep16_]
        rates_16 = self.day_miss_rate_[self.mask_2016_]

        
        # --- Ortak kesinti günleri hariç Eylül aboneye özel eksiklik ---
        non_outage_sep = ~self.sep_common_outage_
        out["ctx_sep_specific_missing"] = isna_sep[:, non_outage_sep].mean(axis=1)
        out["ctx_sep_outage_missing"] = isna_sep[:, self.sep_common_outage_].mean(axis=1) if self.sep_common_outage_.any() else np.zeros(len(index))
        out["ctx_sep_excess_missing"] = (isna_sep - sep_rates[None, :]).mean(axis=1)
        w_sep = 1.0 - sep_rates
        out["ctx_sep_weighted_missing"] = (isna_sep * w_sep[None, :]).sum(axis=1) / (w_sep.sum() + 1e-8)

        # 2016 genel aboneye özel eksiklik
        out["ctx_2016_specific_missing"] = isna_16[:, ~self.all16_common_outage_].mean(axis=1)
        out["ctx_2016_excess_missing"] = (isna_16 - rates_16[None, :]).mean(axis=1)
        out["ctx_2016_rare_missing_cnt"] = isna_16[:, self.rare_miss_days_16_].sum(axis=1)

        
        # --- 2016 şebeke yük korelasyonu ve şebeke kalıntıları ---
        a16 = arr_all[:, self.mask_2016_].astype(np.float64)
        out["ctx_grid_corr_2016"] = _row_corr_with_vector(a16, self.grid_curve_[self.mask_2016_])
        out["ctx_grid_corr_h1_16"] = _row_corr_with_vector(arr_all[:, self.mask_h1_16_].astype(np.float64), self.grid_curve_[self.mask_h1_16_])
        out["ctx_grid_corr_h2_16"] = _row_corr_with_vector(arr_all[:, self.mask_h2_16_].astype(np.float64), self.grid_curve_[self.mask_h2_16_])
        out["ctx_grid_corr_shift_16"] = out["ctx_grid_corr_h2_16"] - out["ctx_grid_corr_h1_16"]
        sep_oct = self.mask_sep16_ | self.mask_oct16_
        out["ctx_grid_corr_sep_oct_16"] = _row_corr_with_vector(arr_all[:, sep_oct].astype(np.float64), self.grid_curve_[sep_oct])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m16 = np.nanmean(a16, axis=1)
        m16_s = np.where(np.isnan(m16), 0.0, m16) + 1.0
        res16 = a16 / m16_s[:, None] - self.grid_curve_norm_16_[None, :]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out["ctx_grid_res_std_16"] = np.nanstd(res16, axis=1)
            out["ctx_grid_res_max_16"] = np.nanmax(res16, axis=1)
            out["ctx_grid_res_min_16"] = np.nanmin(res16, axis=1)
            oct_in_16 = self.dates_all_[self.mask_2016_].month == 10
            out["ctx_grid_res_oct_16"] = np.nanmean(res16[:, oct_in_16], axis=1)
            out["ctx_grid_res_late_early_16"] = np.nanmean(res16[:, -90:], axis=1) - np.nanmean(res16[:, :90], axis=1)

        
        # --- 2016 SVD gömme bileşenleri ---
        emb_n16 = self.svd_norm16_.transform(self._prep_norm_matrix(arr_all[:, self.mask_2016_]))
        for j in range(emb_n16.shape[1]):
            out[f"ctx_svd_n16_{j}"] = emb_n16[:, j]
        emb_m16 = self.svd_miss16_.transform(isna_16.astype(np.float32))
        for j in range(emb_m16.shape[1]):
            out[f"ctx_svd_m16_{j}"] = emb_m16[:, j]

        if self.include_history:
            out["ctx_all_excess_missing"] = (isna_all - self.day_miss_rate_[None, :]).mean(axis=1)
            out["ctx_all_rare_missing_cnt"] = isna_all[:, self.rare_miss_days_all_].sum(axis=1)
            a_all = arr_all.astype(np.float64)
            out["ctx_grid_corr_all"] = _row_corr_with_vector(a_all, self.grid_curve_)
            out["ctx_grid_corr_2015"] = _row_corr_with_vector(arr_all[:, self.mask_2015_].astype(np.float64), self.grid_curve_[self.mask_2015_])
            out["ctx_grid_corr_2014"] = _row_corr_with_vector(arr_all[:, self.mask_2014_].astype(np.float64), self.grid_curve_[self.mask_2014_])
            out["ctx_grid_corr_diff_16_15"] = out["ctx_grid_corr_2016"] - out["ctx_grid_corr_2015"]
            out["ctx_grid_corr_diff_15_14"] = out["ctx_grid_corr_2015"] - out["ctx_grid_corr_2014"]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mall = np.nanmean(a_all, axis=1)
            mall_s = np.where(np.isnan(mall), 0.0, mall) + 1.0
            res_all = a_all / mall_s[:, None] - self.grid_curve_norm_all_[None, :]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                out["ctx_grid_res_std_all"] = np.nanstd(res_all, axis=1)
                out["ctx_grid_res_16_minus_15"] = (
                    np.nanmean(res_all[:, self.mask_2016_], axis=1) - np.nanmean(res_all[:, self.mask_2015_], axis=1)
                )

            emb_na = self.svd_norm_all_.transform(self._prep_norm_matrix(arr_all))
            for j in range(emb_na.shape[1]):
                out[f"ctx_svd_nall_{j}"] = emb_na[:, j]
            emb_ma = self.svd_miss_all_.transform(isna_all.astype(np.float32))
            for j in range(emb_ma.shape[1]):
                out[f"ctx_svd_mall_{j}"] = emb_ma[:, j]
            emb_mas = self.svd_miss_after_.transform(self._prep_miss_after_start(arr_all))
            for j in range(emb_mas.shape[1]):
                out[f"ctx_svd_mafter_{j}"] = emb_mas[:, j]
            emb_za = self.svd_zero_all_.transform((arr_all == 0).astype(np.float32))
            for j in range(emb_za.shape[1]):
                out[f"ctx_svd_zall_{j}"] = emb_za[:, j]
            emb_ra = self.svd_res_all_.transform(self._prep_res_matrix(arr_all))
            for j in range(emb_ra.shape[1]):
                out[f"ctx_svd_rall_{j}"] = emb_ra[:, j]

        return pd.DataFrame(out, index=index, dtype=np.float32).replace([np.inf, -np.inf], np.nan)

    def transform(self, arr_all: np.ndarray, feat_df: pd.DataFrame) -> pd.DataFrame:
        ctx = self._compute_grid_and_svd(arr_all, feat_df.index)
        combined = pd.concat([feat_df, ctx], axis=1)

        seg_vals = combined["ortanca_tuketim"].to_numpy(dtype=np.float64)
        bins = np.clip(np.digitize(np.nan_to_num(seg_vals, nan=0.0), self.peer_edges_) - 1, 0, len(self.peer_edges_) - 2)

        peer_out = {}
        for col in self.peer_cols_:
            vals = combined[col].to_numpy(dtype=np.float64)
            b_stats, g_mu, g_sd = self.peer_stats_[col]
            mu_arr = np.array([b_stats[b][0] for b in bins], dtype=np.float64)
            sd_arr = np.array([b_stats[b][1] for b in bins], dtype=np.float64)
            z = np.clip((vals - mu_arr) / (sd_arr + 1e-6), -15.0, 15.0)
            peer_out[f"peer_z_{col}"] = z.astype(np.float32)

        peer_df = pd.DataFrame(peer_out, index=feat_df.index, dtype=np.float32).replace([np.inf, -np.inf], np.nan)
        return pd.concat([ctx, peer_df], axis=1)

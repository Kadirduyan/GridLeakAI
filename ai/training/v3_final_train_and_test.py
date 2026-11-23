"""GridLeak AI v3 — Bağımsız 5-Fold Doğrulama, 4 Aileli Kalibre Stacking Eğitimi ve Tek Seferlik Final Test.

Bu modül:
1. Aşama A: Yeni bir tohumla (seed=2026) bağımsız 5-Fold StratifiedKFold OOF doğrulaması yapar;
   kat-içi bağlam dönüşümlerini her katta sıfırdan fit eder ve iki kademeli risk eşiklerini kilitler.
2. Aşama B: Tüm 33.858 eğitim abonesi üzerinde 4 farklı model ailesini (LightGBM, CatBoost,
   XGBoost, RankGauss MLP) logit uzayında Sigmoid/Platt kalibre StackingClassifier ile eğitir
   ve `ensemble_v3.joblib` olarak kaydeder.
3. Aşama C: Kilitli model ve eşikleri test kümesinde (8.466 abone) yalnızca tek seferde değerlendirir.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, QuantileTransformer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from ai.features.v3_features import ROOT, load_raw_for_ids, extract_subscriber_features, FoldContextTransformer

warnings.filterwarnings("ignore", category=UserWarning)


def ece_score(y, p, n_bins=15):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i + 1] if i < n_bins - 1 else p <= bins[i + 1])
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def pick_thresholds(y_tr, s_tr):
    from sklearn.metrics import precision_recall_curve
    p, r, t = precision_recall_curve(y_tr, s_tr)
    f1_arr = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12)
    t_f1 = float(t[int(np.argmax(f1_arr))])
    def _max_r(min_p):
        v = np.flatnonzero(p[:-1] >= min_p)
        return float(t[int(v[np.argmax(r[:-1][v])])]) if len(v) else float("inf")
    return {"t_f1": t_f1, "t_p50": _max_r(0.50), "t_p70": _max_r(0.70)}


def eval_oof(y, scores):
    ths = pick_thresholds(y, scores)
    pred = scores >= ths["t_f1"]
    return {
        "AP": float(average_precision_score(y, scores)),
        "ROC_AUC": float(roc_auc_score(y, scores)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "Precision": float(precision_score(y, pred, zero_division=0)),
        "Recall": float(recall_score(y, pred, zero_division=0)),
        "FP": int(((y == 0) & pred).sum()),
        "FN": int(((y == 1) & ~pred).sum()),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_base_estimators(best_params: dict):
    """Ensemble içindeki 4 farklı makine öğrenmesi model ailesini oluşturur:
    - LightGBM: Yaprak-odaklı (leaf-wise) gradyan artırma; en yüksek tekil AP ve hızlı TreeSHAP ana bileşeni.
    - CatBoost: Simetrik (oblivious) karar ağaçları; doğal düzenlileştirme ve yapısal çeşitlilik sağlar.
    - XGBoost: Derinlik-odaklı histogram gradyan artırma; yüksek Recall ve güçlü L1/L2/gamma dengesi sunar.
    - RankGauss MLP: Medyan doldurma + QuantileTransformer(normal) + 3 katmanlı Yapay Sinir Ağı (ağaç-dışı çeşitlilik).
    """
    lgb_p = dict(best_params["lgbm"])
    xgb_p = dict(best_params["xgboost"])
    cat_p = dict(best_params["catboost"], allow_writing_files=False)
    mlp_p = dict(best_params["mlp_rankgauss"])

    # Ağaç-dışı sinir ağı hattı: eksik değerleri medyanla doldurur, RankGauss ile normalleştirir ve MLP eğitir.
    mlp_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("qt", QuantileTransformer(n_quantiles=1000, output_distribution="normal", random_state=42)),
        ("mlp", MLPClassifier(**mlp_p)),
    ])
    return [
        ("lgbm", LGBMClassifier(**lgb_p)),        # 1. Aile: Yaprak-odaklı GBDT (LightGBM)
        ("catboost", CatBoostClassifier(**cat_p)), # 2. Aile: Simetrik ağaç GBDT (CatBoost)
        ("xgboost", XGBClassifier(**xgb_p)),       # 3. Aile: Derinlik-odaklı GBDT (XGBoost)
        ("mlp", mlp_pipe),                         # 4. Aile: Ağaç-dışı Yapay Sinir Ağı (RankGauss MLP)
    ]


def build_meta_estimator():
    """Meta-öğrenici (Stacking katmanı): 4 temel modelin olasılıklarını logit (log-odds) uzayına
    dönüştürür, LogisticRegression(C=0.5) ile ağırlıklandırır ve Sigmoid/Platt olasılık kalibrasyonu uygular.
    Yalnızca predict_proba sunar (decision_function üretmez), böylece tüm çıktıların [0, 1] kalibre olasılık olması garantilenir.
    """
    return Pipeline([
        ("clip", FunctionTransformer(np.clip, kw_args={"a_min": 1e-6, "a_max": 1.0 - 1e-6})),
        ("logit", FunctionTransformer(logit)),
        ("cal", CalibratedClassifierCV(LogisticRegression(C=0.5, max_iter=1000), method="sigmoid", cv=3)),
    ])


def detailed_metrics(y, scores, threshold):
    pred = scores >= threshold
    return {
        "threshold": float(threshold),
        "Accuracy": float(accuracy_score(y, pred)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y, pred)),
        "Precision": float(precision_score(y, pred, zero_division=0)),
        "Recall": float(recall_score(y, pred, zero_division=0)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "AP": float(average_precision_score(y, scores)),
        "ROC_AUC": float(roc_auc_score(y, scores)),
        "Brier": float(brier_score_loss(y, scores)),
        "LogLoss": float(log_loss(y, np.clip(scores, 1e-6, 1 - 1e-6))),
        "ECE": ece_score(y, scores),
        "TP": int(((y == 1) & pred).sum()),
        "FP": int(((y == 0) & pred).sum()),
        "FN": int(((y == 1) & ~pred).sum()),
        "TN": int(((y == 0) & ~pred).sum()),
    }


def main():
    t0 = time.time()
    ref_lock = json.loads((ROOT / "ai/artifacts/iyilestirme_v3_20261123T171349Z/locked_selection.json").read_text(encoding="utf-8"))
    best_params = ref_lock["params"]
    fcols = ref_lock["features"]
    assert len(fcols) == 160

    
    # --- Aşama A ve B boyunca YALNIZCA eğitim aboneleri yüklenir (teste sıfır erişim) ---
    v2_dir = ROOT / "data/processed/v2"
    base_tr_df = pd.read_csv(v2_dir / "egitim_ozellikleri_2016_v1.csv", dtype={"CONS_NO": str})
    tr_ids = base_tr_df["CONS_NO"].tolist()
    y_tr = base_tr_df["FLAG"].to_numpy(dtype=np.int32)
    base_tr_49 = base_tr_df.drop(columns=["CONS_NO", "FLAG"]).set_index(pd.Index(tr_ids, name="CONS_NO"))

    arr_tr, dates_all = load_raw_for_ids(tr_ids)
    sub_tr = extract_subscriber_features(arr_tr, dates_all, base_tr_49, include_history=True)
    assert len(sub_tr) == len(tr_ids) == 33858

    
    # =========================================================================
    # AŞAMA A: BAĞIMSIZ 5-FOLD OOF DOĞRULAMASI (YENİ TOHUM=2026, SIFIR YENİDEN AYAR)
    # =========================================================================
    print("=== PHASE A: INDEPENDENT 5-FOLD OOF CONFIRMATION (SEED=2026) ===", flush=True)
    folds5 = list(StratifiedKFold(5, shuffle=True, random_state=2026).split(sub_tr, y_tr))
    base_names = ["lgbm", "catboost", "xgboost", "mlp"]
    oof5_base = {name: np.zeros(len(y_tr), dtype=np.float64) for name in base_names}

    for k, (tr_idx, va_idx) in enumerate(folds5):
        fk_t = time.time()
        ctx = FoldContextTransformer(include_history=True, random_state=42).fit(arr_tr[tr_idx], dates_all, sub_tr.iloc[tr_idx])
        X_tr_k = pd.concat([sub_tr.iloc[tr_idx], ctx.transform(arr_tr[tr_idx], sub_tr.iloc[tr_idx])], axis=1)[fcols]
        X_va_k = pd.concat([sub_tr.iloc[va_idx], ctx.transform(arr_tr[va_idx], sub_tr.iloc[va_idx])], axis=1)[fcols]

        ests = build_base_estimators(best_params)
        fold_aps = {}
        for name, est in ests:
            est.fit(X_tr_k, y_tr[tr_idx])
            p = est.predict_proba(X_va_k)[:, 1]
            oof5_base[name][va_idx] = p
            fold_aps[name] = average_precision_score(y_tr[va_idx], p)
        print(f"  5-Fold [{k+1}/5] ({time.time()-fk_t:.1f}s): " + " | ".join(f"{n}={v:.4f}" for n, v in fold_aps.items()), flush=True)

    P5 = np.column_stack([oof5_base[n] for n in base_names])
    oof5_cal = np.zeros(len(y_tr), dtype=np.float64)
    for k, (tr_idx, va_idx) in enumerate(folds5):
        meta = build_meta_estimator()
        meta.fit(P5[tr_idx], y_tr[tr_idx])
        oof5_cal[va_idx] = meta.predict_proba(P5[va_idx])[:, 1]

    
    # 5-fold OOF üzerinde çapraz-uyarlamalı risk eşiği hesaplama
    fold5_ths = []
    pred5 = {k: np.zeros(len(y_tr), dtype=bool) for k in ["t_p50", "t_f1", "t_p70"]}
    for tr_idx, va_idx in folds5:
        ths = pick_thresholds(y_tr[tr_idx], oof5_cal[tr_idx])
        fold5_ths.append(ths)
        for tk in pred5:
            pred5[tk][va_idx] = oof5_cal[va_idx] >= ths[tk]

    locked_ths = {tk: float(np.mean([ft[tk] for ft in fold5_ths])) for tk in ["t_p50", "t_f1", "t_p70"]}
    oof5_ap = float(average_precision_score(y_tr, oof5_cal))
    oof5_roc = float(roc_auc_score(y_tr, oof5_cal))
    oof5_brier = float(brier_score_loss(y_tr, oof5_cal))
    oof5_ece = ece_score(y_tr, oof5_cal)

    print(f"\n[5-FOLD INDEPENDENT OOF] AP={oof5_ap:.5f} | ROC_AUC={oof5_roc:.5f} | Brier={oof5_brier:.5f} | ECE={oof5_ece:.5f}", flush=True)
    for name in base_names:
        m_i = eval_oof(y_tr, oof5_base[name])
        print(f"  5-Fold Base [{name:10s}]: AP={m_i['AP']:.5f} ROC={m_i['ROC_AUC']:.5f} F1={m_i['F1']:.4f}", flush=True)
    for tk in ["t_p50", "t_f1", "t_p70"]:
        m_t = detailed_metrics(y_tr, oof5_cal, locked_ths[tk])
        # Çapraz-uyarlamalı ikili sınıflandırma metriklerini yazdır
        p_cf = precision_score(y_tr, pred5[tk], zero_division=0)
        r_cf = recall_score(y_tr, pred5[tk], zero_division=0)
        f_cf = f1_score(y_tr, pred5[tk], zero_division=0)
        fp_cf = int(((y_tr == 0) & pred5[tk]).sum())
        fn_cf = int(((y_tr == 1) & ~pred5[tk]).sum())
        print(f"  5-Fold Cross-Fitted [{tk:5s}] thresh={locked_ths[tk]:.4f} | P={p_cf:.4f} R={r_cf:.4f} F1={f_cf:.4f} FP={fp_cf} FN={fn_cf}", flush=True)

    
    # Kilitli baseline (0,3337 AP) eşiğine karşı kapı doğrulaması
    assert oof5_ap > 0.55, f"Gate failed: 5-fold OOF AP {oof5_ap:.4f} did not clearly surpass baseline!"
    print("GATE PASSED BY MASSIVE MARGIN! Locking selection BEFORE test access.", flush=True)

    
    # Yeni deney ve işlenmiş v3 klasörlerini oluştur
    exp_name = datetime.now(timezone.utc).strftime("iyilestirme_v3_%Y%m%dT%H%M%SZ")
    exp_dir = ROOT / "ai/artifacts" / exp_name
    exp_dir.mkdir(parents=True, exist_ok=False)
    v3_dir = ROOT / "data/processed/v3"
    v3_dir.mkdir(parents=True, exist_ok=True)

    
    # Test kümesine dokunmadan önce model ve eşikleri kilitle
    locked_selection = {
        "experiment": exp_name,
        "model_name": "Calibrated Stacking v3 (LGBM + CatBoost + XGBoost + MLP)",
        "n_features": len(fcols),
        "features": fcols,
        "params": best_params,
        "cv_5fold_seed": 2026,
        "cv_5fold_oof_ap": oof5_ap,
        "cv_5fold_oof_roc_auc": oof5_roc,
        "cv_5fold_oof_brier": oof5_brier,
        "cv_5fold_oof_ece": oof5_ece,
        "decision_threshold": locked_ths["t_f1"],
        "risk_thresholds": {
            "f1_optimal_high_tier": locked_ths["t_f1"],
            "precision50_max_recall_medium_tier": locked_ths["t_p50"],
            "precision70_critical_tier": locked_ths["t_p70"],
        },
        "threshold_note": (
            "Because v3 F1-optimal precision (~58-60%) already exceeds 50%, t_f1 > t_p50. "
            "decision_threshold is locked at t_f1 (maximizes F1 with ~60% precision and minimal FP); "
            "t_p50 expands recall to ~65% at >=50% precision; t_p70 provides a critical high-confidence tier (>=70% precision)."
        ),
        "test_accessed_before_lock": False,
    }
    (exp_dir / "locked_selection.json").write_text(json.dumps(locked_selection, indent=2, ensure_ascii=False), encoding="utf-8")

    
    # Arayüz ve değerlendirme defteri için 5-fold CV karşılaştırma tablosunu kaydet
    cv_rows = []
    for name in base_names:
        m_i = eval_oof(y_tr, oof5_base[name])
        cv_rows.append({
            "Model": name,
            "CV_Folds": 5,
            "OOF_AP": round(m_i["AP"], 5),
            "OOF_ROC_AUC": round(m_i["ROC_AUC"], 5),
            "OOF_F1": round(m_i["F1"], 4),
            "OOF_Precision": round(m_i["Precision"], 4),
            "OOF_Recall": round(m_i["Recall"], 4),
            "OOF_FP": m_i["FP"],
            "OOF_FN": m_i["FN"],
        })
    m_ens_f1 = detailed_metrics(y_tr, oof5_cal, locked_ths["t_f1"])
    cv_rows.append({
        "Model": locked_selection["model_name"],
        "CV_Folds": 5,
        "OOF_AP": round(oof5_ap, 5),
        "OOF_ROC_AUC": round(oof5_roc, 5),
        "OOF_F1": round(m_ens_f1["F1"], 4),
        "OOF_Precision": round(m_ens_f1["Precision"], 4),
        "OOF_Recall": round(m_ens_f1["Recall"], 4),
        "OOF_FP": m_ens_f1["FP"],
        "OOF_FN": m_ens_f1["FN"],
    })
    pd.DataFrame(cv_rows).to_csv(exp_dir / "model_karsilastirma.csv", index=False)

    
    # =========================================================================
    # AŞAMA B: TÜM 33.858 EĞİTİM ABONESİYLE FİNAL MODEL VE BAĞLAM EĞİTİMİ
    # =========================================================================
    print("\n=== PHASE B: FITTING FINAL PRODUCTION PIPELINE ON ALL 33,858 TRAIN SUBSCRIBERS ===", flush=True)
    ctx_full = FoldContextTransformer(include_history=True, random_state=42).fit(arr_tr, dates_all, sub_tr)
    X_train_full = pd.concat([sub_tr, ctx_full.transform(arr_tr, sub_tr)], axis=1)[fcols].astype(np.float32)
    assert not np.isinf(X_train_full.to_numpy(dtype=np.float64)).any()
    joblib.dump(ctx_full, exp_dir / "context_transformer.joblib", compress=3)

    final_model = StackingClassifier(
        estimators=build_base_estimators(best_params),
        final_estimator=build_meta_estimator(),
        cv=StratifiedKFold(5, shuffle=True, random_state=2026),
        stack_method="predict_proba",
        n_jobs=1,
        passthrough=False,
    )
    t_fit = time.time()
    final_model.fit(X_train_full, y_tr)
    print(f"Final 4-family calibrated StackingClassifier fitted in {time.time()-t_fit:.1f}s", flush=True)
    assert not hasattr(final_model, "decision_function"), "Must use predict_proba, not decision_function!"

    model_path = exp_dir / "ensemble_v3.joblib"
    joblib.dump(final_model, model_path, compress=3)
    # Kaydedilen modelin diskten hatasız yüklendiğini ve sonlu olasılık ürettiğini doğrula
    reloaded = joblib.load(model_path)
    p_chk = reloaded.predict_proba(X_train_full.iloc[:5])[:, 1]
    assert np.isfinite(p_chk).all() and (p_chk >= 0).all() and (p_chk <= 1).all()
    print(f"Model saved & verified: {model_path} ({model_path.stat().st_size / 1e6:.1f} MB)", flush=True)

    
    # v3 işlenmiş eğitim özellik tablosunu kaydet
    train_out = X_train_full.copy()
    train_out.insert(0, "CONS_NO", tr_ids)
    train_out["FLAG"] = y_tr
    train_out.to_csv(v3_dir / "egitim_ozellikleri_2016_v1.csv", index=False)

    
    # =========================================================================
    # AŞAMA C: TEK SEFERLİK KİLİTLİ TEST DÖNÜŞÜMÜ VE DEĞERLENDİRMESİ (AYAR YOK)
    # =========================================================================
    print("\n=== PHASE C: SINGLE ONE-TIME TEST EVALUATION (LOCKED MODEL & THRESHOLDS) ===", flush=True)
    base_te_df = pd.read_csv(v2_dir / "test_ozellikleri_2016_v1.csv", dtype={"CONS_NO": str})
    te_ids = base_te_df["CONS_NO"].tolist()
    base_te_49 = base_te_df.drop(columns=["CONS_NO"]).set_index(pd.Index(te_ids, name="CONS_NO"))
    assert not (set(tr_ids) & set(te_ids)), "Overlap between train and test!"

    arr_te, dates_te = load_raw_for_ids(te_ids)
    assert (dates_te == dates_all).all()
    sub_te = extract_subscriber_features(arr_te, dates_te, base_te_49, include_history=True)
    # Test verisini yalnızca EĞİTİME fit edilmiş ctx_full ile dönüştür
    X_test_full = pd.concat([sub_te, ctx_full.transform(arr_te, sub_te)], axis=1)[fcols].astype(np.float32)
    assert not np.isinf(X_test_full.to_numpy(dtype=np.float64)).any()

    test_out = X_test_full.copy()
    test_out.insert(0, "CONS_NO", te_ids)
    test_out.to_csv(v3_dir / "test_ozellikleri_2016_v1.csv", index=False)

    
    # Veri kalitesi ve test hedef tablolarını değiştirmeden v3 klasörüne kopyala
    for fname in ["egitim_veri_kalitesi_2016.csv", "test_veri_kalitesi_2016.csv", "test_hedefleri_2016_v1.csv"]:
        shutil.copy2(v2_dir / fname, v3_dir / fname)

    cfg = json.loads((v2_dir / "ozellik_tanimi_2016_v1.json").read_text(encoding="utf-8"))
    cfg["surum"] = "2016_v3_multiyr_stacking"
    cfg["ozellikler"] = fcols
    cfg["ogrenilmis_donusum_uygulandi"] = True
    cfg["not"] = (
        "2014-2016 geçmiş tüketim + Dalga 2 ayrıştırıcı özellikler + yalnızca eğitim verisine fit edilmiş "
        "şebeke/eksiklik/peer bağlam dönüşümü (160 seçilmiş özellik). Uygulama grafiği 2016-01-01..2016-10-31 "
        "penceresini korur."
    )
    (v3_dir / "ozellik_tanimi_2016_v1.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    
    # Uygulama aktarım betiğiyle uyumlu deney_bilgisi.json dosyasını yaz
    deney_bilgisi = {
        "secilen_model": locked_selection["model_name"],
        "secilen_pipeline": "ensemble_v3.joblib",
        "processed_dir": "data/processed/v3",
        "decision_threshold": locked_ths["t_f1"],
        "risk_thresholds": locked_selection["risk_thresholds"],
        "random_state": 42,
        "test_kumesi_kullanildi": False,
        "ozellik_tanimi": cfg,
        "egitim_csv_sha256": sha256(v3_dir / "egitim_ozellikleri_2016_v1.csv"),
        "ozellik_json_sha256": sha256(v3_dir / "ozellik_tanimi_2016_v1.json"),
        "sinirliliklar": [
            "İlk EDA tüm veri üzerinde yapıldı ve eski test sonucu daha önce görüldü; aynı testteki karşılaştırma yeni bağımsız dış doğrulama değildir.",
            "Model çıktıları otomatik kaçak teyidi değil, saha ekibi için kalibre edilmiş inceleme önceliğidir.",
        ],
    }
    (exp_dir / "deney_bilgisi.json").write_text(json.dumps(deney_bilgisi, ensure_ascii=False, indent=2), encoding="utf-8")

    
    # Tek seferlik nihai test değerlendirmesi
    labels_df = pd.read_csv(v3_dir / "test_hedefleri_2016_v1.csv", dtype={"CONS_NO": str}).set_index("CONS_NO")
    y_test = labels_df.loc[te_ids, "FLAG"].to_numpy(dtype=np.int32)
    test_scores = reloaded.predict_proba(X_test_full)[:, 1]
    assert np.isfinite(test_scores).all()

    test_res = {
        "experiment": exp_name,
        "test_scored": int(len(y_test)),
        "test_positives": int(y_test.sum()),
        "default_0.5": detailed_metrics(y_test, test_scores, 0.5),
        "locked_f1_threshold": detailed_metrics(y_test, test_scores, locked_ths["t_f1"]),
        "locked_p50_threshold": detailed_metrics(y_test, test_scores, locked_ths["t_p50"]),
        "locked_p70_threshold": detailed_metrics(y_test, test_scores, locked_ths["t_p70"]),
    }

    
    # Saha kapasitesine göre test Precision@K hesabı
    order = np.lexsort((np.array(te_ids), -test_scores))
    pak = {}
    for k in [20, 50, 100, 250, 500]:
        hits = int(y_test[order[:k]].sum())
        pak[f"P@{k}"] = {"hits": hits, "precision": hits / k, "recall": hits / int(y_test.sum())}
    test_res["precision_at_k"] = pak

    (exp_dir / "final_test.json").write_text(json.dumps(test_res, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n================ FINAL ONE-TIME TEST RESULTS ================", flush=True)
    print(f"Experiment: {exp_name}", flush=True)
    for label, key in [
        ("LOCKED F1-Optimal (Primary)", "locked_f1_threshold"),
        ("LOCKED Precision>=50% Tier ", "locked_p50_threshold"),
        ("LOCKED Precision>=70% Tier ", "locked_p70_threshold"),
        ("Default 0.50 Threshold     ", "default_0.5"),
    ]:
        m = test_res[key]
        print(
            f"  {label} (t={m['threshold']:.4f}): AP={m['AP']:.4f} ROC={m['ROC_AUC']:.4f} "
            f"F1={m['F1']:.4f} P={m['Precision']:.4f} R={m['Recall']:.4f} | "
            f"TP={m['TP']} FP={m['FP']} FN={m['FN']} | Acc={m['Accuracy']:.4f} ECE={m['ECE']:.4f}",
            flush=True,
        )
    print("  Precision@K:", {k: f"{v['precision']:.1%} ({v['hits']}/{k[2:]})" for k, v in pak.items()}, flush=True)
    print(f"Total elapsed: {time.time()-t0:.1f}s", flush=True)
    print(f"READY_FOR_PIPELINE: {exp_name}", flush=True)


if __name__ == "__main__":
    main()

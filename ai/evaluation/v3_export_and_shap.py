"""GridLeak AI v3 — Başsız Değerlendirme, Hızlı LightGBM TreeSHAP ve Katalog Aktarımı.

Bu betik, önceden eğitilip kaydedilmiş final `ensemble_v3.joblib` modelini yeniden eğitmeden yükler:
1. Ayrılmış test kümesi üzerinde resmi değerlendirme tablolarını ve grafikleri (`Agg` başsız modda) kaydeder.
2. Ensemble'ın ana bileşeni LightGBM üzerinden en yüksek riskli ilk 2.000 abone için kesin TreeSHAP
   açıklamalarını (~6,7 sn) hesaplar.
3. Tüm sonuçları canlı uygulamanın okuduğu `data/application/catalog.sqlite3` dosyasına aktarır.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    PrecisionRecallDisplay,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)

from ai.evaluation.export_application import export

EXP_NAME = "iyilestirme_v3_20261123T171349Z"


def sha256(p: Path) -> str:
    """Dosyanın SHA256 bütünlük özetini hesaplar."""
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    # --- 0. Kayıtlı Deney Bilgisi ve Final Modeli Yükle (Yeniden Eğitim Yok) ---
    t0 = time.time()
    exp_dir = ROOT / "ai/artifacts" / EXP_NAME
    info = json.loads((exp_dir / "deney_bilgisi.json").read_text(encoding="utf-8"))
    processed = ROOT / info["processed_dir"]
    names = info["ozellik_tanimi"]["ozellikler"]
    esik = float(info["decision_threshold"])

    model_path = exp_dir / info["secilen_pipeline"]
    model = joblib.load(model_path)
    assert not hasattr(model, "decision_function")
    pos_col = list(model.classes_).index(1)

    # --- 1. Resmi Test Değerlendirmesi, Metrikler ve Grafikler (degerlendirme_*) ---
    tr_df = pd.read_csv(processed / "egitim_ozellikleri_2016_v1.csv", dtype={"CONS_NO": str})
    te_df = pd.read_csv(processed / "test_ozellikleri_2016_v1.csv", dtype={"CONS_NO": str})
    hedefler = pd.read_csv(processed / "test_hedefleri_2016_v1.csv", dtype={"CONS_NO": str})
    assert not (set(tr_df["CONS_NO"]) & set(te_df["CONS_NO"]))

    # Test özelliklerini gerçek hedeflerle abone kimliği üzerinden birebir eşleştir
    uygun = hedefler["skorlanabilir"].astype(str).str.lower().map({"true": True, "false": False})
    test = te_df.merge(hedefler[["CONS_NO", "FLAG"]], on="CONS_NO", validate="one_to_one")
    X_test = test[names].copy()
    y_test = test["FLAG"].to_numpy(dtype=int)

    # Kilitli eşikle (t=0,2901) test olasılık skorları ve sınıf tahminlerini üret
    skor = model.predict_proba(X_test)[:, pos_col]
    tahmin = (skor >= esik).astype(int)
    assert np.isfinite(skor).all()

    # Model ve "herkese 0" referansı için tüm sınıflandırma metriklerini hesapla
    def metrikler(gercek, sinif, puan):
        return {
            "Accuracy": accuracy_score(gercek, sinif),
            "Balanced accuracy": balanced_accuracy_score(gercek, sinif),
            "Precision": precision_score(gercek, sinif, zero_division=0),
            "Recall": recall_score(gercek, sinif, zero_division=0),
            "F1": f1_score(gercek, sinif, zero_division=0),
            "AP": average_precision_score(gercek, puan),
            "ROC-AUC": roc_auc_score(gercek, puan),
        }

    sonuclar = pd.DataFrame(
        [
            metrikler(y_test, tahmin, skor),
            metrikler(y_test, np.zeros(len(test), dtype=int), np.zeros(len(test))),
        ],
        index=[info["secilen_model"], "Herkese FLAG=0 referansı"],
    )

    # Karışıklık matrisi (TN, FP, FN, TP), kapsama ve abone bazlı hata dökümü
    tn, fp, fn, tp = confusion_matrix(y_test, tahmin, labels=[0, 1]).ravel()
    hata_ozeti = pd.DataFrame({"Sonuç": ["TN", "FP", "FN", "TP"], "Abone": [int(tn), int(fp), int(fn), int(tp)]})
    kapsam = pd.DataFrame({
        "Durum": ["Toplam test", "Skorlanabilir", "Veri yetersiz"],
        "Abone": [len(hedefler), len(test), int((~uygun).sum())],
    })
    abone_sonuclari = test[["CONS_NO", "FLAG"]].copy()
    abone_sonuclari["tahmin"] = tahmin
    abone_sonuclari["model_skoru"] = skor
    abone_sonuclari["sonuc"] = np.select(
        [(y_test == 0) & (tahmin == 0), (y_test == 0) & (tahmin == 1), (y_test == 1) & (tahmin == 0)],
        ["TN", "FP", "FN"],
        default="TP",
    )

    # Saha denetim kapasitelerine (ilk 20, 50, 100, 250, 500 abone) göre Precision@K hesabı
    sirali = abone_sonuclari.sort_values(["model_skoru", "CONS_NO"], ascending=[False, True])
    k_degerleri = [20, 50, 100, 250, 500]
    k_rows = []
    for k in k_degerleri:
        yakalanan = int(sirali.head(k)["FLAG"].sum())
        k_rows.append({"K": k, "Kaçak etiketli": yakalanan, "Precision@K": yakalanan / k, "Recall@K": yakalanan / int(y_test.sum())})
    kapasite_tablosu = pd.DataFrame(k_rows)

    # Karışıklık Matrisi, Precision-Recall ve ROC eğrilerini tek figürde çiz (Agg başsız mod)
    fig, eksenler = plt.subplots(1, 3, figsize=(17, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, tahmin, labels=[0, 1], ax=eksenler[0], colorbar=False)
    PrecisionRecallDisplay.from_predictions(y_test, skor, ax=eksenler[1], name="v3 Ensemble")
    eksenler[1].axhline(y_test.mean(), linestyle="--", color="gray", label="Pozitif sınıf oranı")
    eksenler[1].legend()
    RocCurveDisplay.from_predictions(y_test, skor, ax=eksenler[2], name="v3 Ensemble")
    eksenler[2].plot([0, 1], [0, 1], "--", color="gray")
    fig.tight_layout()

    # Tüm test tablolarını, grafikleri ve SHA256 doğrulama özetini diske kaydet
    ev_dir = exp_dir / ("degerlendirme_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid4().hex[:8])
    ev_dir.mkdir(exist_ok=False)
    sonuclar.to_csv(ev_dir / "test_metrikleri.csv", encoding="utf-8-sig")
    kapsam.to_csv(ev_dir / "kapsama.csv", index=False, encoding="utf-8-sig")
    hata_ozeti.to_csv(ev_dir / "hata_ozeti.csv", index=False, encoding="utf-8-sig")
    abone_sonuclari.to_csv(ev_dir / "test_tahminleri.csv", index=False, encoding="utf-8-sig")
    kapasite_tablosu.to_csv(ev_dir / "precision_at_k.csv", index=False, encoding="utf-8-sig")
    fig.savefig(ev_dir / "degerlendirme_grafikleri.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "deney": exp_dir.name,
        "model": info["secilen_model"],
        "skor_turu": "predict_proba",
        "esik": esik,
        "test_abonesi": len(hedefler),
        "skorlanan": len(test),
        "k_degerleri": k_degerleri,
        "model_sha256": sha256(model_path),
        "test_ozellik_sha256": sha256(processed / "test_ozellikleri_2016_v1.csv"),
        "test_hedef_sha256": sha256(processed / "test_hedefleri_2016_v1.csv"),
        "testte_model_ve_esik_secildi": False,
    }
    (ev_dir / "degerlendirme_bilgisi.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    rapor = (
        f"Seçilen model: {info['secilen_model']}\n"
        f"Test: {len(hedefler)} abone; skorlanan: {len(test)}; veri yetersiz: {len(hedefler)-len(test)}.\n"
        f"AP: {sonuclar.iloc[0]['AP']:.4f}; ROC-AUC: {sonuclar.iloc[0]['ROC-AUC']:.4f}; "
        f"F1: {sonuclar.iloc[0]['F1']:.4f}; Recall: {sonuclar.iloc[0]['Recall']:.4f}; "
        f"Precision: {sonuclar.iloc[0]['Precision']:.4f}.\n"
        f"Yanlış pozitif: {fp}; yanlış negatif: {fn}.\n"
        "Model ve eşik 5-fold eğitim içi OOF üzerinde kilitlendi; testte yeniden ayar yapılmadı.\n"
    )
    (ev_dir / "bulgular.txt").write_text(rapor, encoding="utf-8")
    print(f"[1/3] Değerlendirme kaydedildi: {ev_dir.name} ({time.time()-t0:.1f}s)", flush=True)

    # --- 2. Hızlı LightGBM TreeSHAP (En Yüksek Riskli İlk 2.000 Abone) ---
    t_shap = time.time()
    lgbm_est = model.named_estimators_["lgbm"]
    all_feat = pd.concat([tr_df[["CONS_NO"] + names], te_df[["CONS_NO"] + names]], ignore_index=True)
    all_scores = model.predict_proba(all_feat[names])[:, pos_col]
    top_idx = np.argsort(-all_scores)[:2000]
    top_df = all_feat.iloc[top_idx].reset_index(drop=True)

    # LightGBM'in yerleşik TreeSHAP motoru ile kesin Shapley katkılarını hesapla
    contribs = lgbm_est.booster_.predict(top_df[names], pred_contrib=True)
    shap_vals = contribs[:, :-1]
    base_val = float(np.mean(contribs[:, -1]))
    assert np.isfinite(shap_vals).all()

    shap_dir = exp_dir / ("shap_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid4().hex[:8])
    shap_dir.mkdir(exist_ok=False)

    sh_out = pd.DataFrame(shap_vals, columns=names)
    sh_out.insert(0, "CONS_NO", top_df["CONS_NO"].to_numpy())
    sh_out.to_csv(shap_dir / "orneklem_shap_katkilari.csv", index=False)

    # 1 numaralı en yüksek riskli abone için yerel özellik katkıları
    yerel_id = str(top_df.iloc[0]["CONS_NO"])
    yerel_df = pd.DataFrame({
        "ozellik": names,
        "deger": top_df.iloc[0][names].to_numpy(dtype=float),
        "shap_katkisi": shap_vals[0].astype(float),
    }).sort_values("shap_katkisi", key=lambda s: s.abs(), ascending=False)
    yerel_df.to_csv(shap_dir / "yerel_katkilar.csv", index=False)

    # Küresel özellik önemi (ilk 2.000 abone ortalama mutlak |SHAP| katkısı)
    global_df = pd.DataFrame({
        "ozellik": names,
        "ortalama_mutlak_shap": np.abs(shap_vals).mean(axis=0).astype(float),
        "ortalama_shap": shap_vals.mean(axis=0).astype(float),
    }).sort_values("ortalama_mutlak_shap", ascending=False)
    global_df.to_csv(shap_dir / "genel_ozellik_onemi.csv", index=False)

    # Küresel en etkili 20 özellik yatay çubuk grafiği
    fig2, ax = plt.subplots(figsize=(9, 6))
    top20 = global_df.head(20).iloc[::-1]
    ax.barh(top20["ozellik"], top20["ortalama_mutlak_shap"])
    ax.set_title("LightGBM TreeSHAP — En Etkili 20 Özellik (İlk 2.000 Riskli Abone)")
    ax.set_xlabel("Ortalama |SHAP| (log-odds)")
    fig2.tight_layout()
    fig2.savefig(shap_dir / "genel_onem.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    shap_meta = {
        "deney": exp_dir.name,
        "model": info["secilen_model"],
        "aciklayici": "LightGBM TreeSHAP (pred_contrib, ensemble ana bileşeni, ilk 2.000 yüksek riskli abone)",
        "skor_turu": "lgbm_tree_shap_log_odds",
        "temel_deger": base_val,
        "aciklanan_abone_sayisi": len(top_df),
        "yerel_abone": yerel_id,
        "model_sha256": sha256(model_path),
        "test_ozellik_sha256": sha256(processed / "test_ozellikleri_2016_v1.csv"),
    }
    (shap_dir / "aciklama_bilgisi.json").write_text(json.dumps(shap_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[2/3] Hızlı LightGBM TreeSHAP kaydedildi ({len(top_df)} abone, {time.time()-t_shap:.1f}s) -> {shap_dir.name}", flush=True)

    # --- 3. Tüm Skor, Tüketim ve TreeSHAP Çıktılarını catalog.sqlite3 Kataloğuna Aktar ---
    t_exp = time.time()
    export(exp_dir, root=ROOT)
    print(f"[3/3] Uygulama kataloğu aktarıldı ({time.time()-t_exp:.1f}s, toplam {time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()

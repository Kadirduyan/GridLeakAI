"""GridLeak AI — Uygulama Kataloğu (`catalog.sqlite3`) Dışa Aktarım Modülü.

Seçilen deneyin model dosyasını, 160 özellikli abone tablolarını, 2016 günlük ham tüketim
kayıtlarını, CV/test metriklerini ve hesaplanmış SHAP açıklamalarını SHA256 bütünlük
doğrulamasıyla tek bir salt-okunur SQLite kataloğunda birleştirir.

- Modeli yeniden eğitmez.
- Gerçek etiketler (`FLAG`) kesinlikle uygulama kataloğuna aktarılmaz.
Kullanım: python ai/evaluation/export_application.py --experiment iyilestirme_v3_20261123T171349Z
"""

from contextlib import closing
from pathlib import Path
import argparse
import hashlib
import json
import os
import sqlite3
import tempfile

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    """Dosyanın SHA256 bütünlük özetini (parmak izini) hesaplar."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(experiment, root=ROOT):
    """Seçilen deneyin modelini, skorlarını, SHAP çıktılarını ve tüketim verilerini catalog.sqlite3'e yazar."""

    # --- 1. Deney Bilgilerini Oku ve Dosya Bütünlüğünü (SHA256) Doğrula ---
    experiment = Path(experiment).resolve()
    info = json.loads((experiment / "deney_bilgisi.json").read_text(encoding="utf-8"))
    processed = root / info.get("processed_dir", "data/processed")

    assert digest(processed / "egitim_ozellikleri_2016_v1.csv") == info["egitim_csv_sha256"]
    assert digest(processed / "ozellik_tanimi_2016_v1.json") == info["ozellik_json_sha256"]

    config = info["ozellik_tanimi"]
    names = config["ozellikler"]
    model_path = experiment / info["secilen_pipeline"]
    model = joblib.load(model_path)

    # --- 2. Eğitim ve Test Abonelerinin Özellik ve Veri Kalitesi Tablolarını Birleştir ---
    frames = []
    quality = []

    for prefix, partition in [("egitim", "training"), ("test", "test")]:
        f = pd.read_csv(processed / f"{prefix}_ozellikleri_2016_v1.csv", dtype={"CONS_NO": str})
        f = f[["CONS_NO"] + names].copy()
        f["partition"] = partition
        frames.append(f)

        q = pd.read_csv(processed / f"{prefix}_veri_kalitesi_2016.csv", dtype={"CONS_NO": str})
        q["partition"] = partition
        quality.append(q)

    features = pd.concat(frames, ignore_index=True).set_index("CONS_NO")
    all_quality = pd.concat(quality, ignore_index=True).set_index("CONS_NO")

    assert features.index.is_unique and all_quality.index.is_unique
    assert set(features.index) <= set(all_quality.index)

    # --- 3. Tüm Skorlanabilir Aboneler İçin Risk Skoru, Tahmin ve Öncelik Sırası Üret ---
    X = features[names]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        scale = "decision_function"
    else:
        pos_idx = list(model.classes_).index(1)
        scores = model.predict_proba(X)[:, pos_idx]
        scale = "predict_proba_FLAG_1"

    assert np.isfinite(scores).all()

    threshold = info.get("decision_threshold", 0.0 if scale == "decision_function" else 0.5)
    features["score"] = scores
    features["prediction"] = (scores >= threshold).astype(int)

    # Kimlik (CONS_NO) ile eşit skor bağlarını deterministik kırıyoruz.
    ranking = features.reset_index().sort_values(
        ["score", "CONS_NO"],
        ascending=[False, True],
    )["CONS_NO"]
    ranks = {key: i + 1 for i, key in enumerate(ranking)}

    # --- 4. Uygulama Üst Verisi (Metadata) ve CV Tablosunu Hazırla ---
    dates = pd.date_range(config["donem_baslangic"], config["donem_bitis"], freq="D")

    meta = {
        "status": "connected",
        "source": "sgcc",
        "experiment": experiment.name,
        "model": info["secilen_model"],
        "score_type": scale,
        "decision_threshold": threshold,
        "model_sha256": digest(model_path),
        "total": len(all_quality),
        "scored": len(features),
        "features": names,
        "dates": [d.date().isoformat() for d in dates],
        "period_start": config["donem_baslangic"],
        "period_end": config["donem_bitis"],
        "scope": "Eğitim ve test aboneleri; performans yalnız ayrılmış test grubunda ölçülür.",
        "cv": pd.read_csv(experiment / "model_karsilastirma.csv").to_dict("records"),
        "metrics": [],
        "explanations": [],
        "limitations": info.get("sinirliliklar", []),
    }

    # --- 5. Kayıtlı Test Değerlendirme Metriklerini Doğrula ve Ekle ---
    evaluations = sorted(experiment.glob("degerlendirme_*/degerlendirme_bilgisi.json"))

    if evaluations:
        ep = evaluations[-1]
        ev = json.loads(ep.read_text(encoding="utf-8"))

        assert ev["model_sha256"] == meta["model_sha256"]
        assert ev["test_ozellik_sha256"] == digest(processed / "test_ozellikleri_2016_v1.csv")
        assert ev["test_hedef_sha256"] == digest(processed / "test_hedefleri_2016_v1.csv")

        metrics = pd.read_csv(ep.parent / "test_metrikleri.csv", index_col=0)

        # Kaydedilmiş test metriği ile aynı pipeline'ın yeniden ürettiği skoru karşılaştır.
        from sklearn.metrics import accuracy_score, average_precision_score

        labels = pd.read_csv(
            processed / "test_hedefleri_2016_v1.csv",
            dtype={"CONS_NO": str},
        ).set_index("CONS_NO")

        test_features = features.loc[features["partition"].eq("test")]
        truth = labels.loc[test_features.index, "FLAG"]

        np.testing.assert_allclose(
            average_precision_score(truth, test_features["score"]),
            metrics.loc[info["secilen_model"], "AP"],
            rtol=1e-8,
        )
        np.testing.assert_allclose(
            accuracy_score(truth, test_features["prediction"]),
            metrics.loc[info["secilen_model"], "Accuracy"],
            rtol=1e-8,
        )

        meta["metrics"] = [{"model": idx, **row.to_dict()} for idx, row in metrics.iterrows()]
        meta["evaluation"] = ev
        meta["precision_at_k"] = pd.read_csv(ep.parent / "precision_at_k.csv").to_dict("records")

    # --- 6. Hesaplanmış TreeSHAP Açıklamalarını Abone Kimliklerine Eşle ---
    explanations = {}

    for mp in sorted(experiment.glob("shap_*/aciklama_bilgisi.json")):
        si = json.loads(mp.read_text(encoding="utf-8"))
        if si["model_sha256"] != meta["model_sha256"]:
            continue

        assert si["test_ozellik_sha256"] == digest(processed / "test_ozellikleri_2016_v1.csv")

        sh = pd.read_csv(
            mp.parent / "orneklem_shap_katkilari.csv",
            dtype={"CONS_NO": str},
        ).set_index("CONS_NO")

        for key, row in sh.iterrows():
            explanations[key] = {
                "base_value": si["temel_deger"],
                "score_type": si["skor_turu"],
                "features": [
                    {"feature": name, "contribution": float(row[name])}
                    for name in names
                ],
            }

        local = pd.read_csv(mp.parent / "yerel_katkilar.csv")
        explanations[si["yerel_abone"]] = {
            "base_value": si["temel_deger"],
            "score_type": si["skor_turu"],
            "features": [
                {"feature": r["ozellik"], "contribution": float(r["shap_katkisi"])}
                for _, r in local.iterrows()
            ],
        }
        meta["explanations"] = pd.read_csv(mp.parent / "genel_ozellik_onemi.csv").to_dict("records")

    # --- 7. Ham 2016 Tüketim Kayıtlarıyla Birlikte Atomik SQLite Kataloğunu Oluştur ---
    destination = root / "data/application"
    destination.mkdir(parents=True, exist_ok=True)

    fd, tmp = tempfile.mkstemp(suffix=".sqlite3", dir=destination)
    os.close(fd)

    try:
        with closing(sqlite3.connect(tmp)) as db:
            db.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            db.execute(
                """
                CREATE TABLE subscribers (
                    id TEXT PRIMARY KEY,
                    score REAL,
                    rank INTEGER,
                    quality REAL,
                    recorded INTEGER,
                    total INTEGER,
                    partition TEXT,
                    prediction INTEGER,
                    points TEXT,
                    features TEXT,
                    explanation TEXT
                )
                """
            )
            db.execute("CREATE INDEX rank_order ON subscribers(rank)")
            db.execute(
                "INSERT INTO metadata VALUES (?, ?)",
                ("catalog", json.dumps(meta, ensure_ascii=False, allow_nan=False)),
            )

            # Ham tüketim dosyasını 1.000'er satırlık parçalar halinde okuyup SQLite'a yaz (FLAG okunmaz).
            raw = root / "data/raw/data.csv"
            columns = pd.read_csv(raw, nrows=0).columns
            date_cols = {
                c: pd.to_datetime(c, format="%Y/%m/%d")
                for c in columns
                if c not in ["CONS_NO", "FLAG"]
            }
            date_cols = {
                c: d for c, d in date_cols.items() if dates.min() <= d <= dates.max()
            }

            seen = set()
            reader = pd.read_csv(
                raw,
                usecols=["CONS_NO"] + list(date_cols),
                dtype={"CONS_NO": str},
                chunksize=1000,
            )

            for chunk in reader:
                chunk = chunk.set_index("CONS_NO")
                chunk = chunk.loc[chunk.index.intersection(all_quality.index)]
                chunk = chunk.rename(columns=date_cols).reindex(columns=dates)

                rows = []
                for key, row in chunk.iterrows():
                    assert key not in seen
                    seen.add(key)

                    values = [None if pd.isna(v) else float(v) for v in row]
                    recorded = sum(v is not None for v in values)
                    f = features.loc[key] if key in features.index else None

                    feat_json = (
                        json.dumps(
                            {name: None if pd.isna(f[name]) else float(f[name]) for name in names},
                            allow_nan=False,
                        )
                        if f is not None
                        else "{}"
                    )

                    rows.append(
                        (
                            key,
                            None if f is None else float(f["score"]),
                            ranks.get(key),
                            round(recorded / len(dates) * 100, 2),
                            recorded,
                            len(dates),
                            all_quality.loc[key, "partition"],
                            None if f is None else int(f["prediction"]),
                            json.dumps(values, allow_nan=False),
                            feat_json,
                            json.dumps(explanations.get(key), allow_nan=False),
                        )
                    )

                db.executemany(
                    "INSERT INTO subscribers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )

            assert seen == set(all_quality.index), "Ham tüketim kayıtlarında eksik abone var."
            db.commit()

        # Geçici veritabanını tek adımda canlı catalog.sqlite3 üzerine taşı.
        os.replace(tmp, destination / "catalog.sqlite3")
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

    print("Uygulama kataloğu hazır:", destination / "catalog.sqlite3", "Abone:", len(seen))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()
    export(ROOT / "ai/artifacts" / args.experiment)

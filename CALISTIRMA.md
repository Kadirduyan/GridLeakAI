# GridLeak AI — Çalıştırma Rehberi

## 1. Sunumda Uygulamayı Başlatma (Kopyala / Yapıştır)

Jupyter Terminal veya Windows PowerShell'e aşağıdaki satırı yapıştırıp Enter'a basın. Terminalin hangi klasörde açık olduğu önemli değildir:

```powershell
& "C:\Users\kadir\GridLeakAI\START.cmd"
```

`Uvicorn running on http://127.0.0.1:8000` mesajını gördükten sonra tarayıcıdan açın:

**Uygulama:** http://127.0.0.1:8000/#/review

**API dokümantasyonu:** http://127.0.0.1:8000/docs

- Bu komut backend'i ve hazırlanmış frontend arayüzünü sunar; modeli yeniden eğitmez. Sunum için aşağıdaki eğitim komutlarını çalıştırmanız gerekmez.
- START.cmd tarayıcıyı otomatik açmaz; yukarıdaki uygulama bağlantısını açın.
- Kullanırken sunucunun çalıştığı terminal açık kalsın. Durdurmak için Ctrl+C kullanın.
- `START.cmd tanınmıyor` hatası: Yalnızca dosya adını yazmayın; yukarıdaki tam komutu kullanın. Proje klasöründeyseniz ` .\START.cmd ` de çalışır.
- `10048 / port already in use` hatası: 8000 portu zaten kullanılıyor. Uygulama bağlantısı açılıyorsa tekrar başlatmayın; mevcut sunucuyu kullanın.
- Başka bilgisayarda tam dosya yolunu o bilgisayardaki proje konumuyla değiştirin.

## 2. AI Modelini Yeniden Üretme ve Katalog Aktarımı
```powershell
# AI bağımlılıkları
python -m pip install -r requirements.txt
python -m pip install lightgbm catboost optuna

# Final v3 modelini yeni tohumlu 5-fold OOF ile yeniden eğitme
python ai/training/v3_final_train_and_test.py --processed-output data/processed/v3_rebuild

# Kayıtlı final modeli yeniden eğitmeden değerlendirme, TreeSHAP (ilk 2.000 abone) ve katalog aktarımı
python ai/evaluation/v3_export_and_shap.py
```

Açıklamalı akış, canlı SMOTE kıyaslaması, TreeSHAP ve test grafikleri için `ai/notebooks/01_data_overview_and_eda.ipynb` ile başlayan beş defteri inceleyebilirsiniz.

## 3. Testler ve Frontend Derlemesi
```powershell
# 7 backend API + canlı katalog testi
backend\.venv\Scripts\python.exe -m pytest backend\test_api.py backend\test_live_catalog.py -v

# Frontend TypeScript kontrolü ve üretim derlemesi
cd frontend
npm run build
```

## 4. Eksik Araştırma Aşamalarını Yeniden Üretme

Bu işlemler sunumda uygulamayı açmak için gerekli değildir. Yeni kodlar final
modeli/kataloğu güncellemez; eski deneylerin birebir Optuna kayıtları değildir.

```powershell
# Proje kökünde, AI bağımlılıklarının kurulu olduğu Python ortamında:
python -m pip install -r requirements.txt

# Ham CSV'den bölme ve 49 özellik. Yeni klasör gerekir.
# --include-test yalnız sabit hazırlığı uygular; test başarısı hesaplamaz.
python ai/features/prepare_data.py --output data/processed/rebuild_49 --include-test

# Kat-içi bağlam/SVD + özellik seçimi + dört model Optuna + SMOTE
python ai/training/research_pipeline.py --output ai/artifacts/research_new --mode all --trials 20

# Yalnız kısa yazılım doğrulaması (performans raporu olarak kullanılmaz):
python ai/training/research_pipeline.py --output ai/artifacts/research_smoke --mode all --limit 300 --folds 2 --trials 2 --top-k 20 --smoke

# Orijinal sabit özellik/parametre sözleşmesiyle yeniden eğitim isteğe bağlıdır.
# Bu komut uzun sürer ve sonunda test değerlendirmesi yapar; ayar aramak için çalıştırmayın.
python ai/training/v3_final_train_and_test.py --prepared-dir data/processed/rebuild_49 --processed-output data/processed/v3_rebuild
```

Araştırma çıktıları: `fold_audit.json` (katların kimlikleri), `manifest.json`
(sürümler/bölme özeti), `feature_ranking.csv`, `selected_features.json`,
`selection_comparison.csv`, model başına `*_optuna_trials.csv`, `best_params.json`,
`tuning_metrics.csv`, `smote_comparison.csv` ve OOF tahmin dizileri.
Seçim sonrası OOF skorları geliştirme sonucudur; tamamen bağımsız test sonucu değildir.
Yeni modelin uygulamaya alınması bu komutların parçası değildir.

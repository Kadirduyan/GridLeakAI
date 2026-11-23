# GridLeak AI

## Aktif proje

Bu proje `C:\Users\kadir\GridLeakAI` konumundadır. Masaüstü kopyasıyla eşitlendi; sonraki değişiklikler iki kopya arasında otomatik eşitlenmez. Bu konumdan devam edebilirsiniz.
Jupyter dosya listesinde **GridLeakAI → ai → notebooks** yolunu açın.

## Beş çalışma defteri

1. `01_data_overview_and_eda.ipynb`: veri tanıtımı, kalite, grafikler ve bulgular.
2. `02_preprocessing_and_feature_engineering.ipynb`: bölme, takvim, yedi özellik, eğitim/test hazırlığı ve kayıtlar.
3. `03_model_training_and_tuning.ipynb`: sonraki çalışma; henüz model eğitilmedi.
4. `04_model_evaluation.ipynb`: sonraki değerlendirme.
5. `05_model_explainability_shap.ipynb`: sonraki açıklanabilirlik.

01 ve 02 yukarıdan aşağıya bağımsız kernel ile çalıştırılabilir. 02'de tanımlanan Pipeline henüz fit edilmez.
Ham veri değiştirilmez. Çıktı kaydetme hücreleri aynı adlı üretilmiş dosyaları günceller.
03 eğitim dosyasını kullanır; test hedefleri yalnız son değerlendirme içindir.

## Klasörlerin görevi

- `ai/notebooks`: bütün akademik ML kodları ve açıklamaları; ayrı features/training klasörü yok.
- `ai/artifacts`: ileride eğitilen model ve öğrenilmiş dönüşümler.
- `backend`: main.py, prediction.py, schemas.py, database.py, inspections.py; henüz boş taslaklar.
- `frontend/src/pages`, `components`, `assets`: ekranlar, ortak bileşenler, görseller. api.ts ve styles.css doğrudan src içinde.
- `frontend/public`: sabit dosyalar. Arayüz henüz çalışır uygulama değildir.
- `data/raw`: özgün CSV.
- `data/processed`: eğitim/test özellikleri, test hedefleri, kalite listeleri ve özellik tanımı.
- `data/splits`: sabit abone bölmesi.
- `data/application`: ileride uygulama veritabanı.
- `reports/figures`, `metrics`, `final`: grafik, başarı tabloları ve teslim belgeleri.
- `docs`: yönerge ve proje belgeleri. Eski PDF'lerde eski mimari bulunabilir; güncel klasör haritası klasor_ve_dosya_yapisi.md'dir.
- `tests`: gerektiğinde yazılım doğruluk kontrolleri; ML test kümesi değildir.

## Akademik sınırlar

İlk EDA bütün veride yapıldı; bu sınırlılık raporda yer almalıdır. Sonraki yöntem seçimleri yalnız eğitim/CV içinde yapılır.
39 eğitim ve testteki veri yetersiz aboneler ayrı tutulur; kapsam oranı son raporda belirtilir.
2016 dönemi ilk denemedir; daha uzun dönem karşılaştırması henüz yapılmadı.
Resmi teslim yönergesindeki Word raporu ve kod ZIP'ine veri seti eklenmez.


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

# Final v3 modelini bağımsız 5-fold OOF doğrulamasıyla eğitme
python ai/training/v3_final_train_and_test.py

# Kayıtlı final modeli yeniden eğitmeden değerlendirme, TreeSHAP (ilk 2.000 abone) ve katalog aktarımı
python ai/evaluation/v3_export_and_shap.py
```

Açıklamalı akış, canlı SMOTE kıyaslaması, TreeSHAP ve test grafikleri için `ai/notebooks/01_Veri_Kesfi_ve_EDA.ipynb` ve `ai/notebooks/02_GridLeakAI_v3_Tam_Akis_Model_SMOTE_SHAP.ipynb` defterlerini inceleyebilirsiniz.

## 3. Testler ve Frontend Derlemesi
```powershell
# 7 backend API + canlı katalog testi
backend\.venv\Scripts\python.exe -m pytest backend\test_api.py backend\test_live_catalog.py -v

# Frontend TypeScript kontrolü ve üretim derlemesi
cd frontend
npm run build
```

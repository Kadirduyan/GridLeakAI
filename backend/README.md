# GridLeak backend — yerel geliştirme sürümü

Bu servis frontend ile gelecekteki model arasındaki veri ve kayıt katmanıdır.
Yalnızca üç **sentetik örnek abone** sunar. Gerçek SGCC dosyalarını okumaz,
AI kodunu çalıştırmaz ve model skoru/öncelik sırası uydurmaz: bu alanlar `null` döner.
Frontend bu API’ye bağlıdır; kayıtlar SQLite içinde tutulur. Proje kökündeki START.cmd ile açılır. Sunucu derlenmiş frontend/dist klasörünü de sunar.

## Sade dosya düzeni

| Dosya | Görevi |
|---|---|
| main.py | FastAPI uygulaması, bütün API adresleri, başlangıç ve CORS ayarları |
| schemas.py | Gelen JSON alanlarını, not uzunluğunu, kapasiteyi ve geçerli durumları doğrular |
| subscribers.py | Etiketli örnek abone verileri ve günlük tüketim serisi; gelecekte gerçek veri sağlayıcısıyla değiştirilecek |
| inspections.py | Kapasite, kayıt yeterliliği, plana ekleme/çıkarma, analist notu ve saha sonucu kuralları |
| database.py | SQLite bağlantısı ve tablo oluşturma; SQL sorguları parametreli çalışır |
| test_api.py | Geçici veritabanında davranış testleri |
| requirements.txt | Backend için gerekli Python paketleri |
| __init__.py | backend dizinini Python paketi olarak tanımlar |
| storage/gridleak_demo.sqlite3 | İlk açılışta otomatik oluşan kalıcı örnek kayıt veritabanı |

Ek api/services/database klasörleri yok: görevler dosya düzeyinde ayrıldı.
Boş prediction.py kaldırıldı; eğitimli model entegrasyonu bu sürümün kapsamında değil.

## Kurulum ve açılış

Anaconda Prompt'u açın. Aktif proje için:

```bat
cd /d C:\Users\kadir\GridLeakAI
python -m venv backend\.venv
backend\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
backend\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Masaüstü kopyasını çalıştırmak için ilk satırdaki yolu `C:\Users\kadir\Desktop\GridLeakAI` yapın.
İki kopyayı aynı portta aynı anda çalıştırmayın. İki kopyanın veritabanları birbirinden bağımsızdır.
Sunucuyu kapatmak: Ctrl+C. Yeniden açıldığında kayıtlar korunur.
İsteğe bağlı `GRIDLEAK_DB_PATH` ortam değişkeni SQLite dosyasının yerini değiştirebilir.
Üretim kimlik doğrulaması yok; bu sürümü yalnızca 127.0.0.1 üzerinde yerel kullanım için çalıştırın.

- Etkileşimli API belgesi: http://127.0.0.1:8000/docs
- Sağlık kontrolü: http://127.0.0.1:8000/api/health
- API sözleşmesi: http://127.0.0.1:8000/openapi.json

## API nedir, nereden geldi?

API adresleri başka bir yerden satın alınmadı veya indirilmedi: GridLeak için bizim yazdığımız
Python fonksiyonlarıdır. FastAPI bu fonksiyonlara HTTP üzerinden erişim sağlar.
`GET` okur; `PUT` kaydeder/günceller; `DELETE` plandan çıkarır.
Adreslerin hepsi **main.py** dosyasındaki `@app.get/put/delete` satırlarında tanımlıdır.

| Yöntem / adres | İşlev |
|---|---|
| GET /api/health | Servis/veritabanı ve model bağlantısı durumu |
| GET /api/subscribers?q=104&offset=0&limit=50 | Arama ve sayfalama ile örnek aboneler |
| GET /api/subscribers/{id} | Kayıt doluluğu ve abone özeti; model skoru null |
| GET /api/subscribers/{id}/consumption?days=90 | 1–365 günlük seri ve 7 günlük mevcut kayıt ortalaması |
| GET /api/model/status | Modelin henüz bağlı olmadığını belirtir |
| GET /api/state | Kapasite, plan, analist notları ve sonuçlar |
| GET /api/plan | Kapasite ve plandaki aboneler |
| PUT /api/plan/capacity | Gövde: {"capacity":20} |
| PUT /api/plan/{id} | Elle plana ekler; gövde gerekmez; tekrar ekleme mükerrer kayıt üretmez |
| DELETE /api/plan/{id} | Yalnızca plandan çıkarır; not ve sonuçları korur |
| PUT /api/subscribers/{id}/note | Gövde: {"note":"Sayaç kontrolü planlandı."} |
| GET /api/results | Kayıtlı saha sonuçları |
| PUT /api/results/{id} | Gövde: {"status":"İnceleme sürüyor","note":"Örnek saha notu"} |

Örnek id: AB-104829. Swagger /docs sayfasında adresi açıp **Try it out → Execute** ile deneyebilirsiniz.
PUT/DELETE gerçek yerel SQLite kayıtlarını değiştirir. Saha sonucu için önce plana eklemek gerekir;
plandan çıkarılmış ama daha önce sonuç kaydedilmiş bir abonenin sonucu güncellenebilir.

## Kurallar ve veri anlamı

- Veri kaynağı yanıtlarda `synthetic_demo` olarak belirtilir.
- Kayıt doluluğu = mevcut kayıt / 365 × 100. Ölçüm doğruluğunun garantisi değildir.
- Tüketimde `null` eksik kayıt, `0` ölçülmüş sıfırdır. Eksikler sıfıra çevrilmez.
- İlk sürümde %90 kayıt doluluğu örnek bir planlama kuralıdır; bilimsel olarak seçilmiş eşik değildir.
- Kapasite mevcut plan sayısından aşağı indirilemez. Doluyken ekleme reddedilir.
- Model bağlı olmadığı için otomatik sıralama veya otomatik aday seçimi yapılmaz.
- Geçerli saha durumları: Planlandı, İnceleme sürüyor, Kaçak doğrulandı, Kaçak bulunmadı, Ek inceleme gerekli.
- Saha sonucu hiçbir eğitim etiketini değiştirmez, modeli yeniden eğitmez.
- Zaman damgaları UTC ve saat dilimi bilgisiyle tutulur.
- 404: abone yok; 409: iş kuralı engeli; 422: geçersiz giriş alanı/değeri.

## Test

Proje kökünden:

```bat
backend\.venv\Scripts\python.exe -m pytest backend\test_api.py -q
```

Testler gerçek kayıt dosyasını kullanmaz; geçici veritabanlarında çalışır.

## Kullandığımız hazır araçlar

- FastAPI: HTTP servisi ve otomatik API belgesi — https://fastapi.tiangolo.com/
- Pydantic: JSON alan/tür doğrulama — https://docs.pydantic.dev/
- Uvicorn: Python API sunucusunu çalıştırır — https://www.uvicorn.org/
- SQLite: Python sqlite3 modülüyle kullanılan yerel veritabanı; ayrı sunucu kurulmaz — https://docs.python.org/3/library/sqlite3.html
- pytest ve HTTPX: otomatik API testleri. FastAPI test rehberi: https://fastapi.tiangolo.com/tutorial/testing/

Paketler `pip` ile PyPI'dan kurulur. API adresleri, iş kuralları ve örnek veri bizim proje kodumuzdur.

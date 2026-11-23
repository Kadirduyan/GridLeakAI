# GridLeak — AI dışındaki uygulama akışı

## Açılış

1. Proje ana klasöründeki **START.cmd** dosyasını çift tıklayın.
2. İlk açılış backend/.venv içinde bağımsız Python ortamı oluşturur ve requirements.txt paketlerini PyPI'dan kurar; internet gerekir. Anaconda ortamınız değiştirilmez.
3. Terminalde sunucu başladı mesajını gördüğünüzde **http://127.0.0.1:8000** adresini açın.
4. Terminal açık kalmalıdır. Durdurmak için Ctrl+C.

Hazır dist çıktısı projeye eklendi; normal kullanımda Node veya ayrı frontend sunucusu gerekmez.
Python için önce kullanıcı klasöründeki anaconda3/python.exe, yoksa PATH'teki python kullanılır.
Masaüstü ve kullanıcı klasöründeki projelerin kaynakları aynıdır, veritabanları bağımsızdır. Bir kopya seçip onunla çalışın; iki kopyayı aynı portta aynı anda açmayın.

## Tamamlanan akış

Tarayıcı → frontend/src/api.ts → backend/main.py → iş kuralları → SQLite → tarayıcı.

- Abone listesi, arama, tüketim grafiği ve veri doluluğu.
- Elle denetim planı; kapasite sınırı ve veri yeterliliği kontrolü.
- Analist notları ve saha sonuçları için sunucuda kalıcı kayıt.
- Yeniden açılışta kayıtların geri yüklenmesi; plan dışına çıkan abonenin sonucunun korunması.
- CSV indirme, boş ekranlar, bağlantı ve kayıt hataları, tekrar deneme.
- Masaüstü ve mobil ekran düzeni.

## Henüz bağlanmayan AI

AI defterleri ve artifacts değiştirilmedi. Veri sağlayıcısı üç **sentetik** abonedir; gerçek SGCC veri dosyası kullanılmaz. Gerçek veri/model entegrasyonu AI çalışması ve onun veri sözleşmesi tamamlanınca yapılacak.
Model skoru, otomatik sıra ve SHAP sonuçları yok; arayüzde boş/bekliyor gösterilir. Önceki tasarımda bulunan 87 gibi örnek skorlar artık gerçek sonuçmuş gibi gösterilmez.
Model hazır olana kadar otomatik plan oluşturulmaz. %90 veri doluluğu sadece örnek bir iş kuralıdır.

## Dosyalar

- frontend/src/screens: dört ana ekran.
- frontend/src/shared-components: kartlar, abone listesi, menü ve grafik.
- frontend/src/api.ts: API istekleri, yanıtları arayüz biçimine dönüştürme, zaman aşımı/hata yönetimi.
- frontend/src/data.ts: tipler, ekran metinleri, biçimlendirme ve CSV yardımcıları. Artık örnek abonelerin kaynağı değildir.
- frontend/src/App.tsx: veri yükleme, sayfa seçimi ve sunucuda kayıt işlemlerini koordine eder.
- backend/main.py: bütün /api adresleri; derlenmiş frontend'i de sunar.
- backend/subscribers.py: sentetik abone/tüketim sağlayıcısı.
- backend/inspections.py: denetim ve kayıt kuralları.
- backend/schemas.py: gelen JSON doğrulaması.
- backend/database.py: SQLite bağlantısı ve tablo kurulumu.
- backend/storage/gridleak_demo.sqlite3: kalıcı kayıtlar. Silmeyin; yedeklemek için servis kapalıyken dosyayı kopyalayın.
- backend/test_api.py: geçici veritabanlarıyla testler.

## API adresleri

Sunucu çalışırken http://127.0.0.1:8000/docs adresi interaktif belgedir. Try it out / Execute üzerinden örnek istek yapılabilir. PUT/DELETE işlemleri yerel kayıtları değiştirir.

GET /api/subscribers, GET /api/subscribers/{id}/consumption,
GET /api/state, GET /api/plan, PUT /api/plan/capacity,
PUT veya DELETE /api/plan/{id}, PUT /api/subscribers/{id}/note,
GET /api/results, PUT /api/results/{id}, GET /api/model/status.

API'ler bizim Python kodumuzdur; hazır bir dış hizmetten alınmadı. FastAPI, Pydantic ve Uvicorn hazır kütüphanelerdir; SQLite Python ile kullanılır.

## Geliştirme

Frontend kaynak kodu değiştiğinde frontend klasöründe `npm run build` ile dist'i güncelleyin.
İsteğe bağlı `npm run dev` geliştirme sunucusu 5173 portundan /api isteklerini 8000 portuna yönlendirir; backend açık olmalıdır.
Test: proje kökünde `backend\.venv\Scripts\python.exe -m pytest backend/test_api.py -q`.

Bu sürüm yerel tek kullanıcılıdır; kullanıcı girişi/rol sistemi içermez, internete yayın yapılmadı.
Eski frontend localStorage kayıtları veritabanına otomatik aktarılmaz; yeni kayıtların kaynağı SQLite'tır.

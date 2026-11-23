# GridLeak AI

Elektrik tüketimi analizi ve denetim platformu. React/TypeScript arayüzü, FastAPI backend'i ve Jupyter makine öğrenmesi çalışmaları içerir.

## Durum

Yerel uygulamada abone görüntüleme, denetim planı, kapasite kontrolü, analist notları, saha sonuçları ve CSV dışa aktarma çalışır. Kayıtlar SQLite'ta saklanır.

**Uygulama üç sentetik abone kullanır. Gerçek SGCC verisi ve eğitimli model henüz uygulamaya bağlı değildir.** AI notebook çalışmaları devam etmektedir; model skoru ve SHAP alanları bağlantı bekler.

## GitHub'dan kurulum

Python, Node.js/npm ve Git gereklidir. Windows'ta:

```powershell
git clone https://github.com/Kadirduyan/GridLeakAI.git
cd GridLeakAI
cd frontend
npm ci
npm run build
cd ..
.\START.cmd
```

START.cmd ilk açılışta backend için ayrı Python ortamı oluşturur ve paketleri kurar. Ardından http://127.0.0.1:8000 adresini açın; terminal açık kalmalıdır. API belgesi: http://127.0.0.1:8000/docs.

## Dosya düzeni

| Konum | Görev |
| --- | --- |
| ai/notebooks | Veri analizi, ön işlem, eğitim, değerlendirme ve açıklanabilirlik defterleri |
| backend/main.py | API adresleri ve derlenmiş arayüzün sunulması |
| backend/subscribers.py | Sentetik abone ve tüketim sağlayıcısı |
| backend/inspections.py | Denetim planı, not ve saha sonucu işlemleri |
| backend/schemas.py | İstek doğrulama |
| backend/database.py | SQLite bağlantısı ve tablolar |
| backend/test_api.py | Otomatik API ve statik arayüz testleri |
| frontend/src/screens | Dört ana ekran |
| frontend/src/shared-components | Ortak arayüz bileşenleri |
| frontend/src/api.ts | Backend bağlantısı |
| docs | Proje belgeleri; eski belgeler önceki tasarımları içerebilir |
| reports/figures | Analiz grafikleri |

Ayrıntılar: [Çalıştırma rehberi](CALISTIRMA.md) ve [backend rehberi](backend/README.md).

## Yerel dosyalar

Veri setleri, üretilmiş eğitim tabloları, model dosyaları, SQLite kayıtları, sanal ortamlar, node_modules ve dist Git'e dahil değildir. Boş klasörler Git tarafından saklanmaz.

AI çalışmaları için SGCC veri seti ayrıca edinilip `data/raw/data.csv` konumuna konmalıdır. Notebook'lardaki yerel proje yollarını kendi bilgisayarınıza göre ayarlayın. Uygulama demosu veri seti olmadan çalışır.

## Kontrol ve sürüm takibi

Frontend değişikliklerinden sonra `frontend` içinde `npm run build` çalıştırın.

```powershell
backend\.venv\Scripts\python.exe -m pytest backend/test_api.py -q
git status
git add <degisen-dosyalar>
git commit -m "Yapılan değişikliği açıklayın"
git push
```

Commit yerel geçmişe kaydeder; push bu commit'leri GitHub'a gönderir. İki yerel proje kopyası otomatik eşitlenmez; Git ile takip edilen ana çalışma klasörü `C:\Users\kadir\GridLeakAI` konumudur.

## Akademik sınırlar

İlk EDA bütün veride yapıldı; bu sınırlılık raporda belirtilmelidir. Sonraki yöntem seçimleri yalnız eğitim/çapraz doğrulama içinde yapılır. Veri yetersiz aboneler için kapsam oranı raporlanmalıdır. 2016 dönemi ilk denemedir; nihai model başarısı henüz raporlanmamıştır.

Yerel tek kullanıcı sürümüdür; kullanıcı girişi ve rol yönetimi içermez.

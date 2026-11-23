# GridLeak frontend

React + TypeScript arayüzü FastAPI backend’e bağlıdır. Proje kökündeki START.cmd ile http://127.0.0.1:8000 üzerinden açılır.

## Kaynak dosyaları

- src/screens: dört ana ekran.
- src/shared-components: ortak kartlar, grafik ve menü.
- src/api.ts: sunucu istekleri, zaman aşımı ve hatalar.
- src/data.ts: veri tipleri, ekran metinleri ve CSV yardımcıları.
- src/App.tsx: yükleme, gezinme ve kayıt işlemleri.
- src/styles.css: görünüm ve mobil düzen.
- src/main.tsx: React başlangıcı.
- favicon.svg: sekme simgesi.
- dist: derlenmiş frontend; backend tarafından sunulur.
- node_modules: kurulu geliştirme araçları.

## Geliştirme

npm install
npm run dev

Backend 8000 portunda açık olmalı; Vite /api isteklerini oraya yönlendirir.
Üretim çıktısı: npm run build. Sadece statik dosyayı çift tıklamak API’yi çalıştırmaz.

## Kayıt ve kapsam

Aboneler backend’den gelir ve sentetiktir. Model bağlı değildir; skor, sıra ve SHAP etkileri bekliyor durumundadır.
Analist notu, plan, kapasite ve saha sonucu SQLite’a API üzerinden kaydedilir.
Başarı mesajı sunucu yanıtından sonra gösterilir; bağlantı hatasında yerel veriyle sessiz devam edilmez.
Kaydetmeden ekran/abone değiştirmek taslak notları kaybettirebilir.
Eski localStorage kayıtları otomatik taşınmaz.
Ayrıntılar: proje kökündeki CALISTIRMA.md ve backend/README.md.

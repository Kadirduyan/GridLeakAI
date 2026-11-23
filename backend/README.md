# Güncel yapı
Güncel uçtan uca akış ve dosya görevleri proje kökündeki README.md ve CALISTIRMA.md içinde açıklanır.
Backend `data/application/catalog.sqlite3` dosyasını salt okunur açar. Katalog yoksa 503 döndürür, sentetik sonuç üretmez. Analist kayıtları `backend/storage/gridleak.sqlite3` içinde kalıcıdır.

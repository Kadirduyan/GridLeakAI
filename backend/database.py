"""GridLeak AI — Kalıcı Analist Veritabanı Bağlantısı (`gridleak.sqlite3`).

Denetim kapasitesi (`settings`), denetim planı (`plan`), analist notları (`notes`) ve
saha sonuçları (`results`) tablolarını başlatır; yazma işlemlerinde yarış durumunu önlemek
için `BEGIN IMMEDIATE` kilidiyle atomik işlem (transaction) sağlar.
"""
import sqlite3
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def connection(path: Path, write=False):
    db = sqlite3.connect(path, timeout=15)
    db.row_factory = sqlite3.Row
    try:
        if write:
            # Kapasite kontrolü ve ekleme aynı kilitli işlemde gerçekleşir.
            db.execute("BEGIN IMMEDIATE")
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def initialize(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with connection(path) as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS settings (
            id INTEGER PRIMARY KEY CHECK(id=1),
            capacity INTEGER NOT NULL CHECK(capacity BETWEEN 1 AND 1000)
        );
        INSERT OR IGNORE INTO settings VALUES (1,20);
        CREATE TABLE IF NOT EXISTS plan (
            subscriber_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS notes (
            subscriber_id TEXT PRIMARY KEY,
            note TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS results (
            subscriber_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            note TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """)

"""GridLeak AI — Denetim Planı, Analist Notu ve Saha Sonucu İş Kuralları.

- Kapasite sınırını (`settings.capacity`) ve kayıt doluluğu kontrolünü (`>= %90`) uygular.
- Plana ekleme/çıkarma, analist notu kaydetme ve saha denetim sonucunu kalıcı SQLite'a yazar.
"""
from datetime import datetime, timezone
from fastapi import HTTPException
from .database import connection
from .subscribers import SOURCE


def now():
    return datetime.now(timezone.utc).isoformat()


def state(path):
    with connection(path) as db:
        capacity = db.execute("SELECT capacity FROM settings WHERE id=1").fetchone()[0]
        plan = [dict(row) for row in db.execute("SELECT * FROM plan ORDER BY created_at, subscriber_id")]
        notes = [dict(row) for row in db.execute("SELECT * FROM notes ORDER BY subscriber_id")]
        results = [dict(row) for row in db.execute("SELECT * FROM results ORDER BY subscriber_id")]
    return {"capacity":capacity,"plan":plan,"notes":notes,"results":results,"source":SOURCE}


def set_capacity(path, capacity):
    with connection(path, write=True) as db:
        count = db.execute("SELECT COUNT(*) FROM plan").fetchone()[0]
        if capacity < count:
            raise HTTPException(409, "Kapasite mevcut plan sayısından küçük olamaz. Önce plandan abone çıkarın.")
        db.execute("UPDATE settings SET capacity=? WHERE id=1", (capacity,))
    return {"capacity":capacity}


def add_to_plan(path, subscriber_id, catalog):
    record = catalog.summary(subscriber_id)
    if record["data_quality_percent"] < 90:
        raise HTTPException(409, "Örnek iş kuralı: kayıt doluluğu %90 altında; önce veri kontrolü gerekli.")
    with connection(path, write=True) as db:
        if db.execute("SELECT 1 FROM plan WHERE subscriber_id=?", (subscriber_id,)).fetchone():
            return {"subscriber_id":subscriber_id,"status":"already_planned"}
        capacity = db.execute("SELECT capacity FROM settings WHERE id=1").fetchone()[0]
        count = db.execute("SELECT COUNT(*) FROM plan").fetchone()[0]
        if count >= capacity:
            raise HTTPException(409, "Denetim kapasitesi dolu.")
        db.execute("INSERT INTO plan VALUES (?,?)", (subscriber_id,now()))
    return {"subscriber_id":subscriber_id,"status":"planned"}


def remove_from_plan(path, subscriber_id, catalog):
    catalog.require_subscriber(subscriber_id)
    with connection(path, write=True) as db:
        db.execute("DELETE FROM plan WHERE subscriber_id=?", (subscriber_id,))
    return {"subscriber_id":subscriber_id,"status":"removed","saved_results_preserved":True}


def save_note(path, subscriber_id, note, catalog):
    catalog.require_subscriber(subscriber_id)
    timestamp = now()
    with connection(path, write=True) as db:
        db.execute("INSERT INTO notes VALUES (?,?,?) ON CONFLICT(subscriber_id) DO UPDATE SET note=excluded.note,updated_at=excluded.updated_at", (subscriber_id,note,timestamp))
    return {"subscriber_id":subscriber_id,"note":note,"updated_at":timestamp}


def save_result(path, subscriber_id, status, note, catalog):
    catalog.require_subscriber(subscriber_id)
    timestamp = now()
    with connection(path, write=True) as db:
        planned = db.execute("SELECT 1 FROM plan WHERE subscriber_id=?", (subscriber_id,)).fetchone()
        saved = db.execute("SELECT 1 FROM results WHERE subscriber_id=?", (subscriber_id,)).fetchone()
        if not planned and not saved:
            raise HTTPException(409, "Saha sonucu kaydetmeden önce aboneyi plana ekleyin.")
        db.execute("INSERT INTO results VALUES (?,?,?,?) ON CONFLICT(subscriber_id) DO UPDATE SET status=excluded.status,note=excluded.note,updated_at=excluded.updated_at", (subscriber_id,status,note,timestamp))
    return {"subscriber_id":subscriber_id,"status":status,"note":note,"updated_at":timestamp}

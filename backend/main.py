"""GridLeak API giriş noktası. Proje kökünden: python -m uvicorn backend.main:app"""
import os
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from . import database, inspections, subscribers
from .schemas import CapacityInput, NoteInput, ResultInput


def create_app(db_path=None):
    # Varsayılan dosya backend içinde; çalışma dizini değişse de aynı konum kullanılır.
    path = Path(db_path or os.environ.get("GRIDLEAK_DB_PATH") or Path(__file__).parent / "storage" / "gridleak_demo.sqlite3").resolve()

    @asynccontextmanager
    async def lifespan(app):
        database.initialize(path)
        yield

    app = FastAPI(title="GridLeak API", version="0.1.0", lifespan=lifespan,
                  description="Yerel geliştirme servisi. Abone kayıtları sentetiktir. AI modeli henüz bağlı değildir.")
    app.add_middleware(CORSMiddleware,
        allow_origins=["http://localhost:5173","http://127.0.0.1:5173","http://localhost:5174","http://127.0.0.1:5174"],
        allow_methods=["GET","PUT","DELETE"],allow_headers=["Content-Type"])

    @app.get("/api/health", tags=["Sistem"])
    def health():
        with database.connection(path) as db:
            db.execute("SELECT capacity FROM settings WHERE id=1").fetchone()
        return {"status":"ok","database":"ok","data_source":subscribers.SOURCE,"model_status":"not_connected"}

    @app.get("/api/subscribers", tags=["Aboneler"])
    def list_subscribers(q: str = Query(default="",max_length=100), offset: int = Query(default=0,ge=0), limit: int = Query(default=50,ge=1,le=100)):
        ids = [key for key in subscribers.SUBSCRIBERS if q.strip().lower() in key.lower()]
        return {"items":[subscribers.summary(key) for key in ids[offset:offset+limit]],"total":len(ids),"offset":offset,"limit":limit,"source":subscribers.SOURCE}

    @app.get("/api/subscribers/{subscriber_id}", tags=["Aboneler"])
    def subscriber_detail(subscriber_id: str):
        return subscribers.summary(subscriber_id)

    @app.get("/api/subscribers/{subscriber_id}/consumption", tags=["Aboneler"])
    def subscriber_consumption(subscriber_id: str, days: int = Query(default=90,ge=1,le=365)):
        return subscribers.consumption(subscriber_id,days)

    @app.get("/api/model/status", tags=["Model"])
    def model_status():
        return {"status":"not_connected","model_score":None,"explanations":[],"message":"Eğitimli model henüz bağlanmadı; skor ve otomatik öncelik üretilmiyor."}

    @app.get("/api/state", tags=["Kayıtlar"])
    def saved_state():
        return inspections.state(path)

    @app.get("/api/plan", tags=["Denetim"])
    def plan():
        saved = inspections.state(path)
        return {"capacity":saved["capacity"],"items":saved["plan"],"source":saved["source"]}

    @app.put("/api/plan/capacity", tags=["Denetim"])
    def capacity(payload: CapacityInput):
        return inspections.set_capacity(path,payload.capacity)

    @app.put("/api/plan/{subscriber_id}", tags=["Denetim"])
    def add_plan(subscriber_id: str):
        return inspections.add_to_plan(path,subscriber_id)

    @app.delete("/api/plan/{subscriber_id}", tags=["Denetim"])
    def remove_plan(subscriber_id: str):
        return inspections.remove_from_plan(path,subscriber_id)

    @app.put("/api/subscribers/{subscriber_id}/note", tags=["Kayıtlar"])
    def note(subscriber_id: str,payload: NoteInput):
        return inspections.save_note(path,subscriber_id,payload.note)

    @app.get("/api/results", tags=["Kayıtlar"])
    def results():
        return {"items":inspections.state(path)["results"],"source":subscribers.SOURCE}

    @app.put("/api/results/{subscriber_id}", tags=["Kayıtlar"])
    def result(subscriber_id: str,payload: ResultInput):
        return inspections.save_result(path,subscriber_id,payload.status.value,payload.note)

    # API adresleri önce tanımlanır; yalnızca derlenmiş frontend sunulur.
    frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
    if frontend_dist.is_dir():
        app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")

    return app


app = create_app()

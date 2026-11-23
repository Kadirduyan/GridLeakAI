# ==============================================================================
# GridLeak AI — Çok Aşamalı Üretim Dockerfile (React + Vite + FastAPI + SQLite)
# ==============================================================================

# 1. Aşama: Frontend (React + TypeScript + Vite) derlemesi
FROM node:22-alpine AS frontend-builder
WORKDIR /build/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# 2. Aşama: Backend (FastAPI + Uvicorn) ve derlenmiş arayüz + katalog
FROM python:3.13-slim AS runtime
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend/ ./backend/
COPY --from=frontend-builder /build/frontend/dist ./frontend/dist
COPY data/application/catalog.sqlite3 ./data/application/catalog.sqlite3

# Analist notları, denetim planı ve saha sonuçları için kalıcı veri dizini
RUN mkdir -p /app/backend/storage

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)" || exit 1

CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

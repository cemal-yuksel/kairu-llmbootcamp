# Dockerfile - LLM Backend API
# Multi-stage build kullanarak image boyutunu küçültme

# ============================================================================
# STAGE 1: Builder - Dependencies yükleme
# ============================================================================
FROM python:3.10-slim as builder

# Çalışma dizini
WORKDIR /app

# Sistem paketlerini yükle
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies yükle
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ============================================================================
# STAGE 2: Runtime - Final image
# ============================================================================
FROM python:3.10-slim

# Çalışma dizini
WORKDIR /app

# Builder'dan Python packages kopyala
COPY --from=builder /root/.local /root/.local

# Sistem paketlerini yükle (runtime için minimal)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Uygulama dosyalarını kopyala
COPY 3_fastapi_backend.py .
COPY .env.example .env

# Python path'i ayarla
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Port aç
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Uygulamayı başlat
CMD ["uvicorn", "3_fastapi_backend:app", "--host", "0.0.0.0", "--port", "8000"]


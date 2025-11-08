# Kurulum Rehberi

Bu rehber, hafta 7 içeriğini çalıştırmak için gerekli tüm adımları içerir.

## 📋 Gereksinimler

- Python 3.10 veya üzeri
- pip (Python paket yöneticisi)
- Docker (opsiyonel, deployment için)
- Docker Compose (opsiyonel, multi-container için)

## 🚀 Hızlı Başlangıç

### 1. Virtual Environment Oluştur

```bash
# Klasöre git
cd hafta_7

# Virtual environment oluştur
python -m venv venv

# Aktif et (Windows)
venv\Scripts\activate

# Aktif et (Mac/Linux)
source venv/bin/activate
```

### 2. Paketleri Yükle

```bash
pip install -r requirements.txt
```

### 3. Environment Variables Ayarla

`.env` dosyası oluşturun:

```bash
# .env dosyası oluştur
touch .env
```

`.env` dosyasına şunları ekleyin:

```env
OPENAI_API_KEY=your-openai-api-key-here
HUGGINGFACE_API_KEY=your-huggingface-api-key-here
API_BASE_URL=http://localhost:8000
```

### 4. Uygulamaları Çalıştır

#### Gradio Frontend

```bash
python 1_gradio_frontend.py
```

Tarayıcıda `http://localhost:7860` adresine gidin.

#### Streamlit Frontend

```bash
streamlit run 2_streamlit_frontend.py
```

Tarayıcıda `http://localhost:8501` adresine gidin.

#### FastAPI Backend

```bash
uvicorn 3_fastapi_backend:app --reload
```

Tarayıcıda `http://localhost:8000/docs` adresine gidin (Swagger UI).

#### Frontend-Backend Entegrasyonu

Önce backend'i başlatın:
```bash
uvicorn 3_fastapi_backend:app --reload
```

Sonra frontend'i başlatın:
```bash
# Gradio
python 4_fastapi_integration.py gradio

# Streamlit
streamlit run 4_fastapi_integration.py
```

## 🐳 Docker ile Kurulum

### 1. Docker Image Build Et

```bash
# Backend API
docker build -t llm-backend:latest -f Dockerfile .

# Gradio Frontend
docker build -t llm-gradio:latest -f Dockerfile.gradio .

# Streamlit Frontend
docker build -t llm-streamlit:latest -f Dockerfile.streamlit .
```

### 2. Docker Compose ile Tüm Servisleri Başlat

```bash
docker-compose up -d
```

### 3. Docker Setup Script Kullan

```bash
python 5_docker_setup.py
```

## 📝 Test Etme

### Backend API Test

```bash
# Health check
curl http://localhost:8000/health

# Chat endpoint test
curl -X POST "http://localhost:8000/chat/simple?message=Merhaba&model=gpt-3.5-turbo"
```

### Frontend Test

1. Tarayıcıda ilgili URL'ye gidin
2. Bir mesaj gönderin
3. Yanıtı kontrol edin

## 🐛 Sorun Giderme

### Port Zaten Kullanılıyor

```bash
# Port 8000'i kullanan process'i bul (Mac/Linux)
lsof -ti:8000 | xargs kill -9

# Port 8000'i kullanan process'i bul (Windows)
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### API Key Hatası

`.env` dosyasında API key'inizin doğru olduğundan emin olun:

```bash
# .env dosyasını kontrol et
cat .env
```

### Docker Hatası

```bash
# Docker'ın çalıştığını kontrol et
docker ps

# Docker Desktop'ın çalıştığından emin olun
```

### Import Hatası

```bash
# Virtual environment aktif mi kontrol et
which python  # Mac/Linux
where python  # Windows

# Paketleri yeniden yükle
pip install -r requirements.txt --force-reinstall
```

## 📚 Ek Kaynaklar

- [Gradio Documentation](https://www.gradio.app/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Documentation](https://docs.docker.com/)

## 🎯 Sonraki Adımlar

1. Örnekleri çalıştırın
2. Kod üzerinde değişiklikler yaparak deneyin
3. Kendi uygulamanızı oluşturun
4. Docker ile deploy edin

**Başarılar! 🚀**


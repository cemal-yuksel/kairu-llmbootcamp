# 🐳 Docker ve Container Deployment Dersi
## Hafta 7 - Script 5: Docker Setup ve Production Deployment

---

## 📚 **DERS AKIŞI ve İÇERİK**

### 🎯 **Dersin Amacı**
Bu derste öğrenciler:
- Docker'ın ne olduğunu ve neden kullanıldığını öğrenecek
- Container teknolojisinin avantajlarını anlayacak
- LLM uygulamalarını Docker ile nasıl deploy edeceklerini öğrenecek
- Production-ready deployment süreçlerini kavrayacak

---

## 🏗️ **1. BÖLÜM: Docker Nedir? (15 dakika)**

### 🤔 **Problem: Geleneksel Deployment Sorunları**

**Senaryo:** Bir yazılım geliştirdiniz ve farklı ortamlarda çalıştırmak istiyorsunuz:

```
👨‍💻 Geliştirici Bilgisayarı:
- Python 3.10
- Ubuntu 22.04
- RAM: 16GB
- ✅ Uygulama çalışıyor

🖥️ Test Sunucusu:
- Python 3.8
- CentOS 7
- RAM: 8GB
- ❌ Uygulama çalışmıyor!

🚀 Production Sunucusu:
- Python 3.11
- Windows Server
- RAM: 32GB
- ❌ Uygulama çalışmıyor!
```

**Klasik Sorunlar:**
- "Bende çalışıyor!" sorunu
- Dependency çakışmaları
- Farklı işletim sistemleri
- Farklı Python/library versiyonları

### 🐳 **Çözüm: Docker Container'lar**

Docker, uygulamanızı tüm bağımlılıklarıyla birlikte paketler:

```
📦 Docker Container = Uygulama + Dependencies + OS
├── 🐍 Python 3.10
├── 📚 Requirements (FastAPI, OpenAI, vb.)
├── 🗂️ Uygulama dosyaları
├── 🔧 Sistem kütüphaneleri
└── ⚙️ Çalışma ortamı ayarları
```

**Sonuç:** Herhangi bir Docker destekli sistemde aynı şekilde çalışır!

---

## ⚙️ **2. BÖLÜM: Docker Temelleri (20 dakika)**

### 🧩 **Temel Kavramlar**

#### 1. **Image (Kalıp)**
```dockerfile
# Dockerfile örneği
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "app.py"]
```
- Image = Uygulamanızın kalıbı/şablonu
- Bir kez oluşturulur, birçok kez kullanılır
- Sadece okunabilir (read-only)

#### 2. **Container (Çalışan Instance)**
```bash
# Image'dan container oluştur ve çalıştır
docker run -p 8000:8000 my-app:latest
```
- Container = Image'ın çalışan hali
- Yazılabilir katman ekler
- Her container izole ortamda çalışır

#### 3. **Port Mapping**
```bash
# Host:Container port eşlemesi
docker run -p 8000:8000 my-app    # localhost:8000 → container:8000
docker run -p 9000:8000 my-app    # localhost:9000 → container:8000
```

### 🔧 **Docker Komutları**

```bash
# Image işlemleri
docker build -t my-app:latest .    # Image oluştur
docker images                      # Image'ları listele
docker rmi my-app:latest          # Image sil

# Container işlemleri
docker run -d --name my-container my-app:latest    # Background'da çalıştır
docker ps                         # Çalışan container'ları listele
docker ps -a                      # Tüm container'ları listele
docker stop my-container          # Container'ı durdur
docker start my-container         # Container'ı başlat
docker rm my-container            # Container'ı sil

# Logs ve debugging
docker logs my-container          # Container loglarını gör
docker exec -it my-container bash # Container'a bağlan
```

---

## 🏗️ **3. BÖLÜM: LLM Uygulaması için Docker (25 dakika)**

### 📁 **Proje Yapısı**
```
hafta_7/
├── 📄 Dockerfile                 # Backend için
├── 📄 Dockerfile.gradio          # Gradio frontend için
├── 📄 Dockerfile.streamlit       # Streamlit frontend için
├── 📄 docker-compose.yml         # Orchestration
├── 📄 requirements.txt           # Python dependencies
├── 📄 .env                       # Environment variables
├── 🐍 3_fastapi_backend.py       # Backend API
├── 🐍 1_gradio_frontend.py       # Gradio frontend
└── 🐍 5_docker_setup.py          # Docker automation
```

### 🐍 **Backend Dockerfile Analizi**

```dockerfile
# 1. Base image seç (Python runtime)
FROM python:3.10-slim

# 2. Çalışma dizini oluştur
WORKDIR /app

# 3. Requirements'ı kopyala ve yükle (caching için)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Uygulama kodunu kopyala
COPY . .

# 5. Port bilgisi (dokümantasyon için)
EXPOSE 8000

# 6. Başlangıç komutu
CMD ["python", "3_fastapi_backend.py"]
```

**Neden bu sıra?**
- Requirements önce → Docker cache'ini optimize eder
- Kod değişse bile, dependencies yeniden yüklenmez

### 📊 **docker-compose.yml Analizi**

```yaml
version: '3.8'

services:
  # Backend API
  backend:
    build: 
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    restart: unless-stopped

  # Gradio Frontend
  gradio:
    build:
      context: .
      dockerfile: Dockerfile.gradio
    ports:
      - "7861:7861"
    depends_on:
      - backend
    environment:
      - API_BASE_URL=http://backend:8000
    restart: unless-stopped

  # Streamlit Frontend
  streamlit:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    ports:
      - "8501:8501"
    depends_on:
      - backend
    environment:
      - API_BASE_URL=http://backend:8000
    restart: unless-stopped
```

**Açıklama:**
- **services:** 3 farklı uygulama (backend, gradio, streamlit)
- **depends_on:** Frontend'ler backend'i bekler
- **environment:** Container'a environment variable geçir
- **restart:** Container kapanırsa otomatik yeniden başlat

---

## 🚀 **4. BÖLÜM: Script 5 Analizi (20 dakika)**

### 🎯 **5_docker_setup.py'nin Amacı**

Bu script Docker işlemlerini otomatikleştirir:

```python
def main():
    """Ana menü ile kullanıcı seçenekleri sunar"""
    print("🐳 Docker Setup Menüsü:")
    print("1. Docker kontrol et")
    print("2. Image'ları build et") 
    print("3. Container'ları başlat")
    print("4. Container'ları durdur")
    print("5. Durumları göster")
    print("6. docker-compose ile başlat")
```

### 🔧 **Ana Fonksiyonlar**

#### 1. **Docker Kontrolü**
```python
def check_docker_installed():
    """Docker'ın yüklü olup olmadığını kontrol et"""
    result = run_command("docker --version", check=False)
    if result:
        print(f"✅ Docker yüklü: {result}")
        return True
    else:
        print("❌ Docker yüklü değil!")
        return False
```

#### 2. **Image Build İşlemi**
```python
def build_backend_image():
    """Backend API Docker image'ını build et"""
    print("🔨 Backend image'ı build ediliyor...")
    result = run_command("docker build -t llm-backend:latest .")
    if result is not None:
        print("✅ Backend image başarıyla build edildi")
        return True
    else:
        print("❌ Backend image build edilemedi")
        return False
```

#### 3. **Container Yönetimi**
```python
def start_backend_container():
    """Backend API container'ını başlat"""
    print("🚀 Backend API container başlatılıyor...")
    
    # Önce varsa durdur
    run_command("docker stop llm-backend 2>/dev/null", check=False)
    run_command("docker rm llm-backend 2>/dev/null", check=False)
    
    # Yeni container başlat
    result = run_command(
        "docker run -d "
        "--name llm-backend "
        "-p 8000:8000 "
        "--env-file .env "
        "llm-backend:latest"
    )
```

### 📊 **docker-compose Entegrasyonu**
```python
def start_all_with_compose():
    """docker-compose ile tüm servisleri başlat"""
    print("🚀 docker-compose ile tüm servisler başlatılıyor...")
    result = run_command("docker-compose up -d")
    if result is not None:
        print("✅ Tüm servisler başlatıldı")
        print("   Backend: http://localhost:8000")
        print("   Gradio: http://localhost:7861") 
        print("   Streamlit: http://localhost:8501")
        return True
```

---

## 🎓 **5. BÖLÜM: Pratik Uygulama (30 dakika)**

### 👨‍💻 **Canlı Demo: Adım Adım Deployment**

#### **Adım 1: Hazırlık**
```bash
# Proje dizinine git
cd hafta_7

# Environment dosyası var mı kontrol et
ls -la .env

# Docker çalışıyor mu kontrol et
docker --version
docker ps
```

#### **Adım 2: Script Çalıştırma**
```bash
# Automation script'ini çalıştır
python 5_docker_setup.py
```

**Menü seçenekleri:**
1. **Docker Check** → Docker kurulumu kontrol
2. **Build Images** → Tüm image'ları oluştur
3. **Start Services** → Container'ları başlat
4. **View Status** → Çalışan container'ları gör

#### **Adım 3: Manuel Test**
```bash
# Container'ları listele
docker ps

# Backend test
curl http://localhost:8000/health

# Logs kontrol et
docker logs llm-backend
docker logs llm-gradio
```

#### **Adım 4: docker-compose Kullanımı**
```bash
# Tüm servisleri birden başlat
docker-compose up -d

# Durumu kontrol et
docker-compose ps

# Logları izle
docker-compose logs -f

# Servisleri durdur
docker-compose down
```

### 🌐 **Test Senaryoları**

1. **Backend API Test:**
   ```bash
   curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"messages": [{"role": "user", "content": "Merhaba"}]}'
   ```

2. **Frontend Test:**
   - Gradio: http://localhost:7861
   - Streamlit: http://localhost:8501

3. **Health Check:**
   ```bash
   curl http://localhost:8000/health
   ```

---

## 🎯 **6. BÖLÜM: Production Deployment (15 dakika)**

### 🌍 **Gerçek Dünya Senaryoları**

#### **Scenario 1: AWS EC2'de Deployment**
```bash
# EC2 instance'a bağlan
ssh -i key.pem ubuntu@ec2-xxx.compute.amazonaws.com

# Docker yükle
sudo apt update
sudo apt install docker.io docker-compose

# Projeyi klonla
git clone your-repo.git
cd your-project

# Environment dosyasını oluştur
echo "OPENAI_API_KEY=your-key" > .env

# Deploy
docker-compose up -d
```

#### **Scenario 2: Cloud Platform Deployment**
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  backend:
    image: your-registry/llm-backend:latest
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    deploy:
      replicas: 3  # Load balancing için
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
```

### 🔒 **Security Best Practices**

1. **Secrets Management:**
   ```yaml
   # docker-compose.yml
   services:
     backend:
       environment:
         - OPENAI_API_KEY_FILE=/run/secrets/openai_key
       secrets:
         - openai_key
   
   secrets:
     openai_key:
       file: ./secrets/openai_key.txt
   ```

2. **Network Security:**
   ```yaml
   services:
     backend:
       networks:
         - internal  # Sadece internal network
     
     gradio:
       networks:
         - internal
         - external  # Public access için
   ```

3. **Resource Limits:**
   ```yaml
   services:
     backend:
       deploy:
         resources:
           limits:
             cpus: '1.0'
             memory: 1G
           reservations:
             cpus: '0.5'
             memory: 512M
   ```

---

## 📝 **7. BÖLÜM: Troubleshooting ve Best Practices (10 dakika)**

### 🚨 **Yaygın Problemler ve Çözümleri**

#### **Problem 1: Container başlamıyor**
```bash
# Debug adımları:
docker logs container-name        # Logları kontrol et
docker exec -it container bash   # Container'a gir
docker inspect container-name    # Container detaylarını gör
```

#### **Problem 2: Port conflicts**
```bash
# Port kullanımını kontrol et
netstat -tulpn | grep :8000
lsof -i :8000

# Farklı port kullan
docker run -p 8001:8000 my-app
```

#### **Problem 3: Environment variables**
```bash
# Container içindeki env var'ları kontrol et
docker exec container-name env
docker exec container-name echo $OPENAI_API_KEY
```

### ✅ **Best Practices**

1. **Multi-stage Builds:**
   ```dockerfile
   # Build stage
   FROM python:3.10 as builder
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   
   # Runtime stage
   FROM python:3.10-slim
   COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
   ```

2. **Health Checks:**
   ```dockerfile
   HEALTHCHECK --interval=30s --timeout=3s \
     CMD curl -f http://localhost:8000/health || exit 1
   ```

3. **Proper Logging:**
   ```python
   import logging
   logging.basicConfig(level=logging.INFO)
   logger = logging.getLogger(__name__)
   ```

---

## 🎓 **8. BÖLÜM: Öğrenci Uygulaması ve Q&A (15 dakika)**

### 📚 **Öğrenci Egzersizleri**

#### **Egzersiz 1: Kendi Container'ınızı Oluşturun**
```dockerfile
# Öğrenciler kendi Dockerfile'larını yazacak
FROM python:3.10-slim

# TODO: Öğrenciler tamamlayacak
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "my_app.py"]
```

#### **Egzersiz 2: docker-compose Düzenleyin**
```yaml
# Yeni bir service ekleyin
version: '3.8'
services:
  backend:
    # Mevcut config...
  
  # TODO: Öğrenciler yeni service ekleyecek
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    # Konfigürasyon tamamlanacak
```

### ❓ **Sık Sorulan Sorular**

1. **S: Docker ve VM arasındaki fark nedir?**
   **C:** VM tüm işletim sistemini virtualize eder, Docker sadece uygulama katmanını izole eder. Docker daha hafif ve hızlıdır.

2. **S: Container verisi kaybolur mu?**
   **C:** Evet, container silinirse veriler kaybolur. Persistent data için volume kullanın.

3. **S: Hangi durumlarda Docker kullanmalı?**
   **C:** Microservices, CI/CD, multi-environment deployment, scaling gereken durumlar.

---

## 📊 **DERS ÖZET TABLOSU**

| 🎯 **Konu** | ⏱️ **Süre** | 🛠️ **Aktivite** | 📈 **Seviye** |
|-------------|-------------|------------------|----------------|
| Docker Nedir? | 15 dk | Teori + Örnekler | Başlangıç |
| Docker Temelleri | 20 dk | Komutlar + Demo | Orta |
| LLM App Dockerization | 25 dk | Canlı Kodlama | İleri |
| Script 5 Analizi | 20 dk | Kod İnceleme | İleri |
| Pratik Uygulama | 30 dk | Hands-on | İleri |
| Production Tips | 15 dk | Best Practices | Expert |
| Troubleshooting | 10 dk | Problem Çözme | Orta |
| Q&A | 15 dk | Interaktif | Tüm |

---

## 🎯 **DERS HEDEFLERİ VE BAŞARI METRİKLERİ**

### ✅ **Ders Sonunda Öğrenciler:**

1. **Docker kavramını açıklayabilir** ✓
2. **Dockerfile yazabilir** ✓  
3. **docker-compose kullanabilir** ✓
4. **Container'ları yönetebilir** ✓
5. **Production deployment yapabilir** ✓
6. **Temel troubleshooting becerilerine sahip** ✓

### 📊 **Değerlendirme Kriterleri:**

- **Teori testi:** Docker kavramları (25%)
- **Pratik uygulama:** Container oluşturma (50%)
- **Problem çözme:** Troubleshooting (25%)

---

## 🚀 **İLERİ KONULAR (Bonus)**

### 🔮 **Sonraki Adımlar**

1. **Kubernetes:** Container orchestration
2. **CI/CD:** GitLab/GitHub Actions ile otomatik deployment
3. **Monitoring:** Prometheus + Grafana
4. **Secrets Management:** HashiCorp Vault
5. **Service Mesh:** Istio

---

## 📚 **EK KAYNAKLAR**

### 📖 **Önerilen Okumalar:**
- [Docker Official Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Best Practices for Writing Dockerfiles](https://docs.docker.com/develop/dev-best-practices/)

### 🎥 **Video Kaynaklar:**
- Docker Basics Tutorial
- Container Orchestration with Docker Compose
- Production Docker Deployment

### 🛠️ **Pratik Projeler:**
- Personal Blog Dockerization
- E-commerce App Multi-Container Setup
- ML Model Serving with Docker

---

Bu ders notları ile öğrencileriniz Docker'ı sıfırdan öğrenip, production-ready LLM uygulaması deploy edebilecekler! 🎉
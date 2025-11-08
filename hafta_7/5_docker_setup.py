"""
Docker Setup ve Deployment Script
==================================
Bu script, LLM uygulamalarını Docker container'larında çalıştırmak için 
geliştirilmiş kapsamlı bir yönetim aracıdır.

ANA ÖZELLİKLER:
--------------
1. Docker ve Docker Compose kurulum kontrolü
2. Backend API, Gradio ve Streamlit için Docker image'larını build etme
3. Container'ları başlatma, durdurma ve yönetme
4. Docker Compose ile tüm servisleri tek komutla yönetme
5. Container loglarını izleme ve durum kontrolü

KULLANIM ALANLARI:
-----------------
- Geliştirme ortamında tutarlı çalışma ortamı sağlama
- Production deployment için hazırlık
- Multi-container uygulamaları yönetme
- Takım çalışmasında ortam standardizasyonu
"""

import os  # İşletim sistemi işlemleri için
import subprocess  # Terminal komutlarını çalıştırmak için
import sys  # Sistem işlemleri ve çıkış kodları için
from pathlib import Path  # Dosya yolu işlemleri için modern yaklaşım
from dotenv import load_dotenv  # .env dosyasından çevre değişkenlerini yüklemek için

# ============================================================================
# ÇEVRE DEĞİŞKENLERİNİ YÜKLEME
# ============================================================================
# .env dosyasındaki API anahtarları ve yapılandırmaları yükle
# Bu, hassas bilgilerin kod içinde hardcoded olmasını önler
load_dotenv()

# ============================================================================
# YARDIMCI FONKSİYONLAR - TEMEL ARAÇLAR
# ============================================================================

def run_command(command, check=True):
    """
    Terminal komutu çalıştır ve sonucunu döndür
    
    Bu fonksiyon, Docker komutlarını ve diğer sistem komutlarını 
    güvenli bir şekilde çalıştırmak için merkezi bir yapı sağlar.
    
    PARAMETRELER:
    ------------
    command : str
        Çalıştırılacak terminal komutu (örn: "docker ps")
    check : bool
        True ise hata durumunda exception fırlat, False ise sessizce devam et
    
    DÖNÜŞ DEĞERİ:
    ------------
    str veya None : Komutun çıktısı veya hata durumunda None
    
    ÇALIŞMA MANTĞI:
    --------------
    1. subprocess.run() ile komutu çalıştır
    2. shell=True parametresi ile komutları shell üzerinden çalıştır
    3. capture_output=True ile stdout ve stderr'i yakala
    4. text=True ile çıktıyı string olarak al (byte değil)
    5. Hata durumunda detaylı hata mesajı göster
    
    GÜVENLİK NOTU:
    -------------
    shell=True kullanımı potansiyel güvenlik riski taşır.
    Production ortamında kullanıcı girdisi ile kullanılmamalı.
    """
    try:
        # Komutu çalıştır ve sonucu yakala
        result = subprocess.run(
            command,  # Çalıştırılacak komut
            shell=True,  # Shell'de çalıştır (pipe, redirect vb. için gerekli)
            check=check,  # Hata kontrolü yap
            capture_output=True,  # stdout ve stderr'i yakala
            text=True  # Çıktıyı string olarak al
        )
        # Başarılı ise çıktıyı döndür (baştaki/sondaki boşlukları temizle)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        # Hata durumunda kullanıcıya bilgi ver
        print(f"❌ Hata: {e}")
        print(f"Stderr: {e.stderr}")  # Hata detaylarını göster
        return None


def check_docker_installed():
    """
    Docker'ın sistemde yüklü olup olmadığını kontrol et
    
    Docker, container'ları çalıştırmak için temel gereksinimdir.
    Bu fonksiyon, scriptin çalışması için ön koşulu kontrol eder.
    
    KONTROL MANTĞI:
    --------------
    1. "docker --version" komutu ile Docker'ın varlığını test et
    2. Komut başarılı ise Docker yüklü demektir
    3. check=False ile hata durumunda script'i durdurmadan devam et
    4. Yüklü değilse kullanıcıya kurulum linki ver
    
    DÖNÜŞ DEĞERİ:
    ------------
    bool : Docker yüklü ise True, değilse False
    
    ÖRNEK ÇIKTI:
    -----------
    ✅ Docker yüklü: Docker version 24.0.6, build ed223bc
    """
    # Docker version komutunu çalıştır (hata olsa bile devam et)
    result = run_command("docker --version", check=False)
    if result:
        # Başarılı ise version bilgisini göster
        print(f"✅ Docker yüklü: {result}")
        return True
    else:
        # Docker yüklü değilse kullanıcıyı bilgilendir
        print("❌ Docker yüklü değil!")
        print("Docker'ı yüklemek için: https://docs.docker.com/get-docker/")
        return False


def check_docker_compose_installed():
    """
    Docker Compose'un sistemde yüklü olup olmadığını kontrol et
    
    Docker Compose, multi-container uygulamaları yönetmek için kullanılır.
    docker-compose.yml dosyasındaki servisleri tek komutla başlatır/durdurur.
    
    DOCKER COMPOSE NEDİR?
    --------------------
    Docker Compose, birden fazla container'ı tek bir yapılandırma dosyası
    ile yönetmeyi sağlar. Örneğin:
    - Backend API (port 8000)
    - Gradio Frontend (port 7860)
    - Streamlit Frontend (port 8501)
    
    Bu üç servisi "docker-compose up" ile tek komutla başlatabilirsiniz.
    
    DÖNÜŞ DEĞERİ:
    ------------
    bool : Docker Compose yüklü ise True, değilse False
    """
    # Docker Compose version komutunu çalıştır
    result = run_command("docker-compose --version", check=False)
    if result:
        print(f"✅ Docker Compose yüklü: {result}")
        return True
    else:
        print("❌ Docker Compose yüklü değil!")
        print("Docker Compose'u yüklemek için: https://docs.docker.com/compose/install/")
        return False


def check_env_file():
    """
    .env dosyasının var olup olmadığını kontrol et
    
    .env DOSYASI NEDİR?
    ------------------
    .env dosyası, hassas bilgileri (API anahtarları, şifreler vb.) 
    kod dışında saklamak için kullanılır. Örnek içerik:
    
    OPENAI_API_KEY=sk-xxxxxxxxxxxxx
    HUGGINGFACE_API_KEY=hf_xxxxxxxxxxxxx
    DATABASE_URL=postgresql://...
    
    GÜVENLİK ÖNEMİ:
    --------------
    - .env dosyası asla git'e commit edilmemelidir (.gitignore'da olmalı)
    - Her geliştirici kendi .env dosyasını oluşturur
    - Production'da farklı .env kullanılır
    
    DÖNÜŞ DEĞERİ:
    ------------
    bool : .env dosyası varsa True, yoksa False
    """
    # Path nesnesi ile modern dosya kontrolü
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env dosyası bulundu")
        return True
    else:
        # Dosya yoksa kullanıcıya nasıl oluşturacağını göster
        print("⚠️ .env dosyası bulunamadı!")
        print("Lütfen .env dosyası oluşturun:")
        print("  OPENAI_API_KEY=your-api-key")
        print("  HUGGINGFACE_API_KEY=your-api-key")
        return False


# ============================================================================
# IMAGE BUILD FONKSİYONLARI - DOCKER IMAGE'LARINI OLUŞTURMA
# ============================================================================

def build_backend_image():
    """
    Backend API için Docker image'ını build et
    
    DOCKER IMAGE NEDİR?
    ------------------
    Docker image, container'ların çalışması için gereken tüm dosyaları,
    bağımlılıkları ve yapılandırmaları içeren şablon/template'dir.
    
    BU IMAGE'IN İÇERİĞİ:
    -------------------
    - Python runtime ortamı
    - FastAPI framework ve bağımlılıkları
    - Uygulama kodları (API endpoints)
    - requirements.txt'deki tüm kütüphaneler
    
    BUILD KOMUTU AÇIKLAMASI:
    ------------------------
    docker build 
        -t llm-backend:latest    → Image'a tag/isim ver
        -f Dockerfile            → Hangi Dockerfile kullanılacak
        .                        → Build context (mevcut dizin)
    
    BUILD CONTEXT:
    -------------
    "." (nokta) parametresi, Docker'a hangi dizindeki dosyaları 
    image'a dahil edeceğini söyler. Dockerfile içindeki COPY 
    komutları bu dizini baz alır.
    
    DÖNÜŞ DEĞERİ:
    ------------
    bool : Build başarılı ise True, başarısız ise False
    """
    print("\n🔨 Backend API image'ı build ediliyor...")
    # Docker build komutunu çalıştır
    result = run_command("docker build -t llm-backend:latest -f Dockerfile .")
    if result is not None:
        print("✅ Backend API image başarıyla build edildi")
        return True
    else:
        print("❌ Backend API image build edilemedi")
        return False


def build_gradio_image():
    """
    Gradio frontend için Docker image'ını build et
    
    GRADIO NEDİR?
    ------------
    Gradio, makine öğrenimi modellerine hızlıca kullanıcı arayüzü 
    oluşturmak için Python kütüphanesidir. Kod yazmadan güzel 
    UI'lar oluşturabilirsiniz.
    
    BU IMAGE'IN ÖZELLİKLERİ:
    -----------------------
    - Gradio kütüphanesi
    - Frontend kodları
    - API'ye bağlanmak için gerekli yapılandırmalar
    - Port 7860'da çalışır
    
    DOCKERFILE.GRADIO:
    -----------------
    Farklı bir Dockerfile kullanılır (-f Dockerfile.gradio)
    Çünkü frontend'in gereksinimleri backend'den farklıdır.
    """
    print("\n🔨 Gradio frontend image'ı build ediliyor...")
    result = run_command("docker build -t llm-gradio:latest -f Dockerfile.gradio .")
    if result is not None:
        print("✅ Gradio frontend image başarıyla build edildi")
        return True
    else:
        print("❌ Gradio frontend image build edilemedi")
        return False


def build_streamlit_image():
    """
    Streamlit frontend için Docker image'ını build et
    
    STREAMLIT NEDİR?
    ---------------
    Streamlit, veri bilimi ve makine öğrenimi uygulamaları için 
    hızlı web uygulamaları oluşturmaya yarayan Python framework'üdür.
    
    GRADIO VS STREAMLIT:
    -------------------
    - Gradio: Daha basit, model odaklı UI'lar için
    - Streamlit: Daha karmaşık, dashboard tarzı uygulamalar için
    
    BU PROJEDE:
    ----------
    İki frontend de mevcut, kullanıcı tercihine göre birini seçebilir.
    Her ikisi de aynı Backend API'ye bağlanır (port 8000).
    """
    print("\n🔨 Streamlit frontend image'ı build ediliyor...")
    result = run_command("docker build -t llm-streamlit:latest -f Dockerfile.streamlit .")
    if result is not None:
        print("✅ Streamlit frontend image başarıyla build edildi")
        return True
    else:
        print("❌ Streamlit frontend image build edilemedi")
        return False


# ============================================================================
# CONTAINER BAŞLATMA FONKSİYONLARI
# ============================================================================

def start_backend_container():
    """
    Backend API container'ını başlat
    
    DOCKER CONTAINER NEDİR?
    ----------------------
    Container, Docker image'dan oluşturulan çalışan bir instance'dır.
    Image = Şablon/Template, Container = Çalışan Kopya
    
    DOCKER RUN KOMUTU DETAYI:
    -------------------------
    docker run
        -d                       → Detached mode (arka planda çalış)
        --name llm-backend       → Container'a isim ver
        -p 8000:8000            → Port mapping (host:container)
        --env-file .env         → Environment variables'ı .env'den yükle
        llm-backend:latest      → Kullanılacak image
    
    PORT MAPPING AÇIKLAMASI:
    -----------------------
    -p 8000:8000 → Sol taraf (8000): Host bilgisayarın portu
                → Sağ taraf (8000): Container içindeki port
    localhost:8000'e yapılan istekler container'ın 8000 portuna gider.
    
    ÖNCELİKLE DURDURMA:
    ------------------
    Aynı isimde container varsa çakışma olmaması için önce durdurulur.
    2>/dev/null → Hata mesajlarını gizle (container yoksa hata vermesin)
    
    DÖNÜŞ DEĞERİ:
    ------------
    bool : Container başarıyla başladı ise True
    """
    print("\n🚀 Backend API container başlatılıyor...")
    
    # Önce mevcut container'ı durdur ve sil (varsa)
    # check=False ile hata olsa bile devam et
    run_command("docker stop llm-backend 2>/dev/null", check=False)
    run_command("docker rm llm-backend 2>/dev/null", check=False)
    
    # Yeni container'ı başlat
    result = run_command(
        "docker run -d "  # Detached mode
        "--name llm-backend "  # Container adı
        "-p 8000:8000 "  # Port mapping
        "--env-file .env "  # Environment variables
        "llm-backend:latest"  # Kullanılacak image
    )
    
    if result:
        print("✅ Backend API container başlatıldı")
        print("   URL: http://localhost:8000")
        print("   Docs: http://localhost:8000/docs")  # FastAPI otomatik API dokümantasyonu
        return True
    else:
        print("❌ Backend API container başlatılamadı")
        return False


def start_gradio_container():
    """
    Gradio frontend container'ını başlat
    
    CONTAINER ARASI İLETİŞİM:
    ------------------------
    Frontend (Gradio) → Backend API'ye HTTP istekleri yapar
    
    HOST.DOCKER.INTERNAL AÇIKLAMASI:
    --------------------------------
    Container'lar izole ortamlarda çalışır. Bir container'dan host 
    bilgisayardaki başka bir container'a erişmek için özel DNS kullanılır:
    
    - Linux: host.docker.internal (Docker 20.10+)
    - Windows/Mac: host.docker.internal (varsayılan)
    
    API_BASE_URL=http://host.docker.internal:8000
    → Gradio container'ı, host bilgisayardaki 8000 portuna erişir
    → Orada Backend API dinliyor
    
    ENVIRONMENT VARIABLE:
    --------------------
    -e parametresi ile container içine environment variable geçiyoruz.
    Gradio kodunda bu değişken okunur: os.getenv("API_BASE_URL")
    """
    print("\n🚀 Gradio frontend container başlatılıyor...")
    
    # Önce durdur (varsa)
    run_command("docker stop llm-gradio-frontend 2>/dev/null", check=False)
    run_command("docker rm llm-gradio-frontend 2>/dev/null", check=False)
    
    # Container'ı başlat
    result = run_command(
        "docker run -d "
        "--name llm-gradio-frontend "
        "-p 7860:7860 "  # Gradio varsayılan portu
        "-e API_BASE_URL=http://host.docker.internal:8000 "  # Backend API adresi
        "llm-gradio:latest"
    )
    
    if result:
        print("✅ Gradio frontend container başlatıldı")
        print("   URL: http://localhost:7860")
        return True
    else:
        print("❌ Gradio frontend container başlatılamadı")
        return False


def start_streamlit_container():
    """
    Streamlit frontend container'ını başlat
    
    STREAMLIT PORT:
    --------------
    Streamlit varsayılan olarak 8501 portunda çalışır.
    Gradio'dan farklı port kullanır, böylece iki frontend 
    aynı anda çalışabilir.
    
    ÇOK FRONTEND KULLANIMI:
    ----------------------
    Aynı Backend API'yi kullanan iki farklı frontend:
    - Gradio: http://localhost:7860
    - Streamlit: http://localhost:8501
    - Her ikisi de → http://localhost:8000 (Backend)
    
    Bu yapı, kullanıcıya seçenek sunar ve A/B testi yapmaya olanak verir.
    """
    print("\n🚀 Streamlit frontend container başlatılıyor...")
    
    # Önce durdur (varsa)
    run_command("docker stop llm-streamlit-frontend 2>/dev/null", check=False)
    run_command("docker rm llm-streamlit-frontend 2>/dev/null", check=False)
    
    # Container'ı başlat
    result = run_command(
        "docker run -d "
        "--name llm-streamlit-frontend "
        "-p 8501:8501 "  # Streamlit varsayılan portu
        "-e API_BASE_URL=http://host.docker.internal:8000 "  # Backend API adresi
        "llm-streamlit:latest"
    )
    
    if result:
        print("✅ Streamlit frontend container başlatıldı")
        print("   URL: http://localhost:8501")
        return True
    else:
        print("❌ Streamlit frontend container başlatılamadı")
        return False


# ============================================================================
# DOCKER COMPOSE FONKSİYONLARI
# ============================================================================

def start_with_compose():
    """
    Docker Compose ile tüm servisleri başlat
    
    DOCKER COMPOSE AVANTAJLARI:
    --------------------------
    1. Tek komutla tüm servisleri başlat/durdur
    2. Servisler arası bağımlılıkları yönet
    3. Network otomatik oluşturulur
    4. Volume yönetimi kolaylaşır
    5. Yapılandırma dosyası (docker-compose.yml) versiyon kontrolünde
    
    DOCKER-COMPOSE.YML YAPISI:
    -------------------------
    version: '3.8'
    services:
      backend:           # Backend API servisi
        build: .
        ports:
          - "8000:8000"
        env_file:
          - .env
      
      gradio:            # Gradio frontend servisi
        build:
          context: .
          dockerfile: Dockerfile.gradio
        ports:
          - "7860:7860"
        depends_on:      # Backend başlamadan Gradio başlamasın
          - backend
    
    DOCKER-COMPOSE UP -D AÇIKLAMASI:
    --------------------------------
    - up: Servisleri başlat
    - -d: Detached mode (arka planda çalıştır)
    
    NETWORK OLUŞTURma:
    -----------------
    Docker Compose otomatik olarak bir network oluşturur.
    Bu network'te servisler birbirlerini isimle bulabilir:
    - Backend: http://backend:8000
    - Gradio içinden: requests.get("http://backend:8000/api/...")
    """
    print("\n🚀 Docker Compose ile servisler başlatılıyor...")
    result = run_command("docker-compose up -d")
    
    if result is not None:
        print("✅ Tüm servisler başlatıldı")
        print("\n📊 Servisler:")
        print("   Backend API: http://localhost:8000")
        print("   Backend Docs: http://localhost:8000/docs")
        print("   Gradio Frontend: http://localhost:7860")
        print("   Streamlit Frontend: http://localhost:8501")
        return True
    else:
        print("❌ Servisler başlatılamadı")
        return False


def stop_containers():
    """
    Tüm container'ları durdur ve temizle
    
    İKİ YÖNTEM:
    ----------
    1. Docker Compose ile başlatılanlar: docker-compose down
    2. Manuel başlatılanlar: docker stop + docker rm
    
    DOCKER-COMPOSE DOWN:
    -------------------
    - Container'ları durdurur
    - Container'ları siler
    - Network'leri temizler
    - Volume'lar korunur (silinmez)
    
    DOCKER-COMPOSE DOWN -V:
    ----------------------
    Volume'ları da silmek için -v parametresi eklenir.
    Ancak bu komutta kullanmadık, veriler korunsun diye.
    
    2>/DEV/NULL AÇIKLAMASI:
    ----------------------
    stderr (hata çıktısı) stream'ini /dev/null'a yönlendir.
    Yani hata mesajlarını gizle. Container yoksa hata vermez.
    
    CHECK=FALSE:
    -----------
    run_command'a check=False veriyoruz çünkü container 
    bulunamasa bile hata fırlatmasın, sessizce devam etsin.
    """
    print("\n🛑 Container'lar durduruluyor...")
    # Compose ile başlatılanları durdur
    run_command("docker-compose down", check=False)
    # Manuel başlatılanları durdur
    run_command("docker stop llm-backend llm-gradio-frontend llm-streamlit-frontend 2>/dev/null", check=False)
    print("✅ Container'lar durduruldu")


def show_logs(service=None):
    """
    Container loglarını göster
    
    LOG İZLEME ÖNEMİ:
    ----------------
    - Hataları debug etmek için
    - Performans sorunlarını tespit etmek için
    - Kullanıcı aktivitelerini izlemek için
    - Security olaylarını görmek için
    
    DOCKER LOGS KOMUTU:
    ------------------
    docker logs
        -f              → Follow mode (canlı izle)
        container_name  → Hangi container'ın logları
    
    CTRL+C ile loglardan çıkılır.
    
    İKİ MOD:
    -------
    1. Belirli bir servisin logları: docker logs -f llm-backend
    2. Tüm servislerin logları: docker-compose logs -f
    
    DOCKER-COMPOSE LOGS:
    -------------------
    Compose ile başlatılan tüm servislerin loglarını aynı anda gösterir.
    Renkli çıktı ile hangi log hangi servise ait belli olur.
    """
    if service:
        # Belirli bir servisin loglarını göster
        print(f"\n📋 {service} logları:")
        run_command(f"docker logs -f {service}", check=False)
    else:
        # Tüm servislerin loglarını göster (Compose ile)
        print("\n📋 Tüm loglar:")
        run_command("docker-compose logs -f", check=False)


def show_status():
    """
    Container durumlarını göster
    
    DOCKER PS KOMUTU:
    ----------------
    docker ps → Çalışan container'ları listele
    docker ps -a → Tüm container'ları listele (durdurulmuş olanlar dahil)
    
    FILTER KULLANIMI:
    ----------------
    --filter name=llm-
    → İsmi "llm-" ile başlayan container'ları filtrele
    → Bu projede tüm container isimleri "llm-" ile başlar
    
    ÇIKTI BİLGİLERİ:
    ---------------
    CONTAINER ID  | IMAGE         | STATUS        | PORTS        | NAMES
    abc123        | llm-backend   | Up 2 hours    | 8000->8000   | llm-backend
    
    - CONTAINER ID: Benzersiz kimlik (kısa form)
    - IMAGE: Kullanılan Docker image
    - STATUS: Durum (Up = çalışıyor, Exited = durmuş)
    - PORTS: Port mapping'ler
    - NAMES: Container adı
    
    STATUS ANALİZİ:
    --------------
    - "Up X minutes/hours" → Container sağlıklı çalışıyor
    - "Exited (0)" → Container normal şekilde durdu
    - "Exited (1)" → Container hata ile durdu (log kontrol edin)
    - "Restarting" → Container sürekli yeniden başlıyor (sorun var)
    """
    print("\n📊 Container Durumları:")
    # llm- ile başlayan tüm container'ları göster
    run_command("docker ps -a --filter name=llm-", check=False)


# ============================================================================
# ANA MENÜ VE KULLANICI ARAYÜZÜ
# ============================================================================

def main():
    """
    Ana menü - Script'in giriş noktası
    
    SCRIPT AKIŞI:
    ------------
    1. Başlık ve bilgilendirme
    2. Ön koşul kontrolleri (Docker, Compose, .env)
    3. Menü gösterimi
    4. Kullanıcı seçimine göre ilgili fonksiyonu çağır
    
    MENÜ TASARIM PRENSİPLERİ:
    ------------------------
    - Her seçenek net ve anlaşılır
    - Mantıksal gruplandırma (build, start, manage)
    - 0 ile çıkış (yaygın konvansiyon)
    - Emoji kullanımı ile görsel zenginlik
    
    HATA YÖNETİMİ:
    -------------
    - Geçersiz seçimde uyarı ver
    - Ön koşullar sağlanmadıysa sys.exit(1) ile çık
    - KeyboardInterrupt (Ctrl+C) yakalanır
    
    SYS.EXIT KODLARI:
    ----------------
    - sys.exit(0) → Başarılı çıkış
    - sys.exit(1) → Hatalı çıkış
    Bu kodlar script'i çağıran ortamda kontrol edilebilir.
    """
    # Başlık banner'ı
    print("=" * 60)
    print("🐳 Docker Setup ve Deployment Script")
    print("=" * 60)
    
    # ====== ÖN KOŞUL KONTROLLERİ ======
    # Docker mutlaka yüklü olmalı, yoksa script çalışamaz
    if not check_docker_installed():
        sys.exit(1)  # Kritik hata, çık
    
    # Docker Compose opsiyonel, uyarı ver ama devam et
    check_docker_compose_installed()
    
    # .env dosyası opsiyonel, uyarı ver ama devam et
    # API anahtarları olmadan bazı özellikler çalışmayabilir
    check_env_file()
    
    # ====== MENÜ GÖSTERME ======
    print("\n" + "=" * 60)
    print("Menü:")
    print("1. Backend API image build et")
    print("2. Gradio frontend image build et")
    print("3. Streamlit frontend image build et")
    print("4. Tüm image'ları build et")
    print("5. Backend API container başlat")
    print("6. Gradio frontend container başlat")
    print("7. Streamlit frontend container başlat")
    print("8. Docker Compose ile tüm servisleri başlat")
    print("9. Container'ları durdur")
    print("10. Container durumlarını göster")
    print("11. Logları göster")
    print("0. Çıkış")
    print("=" * 60)
    
    # ====== KULLANICI GİRDİSİ ALMA ======
    # strip() ile baştaki/sondaki boşlukları temizle
    choice = input("\nSeçiminiz (0-11): ").strip()
    
    # ====== SEÇİM BAZINDA ROUTE ETME ======
    # Her seçim ilgili fonksiyonu çağırır
    
    if choice == "1":
        # Sadece backend image'ı build et
        build_backend_image()
        
    elif choice == "2":
        # Sadece Gradio image'ı build et
        build_gradio_image()
        
    elif choice == "3":
        # Sadece Streamlit image'ı build et
        build_streamlit_image()
        
    elif choice == "4":
        # Tüm image'ları sırayla build et
        # Bağımlılık yok, paralel build için Docker BuildKit kullanılabilir
        build_backend_image()
        build_gradio_image()
        build_streamlit_image()
        
    elif choice == "5":
        # Backend container'ı başlat
        # Frontend'ler için backend gerekli, önce backend başlatılmalı
        start_backend_container()
        
    elif choice == "6":
        # Gradio container'ı başlat
        # Not: Backend'in çalışıyor olması beklenir
        start_gradio_container()
        
    elif choice == "7":
        # Streamlit container'ı başlat
        # Not: Backend'in çalışıyor olması beklenir
        start_streamlit_container()
        
    elif choice == "8":
        # Docker Compose ile tüm servisleri başlat
        # En pratik yöntem: Tek komutla her şey hazır
        start_with_compose()
        
    elif choice == "9":
        # Tüm container'ları durdur ve temizle
        stop_containers()
        
    elif choice == "10":
        # Container'ların durumunu göster
        show_status()
        
    elif choice == "11":
        # Logları göster
        # Kullanıcıdan spesifik servis adı istenir
        service = input("Service adı (boş bırakınca tümü): ").strip()
        # Boş string False olarak değerlendirilir
        show_logs(service if service else None)
        
    elif choice == "0":
        # Çıkış
        print("Çıkılıyor...")
        sys.exit(0)
        
    else:
        # Geçersiz seçim
        print("❌ Geçersiz seçim!")


# ============================================================================
# SCRIPT BAŞLATMA NOKTASI
# ============================================================================

if __name__ == "__main__":
    """
    Python script'i doğrudan çalıştırıldığında bu blok çalışır
    
    __NAME__ == "__MAIN__" AÇIKLAMASI:
    ----------------------------------
    - Script doğrudan çalıştırılırsa: __name__ = "__main__"
    - Script import edilirse: __name__ = modül adı
    
    Bu kontrol sayesinde:
    - import edildiğinde otomatik çalışmaz
    - sadece doğrudan çalıştırıldığında main() çağrılır
    
    TRY-EXCEPT BLOĞU:
    ----------------
    KeyboardInterrupt: Kullanıcı Ctrl+C ile iptal ederse
    → Zarif bir çıkış mesajı göster
    → Stack trace gösterme (kullanıcı dostu)
    
    SYS.EXIT(0):
    -----------
    Normal çıkış kodu. İşletim sistemi ve script çağıran 
    programlar bu kodu kontrol edebilir.
    """
    try:
        main()  # Ana fonksiyonu çalıştır
    except KeyboardInterrupt:
        # Kullanıcı Ctrl+C ile iptal etti
        print("\n\n⚠️ İşlem iptal edildi.")
        sys.exit(0)  # Normal çıkış

"""
GENEL MİMARİ AKIŞ:
==================

1. DEVELOPMENT ORTAMI:
   Developer → Python kodu yazar
   Developer → Docker build yapar
   Developer → Container'ları test eder

2. BUILD SÜRECI:
   Dockerfile → Image oluşturulur
   Image → Docker Hub'a push edilir (opsiyonel)
   
3. DEPLOYMENT SÜRECI:
   Production Server → Image'ı pull eder
   Production Server → Container'ı başlatır
   Production Server → Monitoring yapar

4. CONTAINER YAŞAM DÖNGÜSÜ:
   Created → Running → Stopped → Removed
   
5. SERVIS İLETİŞİMİ:
   User → Frontend (7860/8501)
   Frontend → Backend API (8000)
   Backend → External APIs (OpenAI, HuggingFace)
   Backend → Database (opsiyonel)

BEST PRACTICES:
===============
✅ .env dosyasını git'e commit etmeyin
✅ Image'ları tag'leyin (latest, v1.0.0, vb.)
✅ Multi-stage build kullanın (production için)
✅ Health check'ler ekleyin
✅ Log rotation yapılandırın
✅ Resource limit'leri belirleyin (CPU, RAM)
✅ Security scan yapın (Trivy, Clair)

DEPLOYMENT STRATEJİLERİ:
========================
1. Blue-Green Deployment: İki ortam, sıfır downtime
2. Rolling Update: Kademeli güncelleme
3. Canary Deployment: Küçük grup test, sonra tam
4. A/B Testing: İki version aynı anda

MONITORING VE LOGGING:
======================
- Docker stats → Resource kullanımı
- Docker logs → Uygulama logları
- Prometheus + Grafana → Metrik toplama
- ELK Stack → Merkezi log yönetimi
- Sentry → Error tracking

SCALING:
========
- Horizontal: Daha fazla container (docker-compose scale)
- Vertical: Container'a daha fazla kaynak
- Kubernetes: Otomatik scaling ve orchestration
"""


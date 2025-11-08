"""
FastAPI ile Backend API
LLM tabanlı uygulamalar için RESTful API

Bu modül, OpenAI API'yi kullanarak çeşitli LLM tabanlı işlemleri 
gerçekleştiren bir RESTful API sunucusu oluşturur.
"""

# ============================================================================
# KÜTÜPHANE İMPORT İŞLEMLERİ
# ============================================================================

# FastAPI - Modern, hızlı (yüksek performanslı) web framework'ü
from fastapi import FastAPI, HTTPException, Depends, Header
# FastAPI'nin CORS (Cross-Origin Resource Sharing) middleware'i
# Frontend'den backend'e erişim için gerekli
from fastapi.middleware.cors import CORSMiddleware
# Streaming (akış) yanıtlar için kullanılır (örn: ChatGPT gibi cevap yazdırma)
from fastapi.responses import StreamingResponse
# Pydantic - Veri doğrulama ve model oluşturma için
# API'ye gelen ve giden verilerin şemasını tanımlar
from pydantic import BaseModel, Field
# Type hinting (tip belirtme) için - Kodun okunabilirliğini artırır
from typing import Optional, List, Dict, Any
# OpenAI kütüphanesi - GPT modellerine erişim için
from openai import OpenAI
# İşletim sistemi işlemleri için (çevre değişkenlerini okuma)
import os
# .env dosyasından çevre değişkenlerini yüklemek için
from dotenv import load_dotenv
# ASGI sunucusu - FastAPI uygulamasını çalıştırır
import uvicorn
# JSON verilerini işlemek için
import json
# Zaman damgası oluşturmak için
from datetime import datetime

# ============================================================================
# ÇEVRE DEĞİŞKENLERİNİN YÜKLENMESI
# ============================================================================

# .env dosyasındaki değişkenleri (API anahtarları vb.) yükle
# Bu sayede hassas bilgiler kod içinde görünmez
load_dotenv()

# ============================================================================
# OPENAI CLIENT OLUŞTURMA
# ============================================================================

# OpenAI API client'ı başlat
# API anahtarı çevre değişkeninden okunur (güvenlik için)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ============================================================================
# FASTAPI UYGULAMASI OLUŞTURMA
# ============================================================================

# FastAPI uygulaması oluştur
# Bu, tüm API endpoint'lerinin bağlandığı ana nesnedir
app = FastAPI(
    title="LLM Backend API",  # API başlığı (dokümantasyonda görünür)
    description="LLM tabanlı uygulamalar için RESTful API",  # API açıklaması
    version="1.0.0",  # Versiyon numarası
    docs_url="/docs",  # Swagger UI dokümantasyon URL'i
    redoc_url="/redoc"  # ReDoc dokümantasyon URL'i
)

# ============================================================================
# CORS YAPILANDIRMASI (Cross-Origin Resource Sharing)
# ============================================================================

# CORS middleware ekle
# Bu, farklı domain'lerden (örn: localhost:3000) API'ye erişimi sağlar
# Frontend ve backend farklı portlarda çalışıyorsa zorunludur
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tüm origin'lere izin ver (Production'da ["http://example.com"] gibi spesifik olmalı)
    allow_credentials=True,  # Cookie'lere izin ver
    allow_methods=["*"],  # Tüm HTTP metodlarına (GET, POST, PUT, DELETE) izin ver
    allow_headers=["*"],  # Tüm header'lara izin ver
)

# ============================================================================
# PYDANTIC MODELLERİ (Veri Şemaları)
# ============================================================================
# Pydantic modelleri, API'ye gelen ve giden verilerin yapısını tanımlar
# Otomatik veri doğrulama ve API dokümantasyonu sağlar

class ChatMessage(BaseModel):
    """
    Tek bir chat mesajını temsil eder
    Örnek: {"role": "user", "content": "Merhaba!"}
    """
    role: str = Field(..., description="Mesaj rolü: 'user' veya 'assistant'")
    # ... üç nokta, bu alanın zorunlu (required) olduğunu belirtir
    content: str = Field(..., description="Mesaj içeriği")


class ChatRequest(BaseModel):
    """
    Chat endpoint'ine gönderilen istek verilerini tanımlar
    Konuşma geçmişi ve model parametrelerini içerir
    """
    messages: List[ChatMessage] = Field(..., description="Konuşma mesajları")
    # Liste içinde ChatMessage nesneleri bulunur
    model: str = Field(default="gpt-3.5-turbo", description="Kullanılacak model")
    # default: Değer girilmezse kullanılacak varsayılan değer
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Yaratıcılık seviyesi")
    # ge: greater or equal (büyük eşit), le: less or equal (küçük eşit)
    # Temperature: 0'a yakın = tutarlı, 2'ye yakın = yaratıcı
    max_tokens: int = Field(default=150, ge=1, le=4000, description="Maksimum token sayısı")
    # Token: Yaklaşık olarak kelime parçaları (1 token ≈ 0.75 kelime)
    stream: bool = Field(default=False, description="Streaming yanıt isteniyor mu?")
    # True ise yanıt kelime kelime gelir (ChatGPT gibi)


class TextProcessRequest(BaseModel):
    """
    Metin işleme endpoint'i için istek modeli
    Özetleme, çeviri ve analiz işlemleri için kullanılır
    """
    text: str = Field(..., description="İşlenecek metin")
    operation: str = Field(..., description="İşlem türü: 'summarize', 'translate', 'analyze'")
    language: Optional[str] = Field(default=None, description="Hedef dil (çeviri için)")
    # Optional: Bu alan boş bırakılabilir (None olabilir)
    model: str = Field(default="gpt-3.5-turbo", description="Kullanılacak model")


class CodeExplainRequest(BaseModel):
    """
    Kod açıklama endpoint'i için istek modeli
    Programlama kodlarının açıklanması için kullanılır
    """
    code: str = Field(..., description="Açıklanacak kod")
    language: str = Field(..., description="Programlama dili")
    model: str = Field(default="gpt-3.5-turbo", description="Kullanılacak model")


class HealthResponse(BaseModel):
    """
    Health check endpoint'inin yanıt modeli
    Sistemin durumunu kontrol etmek için kullanılır
    """
    status: str  # Örn: "healthy" veya "unhealthy"
    timestamp: str  # İsteğin yapıldığı zaman
    api_key_configured: bool  # OpenAI API anahtarı yapılandırılmış mı?


# ============================================================================
# YARDIMCI FONKSİYONLAR
# ============================================================================

def get_openai_response(messages: List[Dict[str, str]], model: str, temperature: float, max_tokens: int):
    """
    OpenAI API'den standart (non-streaming) yanıt alır
    
    Args:
        messages: Konuşma geçmişi (liste halinde mesajlar)
        model: Kullanılacak GPT modeli
        temperature: Yaratıcılık seviyesi (0.0-2.0)
        max_tokens: Maksimum yanıt uzunluğu
    
    Returns:
        str: AI'ın ürettiği metin yanıtı
    
    Raises:
        HTTPException: API hatası durumunda
    """
    try:
        # OpenAI API'ye istek gönder
        response = client.chat.completions.create(
            model=model,  # Kullanılacak model
            messages=messages,  # Konuşma geçmişi
            temperature=temperature,  # Yaratıcılık seviyesi
            max_tokens=max_tokens  # Maksimum yanıt uzunluğu
        )
        # API yanıtından sadece metin içeriğini döndür
        return response.choices[0].message.content
    except Exception as e:
        # Hata oluşursa HTTP 500 hatası fırlat
        raise HTTPException(status_code=500, detail=f"OpenAI API hatası: {str(e)}")


def stream_openai_response(messages: List[Dict[str, str]], model: str, temperature: float, max_tokens: int):
    """
    OpenAI API'den streaming (akış) yanıt alır
    ChatGPT gibi kelime kelime yanıt üretir
    
    Generator fonksiyondur - yield ile veri parçaları döndürür
    
    Args:
        messages: Konuşma geçmişi
        model: Kullanılacak GPT modeli
        temperature: Yaratıcılık seviyesi
        max_tokens: Maksimum yanıt uzunluğu
    
    Yields:
        str: Server-Sent Events (SSE) formatında veri parçaları
    """
    try:
        # OpenAI API'ye streaming isteği gönder
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True  # Streaming aktif
        )
        
        # Her bir yanıt parçasını (chunk) işle
        for chunk in response:
            # Eğer chunk içinde içerik varsa
            if chunk.choices[0].delta.content:
                # Server-Sent Events formatında gönder
                # "data: " ile başlar, "\n\n" ile biter
                yield f"data: {json.dumps({'content': chunk.choices[0].delta.content})}\n\n"
        
        # Stream tamamlandığında sinyal gönder
        yield "data: [DONE]\n\n"
    except Exception as e:
        # Hata durumunda hata mesajını stream olarak gönder
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


# ============================================================================
# API ENDPOINT'LERİ
# ============================================================================
# Endpoint: API'nin erişilebilir URL'leri
# Her endpoint belirli bir işlevi yerine getirir

@app.get("/", tags=["General"])
async def root():
    """
    Root (kök) endpoint - API hakkında temel bilgi
    
    HTTP Method: GET
    URL: http://localhost:8000/
    
    Returns:
        dict: Hoş geldiniz mesajı ve yönlendirme linkleri
    """
    return {
        "message": "LLM Backend API'ye hoş geldiniz!",
        "docs": "/docs",  # Swagger dokümantasyonu
        "health": "/health"  # Sağlık kontrolü
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Health check endpoint - Sistem durumu kontrolü
    
    HTTP Method: GET
    URL: http://localhost:8000/health
    
    Bu endpoint sistem canlılığını ve API anahtarı durumunu kontrol eder
    Monitoring araçları tarafından kullanılabilir
    
    Returns:
        HealthResponse: Sistem durumu bilgileri
    """
    return HealthResponse(
        status="healthy",  # Sistem durumu
        timestamp=datetime.now().isoformat(),  # ISO 8601 formatında zaman
        api_key_configured=bool(os.getenv("OPENAI_API_KEY"))  # API anahtarı var mı?
    )


@app.post("/chat", tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Ana chat endpoint - Tam özellikli sohbet
    
    HTTP Method: POST
    URL: http://localhost:8000/chat
    
    Bu endpoint:
    - Konuşma geçmişini destekler
    - Model parametrelerini özelleştirebilir
    - Normal veya streaming yanıt verebilir
    
    Args:
        request: ChatRequest modeline uygun JSON verisi
    
    Returns:
        dict veya StreamingResponse: AI yanıtı ve metadata
    """
    try:
        # Pydantic modellerini dict formatına çevir (OpenAI API formatı)
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Streaming isteniyorsa
        if request.stream:
            # StreamingResponse döndür - kelime kelime yanıt
            return StreamingResponse(
                stream_openai_response(messages, request.model, request.temperature, request.max_tokens),
                media_type="text/event-stream"  # SSE content type
            )
        
        # Normal (non-streaming) yanıt
        response_content = get_openai_response(
            messages,
            request.model,
            request.temperature,
            request.max_tokens
        )
        
        # Yanıtı metadata ile birlikte döndür
        return {
            "response": response_content,  # AI'ın cevabı
            "model": request.model,  # Kullanılan model
            "usage": {  # Token kullanım istatistikleri (yaklaşık)
                "prompt_tokens": len(str(messages)),  # Gönderilen mesaj token sayısı
                "completion_tokens": len(response_content.split()),  # Yanıt token sayısı
                "total_tokens": len(str(messages)) + len(response_content.split())  # Toplam
            }
        }
    except Exception as e:
        # Hata durumunda HTTP 500 döndür
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/simple", tags=["Chat"])
async def chat_simple(message: str, model: str = "gpt-3.5-turbo"):
    """
    Basitleştirilmiş chat endpoint - Tek mesaj gönderme
    
    HTTP Method: POST
    URL: http://localhost:8000/chat/simple?message=Merhaba&model=gpt-3.5-turbo
    
    Bu endpoint hızlı test için kullanışlıdır
    Konuşma geçmişi tutmaz, tek seferlik soru-cevap yapar
    
    Args:
        message: Kullanıcı mesajı (query parameter)
        model: Kullanılacak model (opsiyonel)
    
    Returns:
        dict: Mesaj ve AI yanıtı
    """
    try:
        # Basit bir konuşma oluştur
        messages = [
            {"role": "system", "content": "Sen yardımcı bir asistansın."},  # Sistem rolü
            {"role": "user", "content": message}  # Kullanıcı mesajı
        ]
        
        # AI yanıtı al
        response_content = get_openai_response(messages, model, 0.7, 150)
        
        return {
            "message": message,  # Gönderilen mesaj
            "response": response_content,  # AI yanıtı
            "model": model  # Kullanılan model
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/text/process", tags=["Text Processing"])
async def process_text(request: TextProcessRequest):
    """
    Genel metin işleme endpoint - Özetleme, çeviri, analiz
    
    HTTP Method: POST
    URL: http://localhost:8000/text/process
    
    Desteklenen işlemler:
    - summarize: Metni özetle
    - translate: Metni başka dile çevir
    - analyze: Metni analiz et ve yorum yap
    
    Args:
        request: TextProcessRequest modeline uygun JSON
    
    Returns:
        dict: Orijinal metin, işlem ve sonuç
    """
    try:
        # Her işlem türü için farklı system prompt tanımla
        system_prompts = {
            "summarize": "Sen bir metin özetleme uzmanısın. Verilen metni kısa ve öz şekilde özetle.",
            "translate": f"Sen bir çevirmensin. Verilen metni {request.language or 'İngilizce'} diline çevir.",
            "analyze": "Sen bir metin analiz uzmanısın. Verilen metni analiz et ve yorum yap."
        }
        
        # İşlem türüne göre system prompt seç
        system_prompt = system_prompts.get(
            request.operation,
            "Sen yardımcı bir asistansın."  # Varsayılan
        )
        
        # İşlem türüne göre user prompt oluştur
        if request.operation == "translate" and request.language:
            user_prompt = f"Bu metni {request.language} diline çevir:\n\n{request.text}"
        elif request.operation == "summarize":
            user_prompt = f"Bu metni özetle:\n\n{request.text}"
        else:
            user_prompt = f"Bu metni analiz et:\n\n{request.text}"
        
        # Konuşma mesajlarını oluştur
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # AI yanıtı al (düşük temperature - daha tutarlı sonuçlar için)
        response_content = get_openai_response(messages, request.model, 0.5, 200)
        
        return {
            "original_text": request.text,  # Orijinal metin
            "operation": request.operation,  # Yapılan işlem
            "result": response_content,  # İşlem sonucu
            "model": request.model  # Kullanılan model
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/code/explain", tags=["Code"])
async def explain_code(request: CodeExplainRequest):
    """
    Kod açıklama endpoint - Programlama kodlarını açıklar
    
    HTTP Method: POST
    URL: http://localhost:8000/code/explain
    
    Kullanım: Öğrenciler veya geliştiriciler için kod eğitimi
    Verilen kodu satır satır veya genel olarak açıklar
    
    Args:
        request: CodeExplainRequest modeline uygun JSON
    
    Returns:
        dict: Kod, dil ve açıklama
    """
    try:
        # Programlama diline özel system prompt
        system_prompt = f"Sen bir {request.language} programlama uzmanısın. Verilen kodu detaylı şekilde açıkla."
        
        # Kodu markdown kod bloğu içinde gönder (daha iyi formatting)
        user_prompt = f"Bu kodu açıkla:\n\n```{request.language.lower()}\n{request.code}\n```"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # AI yanıtı al (daha uzun yanıt için max_tokens=300)
        response_content = get_openai_response(messages, request.model, 0.5, 300)
        
        return {
            "code": request.code,  # Açıklanan kod
            "language": request.language,  # Programlama dili
            "explanation": response_content,  # Kod açıklaması
            "model": request.model  # Kullanılan model
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/text/summarize", tags=["Text Processing"])
async def summarize_text(text: str, model: str = "gpt-3.5-turbo"):
    """
    Hızlı metin özetleme endpoint
    
    HTTP Method: POST
    URL: http://localhost:8000/text/summarize?text=Uzun metin...
    
    /text/process endpoint'inin özel hali
    Daha basit kullanım için query parameter kabul eder
    
    Args:
        text: Özetlenecek metin (query parameter)
        model: Kullanılacak model (opsiyonel)
    
    Returns:
        dict: Orijinal metin ve özet
    """
    try:
        messages = [
            {"role": "system", "content": "Sen bir metin özetleme uzmanısın. Verilen metni kısa ve öz şekilde özetle."},
            {"role": "user", "content": f"Bu metni özetle:\n\n{text}"}
        ]
        
        # Düşük temperature - tutarlı özetler için
        response_content = get_openai_response(messages, model, 0.5, 150)
        
        return {
            "original_text": text,
            "summary": response_content,
            "model": model
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/text/translate", tags=["Text Processing"])
async def translate_text(text: str, target_language: str = "İngilizce", model: str = "gpt-3.5-turbo"):
    """
    Hızlı metin çeviri endpoint
    
    HTTP Method: POST
    URL: http://localhost:8000/text/translate?text=Hello&target_language=Türkçe
    
    /text/process endpoint'inin özel hali
    Daha basit kullanım için query parameter kabul eder
    
    Args:
        text: Çevrilecek metin (query parameter)
        target_language: Hedef dil (varsayılan: İngilizce)
        model: Kullanılacak model (opsiyonel)
    
    Returns:
        dict: Orijinal metin, hedef dil ve çeviri
    """
    try:
        messages = [
            {"role": "system", "content": f"Sen bir çevirmensin. Verilen metni {target_language} diline çevir."},
            {"role": "user", "content": text}
        ]
        
        # Çok düşük temperature - çeviride yaratıcılık istemiyoruz
        response_content = get_openai_response(messages, model, 0.3, 200)
        
        return {
            "original_text": text,
            "target_language": target_language,
            "translation": response_content,
            "model": model
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ERROR HANDLERS (Hata Yöneticileri)
# ============================================================================
# Özel hata yanıtları tanımlamak için kullanılır

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """
    404 (Not Found) hatası için özel yanıt
    
    Kullanıcı olmayan bir endpoint'e istek yaptığında çalışır
    Örnek: GET /nonexistent
    
    Args:
        request: HTTP isteği nesnesi
        exc: Exception nesnesi
    
    Returns:
        dict: Hata mesajı ve mevcut endpoint'ler
    """
    return {
        "error": "Endpoint bulunamadı",
        "path": str(request.url.path),  # İstek yapılan yol
        "available_endpoints": ["/chat", "/text/process", "/code/explain"]  # Kullanılabilir endpoint'ler
    }


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """
    500 (Internal Server Error) hatası için özel yanıt
    
    Sunucu tarafında beklenmeyen hata oluştuğunda çalışır
    
    Args:
        request: HTTP isteği nesnesi
        exc: Exception nesnesi
    
    Returns:
        dict: Hata mesajı ve detaylar
    """
    return {
        "error": "Internal server error",
        "message": str(exc),  # Hata mesajı
        "path": str(request.url.path)  # Hatanın oluştuğu yol
    }


# ============================================================================
# UYGULAMA ÇALIŞTIRMA
# ============================================================================

if __name__ == "__main__":
    """
    Script doğrudan çalıştırıldığında (python 3_fastapi_backend.py)
    bu blok çalışır
    
    Module olarak import edildiğinde çalışmaz
    """
    uvicorn.run(
        "3_fastapi_backend:app",  # Modül adı ve FastAPI app nesnesi
        host="0.0.0.0",  # Tüm network interface'lerinde dinle (localhost ve IP adresi)
        port=8000,  # Port numarası
        reload=True,  # Kod değiştiğinde otomatik yeniden başlat (development için)
        log_level="info"  # Log seviyesi (info, debug, warning, error)
    )

"""
=============================================================================
KULLANIM ÖRNEKLERİ
=============================================================================

1. UYGULAMAYI ÇALIŞTIRMA:
   Terminal'de: python 3_fastapi_backend.py
   Veya: uvicorn 3_fastapi_backend:app --reload

2. API DOKÜMANTASYONU:
   Swagger UI: http://localhost:8000/docs
   ReDoc: http://localhost:8000/redoc

3. ÖRNEK İSTEKLER (curl veya Postman ile):

   a) Basit Chat:
   curl -X POST "http://localhost:8000/chat/simple?message=Merhaba"

   b) Tam Chat (JSON body ile):
   curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{
       "messages": [
         {"role": "user", "content": "Python nedir?"}
       ],
       "model": "gpt-3.5-turbo",
       "temperature": 0.7
     }'

   c) Metin Özetleme:
   curl -X POST "http://localhost:8000/text/summarize?text=Uzun metin..."

   d) Kod Açıklama:
   curl -X POST "http://localhost:8000/code/explain" \
     -H "Content-Type: application/json" \
     -d '{
       "code": "def hello(): print(\"Hi\")",
       "language": "Python"
     }'

=============================================================================
ÖNEMLİ KAVRAMLAR
=============================================================================

1. ASGI (Asynchronous Server Gateway Interface):
   - Python web uygulamaları için modern standart
   - Async/await desteği
   - WebSocket ve streaming desteği
   - FastAPI ve Uvicorn birlikte ASGI kullanır

2. PYDANTIC:
   - Veri doğrulama kütüphanesi
   - Type hints kullanarak otomatik validasyon
   - JSON Schema üretimi
   - API dokümantasyonu için kritik

3. ENDPOINT:
   - API'nin erişilebilir URL'leri
   - HTTP metodları: GET, POST, PUT, DELETE
   - Her endpoint belirli bir işlevi yerine getirir

4. MIDDLEWARE:
   - Request ve response arasında çalışan katman
   - CORS, authentication, logging için kullanılır
   - Her istek middleware'den geçer

5. STREAMING:
   - Veri parça parça gönderme
   - Server-Sent Events (SSE) protokolü
   - ChatGPT benzeri kelime kelime yanıt için

6. TEMPERATURE (LLM Parametresi):
   - 0.0: Deterministik, her seferinde aynı cevap
   - 0.7: Dengeli (varsayılan)
   - 2.0: Maksimum yaratıcılık, beklenmedik sonuçlar

7. TOKENS:
   - LLM'lerin metin birimi
   - 1 token ≈ 4 karakter veya 0.75 kelime
   - Maliyet ve limit hesaplamalarında kullanılır

=============================================================================
GÜVENLİK ÖNERİLERİ
=============================================================================

1. API anahtarlarını .env dosyasında tutun
2. Production'da CORS origins'i sınırlandırın
3. Rate limiting ekleyin (örn: slowapi kütüphanesi)
4. Authentication/Authorization uygulayın
5. Input validation yapın (Pydantic otomatik yapar)
6. HTTPS kullanın (production'da)
7. API anahtarlarını asla kod içinde yazmayın
8. Environment-specific konfigürasyon kullanın

=============================================================================
PERFORMANS OPTİMİZASYONU
=============================================================================

1. Async/await kullanın (I/O operasyonları için)
2. Connection pooling uygulayın
3. Caching mekanizması ekleyin (Redis, Memcached)
4. Response compression aktifleyin
5. Database query'lerini optimize edin
6. Background tasks kullanın (uzun işlemler için)
7. Load balancing yapın (production'da)

=============================================================================
"""


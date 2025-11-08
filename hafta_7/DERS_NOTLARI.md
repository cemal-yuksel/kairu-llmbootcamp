# Ders Notları - Modül 8: LLM Tabanlı Uygulama Dağıtımı

## 📚 Konu Özeti

Bu modülde, LLM tabanlı uygulamaları production ortamına dağıtmayı öğreneceksiniz. Frontend (Gradio/Streamlit), Backend (FastAPI) ve Deployment (Docker) konularını kapsamlı şekilde ele alacağız.

## 🎯 Öğrenme Hedefleri

Bu modül sonunda:
- ✅ Gradio ile hızlı prototipleme yapabileceksiniz
- ✅ Streamlit ile interaktif uygulamalar geliştirebileceksiniz
- ✅ FastAPI ile RESTful API tasarlayabileceksiniz
- ✅ Frontend ve Backend'i entegre edebileceksiniz
- ✅ Docker ile containerization yapabileceksiniz
- ✅ Production ortamına deploy edebileceksiniz

## 📖 Konu 1: Gradio ile Frontend

### Gradio Nedir?

Gradio, Python tabanlı bir web arayüzü framework'üdür. LLM uygulamaları için hızlı prototipleme yapmanıza olanak sağlar.

### Avantajları

- 🚀 Hızlı prototipleme
- 📱 Otomatik UI oluşturma
- 🌊 Streaming desteği
- 🔄 Real-time feedback
- 📊 Multi-modal input/output

### Temel Kullanım

```python
import gradio as gr

def chatbot(message, history):
    # LLM ile yanıt oluştur
    response = "Yanıt buraya gelecek"
    return response

# Arayüz oluştur
demo = gr.Chatbot(chatbot)
demo.launch()
```

### Önemli Özellikler

1. **Chatbot Interface**: `gr.Chatbot()` ile sohbet arayüzü
2. **Streaming**: `yield` ile real-time yanıtlar
3. **File Upload**: `gr.File()` ile dosya yükleme
4. **Tabs**: `gr.Tabs()` ile çoklu sayfa
5. **Customization**: Theme ve styling özelleştirme

## 📖 Konu 2: Streamlit ile Frontend

### Streamlit Nedir?

Streamlit, data science ve machine learning uygulamaları için özel olarak tasarlanmış bir web framework'üdür.

### Avantajları

- 📊 Data visualization odaklı
- 🎨 Widget'lar ve interaktif öğeler
- 💾 Session state yönetimi
- 📈 Plotly, Matplotlib entegrasyonu
- 🔄 Real-time updates

### Temel Kullanım

```python
import streamlit as st

st.title("Başlık")
message = st.text_input("Mesajınız")
if st.button("Gönder"):
    st.write("Yanıt buraya gelecek")
```

### Önemli Özellikler

1. **Session State**: `st.session_state` ile durum yönetimi
2. **Widgets**: `st.button()`, `st.slider()`, `st.selectbox()`
3. **Chat Interface**: `st.chat_message()` ile sohbet
4. **Caching**: `@st.cache` ile performans optimizasyonu
5. **Layout**: `st.columns()`, `st.sidebar()` ile düzenleme

## 📖 Konu 3: FastAPI ile Backend

### FastAPI Nedir?

FastAPI, modern, hızlı (yüksek performanslı) bir web framework'üdür. Python 3.7+ için standart Python type hints'e dayanır.

### Avantajları

- ⚡ Yüksek performans
- 📝 Otomatik API dokümantasyonu
- ✅ Pydantic ile veri doğrulama
- 🔄 Async/await desteği
- 🔒 Type hints ile tip güvenliği

### Temel Kullanım

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Message(BaseModel):
    content: str

@app.post("/chat")
async def chat(message: Message):
    # LLM ile yanıt oluştur
    response = "Yanıt buraya gelecek"
    return {"response": response}
```

### Önemli Özellikler

1. **Pydantic Models**: Veri doğrulama ve serialization
2. **Async Endpoints**: `async def` ile asenkron işlemler
3. **CORS Middleware**: Cross-origin istekleri
4. **Streaming Responses**: Real-time yanıtlar
5. **OpenAPI Docs**: Otomatik Swagger UI

### API Endpoint Türleri

- **GET**: Veri okuma
- **POST**: Veri gönderme
- **PUT**: Veri güncelleme
- **DELETE**: Veri silme

## 📖 Konu 4: Frontend-Backend Entegrasyonu

### Entegrasyon Yaklaşımı

1. **Backend API**: FastAPI ile RESTful API oluştur
2. **Frontend**: Gradio veya Streamlit ile UI oluştur
3. **HTTP Requests**: `requests` veya `httpx` ile API çağrıları
4. **Error Handling**: Hata yönetimi ve fallback mekanizmaları

### Örnek Entegrasyon

```python
import requests

def chat_with_api(message):
    response = requests.post(
        "http://localhost:8000/chat/simple",
        params={"message": message}
    )
    return response.json()["response"]
```

### Best Practices

- ✅ API URL'leri environment variable'larda sakla
- ✅ Error handling ekle
- ✅ Timeout ayarla
- ✅ Retry mekanizması ekle
- ✅ Loading states göster

## 📖 Konu 5: Docker ve Deployment

### Docker Nedir?

Docker, uygulamaları container'lara paketleyen bir platformdur. Uygulamaları herhangi bir ortamda çalıştırmanıza olanak sağlar.

### Avantajları

- 🔒 İzolasyon
- 🚀 Hızlı deployment
- 📦 Tutarlı ortamlar
- 🔄 Kolay scaling
- 💰 Maliyet tasarrufu

### Dockerfile Yapısı

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0"]
```

### Docker Compose

Multi-container uygulamaları yönetmek için:

```yaml
version: '3.8'
services:
  backend:
    build: .
    ports:
      - "8000:8000"
  frontend:
    build: .
    ports:
      - "7860:7860"
```

### Deployment Adımları

1. **Dockerfile Oluştur**: Uygulama için image tanımla
2. **Image Build Et**: `docker build -t app-name .`
3. **Container Çalıştır**: `docker run -p 8000:8000 app-name`
4. **Docker Compose**: Multi-container için `docker-compose up`

## 🎓 Önemli Konseptler

### Frontend Framework Seçimi

**Gradio kullanın eğer:**
- Hızlı prototipleme yapıyorsanız
- Minimal kod ile çalışmak istiyorsanız
- Streaming output'a ihtiyacınız varsa

**Streamlit kullanın eğer:**
- Data visualization önemliyse
- Widget'lar ve interaktif öğeler gerekiyorsa
- Session state yönetimi gerekiyorsa

### Backend API Tasarımı

- ✅ RESTful prensiplerine uyun
- ✅ Pydantic ile veri doğrulama yapın
- ✅ Async/await kullanın
- ✅ Error handling ekleyin
- ✅ API dokümantasyonu yazın

### Deployment Stratejisi

1. **Development**: Local ortamda test
2. **Staging**: Test ortamında deploy
3. **Production**: Canlı ortamda deploy
4. **Monitoring**: Log ve metrik takibi

## 🔧 Pratik İpuçları

### Performance Optimizasyonu

- **Caching**: Sık kullanılan verileri cache'le
- **Async Operations**: Asenkron işlemler kullan
- **Connection Pooling**: HTTP bağlantılarını pool'la
- **Rate Limiting**: API isteklerini sınırla

### Security Best Practices

- ✅ API key'leri environment variable'larda sakla
- ✅ CORS ayarlarını sınırla
- ✅ Input validation yap
- ✅ HTTPS kullan
- ✅ Rate limiting ekle

### Error Handling

```python
try:
    response = api_call()
except requests.exceptions.ConnectionError:
    return "Backend API'ye bağlanılamadı"
except Exception as e:
    return f"Hata: {str(e)}"
```

## 📊 Karşılaştırma Tablosu

| Özellik | Gradio | Streamlit | FastAPI |
|---------|--------|-----------|---------|
| Kullanım Amacı | Prototipleme | Data Apps | Backend API |
| Öğrenme Eğrisi | Düşük | Düşük | Orta |
| Performans | İyi | İyi | Çok İyi |
| Customization | Sınırlı | Orta | Yüksek |
| Deployment | Kolay | Kolay | Orta |

## 🎯 Ödev Hazırlığı

Ödev için aşağıdaki konuları anladığınızdan emin olun:

1. **Frontend Seçimi**: Ne zaman Gradio, ne zaman Streamlit?
2. **API Tasarımı**: RESTful API nasıl tasarlanır?
3. **Docker Containerization**: Uygulamayı nasıl containerize edersiniz?
4. **Deployment**: Production ortamına nasıl deploy edersiniz?

## 📚 Ek Kaynaklar

- [Gradio Documentation](https://www.gradio.app/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Guide](https://docs.docker.com/compose/)

## 💡 Sonuç

Bu modülde, LLM tabanlı uygulamaları production ortamına dağıtmayı öğrendiniz. Frontend (Gradio/Streamlit), Backend (FastAPI) ve Deployment (Docker) konularında derinlemesine bilgi edindiniz.

**Başarılar! 🚀**


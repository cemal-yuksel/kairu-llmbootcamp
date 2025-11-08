"""
Frontend-Backend Entegrasyonu Ders Notu
========================================

Bu modül, modern web uygulamalarında frontend ve backend ayrımının nasıl yapılacağını gösterir.
İki popüler Python frontend framework'ü (Gradio ve Streamlit) ile FastAPI backend'inin 
nasıl entegre edileceğini öğreneceğiz.

Mimari Yapı:
-----------
Frontend (Kullanıcı Arayüzü)     →  HTTP İstekleri  →  Backend (FastAPI)
├─ Gradio (Port 7861)                                   ├─ REST API Endpoint'leri
└─ Streamlit (Port 8501)                                ├─ İş Mantığı
                                                         └─ AI Modelleri (OpenAI)

Öğrenilecek Konular:
-------------------
1. HTTP istekleri ile API çağrıları (requests kütüphanesi)
2. Frontend-Backend ayrımının avantajları
3. Asenkron iletişim ve hata yönetimi
4. Gradio ve Streamlit ile entegrasyon
5. RESTful API kullanımı best practices
"""

# ============================================================================
# 1. KÜTÜPHANE İMPORT'LARI
# ============================================================================

# Gradio: Hızlı prototipleme için kullanıcı arayüzü kütüphanesi
import gradio as gr

# Streamlit: Veri bilimi ve ML uygulamaları için web framework'ü
import streamlit as st

# Requests: HTTP istekleri göndermek için standart Python kütüphanesi
# Backend API'sine bağlanmak için kullanılır
import requests

# JSON: API yanıtlarını işlemek için (Python'da built-in)
import json

# Type Hints: Kod okunabilirliği ve IDE desteği için tip belirtme
from typing import List, Dict, Any

# OS: İşletim sistemi işlemleri (environment variables okuma)
import os

# Dotenv: .env dosyasından environment variables yükleme
from dotenv import load_dotenv

# ============================================================================
# 2. KONFIGÜRASYON VE BAŞLANGIÇ AYARLARI
# ============================================================================

# .env dosyasından environment variables'ları yükle
# Örnek .env içeriği:
# API_BASE_URL=http://localhost:8000
# OPENAI_API_KEY=sk-...
load_dotenv()

# Backend API'nin temel URL'ini al
# Eğer .env'de yoksa default olarak localhost:8000 kullan
# Bu sayede farklı ortamlarda (development, production) kolayca değiştirebiliriz
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# ============================================================================
# 3. GRADIO + FASTAPI ENTEGRASYONU
# ============================================================================
# Gradio, kullanıcı dostu arayüzler oluşturmak için harika bir araçtır.
# Özellikle ML modellerini hızlıca demo etmek için tercih edilir.
# ============================================================================

def gradio_chat_with_api(message, history):
    """
    Gradio Chatbot Fonksiyonu - Backend API Entegrasyonu
    ====================================================
    
    Bu fonksiyon, kullanıcının mesajını alır ve FastAPI backend'ine gönderir.
    Backend'den gelen yanıtı kullanıcıya döndürür.
    
    Parametreler:
    ------------
    message (str): Kullanıcının gönderdiği mesaj
    history (list): Önceki konuşma geçmişi (Gradio tarafından yönetilir)
                   Her eleman (kullanıcı_mesajı, bot_yanıtı) tuple'ı şeklindedir
    
    Döndürür:
    --------
    str: Backend'den gelen AI yanıtı veya hata mesajı
    
    API Endpoint:
    ------------
    POST /chat/simple
    Query Parameters: message, model
    
    Çalışma Mantığı:
    ---------------
    1. Kullanıcı mesajı alınır
    2. HTTP POST isteği ile backend'e gönderilir
    3. Backend, OpenAI API'sini çağırır
    4. Yanıt alınır ve kullanıcıya gösterilir
    """
    try:
        # HTTP POST isteği oluştur
        # requests.post() fonksiyonu senkron çalışır (yanıt gelene kadar bekler)
        response = requests.post(
            # API endpoint'inin tam URL'i
            f"{API_BASE_URL}/chat/simple",
            
            # Query parametreleri (URL'e ?message=...&model=... şeklinde eklenir)
            params={
                "message": message,              # Kullanıcının mesajı
                "model": "gpt-3.5-turbo"        # Kullanılacak AI modeli
            },
            
            # Timeout: 30 saniye içinde yanıt gelmezse exception fırlat
            # Bu, sonsuz beklemek yerine kullanıcıya hata mesajı göstermemizi sağlar
            timeout=30
        )
        
        # HTTP status code kontrol et
        # 200: OK - İstek başarılı
        # 4xx: Client Error - İstek hatalı
        # 5xx: Server Error - Backend'de hata oluştu
        if response.status_code == 200:
            # JSON yanıtını Python dictionary'sine çevir
            data = response.json()
            
            # "response" anahtarındaki değeri al
            # Eğer yoksa default mesaj döndür
            return data.get("response", "Yanıt alınamadı")
        else:
            # Hata durumunda detaylı bilgi ver
            return f"Hata: {response.status_code} - {response.text}"
            
    except requests.exceptions.ConnectionError:
        # Backend API'ye bağlanılamazsa (API kapalıysa)
        # Bu, en yaygın hata durumudur - kullanıcı dostu mesaj ver
        return "❌ Backend API'ye bağlanılamadı. API'nin çalıştığından emin olun."
        
    except Exception as e:
        # Beklenmeyen tüm diğer hatalar için genel catch bloğu
        # Production'da bu hataları loglama sistemine kaydetmeliyiz
        return f"Hata oluştu: {str(e)}"


def gradio_summarize_with_api(text):
    """
    Metin Özetleme Fonksiyonu - Backend API Entegrasyonu
    ===================================================
    
    Uzun metinleri özetlemek için backend API'sini kullanır.
    
    Parametreler:
    ------------
    text (str): Özetlenecek metin (makale, döküman vb.)
    
    Döndürür:
    --------
    str: Özetlenmiş metin
    
    Kullanım Senaryoları:
    --------------------
    - Uzun makaleleri özetleme
    - Meeting notlarını özetleme
    - Araştırma raporlarını özetleme
    
    API Çağrısı:
    -----------
    POST /text/summarize
    Params: text, model
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/text/summarize",
            params={
                "text": text,                    # Özetlenecek metin
                "model": "gpt-3.5-turbo"        # AI modeli
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("summary", "Özet oluşturulamadı")
        else:
            return f"Hata: {response.status_code} - {response.text}"
            
    except requests.exceptions.ConnectionError:
        return "❌ Backend API'ye bağlanılamadı. API'nin çalıştığından emin olun."
    except Exception as e:
        return f"Hata oluştu: {str(e)}"


def gradio_translate_with_api(text, target_language):
    """
    Metin Çeviri Fonksiyonu - Backend API Entegrasyonu
    =================================================
    
    Metinleri farklı dillere çevirmek için backend API'sini kullanır.
    
    Parametreler:
    ------------
    text (str): Çevrilecek metin
    target_language (str): Hedef dil (örn: "İngilizce", "Fransızca")
    
    Döndürür:
    --------
    str: Çevrilmiş metin
    
    Desteklenen Diller:
    ------------------
    - İngilizce, Fransızca, Almanca, İspanyolca, Japonca, Türkçe vb.
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/text/translate",
            params={
                "text": text,
                "target_language": target_language,
                "model": "gpt-3.5-turbo"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("translation", "Çeviri yapılamadı")
        else:
            return f"Hata: {response.status_code} - {response.text}"
            
    except requests.exceptions.ConnectionError:
        return "❌ Backend API'ye bağlanılamadı. API'nin çalıştığından emin olun."
    except Exception as e:
        return f"Hata oluştu: {str(e)}"


def create_gradio_integration():
    """
    Gradio Arayüz Oluşturma - Tam Entegre Uygulama
    ==============================================
    
    Bu fonksiyon, Gradio Blocks API'sini kullanarak modüler bir web arayüzü oluşturur.
    
    Gradio Blocks:
    -------------
    - Gradio'nun yeni ve güçlü arayüz oluşturma yöntemi
    - Daha fazla kontrol ve özelleştirme imkanı
    - Layout yönetimi (rows, columns, tabs)
    - Event handling ve state management
    
    Döndürür:
    --------
    gr.Blocks: Gradio arayüz nesnesi (launch() ile başlatılır)
    """
    # Gradio Blocks context manager ile arayüz oluştur
    # with bloğu içindeki tüm Gradio component'leri otomatik olarak eklenir
    with gr.Blocks(title="Gradio + FastAPI Entegrasyonu") as demo:
        
        # Markdown: HTML'e çevrilebilen metin formatı
        # Başlıklar, listeler, linkler vb. oluşturabilirsiniz
        gr.Markdown(
            """
            # 🤖 Gradio + FastAPI Entegrasyonu
            
            Bu uygulama Gradio frontend'i ile FastAPI backend'ini birleştirir.
            
            **Özellikler:**
            - 💬 AI Chatbot
            - 📝 Metin Özetleme
            - 🌍 Metin Çeviri
            """
        )
        
        # Tabs: Farklı özellikleri organize etmek için sekmeler
        # Her sekme bağımsız bir özellik sunar
        with gr.Tabs():
            # ================================================================
            # TAB 1: CHATBOT SEKMES İ
            # ================================================================
            with gr.Tab("💬 Chatbot"):
                gr.Markdown("### FastAPI backend ile chatbot")
                
                # Chatbot component: Konuşma geçmişini otomatik yönetir
                # Her mesaj (kullanıcı, bot) çifti olarak saklanır
                chatbot = gr.Chatbot(
                    label="Konuşma",
                    height=400  # Piksel cinsinden yükseklik
                )
                
                # Textbox: Kullanıcının mesaj yazması için
                msg = gr.Textbox(
                    label="Mesajınız",
                    placeholder="Mesajınızı yazın...",
                    lines=2,  # Çok satırlı input
                    autofocus=True  # Sayfa yüklendiğinde otomatik focus
                )
                
                # Button: Mesaj gönderme butonu
                submit_btn = gr.Button(
                    "Gönder", 
                    variant="primary"  # Mavi renkli vurgulu buton
                )
                
                # Clear Button: Konuşma geçmişini temizle
                clear_btn = gr.Button("Temizle")
                
                def respond(message, chat_history):
                    """
                    Chatbot Yanıt Fonksiyonu
                    -----------------------
                    
                    Bu iç fonksiyon, kullanıcı mesajını alır ve bot yanıtını ekler.
                    
                    Parametreler:
                    - message: Yeni kullanıcı mesajı
                    - chat_history: Mevcut konuşma geçmişi (list of tuples)
                    
                    Döndürür:
                    - "": Textbox'ı temizle
                    - chat_history: Güncellenmiş konuşma geçmişi
                    """
                    # Backend'den yanıt al
                    bot_message = gradio_chat_with_api(message, chat_history)
                    
                    # Konuşma geçmişine ekle
                    # Format: (kullanıcı_mesajı, bot_yanıtı)
                    chat_history.append((message, bot_message))
                    
                    # İlk değer ("") textbox'ı temizler
                    # İkinci değer (chat_history) chatbot'u günceller
                    return "", chat_history
                
                # Event Handling: Enter tuşuna basıldığında mesaj gönder
                # submit metodu, Enter tuşu event'ini yakalar
                msg.submit(
                    fn=respond,              # Çağrılacak fonksiyon
                    inputs=[msg, chatbot],   # Input component'leri
                    outputs=[msg, chatbot]   # Output component'leri
                )
                
                # Button click event: Buton tıklandığında mesaj gönder
                submit_btn.click(respond, [msg, chatbot], [msg, chatbot])
                
                # Clear button: Chatbot'u None yaparak geçmişi temizle
                # queue=False: Bu işlem sıraya girmeden anında çalışır
                clear_btn.click(lambda: None, None, chatbot, queue=False)
            
            # ================================================================
            # TAB 2: METİN İŞLEME SEKMESİ
            # ================================================================
            with gr.Tab("📝 Metin İşleme"):
                # Row: Yan yana iki kolon oluştur
                with gr.Row():
                    # Sol Kolon: Metin Özetleme
                    with gr.Column():
                        gr.Markdown("### Metin Özetleme")
                        
                        text_input = gr.Textbox(
                            label="Metin",
                            placeholder="Özetlemek istediğiniz metni yazın...",
                            lines=5
                        )
                        
                        summarize_btn = gr.Button("Özetle", variant="primary")
                        
                        summary_output = gr.Textbox(
                            label="Özet", 
                            lines=5,
                            interactive=False  # Sadece okuma modunda
                        )
                        
                        # Button click → API çağrısı → Output güncelle
                        summarize_btn.click(
                            gradio_summarize_with_api,  # Fonksiyon
                            text_input,                  # Input
                            summary_output               # Output
                        )
                    
                    # Sağ Kolon: Metin Çeviri
                    with gr.Column():
                        gr.Markdown("### Metin Çeviri")
                        
                        translate_input = gr.Textbox(
                            label="Çevrilecek Metin",
                            placeholder="Çevirmek istediğiniz metni yazın...",
                            lines=3
                        )
                        
                        # Dropdown: Hedef dil seçimi
                        language_select = gr.Dropdown(
                            choices=["İngilizce", "Fransızca", "Almanca", "İspanyolca", "Japonca"],
                            label="Hedef Dil",
                            value="İngilizce"  # Default seçim
                        )
                        
                        translate_btn = gr.Button("Çevir", variant="primary")
                        
                        translate_output = gr.Textbox(
                            label="Çeviri", 
                            lines=5,
                            interactive=False
                        )
                        
                        # İki input (metin + dil) → Bir output (çeviri)
                        translate_btn.click(
                            gradio_translate_with_api, 
                            [translate_input, language_select],  # İki input
                            translate_output
                        )
        
        # Footer: Kullanım talimatları
        gr.Markdown(
            f"""
            ---
            **Backend API URL**: `{API_BASE_URL}`
            
            **Not**: Backend API'nin çalıştığından emin olun:
            ```bash
            uvicorn 3_fastapi_backend:app --reload
            ```
            
            **API Dokümantasyonu**: {API_BASE_URL}/docs
            """
        )
    
    return demo


# ============================================================================
# 4. STREAMLIT + FASTAPI ENTEGRASYONU
# ============================================================================
# Streamlit, veri bilimi ve ML uygulamaları için optimize edilmiş bir framework'tür.
# Özellikle veri görselleştirme ve dashboard uygulamaları için tercih edilir.
# ============================================================================

def streamlit_chat_with_api(message: str) -> str:
    """
    Streamlit Chatbot API Çağrısı
    =============================
    
    Gradio versiyonundan farklı olarak, bu fonksiyon sadece mesajı alır.
    Konuşma geçmişi Streamlit session_state tarafından yönetilir.
    
    Type Hints:
    ----------
    - message: str → Gelen parametrenin string olması gerektiğini belirtir
    - -> str → Fonksiyonun string döndüreceğini belirtir
    
    Bu sayede IDE'ler daha iyi autocomplete ve hata kontrolü yapabilir.
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat/simple",
            params={"message": message, "model": "gpt-3.5-turbo"},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("response", "Yanıt alınamadı")
        else:
            return f"Hata: {response.status_code} - {response.text}"
    except requests.exceptions.ConnectionError:
        return "❌ Backend API'ye bağlanılamadı. API'nin çalıştığından emin olun."
    except Exception as e:
        return f"Hata oluştu: {str(e)}"


def streamlit_summarize_with_api(text: str) -> str:
    """Streamlit Metin Özetleme API Çağrısı"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/text/summarize",
            params={"text": text, "model": "gpt-3.5-turbo"},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("summary", "Özet oluşturulamadı")
        else:
            return f"Hata: {response.status_code} - {response.text}"
    except requests.exceptions.ConnectionError:
        return "❌ Backend API'ye bağlanılamadı. API'nin çalıştığından emin olun."
    except Exception as e:
        return f"Hata oluştu: {str(e)}"


def streamlit_translate_with_api(text: str, target_language: str) -> str:
    """Streamlit Metin Çeviri API Çağrısı"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/text/translate",
            params={"text": text, "target_language": target_language, "model": "gpt-3.5-turbo"},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("translation", "Çeviri yapılamadı")
        else:
            return f"Hata: {response.status_code} - {response.text}"
    except requests.exceptions.ConnectionError:
        return "❌ Backend API'ye bağlanılamadı. API'nin çalıştığından emin olun."
    except Exception as e:
        return f"Hata oluştu: {str(e)}"


def create_streamlit_integration():
    """
    Streamlit Arayüz Oluşturma - Tam Entegre Uygulama
    ================================================
    
    Streamlit'in çalışma mantığı:
    ----------------------------
    1. Sayfa her kullanıcı etkileşiminde (button click, input change) yeniden çalışır
    2. st.session_state ile state management yapılır (verileri saklar)
    3. Component'ler üstten alta sırayla oluşturulur
    4. Automatic rerun: Kullanıcı bir şey değiştirdiğinde sayfa otomatik yenilenir
    """
    
    # ========================================================================
    # SAYFA KONFİGÜRASYONU
    # ========================================================================
    # NOT: set_page_config() en başta çağrılmalıdır!
    st.set_page_config(
        page_title="Streamlit + FastAPI Entegrasyonu",  # Tarayıcı sekmesi başlığı
        page_icon="🤖",                                  # Tarayıcı sekmesi ikonu
        layout="wide"                                     # Geniş layout (full width)
    )
    
    # Ana başlık ve açıklama
    st.title("🤖 Streamlit + FastAPI Entegrasyonu")
    st.markdown("Bu uygulama Streamlit frontend'i ile FastAPI backend'ini birleştirir.")
    
    # ========================================================================
    # API DURUMU KONTROLÜ (Health Check)
    # ========================================================================
    # Sayfa yüklendiğinde backend API'nin çalışıp çalışmadığını kontrol et
    try:
        health_response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        
        if health_response.status_code == 200:
            # success: Yeşil renkli başarı mesajı
            st.success(f"✅ Backend API çalışıyor: {API_BASE_URL}")
        else:
            # error: Kırmızı renkli hata mesajı
            st.error(f"❌ Backend API yanıt vermiyor: {health_response.status_code}")
    except Exception:
        st.error(f"❌ Backend API'ye bağlanılamadı: {API_BASE_URL}")
        # info: Mavi renkli bilgi mesajı
        st.info("Backend API'yi başlatmak için: `uvicorn 3_fastapi_backend:app --reload`")
    
    # ========================================================================
    # TAB YAPISININ OLUŞTURULMASI
    # ========================================================================
    # Streamlit tabs: Farklı özellikleri organize etmek için
    # Her tab bağımsız bir sayfa gibi çalışır
    tab1, tab2, tab3 = st.tabs(["💬 Chatbot", "📝 Metin İşleme", "📊 API Durumu"])
    
    # ========================================================================
    # TAB 1: CHATBOT
    # ========================================================================
    with tab1:
        st.header("💬 Chatbot")
        st.markdown("### FastAPI backend ile chatbot")
        
        # Session State: Sayfa yenilendiğinde verileri saklamak için
        # Streamlit her etkileşimde scripti baştan çalıştırır
        # Bu yüzden mesaj geçmişini session_state'de tutmalıyız
        if "integration_messages" not in st.session_state:
            # İlk çalıştırmada boş liste oluştur
            st.session_state.integration_messages = []
        
        # Mevcut mesajları göster
        # Her mesaj {"role": "user/assistant", "content": "metin"} formatında
        for message in st.session_state.integration_messages:
            # chat_message: Kullanıcı veya asistan mesajı için özel tasarım
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat Input: Kullanıcının mesaj yazması için özel input
        # Yeni mesaj yazılıp Enter'a basıldığında bu blok çalışır
        if prompt := st.chat_input("Mesajınızı yazın..."):
            # Walrus operator (:=): Atama ve kontrol aynı satırda
            # prompt değişkenine atama yapar ve None değilse if bloğu çalışır
            
            # Kullanıcı mesajını session_state'e ekle
            st.session_state.integration_messages.append({
                "role": "user", 
                "content": prompt
            })
            
            # Kullanıcı mesajını göster
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Bot yanıtını al ve göster
            with st.chat_message("assistant"):
                # Spinner: Loading animasyonu göster
                with st.spinner("Yanıt bekleniyor..."):
                    response = streamlit_chat_with_api(prompt)
                    st.markdown(response)
                    
                    # Bot yanıtını session_state'e ekle
                    st.session_state.integration_messages.append({
                        "role": "assistant", 
                        "content": response
                    })
        
        # Geçmişi Temizle Butonu
        if st.button("🗑️ Geçmişi Temizle"):
            st.session_state.integration_messages = []
            # rerun: Sayfayı yeniden çalıştır (güncel state ile)
            st.rerun()
    
    # ========================================================================
    # TAB 2: METİN İŞLEME
    # ========================================================================
    with tab2:
        st.header("📝 Metin İşleme")
        
        # Columns: Yan yana iki bölüm oluştur
        # col1 ve col2 eşit genişlikte olacak
        col1, col2 = st.columns(2)
        
        # Sol Kolon: Metin Özetleme
        with col1:
            st.subheader("📄 Metin Özetleme")
            
            # text_area: Çok satırlı metin girişi
            text_input = st.text_area(
                "Özetlemek istediğinez metni yazın:",
                height=200,                              # Piksel cinsinden yükseklik
                placeholder="Metninizi buraya yazın..."  # Placeholder metin
            )
            
            # Button: Primary tip (mavi renkli, vurgulu)
            if st.button("Özetle", type="primary"):
                # Metin boş mu kontrol et
                if text_input:
                    # Spinner ile loading göster
                    with st.spinner("Özetleme yapılıyor..."):
                        summary = streamlit_summarize_with_api(text_input)
                        # Özeti göster (değiştirilemez)
                        st.text_area("Özet:", value=summary, height=150, disabled=True)
                else:
                    # warning: Sarı renkli uyarı mesajı
                    st.warning("Lütfen metin girin!")
        
        # Sağ Kolon: Metin Çeviri
        with col2:
            st.subheader("🌍 Metin Çeviri")
            
            translate_input = st.text_area(
                "Çevirmek istediğiniz metni yazın:",
                height=150,
                placeholder="Çevrilecek metni buraya yazın..."
            )
            
            # selectbox: Dropdown menu
            target_language = st.selectbox(
                "Hedef Dil:",
                ["İngilizce", "Fransızca", "Almanca", "İspanyolca", "Japonca", "Türkçe"]
            )
            
            if st.button("Çevir", type="primary"):
                if translate_input:
                    with st.spinner("Çeviri yapılıyor..."):
                        translation = streamlit_translate_with_api(translate_input, target_language)
                        st.text_area("Çeviri:", value=translation, height=150, disabled=True)
                else:
                    st.warning("Lütfen metin girin!")
    
    # ========================================================================
    # TAB 3: API DURUMU VE TEST ARAÇLARI
    # ========================================================================
    with tab3:
        st.header("📊 API Durumu")
        st.markdown("### Backend API bilgileri ve test araçları")
        
        # API endpoint'lerini tanımla
        # Her endpoint için (method_path, endpoint_path) tuple'ı
        endpoints = [
            ("GET /health", "/health"),                  # Health check
            ("POST /chat/simple", "/chat/simple"),       # Chatbot
            ("POST /text/summarize", "/text/summarize"), # Özetleme
            ("POST /text/translate", "/text/translate"), # Çeviri
        ]
        
        # Her endpoint için test arayüzü oluştur
        for method_path, endpoint in endpoints:
            # expander: Genişletilebilir bölüm (accordion gibi)
            with st.expander(f"{method_path}"):
                # Unique key: Her button için benzersiz key gerekli
                if st.button(f"Test {method_path}", key=endpoint):
                    try:
                        # HTTP method'una göre istek gönder
                        if "GET" in method_path:
                            response = requests.get(
                                f"{API_BASE_URL}{endpoint}", 
                                timeout=5
                            )
                        else:
                            # POST istekleri için test parametreleri
                            response = requests.post(
                                f"{API_BASE_URL}{endpoint}",
                                params={
                                    "message": "test", 
                                    "text": "test", 
                                    "target_language": "İngilizce"
                                },
                                timeout=5
                            )
                        
                        # Yanıtı göster
                        if response.status_code == 200:
                            st.success(f"✅ Başarılı: {response.status_code}")
                            # json: JSON formatında veriyi güzel göster
                            st.json(response.json())
                        else:
                            st.error(f"❌ Hata: {response.status_code}")
                            st.text(response.text)
                    except Exception as e:
                        st.error(f"❌ Bağlantı hatası: {str(e)}")
        
        # Footer bilgileri
        st.markdown(f"**API Base URL**: `{API_BASE_URL}`")
        st.markdown("**API Dokümantasyonu**: `/docs` endpoint'inde Swagger UI mevcut")
        
        # Ek Bilgi: FastAPI'nin otomatik dokümantasyonu
        st.info(f"""
        📚 **API Dokümantasyonuna Erişim:**
        - Swagger UI: {API_BASE_URL}/docs
        - ReDoc: {API_BASE_URL}/redoc
        """)


# ============================================================================
# 5. UYGULAMA ÇALIŞTIRMA VE ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    Ana Çalıştırma Bloğu
    ====================
    
    Bu blok, script doğrudan çalıştırıldığında (import edilmediğinde) çalışır.
    Gradio veya Streamlit uygulamasını başlatmak için kullanılır.
    
    Kullanım:
    --------
    
    1. Gradio Uygulamasını Başlatmak:
       python 4_fastapi_integration.py gradio
       
       Açıklama:
       - sys.argv[1] = "gradio" olduğunda Gradio arayüzü başlatılır
       - Port 7861'de çalışır
       - http://localhost:7861 adresinden erişilir
    
    2. Streamlit Uygulamasını Başlatmak:
       streamlit run 4_fastapi_integration.py
       
       Açıklama:
       - Streamlit kendi command line tool'u ile çalıştırılır
       - Port 8501'de çalışır (Streamlit default)
       - http://localhost:8501 adresinden erişilir
    
    Ön Gereksinimler:
    ----------------
    Backend API'nin çalışıyor olması gerekir:
    uvicorn 3_fastapi_backend:app --reload --port 8000
    
    Mimari:
    ------
    Frontend (7861/8501) → HTTP İstekleri → Backend (8000) → OpenAI API
    """
    
    import sys
    
    # Komut satırı argümanlarını kontrol et
    # sys.argv: [script_adı, arg1, arg2, ...]
    if len(sys.argv) > 1 and sys.argv[1] == "gradio":
        # ====================================================================
        # GRADIO UYGULAMASI
        # ====================================================================
        print("🚀 Gradio uygulaması başlatılıyor...")
        print(f"📡 Backend API: {API_BASE_URL}")
        print("🌐 Arayüz: http://localhost:7861")
        
        # Gradio arayüzünü oluştur
        demo = create_gradio_integration()
        
        # Queue: Asenkron işlemler için kuyruk sistemi
        # Birden fazla kullanıcı aynı anda kullanabilir
        demo.queue()
        
        # Launch: Gradio web sunucusunu başlat
        demo.launch(
            server_name="0.0.0.0",  # Tüm network interface'lerinde dinle
                                     # 0.0.0.0: Dışarıdan erişime izin ver
                                     # 127.0.0.1: Sadece localhost
            server_port=7861,        # Port numarası
            share=False              # Gradio share link oluşturma
                                     # True: Geçici public URL oluşturur
        )
    else:
        # ====================================================================
        # STREAMLIT UYGULAMASI
        # ====================================================================
        # Streamlit uygulaması streamlit CLI ile çalıştırılır
        # Bu blok sadece fonksiyonu çağırır
        create_streamlit_integration()
        
        # Streamlit Çalıştırma Komutları:
        # -------------------------------
        # streamlit run 4_fastapi_integration.py
        # streamlit run 4_fastapi_integration.py --server.port 8502
        # streamlit run 4_fastapi_integration.py --server.address 0.0.0.0

"""
DERS NOTU ÖZET
==============

1. Frontend-Backend Ayrımı:
   - Frontend: Kullanıcı arayüzü (Gradio/Streamlit)
   - Backend: İş mantığı ve AI modelleri (FastAPI)
   - İletişim: HTTP REST API

2. HTTP İstekleri:
   - requests.get(): Veri alma
   - requests.post(): Veri gönderme
   - Status codes: 200 (OK), 4xx (Client Error), 5xx (Server Error)

3. Error Handling:
   - try-except blokları
   - ConnectionError: Backend'e bağlanılamadı
   - Timeout: Belirli sürede yanıt gelmedi
   - Kullanıcı dostu hata mesajları

4. Gradio Özellikleri:
   - Blocks API: Modüler arayüz
   - Event handling: click, submit
   - State management: Otomatik
   - Chatbot component: Konuşma geçmişi

5. Streamlit Özellikleri:
   - Automatic rerun: Her etkileşimde script yeniden çalışır
   - session_state: Veri saklama
   - chat_message: Chatbot arayüzü
   - columns/tabs: Layout yönetimi

6. Best Practices:
   - Environment variables ile konfigürasyon
   - Type hints kullanımı
   - Detaylı yorum satırları
   - Error handling
   - Health check endpoint'i

7. Deployment Considerations:
   - Frontend ve backend ayrı sunucularda çalışabilir
   - CORS ayarları gerekebilir
   - Environment-specific konfigürasyonlar
   - Load balancing ve scaling

Bu entegrasyon örneği, production-ready bir web uygulamasının temellerini gösterir.
"""


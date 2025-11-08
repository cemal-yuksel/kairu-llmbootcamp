"""
Streamlit ile Frontend Uygulaması - Detaylı Eğitim Notu
========================================================

Bu modül, Streamlit framework'ü kullanarak LLM (Large Language Model) tabanlı 
bir chatbot ve çeşitli doğal dil işleme uygulamaları oluşturmayı gösterir.

Kapsanan Konular:
- Streamlit temel bileşenleri ve sayfa yapılandırması
- OpenAI API entegrasyonu
- Session state yönetimi
- Streaming yanıtlar
- Multi-tab uygulama mimarisi
- Veri görselleştirme

Gereksinimler:
- streamlit
- openai
- python-dotenv
- pandas
- plotly
"""

# ============================================================================
# KÜTÜPHANE İMPORTLARI
# ============================================================================

import streamlit as st  # Streamlit: Web uygulaması oluşturmak için ana framework
from openai import OpenAI  # OpenAI: GPT modelleriyle etkileşim için resmi kütüphane
import os  # os: İşletim sistemi fonksiyonları (environment variables için)
from dotenv import load_dotenv  # dotenv: .env dosyasından çevre değişkenlerini yükler
import time  # time: Zaman işlemleri için (şu an kullanılmıyor ama gelecek özellikler için)
import pandas as pd  # pandas: Veri analizi ve manipülasyonu için
import plotly.express as px  # plotly: İnteraktif grafik ve görselleştirme için

# ============================================================================
# BAŞLANGIÇ YAPILANDIRMASI
# ============================================================================

# Environment variables yükle
# .env dosyasından OPENAI_API_KEY gibi gizli bilgileri yükler
# Bu sayede API anahtarlarını kod içinde yazmaya gerek kalmaz
load_dotenv()

# OpenAI client oluştur
# API anahtarı ile OpenAI servislerine bağlantı kurar
# os.getenv(): Çevre değişkenlerinden API anahtarını okur
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ============================================================================
# SAYFA YAPILANDIRMASI
# ============================================================================

# Streamlit sayfasının temel ayarlarını yapılandır
st.set_page_config(
    page_title="LLM Uygulama Demo",  # Tarayıcı sekmesinde görünecek başlık
    page_icon="🤖",  # Tarayıcı sekmesinde görünecek emoji icon
    layout="wide",  # Sayfa düzeni: "wide" tüm ekranı kullanır, "centered" ortalanmış
    initial_sidebar_state="expanded"  # Sidebar başlangıçta açık mı kapalı mı: "expanded" veya "collapsed"
)

# ============================================================================
# SIDEBAR (YAN PANEL) YAPILANDIRMASI
# ============================================================================

# Sidebar: Uygulamanın sol tarafında yer alan ayarlar paneli
with st.sidebar:
    # Sidebar başlığı
    st.title("⚙️ Ayarlar")
    
    # Model seçimi dropdown menüsü
    # Kullanıcının hangi GPT modelini kullanacağını seçmesini sağlar
    model_choice = st.selectbox(
        "Model Seçin:",  # Dropdown etiketi
        ["gpt-3.5-turbo", "gpt-4"],  # Seçenekler listesi
        index=0  # Varsayılan seçim (0 = ilk eleman = gpt-3.5-turbo)
    )
    
    # Temperature slider'ı
    # Temperature: Model yanıtlarının ne kadar yaratıcı/rastgele olacağını kontrol eder
    # 0.0 = Deterministik (her zaman aynı yanıt)
    # 1.0 = Çok yaratıcı (daha rastgele ve çeşitli yanıtlar)
    temperature = st.slider(
        "Temperature (Yaratıcılık):",  # Slider etiketi
        min_value=0.0,  # Minimum değer
        max_value=1.0,  # Maximum değer
        value=0.7,  # Varsayılan değer
        step=0.1  # Artış miktarı
    )
    
    # Max tokens slider'ı
    # Max tokens: Yanıtın maksimum uzunluğunu belirler
    # 1 token ≈ 4 karakter veya 0.75 kelime
    max_tokens = st.slider(
        "Max Tokens (Maksimum uzunluk):",  # Slider etiketi
        min_value=50,  # Minimum 50 token
        max_value=500,  # Maximum 500 token
        value=150,  # Varsayılan 150 token
        step=50  # 50'lik artışlar
    )
    
    # Görsel ayırıcı çizgi
    st.divider()
    
    # API Key kontrolü ve kullanıcıya bilgilendirme
    # Environment variable'dan API key'in yüklenip yüklenmediğini kontrol eder
    if not os.getenv("OPENAI_API_KEY"):
        # Eğer API key yoksa kırmızı hata mesajı göster
        st.error("⚠️ API Key bulunamadı! `.env` dosyasını kontrol edin.")
    else:
        # Eğer API key varsa yeşil başarı mesajı göster
        st.success("✅ API Key yüklendi")
    
    # Başka bir ayırıcı çizgi
    st.divider()
    
    # Tüm sohbet geçmişini temizleme butonu
    if st.button("🗑️ Tüm Geçmişi Temizle"):
        # Session state'teki mesajları boş liste yap
        st.session_state.messages = []
        # Sayfayı yeniden yükle (değişikliklerin görünmesi için)
        st.rerun()

# ============================================================================
# SESSION STATE YÖNETİMİ
# ============================================================================

"""
Session State Nedir?
--------------------
Streamlit her kullanıcı etkileşiminde (buton tıklama, text girişi vb.) 
scripti baştan çalıştırır. Session state, kullanıcı oturumu boyunca
verileri saklamak için kullanılır.

Örnek: Sohbet geçmişini saklamak için session state kullanıyoruz.
Aksi halde her etkileşimde geçmiş mesajlar kaybolur.
"""

# Eğer "messages" anahtarı session state'te yoksa, boş liste oluştur
# Bu anahtar chatbot mesaj geçmişini saklar
if "messages" not in st.session_state:
    st.session_state.messages = []

# Metin özeti saklamak için session state anahtarı
# Özetleme işlemi sonucu burada saklanır
if "text_summary" not in st.session_state:
    st.session_state.text_summary = ""

# Çeviri sonucu saklamak için session state anahtarı
# Çeviri işlemi sonucu burada saklanır
if "translation_result" not in st.session_state:
    st.session_state.translation_result = ""

# ============================================================================
# YARDIMCI FONKSİYONLAR
# ============================================================================

def get_openai_response(prompt, system_prompt="Sen yardımcı bir asistansın.", model="gpt-3.5-turbo"):
    """
    OpenAI API'den yanıt al (Normal Mod)
    
    Bu fonksiyon, OpenAI API'ye istek gönderir ve tam yanıtı bir seferde alır.
    Streaming olmadığı için kullanıcı yanıtın tamamını bekler.
    
    Parametreler:
    -------------
    prompt : str
        Kullanıcının sorusu veya isteği
    system_prompt : str
        Modelin davranışını belirleyen sistem mesajı
        Örn: "Sen bir Python uzmanısın", "Sen Shakespeare gibi konuş"
    model : str
        Kullanılacak OpenAI model adı (gpt-3.5-turbo veya gpt-4)
    
    Döndürür:
    ---------
    str
        Modelin ürettiği yanıt metni veya hata mesajı
    
    Önemli Kavramlar:
    -----------------
    - System Prompt: Modelin kişiliğini ve davranışını belirler
    - User Prompt: Kullanıcının gerçek sorusu
    - Messages: Sohbet geçmişini içeren liste yapısı
    """
    try:
        # OpenAI API'ye chat completion isteği gönder
        response = client.chat.completions.create(
            model=model,  # Kullanılacak model
            messages=[
                # Sistem mesajı: Modelin rolünü tanımlar
                {"role": "system", "content": system_prompt},
                # Kullanıcı mesajı: Gerçek soru/istek
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,  # Maksimum yanıt uzunluğu
            temperature=temperature  # Yaratıcılık seviyesi
        )
        # Yanıtın içeriğini döndür
        # response.choices[0]: İlk (ve genelde tek) yanıt seçeneği
        # .message.content: Yanıtın metin içeriği
        return response.choices[0].message.content
    except Exception as e:
        # Herhangi bir hata olursa (API hatası, network hatası vb.)
        # Hata mesajını döndür
        return f"Hata oluştu: {str(e)}"


def stream_openai_response(prompt, system_prompt="Sen yardımcı bir asistansın.", model="gpt-3.5-turbo"):
    """
    OpenAI API'den streaming yanıt al
    
    Bu fonksiyon, yanıtı parça parça (chunk) alır ve anlık olarak gösterir.
    ChatGPT'nin kelime kelime yazması gibi bir efekt sağlar.
    
    Parametreler:
    -------------
    prompt : str
        Kullanıcının sorusu
    system_prompt : str
        Sistem mesajı
    model : str
        Model adı
    
    Yields (Generator):
    -------------------
    str
        Şu ana kadar oluşturulan tam yanıt metni
        Her chunk'ta bir önceki yanıt + yeni chunk döndürülür
    
    Generator Nedir?
    ----------------
    Normal fonksiyonlar 'return' kullanır ve bir kez değer döndürür.
    Generator fonksiyonlar 'yield' kullanır ve birden fazla kez değer dondürebilir.
    Bu, streaming için idealdir çünkü her chunk'ı ayrı ayrı döndürebiliriz.
    """
    try:
        # Streaming modda API isteği
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            stream=True,  # ÖNEMLI: Streaming modunu aktif eder
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Tam yanıtı saklamak için değişken
        full_response = ""
        
        # Her chunk'ı işle
        # response bir iterator'dür, for döngüsü her chunk'ı tek tek alır
        for chunk in response:
            # Chunk'ta içerik var mı kontrol et
            # Bazı chunk'lar sadece metadata içerir, content olmayabilir
            if chunk.choices[0].delta.content:
                # Yeni chunk'ı mevcut yanıta ekle
                full_response += chunk.choices[0].delta.content
                # Şu ana kadarki tam yanıtı yield et
                # Bu, Streamlit'te anlık güncelleme sağlar
                yield full_response
    except Exception as e:
        # Hata durumunda hata mesajını yield et
        yield f"Hata oluştu: {str(e)}"


# ============================================================================
# ANA SAYFA
# ============================================================================

# Ana sayfa başlığı
st.title("🤖 LLM Tabanlı Uygulama Örnekleri")
# Alt başlık/açıklama
st.markdown("Bu uygulama Streamlit kullanarak çeşitli LLM uygulamalarını gösterir.")

# Tab (Sekme) yapısı oluştur
# Kullanıcı farklı özellikler arasında geçiş yapabilir
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💬 Chatbot",  # Tab 1: Basit chatbot
    "🌊 Streaming Chatbot",  # Tab 2: Streaming özellikli chatbot
    "📝 Metin İşleme",  # Tab 3: Özetleme ve çeviri
    "💻 Kod Açıklama",  # Tab 4: Kod analizi
    "📊 Veri Görselleştirme"  # Tab 5: Veri analizi ve grafikler
])

# ============================================================================
# TAB 1: Basit Chatbot
# ============================================================================

with tab1:
    """
    Basit Chatbot Tab'ı
    -------------------
    Bu tab, geleneksel chatbot arayüzünü gösterir.
    Kullanıcı mesaj gönderir, bot yanıt verir.
    Tüm mesaj geçmişi saklanır ve gösterilir.
    """
    
    # Tab başlığı
    st.header("💬 Basit Chatbot")
    st.markdown("### Basit chatbot arayüzü")
    
    # Mesaj geçmişini göster
    # Session state'teki her mesajı döngü ile göster
    for message in st.session_state.messages:
        # Chat mesaj balonu oluştur
        # Role: "user" (kullanıcı) veya "assistant" (bot)
        with st.chat_message(message["role"]):
            # Mesaj içeriğini göster
            st.markdown(message["content"])
    
    # Yeni mesaj input alanı
    # := operatörü: Walrus operator, atama ve kontrol aynı anda
    # Kullanıcı mesaj girip Enter'a basarsa prompt değişkenine atanır
    if prompt := st.chat_input("Mesajınızı yazın..."):
        # Kullanıcı mesajını session state'e ekle
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Kullanıcı mesajını ekranda göster
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Bot yanıtını al ve göster
        with st.chat_message("assistant"):
            # OpenAI'dan yanıt al (normal mod, streaming değil)
            response = get_openai_response(prompt, model=model_choice)
            # Yanıtı ekranda göster
            st.markdown(response)
            # Bot yanıtını session state'e ekle
            st.session_state.messages.append({"role": "assistant", "content": response})

# ============================================================================
# TAB 2: Streaming Chatbot
# ============================================================================

with tab2:
    """
    Streaming Chatbot Tab'ı
    -----------------------
    Bu tab, yanıtların kelime kelime gösterildiği chatbot'u gösterir.
    ChatGPT'nin gerçek zamanlı yazma efekti burada uygulanır.
    
    Önemli: Tab 1'den ayrı bir mesaj geçmişi kullanır (streaming_messages)
    """
    
    st.header("🌊 Streaming Chatbot")
    st.markdown("### Streaming output ile chatbot")
    
    # Streaming mesaj geçmişi için ayrı session state
    # Her tab'ın kendi geçmişi olması için
    if "streaming_messages" not in st.session_state:
        st.session_state.streaming_messages = []
    
    # Mesaj geçmişini göster
    for message in st.session_state.streaming_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Yeni mesaj input
    if streaming_prompt := st.chat_input("Mesajınızı yazın... (Streaming)"):
        # Kullanıcı mesajını ekle
        st.session_state.streaming_messages.append({"role": "user", "content": streaming_prompt})
        with st.chat_message("user"):
            st.markdown(streaming_prompt)
        
        # Bot streaming yanıtını al ve göster
        with st.chat_message("assistant"):
            # Boş placeholder oluştur
            # Bu placeholder, stream sırasında sürekli güncellenecek
            message_placeholder = st.empty()
            full_response = ""
            
            # Stream generator'ından her chunk'ı al
            for chunk in stream_openai_response(streaming_prompt, model=model_choice):
                full_response = chunk
                # Yanıtı göster, sonuna cursor (▌) ekle
                # Bu, "yazıyor..." efekti verir
                message_placeholder.markdown(full_response + "▌")
            
            # Stream bittikten sonra cursor'ı kaldır
            message_placeholder.markdown(full_response)
            # Tam yanıtı session state'e ekle
            st.session_state.streaming_messages.append({"role": "assistant", "content": full_response})
    
    # Streaming geçmişini temizle butonu
    if st.button("🗑️ Streaming Geçmişini Temizle"):
        st.session_state.streaming_messages = []
        st.rerun()

# ============================================================================
# TAB 3: Metin İşleme
# ============================================================================

with tab3:
    """
    Metin İşleme Tab'ı
    ------------------
    İki ana özellik:
    1. Metin Özetleme: Uzun metinleri özetler
    2. Metin Çevirisi: Metinleri farklı dillere çevirir
    
    İki sütunlu (column) layout kullanır
    """
    
    st.header("📝 Metin İşleme")
    
    # İki sütun oluştur (yan yana layout)
    # Sütunlar eşit genişlikte (varsayılan)
    col1, col2 = st.columns(2)
    
    # ---- SOL SÜTUN: Metin Özetleme ----
    with col1:
        st.subheader("📄 Metin Özetleme")
        
        # Çok satırlı text input alanı
        text_input = st.text_area(
            "Özetlemek istediğiniz metni yazın:",
            height=200,  # Piksel cinsinden yükseklik
            placeholder="Metninizi buraya yazın..."  # Boşken gösterilen ipucu
        )
        
        # Özetle butonu
        # type="primary" mavi renkli, vurgulu buton yapar
        if st.button("Özetle", type="primary"):
            # Metin girilmiş mi kontrol et
            if text_input:
                # Spinner: İşlem sırasında animasyon göster
                with st.spinner("Özetleme yapılıyor..."):
                    # Özel prompt ile OpenAI'dan özet al
                    summary = get_openai_response(
                        f"Bu metni özetle:\n\n{text_input}",
                        "Sen bir metin özetleme uzmanısın. Verilen metni kısa ve öz şekilde özetle.",
                        model=model_choice
                    )
                    # Özeti session state'e kaydet
                    st.session_state.text_summary = summary
                    # Başarı mesajı göster
                    st.success("Özetleme tamamlandı!")
        
        # Eğer özet varsa göster
        if st.session_state.text_summary:
            # Read-only text area (düzenlenemez)
            st.text_area("Özet:", value=st.session_state.text_summary, height=150)
    
    # ---- SAĞ SÜTUN: Metin Çevirisi ----
    with col2:
        st.subheader("🌍 Metin Çeviri")
        
        # Çevrilecek metin input'u
        translate_input = st.text_area(
            "Çevirmek istediğiniz metni yazın:",
            height=150,
            placeholder="Çevrilecek metni buraya yazın..."
        )
        
        # Hedef dil seçimi dropdown
        target_language = st.selectbox(
            "Hedef Dil:",
            ["İngilizce", "Fransızca", "Almanca", "İspanyolca", "Japonca", "Türkçe"]
        )
        
        # Çevir butonu
        if st.button("Çevir", type="primary"):
            if translate_input:
                with st.spinner("Çeviri yapılıyor..."):
                    # System prompt'ta hedef dili belirt
                    translation = get_openai_response(
                        translate_input,
                        f"Sen bir çevirmensin. Verilen metni {target_language} diline çevir.",
                        model=model_choice
                    )
                    # Çeviriyi session state'e kaydet
                    st.session_state.translation_result = translation
                    st.success("Çeviri tamamlandı!")
        
        # Çeviri sonucunu göster
        if st.session_state.translation_result:
            st.text_area("Çeviri:", value=st.session_state.translation_result, height=150)

# ============================================================================
# TAB 4: Kod Açıklama
# ============================================================================

with tab4:
    """
    Kod Açıklama Tab'ı
    -------------------
    Programcılar için kod analiz aracı.
    Girilen kodu satır satır açıklar.
    Farklı programlama dillerini destekler.
    """
    
    st.header("💻 Kod Açıklama")
    st.markdown("### Kod açıklama aracı")
    
    # Programlama dili seçimi
    code_language = st.selectbox(
        "Programlama Dili:",
        ["Python", "JavaScript", "Java", "C++", "Go", "Rust"],
        index=0  # Python varsayılan
    )
    
    # Kod input alanı
    code_input = st.text_area(
        "Açıklamak istediğiniz kodu yazın:",
        height=300,  # Yüksek alan (kod için)
        placeholder=f"# {code_language} kodunuzu buraya yazın..."
    )
    
    # Açıkla butonu
    if st.button("Açıkla", type="primary"):
        if code_input:
            with st.spinner("Kod açıklaması oluşturuluyor..."):
                # Kodu markdown code block formatında gönder
                # Backtick'ler (```) ile kod bloğu oluştur
                explanation = get_openai_response(
                    f"Bu kodu açıkla:\n\n```{code_language.lower()}\n{code_input}\n```",
                    f"Sen bir {code_language} programlama uzmanısın. Verilen kodu detaylı şekilde açıkla.",
                    model=model_choice
                )
                st.success("Açıklama oluşturuldu!")
                st.markdown("### 📖 Açıklama:")
                # Markdown format destekler (bold, italik, kod blokları vb.)
                st.markdown(explanation)
        else:
            # Kod girilmemişse uyarı göster
            st.warning("Lütfen kod girin!")

# ============================================================================
# TAB 5: Veri Görselleştirme
# ============================================================================

with tab5:
    """
    Veri Görselleştirme Tab'ı
    -------------------------
    LLM ile veri analizi ve Plotly ile görselleştirme kombinasyonu.
    
    Özellikler:
    - Örnek veri gösterimi
    - LLM ile veri analizi
    - İnteraktif grafikler (Plotly)
    """
    
    st.header("📊 Veri Görselleştirme")
    st.markdown("### LLM ile veri analizi ve görselleştirme")
    
    # Örnek veri dictionary'si oluştur
    sample_data = {
        "Ürün": ["A", "B", "C", "D", "E"],  # Ürün isimleri
        "Satış": [100, 150, 200, 120, 180],  # Satış rakamları
        "Kategori": ["Elektronik", "Giyim", "Elektronik", "Giyim", "Elektronik"]  # Kategoriler
    }
    
    # Dictionary'yi pandas DataFrame'e çevir
    # DataFrame: Tablo formatında veri yapısı (Excel gibi)
    df = pd.DataFrame(sample_data)
    
    # Veri tablosunu göster
    st.subheader("Örnek Veri")
    # use_container_width yerine width='stretch' kullanımı
    st.dataframe(df, width='stretch')
    
    # Veri analizi için kullanıcı sorusu
    analysis_prompt = st.text_area(
        "Veri analizi için soru sorun:",
        placeholder="Örn: Bu verilerde hangi kategoride en çok satış var?",
        height=100
    )
    
    # Analiz Et butonu
    if st.button("Analiz Et", type="primary"):
        if analysis_prompt:
            with st.spinner("Analiz yapılıyor..."):
                # DataFrame'i string formatına çevir
                # LLM'in okuyabilmesi için tablo formatında string
                data_str = df.to_string()
                
                # Veri ve soruyu LLM'e gönder
                response = get_openai_response(
                    f"Bu veri tablosunu analiz et:\n\n{data_str}\n\nSoru: {analysis_prompt}",
                    "Sen bir veri analiz uzmanısın. Verilen veriyi analiz et ve yorum yap.",
                    model=model_choice
                )
                
                # Analiz sonucunu göster
                st.markdown("### 📊 Analiz Sonucu:")
                st.markdown(response)
                
                # Görselleştirmeler
                st.markdown("### 📈 Görselleştirme:")
                
                # Bar chart (Sütun grafiği)
                # Plotly Express: Hızlı ve kolay grafik oluşturma
                fig_bar = px.bar(
                    df,  # Veri kaynağı
                    x="Ürün",  # X ekseni
                    y="Satış",  # Y ekseni
                    color="Kategori",  # Renk kategorisi
                    title="Ürün Satışları"  # Grafik başlığı
                )
                # Grafiği göster, container genişliğini kullan
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # Pie chart (Pasta grafiği)
                # Önce kategorilere göre topla
                category_sales = df.groupby("Kategori")["Satış"].sum().reset_index()
                
                fig_pie = px.pie(
                    category_sales,  # Toplam veri
                    values="Satış",  # Değerler
                    names="Kategori",  # Etiketler
                    title="Kategori Bazında Satış Dağılımı"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.warning("Lütfen bir soru girin!")

# ============================================================================
# FOOTER (Sayfa Altı)
# ============================================================================

# Görsel ayırıcı çizgi
st.divider()

# Alt bilgi notu
st.markdown(
    """
    ---
    **Not**: Bu uygulama OpenAI API kullanmaktadır. API key'inizi `.env` dosyasına eklemeyi unutmayın.
    
    **Geliştirici İpuçları:**
    - Session state: Kullanıcı verilerini saklar
    - st.rerun(): Sayfayı yeniden yükler
    - Generator fonksiyonlar: Streaming için idealdir
    - Plotly: İnteraktif grafikler için kullanılır
    - Tabs: Farklı özellikleri organize eder
    """
)

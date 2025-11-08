"""
Gradio ile Frontend Uygulaması
===============================
Bu modül, LLM (Large Language Model) tabanlı çeşitli uygulamalar için 
Gradio kütüphanesi kullanarak modern web arayüzü oluşturur.

Gradio Nedir?
-------------
Gradio, makine öğrenmesi modellerini hızlıca test etmek ve paylaşmak için
web arayüzü oluşturmayı sağlayan bir Python kütüphanesidir.

Bu Uygulama Şunları İçerir:
---------------------------
1. Basit Chatbot (Soru-Cevap)
2. Streaming Chatbot (Canlı yazım efekti)
3. Metin İşleme (Özetleme ve Çeviri)
4. Kod Açıklama Aracı
5. Dosya Analizi (PDF, TXT vb.)
"""

# ============================================================================
# KÜTÜPHANE İÇE AKTARIMLARI
# ============================================================================

import gradio as gr           # Web arayüzü oluşturmak için
from openai import OpenAI     # OpenAI API ile iletişim için
import os                     # İşletim sistemi işlemleri için
from dotenv import load_dotenv # .env dosyasından çevre değişkenleri yüklemek için
import time                   # Zaman işlemleri için (ileride kullanılabilir)

# ============================================================================
# ÇEVRE DEĞİŞKENLERİ YÜKLEME
# ============================================================================

# .env dosyasından API anahtarlarını ve diğer hassas bilgileri yükle
# Bu sayede API anahtarları kod içinde görünmez
load_dotenv()

# ============================================================================
# OPENAI CLIENT OLUŞTURMA
# ============================================================================

# OpenAI API'sine bağlanmak için client oluştur
# API anahtarı çevre değişkenlerinden alınır
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ============================================================================
# ÖRNEK 1: BASİT CHATBOT ARAYÜZÜ
# ============================================================================

def simple_chatbot(message, history):
    """
    Basit Chatbot Fonksiyonu
    ========================
    
    Bu fonksiyon kullanıcının mesajını alır, geçmiş konuşmaları da dikkate alarak
    OpenAI API'sine gönderir ve yanıt üretir.
    
    Parametreler:
    ------------
    message : str
        Kullanıcının gönderdiği son mesaj
    history : list
        Önceki konuşmaların listesi. Her eleman bir mesaj objesi veya tuple'dır.
        
    Döndürür:
    ---------
    str
        AI asistanın ürettiği yanıt metni
        
    Çalışma Mantığı:
    ---------------
    1. System mesajı ile AI'nin rolü belirlenir
    2. Geçmiş konuşmalar OpenAI formatına çevrilir
    3. Kullanıcının son mesajı eklenir
    4. API'ye istek gönderilir
    5. Yanıt döndürülür
    """
    try:
        # ADIM 1: Sistem mesajı ile başla
        # System mesajı AI'nin davranışını ve kişiliğini belirler
        messages = [{"role": "system", "content": "Sen yardımcı bir asistansın. Kısa ve net cevaplar ver."}]
        
        # ADIM 2: Geçmiş konuşmaları ekle
        # History varsa, onu OpenAI'nin beklediği formata çevir
        if history:
            for msg in history:
                # Eğer mesaj zaten doğru formattaysa (dict ve role/content içeriyorsa)
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    messages.append(msg)
                # Eski tuple formatı için geriye dönük uyumluluk
                # Tuple: (kullanıcı_mesajı, ai_yanıtı)
                elif isinstance(msg, (list, tuple)) and len(msg) == 2:
                    messages.append({"role": "user", "content": msg[0]})
                    messages.append({"role": "assistant", "content": msg[1]})
        
        # ADIM 3: Mevcut kullanıcı mesajını ekle
        messages.append({"role": "user", "content": message})
        
        # ADIM 4: OpenAI API'sine istek gönder
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",      # Kullanılacak model
            messages=messages,           # Tüm konuşma geçmişi
            max_tokens=150,              # Maksimum yanıt uzunluğu (token cinsinden)
            temperature=0.7              # Yaratıcılık seviyesi (0-2 arası, yüksek = daha yaratıcı)
        )
        
        # ADIM 5: AI'nin yanıtını döndür
        return response.choices[0].message.content
        
    except Exception as e:
        # Hata durumunda kullanıcıya bilgi ver
        return f"Hata oluştu: {str(e)}"


# ============================================================================
# ÖRNEK 2: STREAMING OUTPUT İLE CHATBOT
# ============================================================================

def streaming_chatbot(message, history):
    """
    Streaming Chatbot Fonksiyonu
    ============================
    
    Bu fonksiyon yanıtı tek seferde döndürmek yerine, kelime kelime akışı
    sağlar. Böylece kullanıcı yanıtın oluşmasını canlı olarak izleyebilir.
    
    Streaming Nedir?
    ---------------
    Normal API çağrısında tüm yanıt oluşturulup bir kerede gönderilir.
    Streaming'de ise yanıt parça parça (chunk) gelir ve anlık gösterilir.
    Bu, uzun yanıtlarda daha iyi kullanıcı deneyimi sağlar.
    
    Parametreler:
    ------------
    message : str
        Kullanıcının mesajı
    history : list
        Konuşma geçmişi
        
    Yield Eder:
    ----------
    str
        Sürekli güncellenen, birikimli yanıt metni
    """
    try:
        # ADIM 1: Sistem mesajı hazırla
        messages = [{"role": "system", "content": "Sen yardımcı bir asistansın."}]
        
        # ADIM 2: Konuşma geçmişini ekle
        if history:
            for msg in history:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    messages.append(msg)
                elif isinstance(msg, (list, tuple)) and len(msg) == 2:
                    messages.append({"role": "user", "content": msg[0]})
                    messages.append({"role": "assistant", "content": msg[1]})
        
        # ADIM 3: Kullanıcı mesajını ekle
        messages.append({"role": "user", "content": message})
        
        # ADIM 4: Streaming API çağrısı yap
        # stream=True parametresi ile yanıt parça parça gelir
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            stream=True,                 # Streaming'i aktif et
            max_tokens=200,
            temperature=0.7
        )
        
        # ADIM 5: Her gelen chunk'ı işle ve yield et
        full_response = ""  # Tüm yanıtı biriktirmek için
        
        for chunk in response:
            # Her chunk'ta yeni içerik var mı kontrol et
            if chunk.choices[0].delta.content:
                # Yeni içeriği ekle
                full_response += chunk.choices[0].delta.content
                # Güncel halini yield et (generator fonksiyonu)
                # Yield, fonksiyonu durdurmadan ara değer döndürür
                yield full_response
                
    except Exception as e:
        yield f"Hata oluştu: {str(e)}"


# ============================================================================
# ÖRNEK 3: METİN İŞLEME UYGULAMASI
# ============================================================================

def text_summarizer(text):
    """
    Metin Özetleme Fonksiyonu
    =========================
    
    Uzun metinleri kısa ve öz şekilde özetler.
    
    Kullanım Alanları:
    -----------------
    - Makale özetleme
    - Rapor özeti çıkarma
    - Uzun e-postaları özetleme
    
    Parametreler:
    ------------
    text : str
        Özetlenecek uzun metin
        
    Döndürür:
    ---------
    str
        Metnin özet hali
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                # System mesajı ile AI'nin rolünü belirt
                {"role": "system", "content": "Sen bir metin özetleme uzmanısın. Verilen metni kısa ve öz şekilde özetle."},
                # User mesajı ile görevi tanımla
                {"role": "user", "content": f"Bu metni özetle:\n\n{text}"}
            ],
            max_tokens=150,
            temperature=0.5  # Özetlemede daha tutarlı olması için düşük temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Hata oluştu: {str(e)}"


def text_translator(text, target_language):
    """
    Metin Çeviri Fonksiyonu
    =====================
    
    Verilen metni istenen dile çevirir.
    
    Parametreler:
    ------------
    text : str
        Çevrilecek metin
    target_language : str
        Hedef dil (örn: "İngilizce", "Fransızca")
        
    Döndürür:
    ---------
    str
        Çevrilmiş metin
        
    Not:
    ----
    Temperature 0.3 ile daha tutarlı ve doğru çeviriler elde edilir.
    Yaratıcılık yerine doğruluk önemli olduğu için düşük tutulmuştur.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"Sen bir çevirmensin. Verilen metni {target_language} diline çevir."},
                {"role": "user", "content": text}
            ],
            max_tokens=200,
            temperature=0.3  # Çeviride tutarlılık için düşük
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Hata oluştu: {str(e)}"


# ============================================================================
# ÖRNEK 4: MULTI-INPUT UYGULAMASI (Çoklu Girdi)
# ============================================================================

def code_explainer(code, language):
    """
    Kod Açıklama Fonksiyonu
    =====================
    
    Verilen program kodunu analiz edip detaylı açıklama yapar.
    
    Özellikler:
    ----------
    - Kod satır satır açıklanır
    - Algoritma mantığı anlatılır
    - Kullanılan yapılar açıklanır
    
    Parametreler:
    ------------
    code : str
        Açıklanacak kod bloğu
    language : str
        Programlama dili (örn: "Python", "JavaScript")
        
    Döndürür:
    ---------
    str
        Kodun detaylı açıklaması
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                # Dil uzmanı olarak sistem mesajı
                {"role": "system", "content": f"Sen bir {language} programlama uzmanısın. Verilen kodu detaylı şekilde açıkla."},
                # Markdown code block formatında kodu gönder
                {"role": "user", "content": f"Bu kodu açıkla:\n\n```{language}\n{code}\n```"}
            ],
            max_tokens=300,
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Hata oluştu: {str(e)}"


# ============================================================================
# ÖRNEK 5: DOSYA YÜKLEME VE İŞLEME
# ============================================================================

def file_processor(file):
    """
    Dosya İşleme Fonksiyonu
    =======================
    
    Yüklenen dosyaları okur, içeriğini analiz eder ve özet çıkarır.
    
    Desteklenen Formatlar:
    ---------------------
    - PDF: pdfplumber veya PyPDF2 ile
    - TXT: Çeşitli karakter kodlamaları ile
    - Kod dosyaları: .py, .js, vb.
    - Diğer metin tabanlı formatlar
    
    Parametreler:
    ------------
    file : File object
        Gradio'nun yüklediği dosya objesi
        
    Döndürür:
    ---------
    str
        Dosyanın analiz sonucu ve özeti
        
    Çalışma Adımları:
    ----------------
    1. Dosya varlığı kontrol edilir
    2. Dosya uzantısı belirlenir
    3. Uygun okuma yöntemi seçilir
    4. İçerik okunur
    5. AI ile analiz edilir
    """
    # ADIM 1: Dosya kontrolü
    if file is None:
        return "Lütfen bir dosya yükleyin."
    
    try:
        # ADIM 2: Dosya yolunu ve bilgilerini al
        # Gradio dosya objesi 'name' attribute'una sahiptir
        file_path = file.name if hasattr(file, 'name') else file
        
        # Dosya adı ve uzantısını ayır
        import os
        filename = os.path.basename(file_path)
        file_extension = os.path.splitext(filename)[1].lower()
        
        content = ""  # Dosya içeriğini tutacak değişken
        
        # ADIM 3: PDF Dosyalarını İşle
        if file_extension == '.pdf':
            try:
                # Önce pdfplumber kütüphanesini dene (daha iyi metin çıkarma)
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    # İlk 5 sayfayı işle (performans için)
                    for page in pdf.pages[:5]:
                        page_text = page.extract_text()
                        if page_text:
                            content += page_text + "\n\n"
                
                # Eğer içerik boşsa alternatif yöntem dene
                if not content:
                    import PyPDF2
                    with open(file_path, 'rb') as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        # Maksimum 5 sayfa veya toplam sayfa sayısı kadar
                        for page_num in range(min(5, len(pdf_reader.pages))):
                            page = pdf_reader.pages[page_num]
                            content += page.extract_text() + "\n\n"
                            
            except Exception as pdf_error:
                return f"PDF okuma hatası: {str(pdf_error)}"
        
        # ADIM 4: Metin Tabanlı Dosyaları İşle
        else:
            # Farklı karakter kodlamalarını dene
            # Çünkü dosyalar farklı encoding'lerde olabilir
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break  # Başarılı okumadan sonra döngüden çık
                except UnicodeDecodeError:
                    # Bu encoding çalışmadı, sonrakini dene
                    continue
        
        # ADIM 5: İçerik kontrolü
        if not content:
            return "Dosya okunamadı. Desteklenmeyen format veya karakter kodlaması."
        
        # ADIM 6: İçeriği sınırla (API limitleri için)
        # İlk 2000 karakter yeterli, ayrıca token limitini aşmamak için
        content_preview = content[:2000]
        
        # ADIM 7: AI ile dosyayı analiz et
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Sen bir dosya analiz uzmanısın. Verilen dosya içeriğini analiz et, özetini çıkar ve ana konuları belirt."},
                {"role": "user", "content": f"Dosya adı: {filename}\nDosya tipi: {file_extension}\n\nDosya içeriğini analiz et:\n\n{content_preview}"}
            ],
            max_tokens=400,
            temperature=0.5
        )
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Hata oluştu: {str(e)}"


# ============================================================================
# GRADIO ARAYÜZÜ OLUŞTURMA
# ============================================================================

def create_gradio_interface():
    """
    Gradio Web Arayüzü Oluşturma Fonksiyonu
    =====================================
    
    Bu fonksiyon tüm uygulamaları içeren tab'lı bir web arayüzü oluşturur.
    
    Gradio Bileşenleri:
    ------------------
    - Blocks: Ana container, özel layout için
    - Tabs: Sekme yapısı
    - Tab: Her bir sekme
    - Chatbot: Sohbet arayüzü
    - Textbox: Metin giriş/çıkış kutusu
    - Button: Düğme
    - Dropdown: Açılır menü
    - File: Dosya yükleme
    - Code: Kod editörü
    
    Döndürür:
    ---------
    gr.Blocks
        Oluşturulan Gradio arayüzü
    """
    
    # ========================================================================
    # TEMA VE STİL AYARLARI
    # ========================================================================
    
    # Gradio'nun önceden tanımlı temalarından Soft teması
    theme = gr.themes.Soft(
        primary_hue="blue",       # Ana renk: Mavi tonları
        secondary_hue="gray",     # İkincil renk: Gri tonları
        font=("Arial", "sans-serif")  # Font ailesi
    )
    
    # ========================================================================
    # ANA BLOCKS CONTAINER OLUŞTURMA
    # ========================================================================
    
    # Blocks: Özelleştirilebilir layout için en esnek yapı
    with gr.Blocks(theme=theme, title="LLM Uygulama Demo") as demo:
        
        # Ana başlık (Markdown formatında)
        gr.Markdown(
            """
            # 🤖 LLM Tabanlı Uygulama Örnekleri
            
            Bu uygulama Gradio kullanarak çeşitli LLM uygulamalarını gösterir.
            """
        )
        
        # ====================================================================
        # SEKMELER (TABS) YAPISI
        # ====================================================================
        
        with gr.Tabs():
            
            # ================================================================
            # TAB 1: BASİT CHATBOT
            # ================================================================
            
            with gr.Tab("💬 Basit Chatbot"):
                gr.Markdown("### Basit chatbot arayüzü")
                
                # Chatbot bileşeni: Mesajları gösterir
                # type="messages": Yeni format, her mesaj {"role": "...", "content": "..."} şeklinde
                chatbot = gr.Chatbot(label="Konuşma", type="messages")
                
                # Kullanıcı giriş kutusu
                msg = gr.Textbox(
                    label="Mesajınız",
                    placeholder="Mesajınızı yazın...",
                    lines=2  # 2 satırlık giriş kutusu
                )
                
                # Gönder düğmesi
                submit_btn = gr.Button("Gönder", variant="primary")  # primary: vurgulu görünüm
                
                # Temizle düğmesi
                clear_btn = gr.Button("Temizle")
                
                # ============================================================
                # CHATBOT İŞLEVİ
                # ============================================================
                
                def respond(message, chat_history):
                    """
                    Chatbot Yanıt Fonksiyonu
                    ------------------------
                    
                    Bu fonksiyon:
                    1. Kullanıcı mesajını alır
                    2. AI'den yanıt üretir
                    3. Her ikisini de history'ye ekler
                    4. Güncellenmiş history'yi döndürür
                    
                    Parametreler:
                    ------------
                    message : str
                        Kullanıcının mesajı
                    chat_history : list
                        Mevcut konuşma geçmişi
                        
                    Döndürür:
                    ---------
                    tuple : ("", güncellenmiş_history)
                        İlk değer boş string (input kutusunu temizlemek için)
                        İkinci değer güncellenmiş konuşma geçmişi
                    """
                    # AI'den yanıt al
                    bot_message = simple_chatbot(message, chat_history)
                    
                    # Kullanıcı mesajını history'ye ekle
                    chat_history.append({"role": "user", "content": message})
                    
                    # AI yanıtını history'ye ekle
                    chat_history.append({"role": "assistant", "content": bot_message})
                    
                    # Boş string ile input'u temizle, güncellenmiş history'yi döndür
                    return "", chat_history
                
                # Olayları (events) bağla
                # Submit (Enter tuşu) ile mesaj gönder
                msg.submit(respond, [msg, chatbot], [msg, chatbot])
                
                # Gönder düğmesine tıklama ile mesaj gönder
                submit_btn.click(respond, [msg, chatbot], [msg, chatbot])
                
                # Temizle düğmesi: history'yi boş liste yap
                # queue=False: Kuyruk sistemini atla, hemen çalıştır
                clear_btn.click(lambda: [], None, chatbot, queue=False)
            
            # ================================================================
            # TAB 2: STREAMING CHATBOT
            # ================================================================
            
            with gr.Tab("🌊 Streaming Chatbot"):
                gr.Markdown("### Streaming output ile chatbot")
                
                # Streaming chatbot için arayüz bileşenleri
                streaming_chatbot_ui = gr.Chatbot(label="Konuşma", type="messages")
                streaming_msg = gr.Textbox(
                    label="Mesajınız",
                    placeholder="Mesajınızı yazın...",
                    lines=2
                )
                streaming_submit = gr.Button("Gönder", variant="primary")
                streaming_clear = gr.Button("Temizle")
                
                # ============================================================
                # STREAMING CHATBOT İŞLEVİ
                # ============================================================
                
                def streaming_respond(message, chat_history):
                    """
                    Streaming Yanıt Fonksiyonu
                    --------------------------
                    
                    Generator fonksiyonu: yield kullanarak ara sonuçlar döndürür
                    
                    Çalışma Mantığı:
                    ---------------
                    1. Kullanıcı mesajını history'ye ekle
                    2. Boş bir assistant mesajı ekle
                    3. Her chunk için bu boş mesajı güncelle
                    4. Güncellenmiş chat_history'yi yield et
                    """
                    # Kullanıcı mesajını ekle
                    chat_history.append({"role": "user", "content": message})
                    
                    # Boş assistant mesajı ekle (doldurulacak)
                    chat_history.append({"role": "assistant", "content": ""})
                    
                    # Streaming yanıt al
                    # chat_history[:-2]: Son iki mesajı hariç tut (şu anki exchange)
                    for response in streaming_chatbot(message, chat_history[:-2]):
                        # Son mesajı (assistant) güncelle
                        chat_history[-1] = {"role": "assistant", "content": response}
                        # Güncellenmiş history'yi yield et (arayüzde canlı güncelleme için)
                        yield chat_history
                
                # Olayları bağla
                streaming_msg.submit(streaming_respond, [streaming_msg, streaming_chatbot_ui], streaming_chatbot_ui)
                streaming_submit.click(streaming_respond, [streaming_msg, streaming_chatbot_ui], streaming_chatbot_ui)
                streaming_clear.click(lambda: [], None, streaming_chatbot_ui, queue=False)
            
            # ================================================================
            # TAB 3: METİN İŞLEME
            # ================================================================
            
            with gr.Tab("📝 Metin İşleme"):
                
                # Row: Yatay olarak yan yana yerleştirme
                with gr.Row():
                    
                    # SOL KOLON: METİN ÖZETLEME
                    with gr.Column():
                        gr.Markdown("### Metin Özetleme")
                        
                        # Özetlenecek metin girişi
                        text_input = gr.Textbox(
                            label="Metin",
                            placeholder="Özetlemek istediğiniz metni yazın...",
                            lines=5
                        )
                        
                        # Özetle düğmesi
                        summarize_btn = gr.Button("Özetle", variant="primary")
                        
                        # Özet çıktısı
                        summary_output = gr.Textbox(label="Özet", lines=5)
                        
                        # Düğme tıklamasını işleve bağla
                        # text_input -> text_summarizer -> summary_output
                        summarize_btn.click(text_summarizer, text_input, summary_output)
                    
                    # SAĞ KOLON: METİN ÇEVİRİ
                    with gr.Column():
                        gr.Markdown("### Metin Çeviri")
                        
                        # Çevrilecek metin
                        translate_input = gr.Textbox(
                            label="Çevrilecek Metin",
                            placeholder="Çevirmek istediğiniz metni yazın...",
                            lines=3
                        )
                        
                        # Hedef dil seçimi (Dropdown)
                        language_select = gr.Dropdown(
                            choices=["İngilizce", "Fransızca", "Almanca", "İspanyolca", "Japonca"],
                            label="Hedef Dil",
                            value="İngilizce"  # Varsayılan seçim
                        )
                        
                        # Çevir düğmesi
                        translate_btn = gr.Button("Çevir", variant="primary")
                        
                        # Çeviri çıktısı
                        translate_output = gr.Textbox(label="Çeviri", lines=5)
                        
                        # İki giriş (metin ve dil) bir fonksiyona, bir çıktıya
                        translate_btn.click(
                            text_translator, 
                            [translate_input, language_select],  # Girdiler listesi
                            translate_output
                        )
            
            # ================================================================
            # TAB 4: KOD AÇIKLAMA
            # ================================================================
            
            with gr.Tab("💻 Kod Açıklama"):
                gr.Markdown("### Kod açıklama aracı")
                
                # Code bileşeni: Syntax highlighting ile kod girişi
                code_input = gr.Code(
                    label="Kod",
                    language="python"  # Varsayılan dil
                )
                
                # Programlama dili seçimi
                code_language = gr.Dropdown(
                    choices=["Python", "JavaScript", "Java", "C++", "Go"],
                    label="Programlama Dili",
                    value="Python"
                )
                
                # Açıkla düğmesi
                explain_btn = gr.Button("Açıkla", variant="primary")
                
                # Açıklama çıktısı (uzun olabilir, 10 satır)
                code_explanation = gr.Textbox(label="Açıklama", lines=10)
                
                # Kod ve dil bilgisini al, açıkla
                explain_btn.click(
                    code_explainer, 
                    [code_input, code_language], 
                    code_explanation
                )
            
            # ================================================================
            # TAB 5: DOSYA İŞLEME
            # ================================================================
            
            with gr.Tab("📁 Dosya İşleme"):
                gr.Markdown(
                    """
                    ### Dosya içeriği analizi
                    Desteklenen formatlar: PDF, TXT, Python, JavaScript, Markdown, 
                    JSON, CSV, HTML, CSS, YAML, XML
                    """
                )
                
                # File bileşeni: Dosya yükleme
                file_input = gr.File(
                    label="Dosya Yükle",
                    # Kabul edilen dosya türleri
                    file_types=[
                        ".txt", ".py", ".js", ".md", ".json", 
                        ".csv", ".html", ".css", ".yaml", ".yml", 
                        ".xml", ".pdf"
                    ]
                )
                
                # İşle düğmesi
                process_btn = gr.Button("İşle", variant="primary")
                
                # Analiz sonucu çıktısı
                file_output = gr.Textbox(label="Analiz Sonucu", lines=10)
                
                # Dosyayı işle
                process_btn.click(file_processor, file_input, file_output)
        
        # ====================================================================
        # FOOTER (Alt Bilgi)
        # ====================================================================
        
        gr.Markdown(
            """
            ---
            **Not**: Bu uygulama OpenAI API kullanmaktadır. 
            API key'inizi `.env` dosyasına eklemeyi unutmayın.
            """
        )
    
    # Oluşturulan demo'yu döndür
    return demo


# ============================================================================
# UYGULAMA ÇALIŞTIRMA (Ana Program)
# ============================================================================

if __name__ == "__main__":
    """
    Program buradan başlar.
    
    __name__ == "__main__": 
    Bu kontrol, dosyanın doğrudan çalıştırıldığını 
    (import edilmediğini) doğrular.
    """
    
    # Başlangıç mesajı
    print("\n" + "="*60)
    print("🚀 LLM Uygulama Demo Başlatılıyor...")
    print("="*60 + "\n")
    
    # ADIM 1: Gradio arayüzünü oluştur
    demo = create_gradio_interface()
    
    # ADIM 2: Queue (Kuyruk) sistemi aktif et
    demo.queue()
    
    # ADIM 3: Uygulamayı başlat (Web sunucusunu çalıştır)
    demo.launch(
        server_name="0.0.0.0",  # Tüm network interface'lerinde dinle
        server_port=7861,       # Port numarası
        share=False,            # Public link oluşturma
        show_error=True,        # Hataları kullanıcıya göster
        inbrowser=True          # Otomatik tarayıcıda aç
    )
    
    """
    Uygulama Erişim Bilgileri:
    -------------------------
    🌐 Tarayıcınızda şu adresleri kullanın:
       - http://localhost:7861
       - http://127.0.0.1:7861
    
    ⚠️  Uygulamayı durdurmak için: Ctrl+C
    """


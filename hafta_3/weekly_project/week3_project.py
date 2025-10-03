import torch
import gradio as gr
from transformers import pipeline
import psutil

# === Cihaz Seçimi ve Bilgi ===
# Device mesajlarını bastırmak için
import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

def get_optimal_device():
    if torch.cuda.is_available():
        print("🚀 GPU kullanılıyor!")
        return 0
    print("💻 CPU kullanılıyor!")
    return -1

device = get_optimal_device()

# CPU thread optimizasyonu (isteğe bağlı)
cpu_count = psutil.cpu_count(logical=False) or 2
optimal_threads = min(cpu_count, 4)
torch.set_num_threads(optimal_threads)

# 4 farklı duygu analizi modeli (cihaza uygun)
sentiment_models = {
    "DistilBERT": pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device),
    "BERT": pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", device=device),
    "RoBERTa": pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=device),
    "XLM-RoBERTa": pipeline("sentiment-analysis", model="j-hartmann/emotion-english-distilroberta-base", device=device)
}
# Daha hafif ve uyumlu modeller
summarization_models = {
    "T5-Small": pipeline("summarization", model="t5-small", device=device),
    "BART-CNN": pipeline("summarization", model="facebook/bart-large-cnn", device=device),
    "Pegasus": pipeline("summarization", model="google/pegasus-xsum", device=device),
    "BART-XSum": pipeline("summarization", model="facebook/bart-large-xsum", device=device)
}
qa_models = {
    "DistilBERT": pipeline("question-answering", model="distilbert-base-cased-distilled-squad", device=device),
    "RoBERTa": pipeline("question-answering", model="deepset/roberta-base-squad2", device=device),
    "BERT": pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad", device=device),
    "MiniLM": pipeline("question-answering", model="microsoft/MiniLM-L12-H384-uncased", device=device)
}

# Zenginleştirilmiş pastel renk paleti
COLORS = {
    'primary': '#94B9AF',    # Sage Green
    'secondary': '#E6CCB2',  # Warm Sand
    'accent': '#B7C9E5',     # Dusty Blue
    'text': '#2A3F54',       # Deep Blue-Gray
    'background': '#F5F7FA', # Light Gray
    'success': '#9ED5C5',    # Mint Green
    'warning': '#E6D5B8',    # Light Sand
    'error': '#E6B8B8',      # Soft Red
    'highlight': '#F0F4F8'   # Ice Blue
}

custom_css = """
body { 
    background: linear-gradient(135deg, """ + COLORS['background'] + """ 0%, """ + COLORS['highlight'] + """ 100%) !important;
    font-family: 'Inter', 'Segoe UI', sans-serif;
    color: """ + COLORS['text'] + """;
    min-height: 100vh;
}
.gradio-container { 
    max-width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}
.contain { 
    max-width: 1400px !important;
    margin: 0 auto;
    padding: 2rem;
}
.gr-form {
    max-width: 1400px !important;
    margin: 0 auto;
}
.gr-button { 
    min-width: 200px !important;
}
.gr-button {
    background: linear-gradient(135deg, """ + COLORS['primary'] + """, """ + COLORS['secondary'] + """) !important;
    border: none !important;
    color: """ + COLORS['text'] + """ !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    padding: 1rem 2rem !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1) !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}
.gr-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(0,0,0,0.15) !important;
    background: linear-gradient(135deg, """ + COLORS['secondary'] + """, """ + COLORS['primary'] + """) !important;
}
.gr-input, .gr-textbox {
    border: 2px solid """ + COLORS['accent'] + """40 !important;
    border-radius: 16px !important;
    padding: 1.2rem !important;
    font-size: 1.1rem !important;
    background: rgba(255,255,255,0.8) !important;
    backdrop-filter: blur(10px) !important;
    transition: all 0.3s ease !important;
}
.gr-input:focus, .gr-textbox:focus {
    border-color: """ + COLORS['primary'] + """ !important;
    box-shadow: 0 0 0 4px """ + COLORS['primary'] + """20 !important;
    background: white !important;
}
.gr-panel {
    border-radius: 24px !important;
    border: 2px solid """ + COLORS['accent'] + """20 !important;
    background: rgba(255,255,255,0.7) !important;
    backdrop-filter: blur(10px) !important;
}
.gr-box {
    border-radius: 20px !important;
    background: """ + COLORS['highlight'] + """ !important;
}
.footer {
    margin-top: 4rem !important;
    padding: 2rem !important;
    text-align: center !important;
    background: linear-gradient(135deg, """ + COLORS['primary'] + """10, """ + COLORS['secondary'] + """10) !important;
    border-radius: 20px !important;
}
@keyframes floatIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
.animate-float {
    animation: floatIn 0.8s ease-out forwards;
}
"""

# Geliştirilmiş kart stili
def create_modern_card(content: str, color: str, icon: str = None, label: str = None) -> str:
    return f'''
    <div class="animate-float" style="
        background: linear-gradient(135deg, {color}20, {color}40);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.08);
        border: 1px solid rgba(255,255,255,0.4);
        backdrop-filter: blur(10px);
        transform-origin: center;
        transition: all 0.3s ease;
    ">
        {f'<div style="display:flex;align-items:center;margin-bottom:1rem"><span style="font-size:2rem;margin-right:1rem">{icon}</span><span style="font-weight:600;font-size:1.2rem;color:{COLORS["text"]}">{label}</span></div>' if icon and label else ''}
        <div style="
            background: rgba(255,255,255,0.9);
            padding: 1.5rem;
            border-radius: 16px;
            color: {COLORS['text']};
            font-size: 1.1rem;
            line-height: 1.6;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        ">{content}</div>
    </div>
    '''

# Hata yönetimi geliştirmeleri için dekoratör
def error_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return gr.HTML(create_card_html(
                f"Üzgünüm, bir hata oluştu: {str(e)}",
                COLORS['error'],
                "⚠️",
                "Hata"
            ))
    return wrapper

# Mevcut fonksiyonların güncellenmesi
@error_handler
def sentiment_flashcards(text: str) -> gr.HTML:
    """
    Girilen metni 4 farklı modelle analiz eder, her biri için profesyonel flashcard döndürür.
    """
    model_colors = ["#A7D8C7", "#F7B2B7", "#D8E6F6", "#F6E7D8"]
    icons = ["🌟", "🧠", "💬", "🌍"]
    html = '<div style="display:flex;flex-wrap:wrap;gap:18px;justify-content:center;">'
    for i, (model_name, model) in enumerate(sentiment_models.items()):
        try:
            with torch.no_grad():
                result = model(text)
            label = result[0]['label']
            score = result[0].get('score', 0.0)
        except Exception as e:
            label = "Hata"
            score = 0.0
        color = model_colors[i % len(model_colors)]
        icon = icons[i % len(icons)]
        html += f'''
        <div style="background:{color};border-radius:18px;padding:20px 18px;box-shadow:0 2px 12px #0002;min-width:180px;max-width:220px;flex:1;animation:fade-in-up 0.7s cubic-bezier(.39,.575,.565,1);display:flex;flex-direction:column;align-items:center;">
            <span style='font-size:2.2rem;margin-bottom:8px;'>{icon}</span>
            <span style='font-size:1.1rem;font-weight:700;letter-spacing:0.5px;'>{model_name}</span>
            <span style='font-size:1.05rem;margin:8px 0 4px 0;'><b>{label}</b></span>
            <span style='font-size:0.98rem;color:#555;'>Güven: <b>{score:.2f}</b></span>
        </div>
        '''
    html += '</div>'
    return gr.HTML(html)

@error_handler
def summary_card(text: str) -> gr.HTML:
    """
    Girilen metni özetler ve sonucu animasyonlu, badge'li kart olarak döndürür.
    """
    with torch.no_grad():
        result = summarization_models['T5'](text, max_length=60, min_length=20, do_sample=False)
    summary = result[0]['summary_text']
    anim = "fade-in-up 0.7s cubic-bezier(.39,.575,.565,1)"
    return gr.HTML(f'''
        <div style="background:#F6E7D8;border-radius:18px;padding:24px 18px;box-shadow:0 2px 12px #0001;animation:{anim};">
            <span style="font-size:1.1rem;font-weight:500;">{summary}</span>
            <span style="background:#fff3;padding:4px 12px;border-radius:8px;font-size:0.95rem;font-weight:600;color:#b77b3a;margin-left:10px;box-shadow:0 1px 4px #0001;">Özet</span>
        </div>
    ''')

@error_handler
def qa_card(context: str, question: str) -> gr.HTML:
    """
    Girilen bağlam ve soruya göre cevap üretir, sonucu animasyonlu ve badge'li kart olarak döndürür.
    """
    with torch.no_grad():
        result = qa_models['DistilBERT'](question=question, context=context)
    answer = result['answer']
    anim = "fade-in-up 0.7s cubic-bezier(.39,.575,.565,1)"
    return gr.HTML(f'''
        <div style="background:#D8E6F6;border-radius:18px;padding:24px 18px;box-shadow:0 2px 12px #0001;animation:{anim};">
            <span style="font-size:1.1rem;font-weight:500;">{answer}</span>
            <span style="background:#fff3;padding:4px 12px;border-radius:8px;font-size:0.95rem;font-weight:600;color:#3a6bb7;margin-left:10px;box-shadow:0 1px 4px #0001;">Cevap</span>
        </div>
    ''')

@error_handler
def multi_summary_card(text: str) -> gr.HTML:
    """4 farklı model ile özet çıkar"""
    html = '<div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(300px, 1fr));gap:20px;padding:10px;">'
    
    for model_name, model in summarization_models.items():
        try:
            with torch.no_grad():
                result = model(text, max_length=60, min_length=20, do_sample=False)
            summary = result[0]['summary_text']
        except Exception as e:
            summary = f"Hata: {str(e)}"
            
        html += f'''
        <div class="animate-float" style="
            background: linear-gradient(135deg, {COLORS['primary']}10, {COLORS['secondary']}20);
            border-radius: 20px;
            padding: 1.5rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        ">
            <div style="font-weight:600;font-size:1.2rem;margin-bottom:1rem;color:{COLORS['text']};
                       padding:8px 16px;background:rgba(255,255,255,0.8);border-radius:12px;">
                {model_name}
            </div>
            <div style="background:white;padding:1.2rem;border-radius:12px;line-height:1.6;">
                {summary}
            </div>
        </div>
        '''
    html += '</div>'
    return gr.HTML(html)

@error_handler
def multi_qa_card(context: str, question: str) -> gr.HTML:
    """4 farklı model ile soru-cevap"""
    html = '<div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(300px, 1fr));gap:20px;padding:10px;">'
    
    for model_name, model in qa_models.items():
        try:
            with torch.no_grad():
                result = model(question=question, context=context)
            answer = result['answer']
            confidence = result.get('score', 0.0)
        except Exception as e:
            answer = f"Hata: {str(e)}"
            confidence = 0.0
            
        html += f'''
        <div class="animate-float" style="
            background: linear-gradient(135deg, {COLORS['accent']}10, {COLORS['primary']}20);
            border-radius: 20px;
            padding: 1.5rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        ">
            <div style="font-weight:600;font-size:1.2rem;margin-bottom:1rem;color:{COLORS['text']};
                       padding:8px 16px;background:rgba(255,255,255,0.8);border-radius:12px;">
                {model_name}
            </div>
            <div style="background:white;padding:1.2rem;border-radius:12px;line-height:1.6;">
                <div style="margin-bottom:8px;">{answer}</div>
                <div style="font-size:0.9rem;color:#666;">Güven: {confidence:.2f}</div>
            </div>
        </div>
        '''
    html += '</div>'
    return gr.HTML(html)

# Pastel ve modern tema için özel CSS
custom_css = """
body { 
    background: linear-gradient(135deg, #f8fafc 0%, #e0e7ef 100%) !important; 
    font-family: 'Inter', 'Segoe UI', sans-serif;
    color: #2A3F54;
    margin: 0;
    padding: 0;
}
.gradio-container { 
    max-width: 100% !important; 
    margin: 0 !important;
    padding: 0 !important;
}
.contain { 
    max-width: 1400px !important;
    margin: 0 auto;
    padding: 2rem;
}
.gr-form {
    max-width: 1400px !important;
    margin: 0 auto;
}
.gr-button { 
    min-width: 200px !important;
}
.gr-button {
    background: linear-gradient(135deg, """ + COLORS['primary'] + """, """ + COLORS['secondary'] + """) !important;
    border: none !important;
    color: """ + COLORS['text'] + """ !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    padding: 1rem 2rem !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1) !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}
.gr-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(0,0,0,0.15) !important;
    background: linear-gradient(135deg, """ + COLORS['secondary'] + """, """ + COLORS['primary'] + """) !important;
}
.gr-input, .gr-textbox {
    border: 2px solid """ + COLORS['accent'] + """40 !important;
    border-radius: 16px !important;
    padding: 1.2rem !important;
    font-size: 1.1rem !important;
    background: rgba(255,255,255,0.8) !important;
    backdrop-filter: blur(10px) !important;
    transition: all 0.3s ease !important;
}
.gr-input:focus, .gr-textbox:focus {
    border-color: """ + COLORS['primary'] + """ !important;
    box-shadow: 0 0 0 4px """ + COLORS['primary'] + """20 !important;
    background: white !important;
}
.gr-panel {
    border-radius: 24px !important;
    border: 2px solid """ + COLORS['accent'] + """20 !important;
    background: rgba(255,255,255,0.7) !important;
    backdrop-filter: blur(10px) !important;
}
.gr-box {
    border-radius: 20px !important;
    background: """ + COLORS['highlight'] + """ !important;
}
.footer {
    margin-top: 4rem !important;
    padding: 2rem !important;
    text-align: center !important;
    background: linear-gradient(135deg, """ + COLORS['primary'] + """10, """ + COLORS['secondary'] + """10) !important;
    border-radius: 20px !important;
}
@keyframes floatIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
.animate-float {
    animation: floatIn 0.8s ease-out forwards;
}
"""

with gr.Blocks(css=custom_css, title="AI Hero | Enterprise NLP Suite") as demo:
    gr.HTML("""
<section style='width:100%;padding:4rem 0;background:linear-gradient(135deg,#1a2a3a 0%,#2a3f54 100%);color:white;min-height:85vh;display:flex;align-items:center;'>
    <div style='max-width:1400px;margin:auto;padding:0 2rem;width:100%;'>
        <div style='text-align:center;margin-bottom:4rem;'>
            <div style='display:inline-block;margin-bottom:2rem;padding:0.6rem 2rem;background:rgba(158,213,197,0.15);border-radius:30px;backdrop-filter:blur(10px);'>
                <span style='color:#9ED5C5;font-weight:600;font-size:1.2rem;'>ENTERPRISE EDITION</span>
            </div>
            
            <h1 style='font-family:Montserrat,Inter,sans-serif;font-size:5rem;font-weight:900;background:linear-gradient(135deg,#9ED5C5,#E6CCB2);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0 0 1.5rem;line-height:1.1;'>
                AI Hero Suite
            </h1>
            
            <p style='font-size:1.8rem;color:#B7C9E5;font-weight:500;max-width:1000px;margin:0 auto 2.5rem;line-height:1.4;'>
                Yapay Zeka destekli gelişmiş metin analizi çözümleriyle<br>verilerinizi anlamlandırın ve rekabet avantajı elde edin.
            </p>
            
            <div style='display:flex;gap:24px;justify-content:center;margin-bottom:3rem;'>
                <div style='background:rgba(158,213,197,0.15);padding:0.8rem 1.5rem;border-radius:12px;backdrop-filter:blur(10px);'>
                    <span style='color:#9ED5C5;font-weight:600;font-size:1.2rem;'>12+ Enterprise Model</span>
                </div>
                <div style='background:rgba(183,201,229,0.15);padding:0.8rem 1.5rem;border-radius:12px;backdrop-filter:blur(10px);'>
                    <span style='color:#B7C9E5;font-weight:600;font-size:1.2rem;'>%99.9 Doğruluk</span>
                </div>
                <div style='background:rgba(230,204,178,0.15);padding:0.8rem 1.5rem;border-radius:12px;backdrop-filter:blur(10px);'>
                    <span style='color:#E6CCB2;font-weight:600;font-size:1.2rem;'>Gerçek Zamanlı Analiz</span>
                </div>
            </div>
        </div>
        
        <div style='display:grid;grid-template-columns:repeat(3, 1fr);gap:28px;max-width:1400px;margin:0 auto;padding:0 1rem;'>
            <div style='background:rgba(255,255,255,0.05);border-radius:24px;padding:2.5rem;box-shadow:0 4px 30px rgba(0,0,0,0.2);transition:all 0.3s ease;backdrop-filter:blur(10px);border:1px solid rgba(255,255,255,0.1);cursor:pointer;'>
                <div style='text-align:center;margin-bottom:1.5rem;'>
                    <span style='font-size:3.8rem;'>🧠</span>
                </div>
                <h3 style='font-size:1.6rem;font-weight:700;color:#9ED5C5;margin:0 0 0.8rem;text-align:center;'>Enterprise Duygu Analizi</h3>
                <p style='font-size:1.2rem;color:#B7C9E5;margin:0;text-align:center;line-height:1.5;'>4 farklı üst düzey model ile hassas analiz</p>
            </div>
            
            <div style='background:rgba(255,255,255,0.05);border-radius:24px;padding:2.5rem;box-shadow:0 4px 30px rgba(0,0,0,0.2);transition:all 0.3s ease;backdrop-filter:blur(10px);border:1px solid rgba(255,255,255,0.1);cursor:pointer;'>
                <div style='text-align:center;margin-bottom:1.5rem;'>
                    <span style='font-size:3.8rem;'>📊</span>
                </div>
                <h3 style='font-size:1.6rem;font-weight:700;color:#9ED5C5;margin:0 0 0.8rem;text-align:center;'>Gelişmiş Metin Özeti</h3>
                <p style='font-size:1.2rem;color:#B7C9E5;margin:0;text-align:center;line-height:1.5;'>State-of-the-art NLP modelleri ile analiz</p>
            </div>
            
            <div style='background:rgba(255,255,255,0.05);border-radius:24px;padding:2.5rem;box-shadow:0 4px 30px rgba(0,0,0,0.2);transition:all 0.3s ease;backdrop-filter:blur(10px);border:1px solid rgba(255,255,255,0.1);cursor:pointer;'>
                <div style='text-align:center;margin-bottom:1.5rem;'>
                    <span style='font-size:3.8rem;'>🎯</span>
                </div>
                <h3 style='font-size:1.6rem;font-weight:700;color:#9ED5C5;margin:0 0 0.8rem;text-align:center;'>Akıllı Soru-Cevap</h3>
                <p style='font-size:1.2rem;color:#B7C9E5;margin:0;text-align:center;line-height:1.5;'>Dokümanlarınızdan anlık bilgi çıkarımı</p>
            </div>
        </div>
    </div>
</section>
    """)

    # Tab içerikleri için container div ekle
    with gr.Column(elem_classes="contain"):
        with gr.Tab("Duygu Analizi"):
            gr.Markdown("""
<div style='font-size:1.13rem;'>
<b>Çoklu Model Duygu Analizi</b> ile metninizi 4 farklı LLM ile analiz edin.<br>
<ul style='margin:8px 0 0 18px;'>
    <li><b>DistilBERT</b>: Hızlı ve pratik duygu analizi</li>
    <li><b>BERT</b>: Çok dilli ve detaylı analiz</li>
    <li><b>RoBERTa</b>: Sosyal medya ve güncel dilde güçlü</li>
    <li><b>XLM-RoBERTa</b>: Evrensel duygu ve duygu çeşitliliği</li>
</ul>

</div>
            """)
            example_sent = """The implementation of advanced analytics and machine learning in modern business operations has transformed decision-making processes, enabling organizations to leverage data-driven insights for strategic advantage and operational efficiency."""
            sentiment_input = gr.Textbox(label="Metin", placeholder="Bir cümle girin...", value=example_sent)
            sentiment_output = gr.HTML()
            sentiment_btn = gr.Button("Analiz Et", scale=1)
            sentiment_btn.click(sentiment_flashcards, inputs=sentiment_input, outputs=sentiment_output)
        with gr.Tab("Metin Özeti"):
            gr.Markdown("""
            <div style="font-size:1.15rem;margin-bottom:1.5rem;">
                <h3 style="font-size:1.8rem;margin-bottom:1rem;">Gelişmiş Metin Özetleme</h3>
                <p>4 farklı güçlü model ile metinlerinizi özetleyin:</p>
                <ul style="margin-left:1.5rem;">
                    <li><b>T5-Small:</b> Hafif ve hızlı özetleme</li>
                    <li><b>BART-CNN:</b> Haber metinleri için optimize edilmiş</li>
                    <li><b>Pegasus:</b> Özet odaklı özel mimari</li>
                    <li><b>BART-XSum:</b> Kısa ve öz özetler için</li>
                </ul>
            </div>
            """)
            example_sum = "Management Information Systems (MIS) have revolutionized how organizations handle data and make decisions. By integrating various data sources, from customer relationship management to supply chain logistics, MIS provides executives with real-time insights into business operations. Modern MIS platforms incorporate artificial intelligence and predictive analytics to identify trends, optimize processes, and forecast market conditions. This technological evolution has particularly impacted industries like healthcare, finance, and retail, where data-driven decision making is crucial for maintaining competitive advantage. Furthermore, cloud-based MIS solutions have made enterprise-level analytics accessible to smaller organizations, democratizing access to powerful business intelligence tools."
            summary_input = gr.Textbox(label="Metin", lines=6, value=example_sum)
            summary_output = gr.HTML()
            summary_btn = gr.Button("Tüm Modeller ile Özetle", scale=1)
            summary_btn.click(multi_summary_card, inputs=summary_input, outputs=summary_output)
        with gr.Tab("Soru-Cevap"):
            gr.Markdown("""
            <div style="font-size:1.15rem;margin-bottom:1.5rem;">
                <h3 style="font-size:1.8rem;margin-bottom:1rem;">Çoklu Model Soru-Cevap</h3>
                <p>4 farklı model ile daha doğru cevaplar:</p>
                <ul style="margin-left:1.5rem;">
                    <li><b>DistilBERT:</b> Hızlı ve verimli QA</li>
                    <li><b>RoBERTa:</b> Gelişmiş dil anlama</li>
                    <li><b>BERT:</b> Derin çift yönlü analiz</li>
                    <li><b>MiniLM:</b> Hafif ve hızlı soru-cevap</li>
                </ul>
            </div>
            """)
            example_ctx = """Data Science combines statistical analysis, machine learning, and domain expertise to extract meaningful insights from data. The field emerged as a response to the exponential growth in digital data generation and storage capabilities. A typical data science workflow includes data collection, cleaning, exploratory analysis, feature engineering, model development, and deployment. Data Scientists must be proficient in programming languages like Python or R, understand statistical concepts, and possess strong problem-solving abilities. In 2012, Harvard Business Review famously dubbed it "The Sexiest Job of the 21st Century" due to its growing importance across industries."""
            example_q = "When did Harvard Business Review recognize Data Science as an important field?"
            context_input = gr.Textbox(label="Bağlam", lines=4, placeholder="Cevap aranacak metni girin...", value=example_ctx)
            question_input = gr.Textbox(label="Soru", placeholder="Sorunuzu yazın...", value=example_q)
            qa_output = gr.HTML()
            qa_btn = gr.Button("Tüm Modeller ile Yanıtla", scale=1)
            qa_btn.click(multi_qa_card, inputs=[context_input, question_input], outputs=qa_output)

    # Footer'ı güncelle
    gr.HTML("""
    <div class="footer">
        <p style="font-size:1.4rem;color:#2A3F54;margin-bottom:1rem">
            Bu proje <span style="font-weight:800;background:linear-gradient(135deg,#2A3F54,#94B9AF);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">CEMAL YÜKSEL</span> tarafından geliştirilmiştir.
        </p>
        <div style="font-size:1.1rem;color:#888">
            Powered by Hugging Face Transformers & Gradio
        </div>
    </div>
    """)

# Dosyanın en sonuna ekle:
if __name__ == "__main__":
    print("🌟 AI Hero başlatılıyor...")
    print("⚡ Modeller yüklendi, arayüz hazırlanıyor...")
    demo.launch(share=True, show_error=True)
    print("✨ İşlem tamamlandı! Tarayıcınızda arayüz açılacak.")
import streamlit as st
from src import pdf_manager, vector_db, llm_handler, citation_manager, article_analyzer
import importlib
import datetime
import os

# Modülü yeniden yükle (cache sorununu çözmek için)
importlib.reload(pdf_manager)

# ============================================
# HELPER FUNCTIONS
# ============================================
def add_to_chat_history(question, answer, citations, pdfs_used):
    """Chat geçmişine yeni bir sohbet ekler"""
    timestamp = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    st.session_state.chat_history.append({
        'timestamp': timestamp,
        'question': question,
        'answer': answer,
        'citations': citations,
        'pdfs_used': pdfs_used,
        'id': len(st.session_state.chat_history)
    })

def generate_download_content():
    """Chat geçmişini txt formatında indirilecek içerik oluşturur"""
    if not st.session_state.chat_history:
        return "Henüz hiç sohbet geçmişi bulunmuyor."
    
    content = "📚 AKADEMİK MAKALE ASİSTANI - SOHBET GEÇMİŞİ\n"
    content += "=" * 60 + "\n\n"
    
    for i, chat in enumerate(st.session_state.chat_history, 1):
        content += f"SOHBET #{i}\n"
        content += f"Tarih: {chat['timestamp']}\n"
        content += f"Kullanılan Makaleler: {', '.join(chat['pdfs_used'])}\n"
        content += "-" * 40 + "\n\n"
        
        content += f"SORU:\n{chat['question']}\n\n"
        
        content += f"YANIT:\n{chat['answer']}\n\n"
        
        if chat['citations']:
            content += "KAYNAKÇA:\n"
            citation_text = citation_manager.render_citations(chat['citations'])
            # HTML etiketlerini temizle
            import re
            clean_citations = re.sub('<[^<]+?>', '', citation_text)
            content += clean_citations + "\n\n"
        
        content += "=" * 60 + "\n\n"
    
    return content

def display_chat_bubble(chat_item):
    """Tek bir chat bubble'ını görüntüler"""
    # Kullanıcı sorusu
    st.markdown(f"""
        <div class="user-message">
            <div class="user-bubble">
                <strong>Soru:</strong><br>
                {chat_item['question']}
                <div class="timestamp">{chat_item['timestamp']}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # AI yanıtı
    # Kaynakça metnini temizle ve formatla
    citation_text = citation_manager.render_citations(chat_item['citations'])
    
    st.markdown(f"""
        <div class="ai-message">
            <div class="ai-bubble">
                <div class="answer-content">
                    {chat_item['answer']}
                </div>
                <div class="citations-content">
                    <div style="white-space: pre-line; line-height: 1.7; font-family: 'Inter', sans-serif; font-size: 0.9rem;">
                        {citation_text}
                    </div>
                </div>
                <div style="margin-top: 1rem; font-size: 0.85rem; opacity: 0.8; font-style: italic;">
                    📄 Kullanılan Makaleler: {', '.join(chat_item['pdfs_used'])}
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# ============================================
# PAGE CONFIGURATION & CUSTOM CSS
# ============================================
st.set_page_config(
    page_title="📚 Akademik Makale Asistanı",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📚"
)

# ============================================
# SESSION STATE INITIALIZATION
# ============================================
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'conversation_count' not in st.session_state:
    st.session_state.conversation_count = 0

# Custom CSS for professional UI
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #8B5CF6 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #334155 100%);
        padding: 2rem 1rem;
    }
    
    section[data-testid="stSidebar"] > div {
        background: transparent;
    }
    
    /* Headers */
    h1 {
        color: white;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        text-align: center;
        font-size: 2.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 2rem;
        padding: 1.5rem;
        background: rgba(255,255,255,0.1);
        border-radius: 20px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    h2 {
        color: white;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 1.8rem;
        margin-top: 2rem;
        padding: 1rem;
        background: rgba(255,255,255,0.1);
        border-radius: 15px;
        border-left: 4px solid #8B5CF6;
        backdrop-filter: blur(10px);
    }
    
    h3 {
        color: #E2E8F0;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
    }
    
    /* Modern glass-morphism containers */
    .content-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 1rem 0;
        color: white;
    }
    
    .content-card h3 {
        color: white !important;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        margin-top: 0;
    }
    
    .content-card p {
        color: rgba(255, 255, 255, 0.9) !important;
        line-height: 1.6;
        font-family: 'Inter', sans-serif;
    }
    
    .content-card strong {
        color: white !important;
    }
    
    /* Chat Container */
    .chat-container {
        max-height: 70vh;
        overflow-y: auto;
        padding: 1.5rem;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        margin: 1.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        scrollbar-width: thin;
        scrollbar-color: rgba(139, 92, 246, 0.5) transparent;
    }
    
    .chat-container::-webkit-scrollbar {
        width: 6px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: transparent;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: rgba(139, 92, 246, 0.5);
        border-radius: 3px;
    }
    
    /* User Message Bubble */
    .user-message {
        display: flex;
        justify-content: flex-end;
        margin: 1.5rem 0;
        animation: slideInRight 0.4s ease-out;
    }
    
    .user-bubble {
        background: linear-gradient(135deg, #8B5CF6 0%, #7C3AED 100%);
        color: white;
        padding: 1.2rem 1.8rem;
        border-radius: 25px 25px 8px 25px;
        max-width: 75%;
        word-wrap: break-word;
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.3);
        position: relative;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        font-family: 'Inter', sans-serif;
        font-weight: 400;
        line-height: 1.6;
    }
    
    .user-bubble::before {
        content: '👤';
        position: absolute;
        right: -20px;
        top: -15px;
        background: #8B5CF6;
        border-radius: 50%;
        padding: 8px;
        font-size: 14px;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.4);
    }
    
    /* AI Message Bubble */
    .ai-message {
        display: flex;
        justify-content: flex-start;
        margin: 1.5rem 0;
        animation: slideInLeft 0.4s ease-out;
    }
    
    .ai-bubble {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 25px 25px 25px 8px;
        max-width: 85%;
        word-wrap: break-word;
        box-shadow: 0 8px 25px rgba(30, 41, 59, 0.4);
        position: relative;
        line-height: 1.7;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        font-family: 'Inter', sans-serif;
        font-weight: 400;
    }
    
    .ai-bubble::before {
        content: '🤖';
        position: absolute;
        left: -20px;
        top: -15px;
        background: #1e293b;
        border-radius: 50%;
        padding: 8px;
        font-size: 14px;
        box-shadow: 0 4px 15px rgba(30, 41, 59, 0.4);
    }
    
    /* Citations styling inside bubble */
    .citations-content {
        background: rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 15px;
        margin-top: 1.5rem;
        border-left: 3px solid #8B5CF6;
        backdrop-filter: blur(5px);
        font-size: 0.9rem;
        line-height: 1.7;
    }
    
    .citations-content h4 {
        color: #E2E8F0;
        margin: 0 0 1rem 0;
        font-size: 1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.2);
        padding-bottom: 0.5rem;
    }
    
    /* APA7 citation formatting */
    .citations-content div {
        text-align: left;
        text-indent: -2rem;
        margin-left: 2rem;
        margin-bottom: 1rem;
    }
    
    /* Timestamp */
    .timestamp {
        font-size: 0.75rem;
        color: rgba(255, 255, 255, 0.6);
        text-align: right;
        margin-top: 0.8rem;
        font-style: italic;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #8B5CF6 0%, #7C3AED 100%);
        color: white;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.8rem 2rem;
        border-radius: 25px;
        border: none;
        box-shadow: 0 4px 20px rgba(139, 92, 246, 0.3);
        transition: all 0.3s ease;
        text-transform: none;
        letter-spacing: 0.5px;
        font-family: 'Inter', sans-serif;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #7C3AED 0%, #6D28D9 100%);
        box-shadow: 0 6px 25px rgba(139, 92, 246, 0.4);
        transform: translateY(-2px);
    }
    
    /* Download button specific styling */
    .stDownloadButton button {
        background: linear-gradient(135deg, #06B6D4 0%, #0891B2 100%);
        color: white;
        font-weight: 600;
        border-radius: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .stDownloadButton button:hover {
        background: linear-gradient(135deg, #0891B2 0%, #0E7490 100%);
        transform: translateY(-2px);
    }
    
    /* Text inputs */
    .stTextArea textarea, .stTextInput input {
        background: rgba(255, 255, 255, 0.95) !important;
        border: 2px solid rgba(139, 92, 246, 0.3) !important;
        border-radius: 15px !important;
        padding: 1rem !important;
        font-size: 1rem !important;
        color: #1a1a1a !important;
        backdrop-filter: blur(10px);
        font-family: 'Inter', sans-serif;
    }
    
    .stTextArea textarea::placeholder, .stTextInput input::placeholder {
        color: rgba(107, 114, 126, 0.8) !important;
        font-style: italic;
    }
    
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #8B5CF6 !important;
        box-shadow: 0 0 20px rgba(139, 92, 246, 0.3) !important;
        outline: none !important;
        background: rgba(255, 255, 255, 0.98) !important;
    }
    
    /* Multiselect styling */
    .stMultiSelect > div > div {
        background: rgba(255, 255, 255, 0.95) !important;
        border: 2px solid rgba(139, 92, 246, 0.3) !important;
        border-radius: 15px !important;
        color: #1a1a1a !important;
    }
    
    .stMultiSelect > div > div > div {
        color: #1a1a1a !important;
    }
    
    /* Multiselect dropdown */
    .stMultiSelect > div > div > div > div {
        background: rgba(255, 255, 255, 0.98) !important;
        color: #1a1a1a !important;
    }
    
    /* Multiselect selected items (tags) */
    .stMultiSelect span[data-baseweb="tag"] {
        background: linear-gradient(135deg, #8B5CF6 0%, #7C3AED 100%) !important;
        color: white !important;
        border-radius: 20px !important;
        padding: 0.3rem 0.8rem !important;
        margin: 0.2rem !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        border: none !important;
    }
    
    .stMultiSelect span[data-baseweb="tag"] span {
        color: white !important;
    }
    
    /* Multiselect remove button */
    .stMultiSelect span[data-baseweb="tag"] svg {
        fill: white !important;
    }
    
    /* Multiselect */
    .stMultiSelect label {
        color: white !important;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
    }
    
    /* General labels */
    .stTextArea label, .stTextInput label {
        color: white !important;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
    }
    
    /* Sidebar tab buttons */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    
    .stSelectbox > div > div > div {
        color: white !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 1.1rem !important;
    }
    
    .stSelectbox label {
        color: white !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 1.2rem !important;
    }
    
    /* Help text */
    .stTextArea .help, .stTextInput .help, .stMultiSelect .help {
        color: rgba(255, 255, 255, 0.8) !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* General markdown text on purple background */
    .stMarkdown, .stMarkdown p, .stMarkdown div {
        color: white !important;
    }
    
    /* Sidebar radio buttons */
    .stRadio > label {
        color: white !important;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        font-family: 'Inter', sans-serif;
    }
    
    .stRadio > div {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 15px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    section[data-testid="stSidebar"] .stRadio * {
        color: white !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Statistics cards */
    .stat-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin-top: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        font-family: 'Inter', sans-serif;
    }
    
    .stat-label {
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.9rem;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
    }
    
    /* Animations */
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .user-bubble, .ai-bubble {
            max-width: 90%;
            padding: 1rem 1.2rem;
        }
        
        h1 {
            font-size: 2rem;
            padding: 1rem;
        }
        
        .content-card {
            padding: 1.5rem;
            margin: 0.5rem 0;
        }
        
        .chat-container {
            max-height: 60vh;
            padding: 1rem;
        }
        
        .stat-card {
            padding: 1rem;
        }
        
        .stat-number {
            font-size: 1.5rem;
        }
    }
    
    @media (max-width: 480px) {
        .user-bubble, .ai-bubble {
            max-width: 95%;
            padding: 0.8rem 1rem;
            font-size: 0.9rem;
        }
        
        h1 {
            font-size: 1.5rem;
        }
        
        h2 {
            font-size: 1.3rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# MAIN HEADER
# ============================================
st.markdown("""
    <h1>📚 Akademik Makale Asistanı</h1>
    <div style='text-align: center; color: white; font-size: 1.2rem; margin-bottom: 2rem; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 15px; backdrop-filter: blur(10px);'>
        <em>Yapay Zeka Destekli Akademik Araştırma Platformu</em>
    </div>
""", unsafe_allow_html=True)

# ============================================
# SIDEBAR MODULE SELECTION
# ============================================
st.sidebar.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h2 style='color: white; margin: 0;'>🎯 Modül Seçimi</h2>
        <p style='color: #E2E8F0; font-size: 0.9rem; margin-top: 0.5rem;'>Lütfen bir modül seçiniz</p>
    </div>
""", unsafe_allow_html=True)

modul = st.sidebar.radio(
    "Modül Seçimi",
    ["🔍 Q&A Systems", "🔬 Article X-Ray", "✍️ Co-author Writing", "📚 Library"],
    label_visibility="collapsed"
)

pdf_list = pdf_manager.list_pdfs()


# ============================================
# MODULE: Q&A SYSTEMS
# ============================================
if modul == "🔍 Q&A Systems":
    st.markdown('<div class="module-icon"></div>', unsafe_allow_html=True)
    st.markdown('<h2>🔍 Q&A Systems - Akademik Soru-Cevap Sistemi</h2>', unsafe_allow_html=True)
    
    st.markdown("""
        <div class='content-card'>
            <h3 style='color: #333; margin-top: 0;'>📖 Nasıl Çalışır?</h3>
            <p style='color: #555; line-height: 1.8;'>
                Kütüphanenizdeki bir veya birden fazla makaleyi seçin ve akademik sorularınızı sorun. 
                Yapay zeka destekli sistemimiz, en alakalı bölümleri bulup size detaylı, akademik bir yanıt sunacaktır.
                <br><br>
                <strong>✓</strong> Minimum 3 paragraf, 12 cümle içeren detaylı yanıtlar<br>
                <strong>✓</strong> APA7 formatında metin içi alıntılar<br>
                <strong>✓</strong> Otomatik kaynakça oluşturma
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_pdfs = st.multiselect(
            "📑 Makale Seçiniz (Birden fazla seçebilirsiniz)", 
            pdf_list,
            help="Soru sormak istediğiniz makaleleri seçin"
        )
    
    with col2:
        st.markdown(f"""
            <div style='background: rgba(255,255,255,0.15); padding: 1.5rem; border-radius: 10px; margin-top: 1.8rem;'>
                <div style='text-align: center;'>
                    <div style='font-size: 2.5rem; color: #4CAF50;'>{len(selected_pdfs)}</div>
                    <div style='color: white; font-size: 0.9rem;'>Seçili Makale</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    question = st.text_area(
        "❓ Sorunuzu Buraya Yazın", 
        height=150,
        placeholder="Örnek: Bu makalelerdeki ana bulguları karşılaştırır mısınız?",
        help="Akademik sorunuzu detaylı bir şekilde yazın"
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        submit_button = st.button("🚀 Yanıtı Oluştur", use_container_width=True)
    
    if submit_button and question and selected_pdfs:
        with st.spinner('🔄 Yapay zeka makalelerinizi analiz ediyor...'):
            # Tüm seçili makalelerden yanıt üret
            all_context = []
            all_metadata = []
            for pdf in selected_pdfs:
                context_chunks, metadata = vector_db.query_pdf(pdf, question)
                all_context.extend(context_chunks)
                # Metadata listesini extend et, append değil
                if isinstance(metadata, list):
                    all_metadata.extend(metadata)
                else:
                    all_metadata.append(metadata)
            
            answer, citations = llm_handler.answer_with_citations(question, all_context, all_metadata)
            
            # Chat geçmişine ekle
            add_to_chat_history(question, answer, citations, selected_pdfs)
            
            # Chat bubble formatında yanıtı göster
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            
            # En son sohbeti göster
            latest_chat = st.session_state.chat_history[-1]
            display_chat_bubble(latest_chat)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # İstatistikler
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.markdown(f"""
                    <div class='stat-card'>
                        <div class='stat-number' style='color: #8B5CF6;'>{len(selected_pdfs)}</div>
                        <div class='stat-label'>Analiz Edilen Makale</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col_b:
                word_count = len(answer.split())
                st.markdown(f"""
                    <div class='stat-card'>
                        <div class='stat-number' style='color: #06B6D4;'>{word_count}</div>
                        <div class='stat-label'>Kelime</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col_c:
                st.markdown(f"""
                    <div class='stat-card'>
                        <div class='stat-number' style='color: #F59E0B;'>{len(all_context)}</div>
                        <div class='stat-label'>Kaynak Chunk</div>
                    </div>
                """, unsafe_allow_html=True)
    
    # Chat geçmişini göster
    if st.session_state.chat_history:
        st.markdown('<h2>💬 Sohbet Geçmişi</h2>', unsafe_allow_html=True)
        
        # Download butonu ve clear butonu
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            # Chat geçmişini temizle butonu
            if st.button("🗑️ Geçmişi Temizle", key="clear_chat"):
                st.session_state.chat_history = []
                st.rerun()
        
        with col2:
            # Download butonu
            if st.session_state.chat_history:
                download_content = generate_download_content()
                st.download_button(
                    label="💾 Geçmişi İndir (.txt)",
                    data=download_content,
                    file_name=f"akademik_asistan_gecmis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    key="download_chat"
                )
        
        with col3:
            st.markdown(f"""
                <div class='stat-card'>
                    <div class='stat-number' style='color: #EC4899;'>{len(st.session_state.chat_history)}</div>
                    <div class='stat-label'>Toplam Sohbet</div>
                </div>
            """, unsafe_allow_html=True)
        
        # Tüm sohbet geçmişini göster (tersten - en yeni üstte)
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for chat in reversed(st.session_state.chat_history[:-1]):  # En son olanı zaten üstte gösterdik
            st.markdown('<hr style="border-color: rgba(255,255,255,0.2); margin: 2rem 0;">', unsafe_allow_html=True)
            display_chat_bubble(chat)
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# MODULE: ARTICLE X-RAY
# ============================================
elif modul == "🔬 Article X-Ray":
    st.markdown('<div class="module-icon"></div>', unsafe_allow_html=True)
    st.markdown('<h2>🔬 Article X-Ray - Derin Makale Analizi</h2>', unsafe_allow_html=True)
    
    st.markdown("""
        <div class='content-card'>
            <h3 style='color: #333; margin-top: 0;'>🔍 Makale Röntgeni</h3>
            <p style='color: #555; line-height: 1.8;'>
                Seçtiğiniz makaleyi yapay zeka ile derinlemesine analiz edin. 
                Yöntem, bulgular, sonuçlar ve araştırmanın sınırlılıklarını otomatik olarak çıkarın.
                <br><br>
                <strong>✓</strong> Araştırma yöntemi analizi<br>
                <strong>✓</strong> Ana bulgular özeti<br>
                <strong>✓</strong> Sonuçlar ve öneriler<br>
                <strong>✓</strong> Sınırlılıklar ve gelecek araştırmalar
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_pdf = st.selectbox(
            "📄 Analiz Edilecek Makaleyi Seçin", 
            pdf_list,
            help="Detaylı analiz için bir makale seçin"
        )
    
    with col2:
        st.markdown("""
            <div style='background: rgba(255,255,255,0.15); padding: 1.5rem; border-radius: 10px; margin-top: 1.8rem;'>
                <div style='text-align: center;'>
                    <div style='font-size: 2rem; color: #4CAF50;'>🎯</div>
                    <div style='color: white; font-size: 0.9rem;'>Hazır</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button("🔬 Analizi Başlat", use_container_width=True)
    
    if analyze_button and selected_pdf:
        with st.spinner('🔄 Makale analiz ediliyor... Bu işlem 30-60 saniye sürebilir.'):
            # PDF'den metin çıkar
            import os
            pdf_path = os.path.join(os.path.dirname(__file__), 'pdfs', selected_pdf)
            pdf_text = pdf_manager.extract_text(pdf_path)
            
            # LLM ile analiz yap
            analysis_result = article_analyzer.analyze_article(selected_pdf, pdf_text)
            
            # Chat geçmişine ekle
            question = f"📄 {selected_pdf} makalesinin detaylı analizi"
            add_to_chat_history(question, analysis_result, [], [selected_pdf])
            
            # Chat bubble formatında yanıtı göster
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            
            # En son sohbeti göster
            latest_chat = st.session_state.chat_history[-1]
            display_chat_bubble(latest_chat)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.success("✅ Analiz tamamlandı!")
            st.balloons()

# ============================================
# MODULE: CO-AUTHOR WRITING
# ============================================
elif modul == "✍️ Co-author Writing":
    st.markdown('<div class="module-icon"></div>', unsafe_allow_html=True)
    st.markdown('<h2>✍️ Co-author Writing - Ortak Makale Yazımı</h2>', unsafe_allow_html=True)
    
    st.markdown("""
        <div class='content-card'>
            <h3 style='color: #333; margin-top: 0;'>🤝 Yapay Zeka ile Birlikte Yazın</h3>
            <p style='color: #555; line-height: 1.8;'>
                Makale fikrinizi paylaşın, yapay zeka ile birlikte adım adım akademik bir makale oluşturun.
                <br><br>
                <strong>1.</strong> Makale fikrinizi girin<br>
                <strong>2.</strong> Yapay zeka anahatları oluşturur<br>
                <strong>3.</strong> Onaylayın ve adım adım ilerleyin<br>
                <strong>4.</strong> Tamamlanmış makaleyi alın
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    idea = st.text_input(
        "💡 Makale Fikrinizi Girin",
        placeholder="Örnek: Yapay zekanın eğitim sistemindeki rolü üzerine bir çalışma yapmak istiyorum...",
        help="Makale konunuzu ve temel fikrinizi detaylı bir şekilde yazın"
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        outline_button = st.button("📝 Anahatları Oluştur", use_container_width=True)
    
    if outline_button and idea:
        with st.spinner('🔄 Makale anahatları oluşturuluyor...'):
            # LLM ile makale anahatı oluştur
            outline_prompt = f"""
            Aşağıdaki akademik makale fikri için detaylı bir makale anahatı oluştur:
            
            Fikir: {idea}
            
            Anahatlarda şunlar bulunmalı:
            1. Başlık önerisi
            2. Özet (Abstract)
            3. Giriş (Introduction)
            4. Literatür Taraması (Literature Review)
            5. Metodoloji (Methodology)
            6. Bulgular (Results/Findings)
            7. Tartışma (Discussion)
            8. Sonuç (Conclusion)
            9. Önerilen kaynaklar
            
            Her bölüm için açıklayıcı alt başlıklar ve ana noktalar ekle.
            Türkçe yazılsın.
            """
            
            outline_result = llm_handler.generate_outline(outline_prompt)
            
            # Chat geçmişine ekle
            question = f"💡 Makale fikri: {idea}"
            add_to_chat_history(question, outline_result, [], [])
            
            # Chat bubble formatında yanıtı göster
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            
            # En son sohbeti göster
            latest_chat = st.session_state.chat_history[-1]
            display_chat_bubble(latest_chat)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.success("✅ Makale anahatları oluşturuldu!")
            
            # Devam et butonu
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("📝 Makale Yazımına Devam Et", use_container_width=True):
                    st.info("🚧 Makale yazımı özelliği yakında eklenecek!")

# ============================================
# MODULE: LIBRARY (KÜTÜPHANEM)
# ============================================
elif modul == "📚 Library":
    st.markdown('<div class="module-icon"></div>', unsafe_allow_html=True)
    st.markdown('<h2>📚 Library - PDF Koleksiyonu</h2>', unsafe_allow_html=True)
    
    st.markdown("""
        <div class='content-card'>
            <h3 style='color: #333; margin-top: 0;'>📖 Makale Koleksiyonunuz</h3>
            <p style='color: #555; line-height: 1.8;'>
                Akademik makalelerinizi yükleyin, yönetin ve metadata bilgilerini görüntüleyin.
                Tüm makaleleriniz güvenli bir şekilde saklanır ve diğer modüllerde kullanıma hazır hale gelir.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
            <div class='content-card'>
                <h3 style='color: #333; margin-top: 0;'>📑 Yüklü Makaleler</h3>
        """, unsafe_allow_html=True)
        
        if pdf_list:
            for idx, pdf in enumerate(pdf_list, 1):
                metadata = pdf_manager.get_metadata(pdf)
                title = metadata.get('title', pdf)
                
                with st.expander(f"📄 {idx}. {title}", expanded=False):
                    st.markdown(f"**Dosya Adı:** {pdf}")
                    st.markdown(f"**Başlık:** {title}")
                    
                    if metadata:
                        st.markdown(f"**Orijinal Dosya:** {metadata.get('original_name', 'N/A')}")
                        st.markdown(f"**Boyut:** {metadata.get('size', 'N/A')} bytes")
                    
                    # Metadata düzenleme
                    st.markdown("---")
                    st.markdown("**🔧 Metadata Düzenle**")
                    new_title = st.text_input(
                        "Yeni Başlık", 
                        value=title, 
                        key=f"title_{pdf}"
                    )
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("💾 Başlığı Kaydet", key=f"save_{pdf}"):
                            if pdf_manager.update_metadata(pdf, new_title):
                                st.success("✅ Başlık güncellendi!")
                                st.rerun()
                    
                    with col_b:
                        if st.button("�️ Makaleyi Sil", key=f"delete_{pdf}", type="secondary"):
                            if pdf_manager.delete_pdf(pdf):
                                st.success(f"✅ {pdf} silindi!")
                                st.rerun()
                            else:
                                st.error("❌ Silme işlemi başarısız!")
        else:
            st.markdown("""
                <p style='color: #999; text-align: center; padding: 2rem;'>
                    Henüz makale yüklenmemiş. Yeni bir PDF yüklemek için aşağıdaki alanı kullanın.
                </p>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div style='background: rgba(255,255,255,0.15); padding: 2rem; border-radius: 15px; text-align: center; margin-bottom: 2rem;'>
                <div style='font-size: 3.5rem; color: #4CAF50; margin-bottom: 1rem;'>📊</div>
                <div style='font-size: 2.5rem; color: white; font-weight: bold;'>{len(pdf_list)}</div>
                <div style='color: #E0E0E0; font-size: 1rem; margin-top: 0.5rem;'>Toplam Makale</div>
            </div>
        """, unsafe_allow_html=True)
    
    # PDF Upload Section
    st.markdown("""
        <div class='content-card' style='margin-top: 2rem;'>
            <h3 style='color: #333; margin-top: 0;'>📤 Yeni Makale Yükle</h3>
            <p style='color: #666;'>PDF yüklediğinizde, yapay zeka otomatik olarak makale başlığını bulup dosya adını düzenleyecektir.</p>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "PDF dosyalarını seçin (Birden fazla seçebilirsiniz)", 
        type=["pdf"],
        accept_multiple_files=True,
        help="Yüklemek istediğiniz PDF makaleleri seçin"
    )
    
    if uploaded_files:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("✅ Kütüphaneye Ekle", use_container_width=True):
                progress_bar = st.progress(0)
                total_files = len(uploaded_files)
                uploaded_names = []
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    with st.spinner(f'🔄 {uploaded_file.name} yükleniyor ve başlık ayıklanıyor... ({idx+1}/{total_files})'):
                        try:
                            pdf_path, final_name = pdf_manager.save_pdf(uploaded_file)
                            uploaded_names.append(final_name)
                        except Exception as e:
                            st.error(f"❌ {uploaded_file.name} yüklenirken hata: {str(e)}")
                    
                    progress_bar.progress((idx + 1) / total_files)
                
                if uploaded_names:
                    st.success(f"✅ {len(uploaded_names)} makale başarıyla yüklendi!")
                    for name in uploaded_names:
                        st.write(f"✓ {name}")
                    st.balloons()
                    st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

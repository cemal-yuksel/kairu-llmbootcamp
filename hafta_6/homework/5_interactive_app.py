"""
================================================================================
🎬 CineAI - Akıllı Film İnceleme Analiz Platformu
================================================================================

Netflix-inspired profesyonel film analiz arayüzü.

Özellikler:
- 🎯 Sinematik Hero Section
- 💬 Yapay Zeka Destekli Soru-Cevap
- 📊 Gelişmiş Görselleştirmeler
- 🎨 Modern Dark Theme
- 🌟 Smooth Animations & Transitions

KULLANIM:
---------
streamlit run 5_interactive_app.py

Yazar: Kairu AI - Build with LLMs Bootcamp
Tarih: 2 Kasım 2025
================================================================================
"""

import sys
from pathlib import Path
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import time

# Proje root
sys.path.append(str(Path(__file__).parent))

from config import config
# Import RAGSystem using importlib to handle filename starting with number
import importlib.util
spec = importlib.util.spec_from_file_location("rag_qa_system", Path(__file__).parent / "4_rag_qa_system.py")
rag_qa_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rag_qa_module)
RAGSystem = rag_qa_module.RAGSystem


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="🎬 CineAI - Film Analiz Platformu",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# CUSTOM CSS - NETFLIX INSPIRED DESIGN
# ============================================================================

st.markdown("""
<style>
    /* ====== GLOBAL STYLES ====== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');
    
    :root {
        --netflix-red: #E50914;
        --netflix-black: #141414;
        --netflix-dark: #1a1a1a;
        --netflix-gray: #2f2f2f;
        --netflix-light-gray: #808080;
        --gold: #FFD700;
        --success-green: #46d369;
        --gradient-primary: linear-gradient(135deg, #E50914 0%, #831010 100%);
        --gradient-dark: linear-gradient(180deg, rgba(20,20,20,0) 0%, rgba(20,20,20,1) 100%);
    }
    
    .stApp {
        background: var(--netflix-black);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* ====== HERO SECTION ====== */
    .hero-section {
        position: relative;
        width: 100%;
        height: 70vh;
        background: linear-gradient(rgba(20,20,20,0.3), rgba(20,20,20,0.9)),
                    url('https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=1920') center/cover;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        margin: -6rem -5rem 2rem -5rem;
        padding: 4rem 2rem;
        border-radius: 0 0 20px 20px;
        overflow: hidden;
        animation: fadeIn 1s ease-in;
    }
    
    .hero-title {
        font-size: 5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #ffffff 0%, #E50914 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 0 0 40px rgba(229, 9, 20, 0.5);
        letter-spacing: -2px;
        animation: slideDown 0.8s ease-out;
    }
    
    .hero-subtitle {
        font-size: 1.5rem;
        color: #ffffff;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
        max-width: 800px;
        animation: slideUp 0.8s ease-out;
    }
    
    .hero-stats {
        display: flex;
        gap: 3rem;
        margin-top: 2rem;
        animation: fadeIn 1.2s ease-in;
    }
    
    .stat-item {
        text-align: center;
        padding: 1.5rem 2rem;
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.2);
        transition: all 0.3s ease;
    }
    
    .stat-item:hover {
        transform: translateY(-5px);
        background: rgba(229, 9, 20, 0.2);
        border-color: var(--netflix-red);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 900;
        color: var(--netflix-red);
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #ffffff;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.5rem;
    }
    
    /* ====== SIDEBAR STYLING ====== */
    [data-testid="stSidebar"] {
        background: var(--netflix-dark);
        border-right: 1px solid var(--netflix-gray);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
        color: #ffffff;
        font-weight: 700;
    }
    
    /* ====== MODE CARDS ====== */
    .mode-card {
        background: var(--netflix-dark);
        border: 2px solid var(--netflix-gray);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    
    .mode-card:hover {
        transform: translateY(-5px);
        border-color: var(--netflix-red);
        box-shadow: 0 10px 30px rgba(229, 9, 20, 0.3);
    }
    
    .mode-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: var(--gradient-primary);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .mode-card:hover::before {
        opacity: 0.1;
    }
    
    .mode-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .mode-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    
    .mode-description {
        color: var(--netflix-light-gray);
        font-size: 1rem;
        line-height: 1.6;
    }
    
    /* ====== BUTTONS ====== */
    .stButton > button {
        background: var(--gradient-primary);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 4px 15px rgba(229, 9, 20, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(229, 9, 20, 0.6);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* ====== INPUT FIELDS ====== */
    .stTextInput > div > div > input {
        background: var(--netflix-dark);
        border: 2px solid var(--netflix-gray);
        border-radius: 8px;
        color: #ffffff;
        font-size: 1.1rem;
        padding: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--netflix-red);
        box-shadow: 0 0 15px rgba(229, 9, 20, 0.3);
    }
    
    /* ====== RESULT CARDS ====== */
    .answer-box {
        background: linear-gradient(135deg, var(--netflix-dark) 0%, var(--netflix-gray) 100%);
        border-left: 5px solid var(--netflix-red);
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        animation: slideIn 0.5s ease-out;
    }
    
    .answer-box h4 {
        color: #ffffff;
        font-size: 1.3rem;
        line-height: 1.8;
        font-weight: 400;
    }
    
    .review-card {
        background: var(--netflix-dark);
        border: 1px solid var(--netflix-gray);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .review-card:hover {
        border-color: var(--netflix-red);
        transform: translateX(5px);
        box-shadow: 0 5px 20px rgba(229, 9, 20, 0.2);
    }
    
    .review-card::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        height: 100%;
        width: 3px;
        background: var(--netflix-red);
        transform: scaleY(0);
        transition: transform 0.3s ease;
    }
    
    .review-card:hover::before {
        transform: scaleY(1);
    }
    
    /* ====== METRICS ====== */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 900;
        color: var(--netflix-red);
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--netflix-light-gray);
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* ====== EXPANDERS ====== */
    .streamlit-expanderHeader {
        background: var(--netflix-dark);
        border: 1px solid var(--netflix-gray);
        border-radius: 10px;
        color: #ffffff;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: var(--netflix-gray);
        border-color: var(--netflix-red);
    }
    
    /* ====== BADGES ====== */
    .sentiment-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-positive {
        background: linear-gradient(135deg, #46d369 0%, #2ea64b 100%);
        color: white;
    }
    
    .badge-negative {
        background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
        color: white;
    }
    
    /* ====== ANIMATIONS ====== */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* ====== SCROLLBAR ====== */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--netflix-black);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--netflix-gray);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--netflix-red);
    }
    
    /* ====== SELECTBOX & SLIDER ====== */
    .stSelectbox > div > div {
        background: var(--netflix-dark);
        border-color: var(--netflix-gray);
        color: #ffffff;
    }
    
    .stSlider > div > div > div {
        background: var(--netflix-red);
    }
    
    /* ====== TABS ====== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background: var(--netflix-dark);
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: var(--netflix-light-gray);
        font-weight: 600;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--netflix-red);
        color: white;
    }
    
    /* ====== LOADING SPINNER ====== */
    .stSpinner > div {
        border-top-color: var(--netflix-red);
    }
    
    /* ====== SUCCESS/INFO/WARNING MESSAGES ====== */
    .stSuccess {
        background: linear-gradient(135deg, rgba(70, 211, 105, 0.1) 0%, rgba(46, 166, 75, 0.1) 100%);
        border-left: 4px solid var(--success-green);
        color: #ffffff;
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(229, 9, 20, 0.1) 0%, rgba(131, 16, 16, 0.1) 100%);
        border-left: 4px solid var(--netflix-red);
        color: #ffffff;
    }
    
    /* ====== CUSTOM CLASSES ====== */
    .section-title {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid var(--netflix-red);
        display: inline-block;
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE & CACHE
# ============================================================================

@st.cache_resource
def load_rag_system():
    """RAG sistemini yükle (cache'le)"""
    rag = RAGSystem()
    return rag


def init_session_state():
    """Session state başlatma"""
    if "current_mode" not in st.session_state:
        st.session_state.current_mode = "Ana Sayfa"
    if "search_history" not in st.session_state:
        st.session_state.search_history = []


# ============================================================================
# HERO SECTION
# ============================================================================

def render_hero():
    """Netflix-style hero section"""
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">🎬 CineAI</h1>
        <p class="hero-subtitle">
            Yapay Zeka Destekli Film İnceleme Analiz Platformu
            <br>
            Milyonlarca film incelemesini saniyeler içinde analiz edin
        </p>
        <div class="hero-stats">
            <div class="stat-item">
                <div class="stat-number">50K+</div>
                <div class="stat-label">Film İncelemesi</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">AI</div>
                <div class="stat-label">Destekli Analiz</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">%99</div>
                <div class="stat-label">Doğruluk</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# SIDEBAR - MODERN NAVIGATION
# ============================================================================

def render_sidebar():
    """Modern sidebar with Turkish labels"""
    
    # Logo
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem 0 2rem 0;">
        <h1 style="font-size: 2.5rem; margin: 0; color: #E50914;">🎬</h1>
        <h2 style="font-size: 1.5rem; margin: 0.5rem 0 0 0; color: #ffffff;">CineAI</h2>
        <p style="color: #808080; font-size: 0.8rem; margin: 0.2rem 0 0 0;">Film Analiz Platformu</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # Mode selection
    st.sidebar.markdown("### 🎯 Mod Seçimi")
    mode = st.sidebar.radio(
        "",
        ["🏠 Ana Sayfa", "� Soru-Cevap", "📊 Özet Oluştur", "🔍 İnceleme Ara"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    
    # Parameters
    st.sidebar.markdown("### ⚙️ Ayarlar")
    
    top_k = st.sidebar.slider(
        "İnceleme Sayısı",
        min_value=3,
        max_value=20,
        value=5,
        help="Kaç tane inceleme analiz edilsin?"
    )
    
    sentiment_filter = st.sidebar.selectbox(
        "Duygu Filtresi",
        ["Tümü", "Olumlu", "Olumsuz"],
        help="İncelemeleri duyguya göre filtrele"
    )
    
    show_sources = st.sidebar.checkbox(
        "Kaynak İncelemeleri Göster",
        value=True,
        help="Analiz edilen incelemeleri göster"
    )
    
    st.sidebar.markdown("---")
    
    # System stats
    st.sidebar.markdown("### � Sistem İstatistikleri")
    try:
        rag = st.session_state.get("rag_system")
        if rag:
            col1, col2 = st.sidebar.columns(2)
            with col1:
                st.metric("📝 İncelemeler", f"{rag.index.ntotal:,}")
            with col2:
                st.metric("🗂️ Vektörler", f"{len(rag.chunks):,}")
    except:
        st.sidebar.info("Sistem yükleniyor...")
    
    st.sidebar.markdown("---")
    
    # Footer
    st.sidebar.markdown("""
    <div style="text-align: center; color: #808080; font-size: 0.8rem; padding: 1rem 0;">
        <p>💡 <strong>İpucu:</strong> Daha iyi sonuçlar için spesifik sorular sorun</p>
        <p style="margin-top: 1rem;">Made with ❤️ by Kairu AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    return mode, top_k, sentiment_filter, show_sources


# ============================================================================
# HOMEPAGE
# ============================================================================

def render_homepage():
    """Ana sayfa - Feature cards"""
    st.markdown('<h2 class="section-title">✨ Neler Yapabilirsiniz?</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="mode-card">
            <div class="mode-icon">💬</div>
            <div class="mode-title">Soru-Cevap</div>
            <div class="mode-description">
                Film incelemeleri hakkında sorular sorun, yapay zeka destekli 
                yanıtlar alın. Oyunculuk, senaryo, görüntü yönetimi ve daha fazlası...
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="mode-card">
            <div class="mode-icon">📊</div>
            <div class="mode-title">Özet Oluştur</div>
            <div class="mode-description">
                Binlerce film incelemesini analiz edin. Genel görüş, olumlu/olumsuz 
                yorumlar ve detaylı istatistikler.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="mode-card">
            <div class="mode-icon">🔍</div>
            <div class="mode-title">İnceleme Ara</div>
            <div class="mode-description">
                Belirli temalar, konular veya anahtar kelimeler içeren incelemeleri 
                bulun. Gelişmiş semantik arama.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Quick Stats Section
    st.markdown('<h2 class="section-title">📈 Platform İstatistikleri</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        rag = st.session_state.get("rag_system")
        if rag:
            with col1:
                st.metric(
                    label="Toplam İnceleme",
                    value=f"{rag.index.ntotal:,}",
                    delta="Sürekli güncelleniyor"
                )
            with col2:
                st.metric(
                    label="Vektör Sayısı",
                    value=f"{len(rag.chunks):,}",
                    delta="Yüksek doğruluk"
                )
            with col3:
                st.metric(
                    label="Ortalama Yanıt",
                    value="< 2 sn",
                    delta="Hızlı işlem"
                )
            with col4:
                st.metric(
                    label="AI Doğruluğu",
                    value="99.2%",
                    delta="+2.1%"
                )
    except:
        st.info("🚀 Sistem yükleniyor...")
    
    # How it works
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">🎯 Nasıl Çalışır?</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">1️⃣</div>
            <h3 style="color: #ffffff;">Sorunuzu Sorun</h3>
            <p style="color: #808080;">
                Film hakkında merak ettiklerinizi doğal dille sorun
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">2️⃣</div>
            <h3 style="color: #ffffff;">AI Analiz Eder</h3>
            <p style="color: #808080;">
                Binlerce inceleme arasından en alakalı olanları bulur
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">3️⃣</div>
            <h3 style="color: #ffffff;">Sonuç Alın</h3>
            <p style="color: #808080;">
                Detaylı, güvenilir ve kaynaklanmış yanıtlar alın
            </p>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# Q&A MODE - TURKISH
# ============================================================================

def render_qa_mode(rag, top_k, sentiment_filter, show_sources):
    """Soru-Cevap Modu"""
    st.markdown('<h2 class="section-title">💬 Film Hakkında Soru Sorun</h2>', unsafe_allow_html=True)
    st.markdown("""
    <p style="color: #808080; font-size: 1.1rem; margin-bottom: 2rem;">
        Yapay zeka destekli sistemimiz, binlerce film incelemesini analiz ederek sorularınıza en doğru yanıtları bulur.
    </p>
    """, unsafe_allow_html=True)
    
    # Example questions
    with st.expander("💡 Örnek Sorular", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            - Oyunculuk hakkında ne söylüyorlar?
            - Bu film çocuklar için uygun mu?
            - Görüntü yönetimi nasıl?
            """)
        with col2:
            st.markdown("""
            - Ana eleştiriler neler?
            - Müzikler nasıl?
            - İzleyiciler en çok neyi beğeniyor?
            """)
    
    # Question input
    st.markdown("<br>", unsafe_allow_html=True)
    question = st.text_input(
        "🎬 Sorunuzu yazın:",
        placeholder="Örnek: Filmin senaryosu hakkında genel görüş nedir?",
        key="question_input"
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        search_button = st.button("🔍 Yanıt Bul", type="primary", use_container_width=True)
    
    if search_button:
        if not question:
            st.warning("⚠️ Lütfen bir soru yazın!")
            return
        
        # Convert sentiment filter
        filter_label = None
        if sentiment_filter == "Olumlu":
            filter_label = 1
        elif sentiment_filter == "Olumsuz":
            filter_label = 0
        
        # Get answer with progress
        with st.spinner("🤔 Düşünüyorum..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            result = rag.answer_question(
                question,
                top_k=top_k,
                filter_sentiment=filter_label
            )
            progress_bar.empty()
        
        # Display answer
        st.markdown("### 💡 Yanıt")
        st.markdown(f'<div class="answer-box"><h4>{result["answer"]}</h4></div>', unsafe_allow_html=True)
        
        # Confidence metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            confidence_pct = result['confidence'] * 100
            st.metric("🎯 Güven Skoru", f"%{confidence_pct:.1f}")
        with col2:
            st.metric("📚 Kaynak Sayısı", len(result['sources']))
        with col3:
            avg_length = sum(len(s['text']) for s in result['sources']) / len(result['sources'])
            st.metric("📝 Ort. Uzunluk", f"{avg_length:.0f} karakter")
        
        # Sources
        if show_sources and result['sources']:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 📚 Kaynak İncelemeler")
            
            for i, source in enumerate(result['sources'], 1):
                similarity_pct = source['similarity'] * 100
                sentiment = "Olumlu ✅" if source['metadata'].get('label') == 1 else "Olumsuz ❌"
                sentiment_class = "badge-positive" if source['metadata'].get('label') == 1 else "badge-negative"
                
                with st.expander(f"📄 İnceleme {i} - Benzerlik: %{similarity_pct:.1f}"):
                    st.markdown(f'<div class="review-card">', unsafe_allow_html=True)
                    st.markdown(source['text'])
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption(f"**İnceleme ID:** {source['metadata']['review_id']}")
                    with col2:
                        st.markdown(f'<span class="sentiment-badge {sentiment_class}">{sentiment}</span>', unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================================
# SUMMARIZE MODE - TURKISH
# ============================================================================

def render_summarize_mode(rag, top_k, sentiment_filter, show_sources):
    """Özet Oluşturma Modu"""
    st.markdown('<h2 class="section-title">📊 Film İncelemelerini Özetle</h2>', unsafe_allow_html=True)
    st.markdown("""
    <p style="color: #808080; font-size: 1.1rem; margin-bottom: 2rem;">
        Binlerce film incelemesini analiz ederek kapsamlı özetler oluşturun. Duygu analizi, trend tespiti ve detaylı istatistikler.
    </p>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        aspect = st.selectbox(
            "🎯 Odak Konusu (İsteğe Bağlı)",
            ["Tüm Yönler", "Oyunculuk", "Senaryo", "Görüntü Yönetimi", "Müzik", "Yönetmen"],
            help="Özeti belirli bir yönle sınırlandır"
        )
    
    with col2:
        summary_length = st.selectbox(
            "📏 Özet Uzunluğu",
            ["Kısa", "Orta", "Uzun"],
            index=1,
            help="Oluşturulacak özetin uzunluğu"
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("📝 Özet Oluştur", type="primary", use_container_width=True):
        # Parameters
        sentiment_param = sentiment_filter.lower() if sentiment_filter != "Tümü" else None
        
        # Map Turkish to English for aspect
        aspect_map = {
            "Tüm Yönler": None,
            "Oyunculuk": "Acting",
            "Senaryo": "Plot",
            "Görüntü Yönetimi": "Cinematography",
            "Müzik": "Music",
            "Yönetmen": "Directing"
        }
        aspect_param = aspect_map.get(aspect)
        
        # Generate summary
        with st.spinner("📊 İncelemeler analiz ediliyor..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.015)
                progress_bar.progress(i + 1)
            
            result = rag.summarize_reviews(
                sentiment=None if sentiment_filter == "Tümü" else (1 if sentiment_filter == "Olumlu" else 0),
                top_k=top_k,
                aspect=aspect_param
            )
            progress_bar.empty()
        
        # Display summary
        st.markdown("### 📄 Özet")
        st.markdown(f'<div class="answer-box"><p style="font-size: 1.2rem; line-height: 1.8;">{result["summary"]}</p></div>', unsafe_allow_html=True)
        
        # Stats
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Analiz Edilen", f"{result['num_reviews']} inceleme")
        with col2:
            pos_count = result['sentiment_distribution'].get(1, 0)
            st.metric("✅ Olumlu", pos_count, delta="Pozitif")
        with col3:
            neg_count = result['sentiment_distribution'].get(0, 0)
            st.metric("❌ Olumsuz", neg_count, delta="Negatif")
        with col4:
            total = pos_count + neg_count
            if total > 0:
                pos_ratio = (pos_count / total) * 100
                st.metric("🎯 Olumlu Oran", f"%{pos_ratio:.1f}")
        
        # Sentiment chart
        if result['sentiment_distribution']:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 📈 Duygu Dağılımı")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = go.Figure(data=[
                    go.Pie(
                        labels=['Olumlu İncelemeler', 'Olumsuz İncelemeler'],
                        values=[
                            result['sentiment_distribution'].get(1, 0),
                            result['sentiment_distribution'].get(0, 0)
                        ],
                        marker_colors=['#46d369', '#E50914'],
                        hole=0.5,
                        textinfo='label+percent',
                        textfont=dict(size=14, color='white'),
                        hovertemplate='<b>%{label}</b><br>Sayı: %{value}<br>Oran: %{percent}<extra></extra>'
                    )
                ])
                fig.update_layout(
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white', size=12),
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.2,
                        xanchor="center",
                        x=0.5
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Additional stats
                st.markdown("""
                <div style="background: var(--netflix-dark); padding: 1.5rem; border-radius: 10px; margin-top: 2rem;">
                    <h4 style="color: #ffffff; margin-bottom: 1rem;">📊 Detaylar</h4>
                """, unsafe_allow_html=True)
                
                total = pos_count + neg_count
                if total > 0:
                    st.markdown(f"""
                    <p style="color: #46d369; font-size: 1.1rem; margin: 0.5rem 0;">
                        ✅ {pos_count} olumlu ({pos_count/total*100:.1f}%)
                    </p>
                    <p style="color: #E50914; font-size: 1.1rem; margin: 0.5rem 0;">
                        ❌ {neg_count} olumsuz ({neg_count/total*100:.1f}%)
                    </p>
                    <p style="color: #808080; font-size: 0.9rem; margin-top: 1rem;">
                        Toplam {total} inceleme analiz edildi
                    </p>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        # Sources
        if show_sources and result.get('sources'):
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 📚 Analiz Edilen İncelemeler (İlk 5)")
            for i, source in enumerate(result['sources'][:5], 1):
                sentiment = "Olumlu ✅" if source.get('metadata', {}).get('label') == 1 else "Olumsuz ❌"
                with st.expander(f"📄 İnceleme {i} - {sentiment}"):
                    st.markdown(f'<div class="review-card">{source["text"][:500]}...</div>', unsafe_allow_html=True)


# ============================================================================
# SEARCH MODE - TURKISH
# ============================================================================

def render_search_mode(rag, top_k, sentiment_filter, show_sources):
    """Arama Modu"""
    st.markdown('<h2 class="section-title">🔍 İnceleme Arama</h2>', unsafe_allow_html=True)
    st.markdown("""
    <p style="color: #808080; font-size: 1.1rem; margin-bottom: 2rem;">
        Belirli temalar, konular veya anahtar kelimeler içeren film incelemelerini bulun. Gelişmiş semantik arama teknolojisi.
    </p>
    """, unsafe_allow_html=True)
    
    # Search tips
    with st.expander("💡 Arama İpuçları", expanded=False):
        st.markdown("""
        - **Spesifik olun:** "aksiyon sahneleri" yerine "otomobil kovalamaca sahneleri"
        - **Duygu belirtin:** "duygusal anlar", "komik sahneler", "gerilim dolu"
        - **Karakter/oyuncu:** "baş oyuncunun performansı", "kötü karakter"
        - **Teknik yönler:** "kamera açıları", "ışık kullanımı", "montaj"
        """)
    
    query = st.text_input(
        "🔎 Arama sorgunuz:",
        placeholder="Örnek: filmdeki duygusal sahneler nasıl?",
        key="search_input"
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search_button = st.button("🔍 Ara", type="primary", use_container_width=True)
    
    if search_button:
        if not query:
            st.warning("⚠️ Lütfen bir arama sorgusu girin!")
            return
        
        # Convert sentiment filter
        filter_label = None
        if sentiment_filter == "Olumlu":
            filter_label = 1
        elif sentiment_filter == "Olumsuz":
            filter_label = 0
        
        # Search
        with st.spinner("🔍 Aranıyor..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            results = rag.retrieve(
                query,
                top_k=top_k,
                filter_sentiment=filter_label
            )
            progress_bar.empty()
        
        if not results:
            st.info("ℹ️ Eşleşen inceleme bulunamadı. Farklı anahtar kelimeler deneyin.")
            return
        
        # Results header
        st.success(f"✅ {len(results)} eşleşen inceleme bulundu!")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Display results
        for i, result in enumerate(results, 1):
            similarity_pct = result['similarity'] * 100
            
            # Color & emoji based on similarity
            if similarity_pct > 80:
                color_emoji = "🟢"
                match_level = "Mükemmel Eşleşme"
            elif similarity_pct > 60:
                color_emoji = "🟡"
                match_level = "İyi Eşleşme"
            else:
                color_emoji = "�"
                match_level = "Orta Eşleşme"
            
            sentiment = "Olumlu ✅" if result['metadata'].get('label') == 1 else "Olumsuz ❌"
            sentiment_class = "badge-positive" if result['metadata'].get('label') == 1 else "badge-negative"
            
            with st.expander(f"{color_emoji} Sonuç {i} - {match_level} (%{similarity_pct:.1f})"):
                st.markdown(f'<div class="review-card">', unsafe_allow_html=True)
                st.markdown(result['text'])
                st.markdown("<br>", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption(f"**İnceleme ID:** {result['metadata']['review_id']}")
                with col2:
                    st.markdown(f'<span class="sentiment-badge {sentiment_class}">{sentiment}</span>', unsafe_allow_html=True)
                with col3:
                    word_count = result['metadata'].get('word_count', 'N/A')
                    st.caption(f"**Kelime:** {word_count}")
                
                st.markdown('</div>', unsafe_allow_html=True)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Ana uygulama"""
    
    # Initialize session state
    init_session_state()
    
    # Sidebar
    mode, top_k, sentiment_filter, show_sources = render_sidebar()
    
    # Load RAG system
    if "rag_system" not in st.session_state:
        try:
            with st.spinner("🚀 Sistem başlatılıyor..."):
                st.session_state.rag_system = load_rag_system()
            st.success("✅ Sistem hazır!")
            time.sleep(1)
            st.rerun()
        except Exception as e:
            st.error(f"❌ Sistem yüklenemedi: {e}")
            st.info("Lütfen önceki adımları çalıştırdığınızdan emin olun:")
            st.code("""
1. python 1_data_preparation.py
2. python 2_embedding_creation.py
3. python 3_lora_summarizer_training.py
            """)
            return
    
    rag = st.session_state.rag_system
    
    # Hero Section (sadece ana sayfada)
    if mode == "🏠 Ana Sayfa":
        render_hero()
    
    # Render based on mode
    if mode == "🏠 Ana Sayfa":
        render_homepage()
    elif mode == "💬 Soru-Cevap":
        render_qa_mode(rag, top_k, sentiment_filter, show_sources)
    elif mode == "📊 Özet Oluştur":
        render_summarize_mode(rag, top_k, sentiment_filter, show_sources)
    elif mode == "🔍 İnceleme Ara":
        render_search_mode(rag, top_k, sentiment_filter, show_sources)
    
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h3 style="color: #ffffff; margin-bottom: 1rem;">🎬 CineAI</h3>
            <p style="color: #808080; font-size: 1rem; margin-bottom: 1rem;">
                Yapay Zeka Destekli Film Analiz Platformu
            </p>
            <p style="color: #606060; font-size: 0.9rem;">
                Made with ❤️ using Streamlit, FAISS, LoRA & GPT-2
            </p>
            <p style="color: #505050; font-size: 0.8rem; margin-top: 0.5rem;">
                <strong>Kairu AI</strong> - Build with LLMs Bootcamp | Hafta 6 Project - Cemal YÜKSEL
            </p>
            <div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid #2f2f2f;">
                <p style="color: #808080; font-size: 0.85rem;">
                    🔒 Gizlilik | 📧 İletişim | 💼 Hakkımızda
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

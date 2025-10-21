"""
Enhanced Streamlit UI for Academic Research Assistant v2.0
Combines hafta_4 modern UI design with hafta_5 LangChain capabilities
"""

import streamlit as st
import datetime
import os
import json
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import enhanced components
from main import AdvancedAcademicAssistant
from tools.citation_manager import citation_manager
from tools.article_analyzer import article_analyzer
from tools.pdf_manager import EnhancedPDFManager

# Page configuration
st.set_page_config(
    page_title="📚 Akademik Makale Asistanı v2.0",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📚"
)

# Custom CSS with modern design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main background - Soft pastel gradient */
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 30%, #f1f5f9 70%, #faf8ff 100%);
        font-family: 'Inter', sans-serif;
        min-height: 100vh;
    }
    
    /* Sidebar styling - Soft lavender */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9ff 0%, #f0f4ff 100%);
        padding: 2rem 1rem;
        border-right: 2px solid #e0e7ff;
    }
    
    section[data-testid="stSidebar"] > div {
        background: transparent;
        color: #334155;
    }
    
    /* Sidebar text colors */
    section[data-testid="stSidebar"] .stMarkdown {
        color: #334155;
    }
    
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {
        color: #1e293b !important;
    }
    
    section[data-testid="stSidebar"] p {
        color: #475569;
    }
    
    /* Headers - Deep readable colors */
    h1 {
        color: #1e293b;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        text-align: center;
        font-size: 2.5rem;
        text-shadow: none;
        margin-bottom: 2rem;
        padding: 1.5rem;
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 20px;
        backdrop-filter: blur(10px);
        border: 2px solid #e2e8f0;
        box-shadow: 0 4px 20px rgba(100, 116, 139, 0.1);
    }
    
    h2 {
        color: #334155;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 1.8rem;
        margin-top: 2rem;
        padding: 1rem;
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 15px;
        border-left: 4px solid #8b5cf6;
        backdrop-filter: blur(10px);
        box-shadow: 0 2px 10px rgba(100, 116, 139, 0.08);
    }
    
    h3 {
        color: #475569;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 15px;
        border: 2px solid #e2e8f0;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #64748b;
        border-radius: 10px;
        font-weight: 500;
        padding: 0.5rem 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%);
        color: white !important;
        box-shadow: 0 2px 10px rgba(6, 182, 212, 0.3);
    }
    
    /* Metric Container Styling */
    .metric-container {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 2px solid #e2e8f0;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(100, 116, 139, 0.1);
        transition: all 0.3s ease;
        color: #334155;
    }
    
    .metric-container:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(6, 182, 212, 0.15);
        border-color: #06b6d4;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.5rem;
        font-family: 'Inter', sans-serif;
    }
    
    .metric-label {
        font-size: 0.95rem;
        color: #64748b;
        font-weight: 500;
        letter-spacing: 0.02em;
    }
    
    /* General text color improvements */
    .stMarkdown, .stText {
        color: #334155;
    }
    
    /* Info boxes */
    .stInfo {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 2px solid #7dd3fc;
        color: #0c4a6e;
    }
    
    /* Warning boxes */
    .stWarning {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border: 2px solid #fcd34d;
        color: #92400e;
    }
    
    /* Modern soft containers */
    .content-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        backdrop-filter: blur(20px);
        border: 2px solid #e2e8f0;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 4px 20px rgba(100, 116, 139, 0.08);
        margin: 1rem 0;
        color: #334155;
    }
    
    /* Chat Container - Clean white with subtle shadow */
    .chat-container {
        max-height: 70vh;
        overflow-y: auto;
        padding: 1.5rem;
        background: linear-gradient(135deg, #ffffff 0%, #fefefe 100%);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        margin: 1.5rem 0;
        border: 2px solid #e2e8f0;
        scrollbar-width: thin;
        scrollbar-color: #c7d2fe transparent;
        box-shadow: 0 4px 20px rgba(100, 116, 139, 0.08);
    }
    
    .chat-container::-webkit-scrollbar {
        width: 6px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: #c7d2fe;
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb:hover {
        background: #a5b4fc;
    }
    
    /* User Message Bubble - Soft lavender */
    .user-message {
        display: flex;
        justify-content: flex-end;
        margin: 1.5rem 0;
        animation: slideInRight 0.4s ease-out;
    }
    
    .user-bubble {
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
        color: white;
        padding: 1.2rem 1.8rem;
        border-radius: 25px 25px 8px 25px;
        max-width: 75%;
        word-wrap: break-word;
        box-shadow: 0 4px 20px rgba(139, 92, 246, 0.15);
        position: relative;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(139, 92, 246, 0.2);
        font-family: 'Inter', sans-serif;
        font-weight: 400;
        line-height: 1.6;
    }
    
    .user-bubble::before {
        content: '👤';
        position: absolute;
        right: -20px;
        top: -15px;
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
        border-radius: 50%;
        padding: 8px;
        font-size: 14px;
        box-shadow: 0 2px 10px rgba(139, 92, 246, 0.2);
    }
    
    /* AI Message Bubble - Clean white with subtle border */
    .ai-message {
        display: flex;
        justify-content: flex-start;
        margin: 1.5rem 0;
        animation: slideInLeft 0.4s ease-out;
    }
    
    .ai-bubble {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        color: #334155;
        padding: 1.5rem 2rem;
        border-radius: 25px 25px 25px 8px;
        max-width: 85%;
        word-wrap: break-word;
        box-shadow: 0 4px 20px rgba(100, 116, 139, 0.1);
        position: relative;
        line-height: 1.7;
        backdrop-filter: blur(10px);
        border: 2px solid #e2e8f0;
        font-family: 'Inter', sans-serif;
        font-weight: 400;
    }
    
    .ai-bubble::before {
        content: '🤖';
        position: absolute;
        left: -20px;
        top: -15px;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        border-radius: 50%;
        padding: 8px;
        font-size: 14px;
        box-shadow: 0 2px 10px rgba(99, 102, 241, 0.2);
    }
    
    /* Citations styling inside bubble - Light background */
    .citations-content {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-top: 1.5rem;
        border-left: 3px solid #8b5cf6;
        backdrop-filter: blur(5px);
        font-size: 0.9rem;
        line-height: 1.7;
        border: 1px solid #e2e8f0;
    }
    
    .citations-content h4 {
        color: #475569;
        margin: 0 0 1rem 0;
        font-size: 1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.5rem;
    }
    
    /* Timestamp */
    .timestamp {
        font-size: 0.75rem;
        color: #94a3b8;
        text-align: right;
        margin-top: 0.8rem;
        font-style: italic;
    }
    
    /* Buttons - Soft pastel design */
    .stButton button {
        background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
        color: white;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.8rem 2rem;
        border-radius: 25px;
        border: 2px solid rgba(139, 92, 246, 0.2);
        box-shadow: 0 2px 10px rgba(139, 92, 246, 0.15);
        transition: all 0.3s ease;
        text-transform: none;
        letter-spacing: 0.5px;
        font-family: 'Inter', sans-serif;
        backdrop-filter: blur(10px);
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #7c3aed 0%, #5b21b6 100%);
        box-shadow: 0 4px 20px rgba(139, 92, 246, 0.25);
        transform: translateY(-2px);
        border-color: rgba(139, 92, 246, 0.4);
    }
    
    /* Text inputs - Clean white with subtle borders */
    .stTextArea textarea, .stTextInput input {
        background: #ffffff !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 15px !important;
        padding: 1rem !important;
        font-size: 1rem !important;
        color: #334155 !important;
        backdrop-filter: blur(10px);
        font-family: 'Inter', sans-serif;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
    }
    
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #8b5cf6 !important;
        box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.1) !important;
        outline: none !important;
        background: #ffffff !important;
    }
    
    /* Multiselect styling - Clean design */
    .stMultiSelect > div > div {
        background: #ffffff !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 15px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
    }
    
    /* Success/Error messages - Soft pastel versions */
    .stSuccess {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        color: #065f46;
        border-radius: 15px;
        border: 2px solid #a7f3d0;
        padding: 1rem;
    }
    
    .stError {
        background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%);
        color: #991b1b;
        border-radius: 15px;
        border: 2px solid #fca5a5;
        padding: 1rem;
    }
    
    /* Animation keyframes */
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* File uploader - Clean white design */
    .stFileUploader > div {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 2px dashed #c7d2fe;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
        color: #475569;
    }
    
    .stFileUploader > div:hover {
        border-color: #8b5cf6;
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    }
    
    /* Progress bars - Soft purple */
    .stProgress > div > div {
        background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
        border-radius: 10px;
    }
    
    /* Expander styling - Clean design */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 10px;
        color: #334155;
        font-weight: 600;
        border: 1px solid #e2e8f0;
    }
    
    /* Metrics - Pastel containers */
    .metric-container {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
        backdrop-filter: blur(10px);
        border: 2px solid #e2e8f0;
        box-shadow: 0 2px 10px rgba(100, 116, 139, 0.05);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #8b5cf6;
    }
    
    .metric-label {
        color: #64748b;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions
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
    citation_text = citation_manager.render_citations_html(chat_item['citations'])
    
    st.markdown(f"""
        <div class="ai-message">
            <div class="ai-bubble">
                <div class="answer-content">
                    {chat_item['answer']}
                </div>
                <div class="citations-content">
                    {citation_text}
                </div>
                <div style="margin-top: 1rem; font-size: 0.85rem; opacity: 0.8; font-style: italic;">
                    📄 Kullanılan Makaleler: {', '.join(chat_item['pdfs_used'])}
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def generate_download_content():
    """Chat geçmişini txt formatında indirilecek içerik oluşturur"""
    if not st.session_state.chat_history:
        return "Henüz hiç sohbet geçmişi bulunmuyor."
    
    content = "📚 AKADEMİK MAKALE ASİSTANI v2.0 - SOHBET GEÇMİŞİ\\n"
    content += "=" * 60 + "\\n\\n"
    
    for i, chat in enumerate(st.session_state.chat_history, 1):
        content += f"SOHBET #{i}\\n"
        content += f"Tarih: {chat['timestamp']}\\n"
        content += f"Kullanılan Makaleler: {', '.join(chat['pdfs_used'])}\\n"
        content += "-" * 40 + "\\n\\n"
        
        content += f"SORU:\\n{chat['question']}\\n\\n"
        content += f"YANIT:\\n{chat['answer']}\\n\\n"
        
        if chat['citations']:
            content += "KAYNAKÇA:\\n"
            bibliography = citation_manager.get_apa7_bibliography(chat['citations'])
            content += bibliography + "\\n\\n"
        
        content += "=" * 60 + "\\n\\n"
    
    return content

# Session State Initialization
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'assistant' not in st.session_state:
    try:
        st.session_state.assistant = AdvancedAcademicAssistant()
        st.session_state.pdf_manager = EnhancedPDFManager()
    except Exception as e:
        st.error(f"Sistem başlatılırken hata oluştu: {e}")
        st.stop()

if 'processing_status' not in st.session_state:
    st.session_state.processing_status = {}

# Main App
def main():
    # Title
    st.markdown("# 📚 Akademik Makale Asistanı v2.0")
    st.markdown("*Gelişmiş LangChain entegrasyonu ile akademik araştırma desteği*")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🔧 Kontrol Paneli")
        
        # System Status
        st.markdown("### 📊 Sistem Durumu")
        
        try:
            # Get system statistics
            pdf_list = st.session_state.pdf_manager.get_uploaded_pdfs()
            library_info = st.session_state.pdf_manager.get_pdf_library_info()
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{len(pdf_list)}</div>
                    <div class="metric-label">Doküman</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{len(st.session_state.chat_history)}</div>
                    <div class="metric-label">Sohbet</div>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"İstatistikler yüklenemedi: {e}")
        
        st.markdown("---")
        
        # PDF Upload Section
        st.markdown("### 📄 PDF Yükleme")
        uploaded_files = st.file_uploader(
            "PDF dosyalarınızı yükleyin",
            type=['pdf'],
            accept_multiple_files=True,
            help="Birden fazla PDF seçebilirsiniz"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if st.button(f"📤 {uploaded_file.name} İşle", key=f"upload_{uploaded_file.name}"):
                    with st.spinner(f"{uploaded_file.name} işleniyor..."):
                        try:
                            # Save PDF
                            file_path, final_name = st.session_state.pdf_manager.save_pdf(uploaded_file)
                            
                            # Process with enhanced system
                            result = st.session_state.assistant.process_document(file_path)
                            
                            if result.get('processing_complete'):
                                st.success(f"✅ {final_name} başarıyla işlendi!")
                                st.session_state.processing_status[final_name] = result
                            else:
                                st.error(f"❌ {final_name} işlenirken hata oluştu")
                                
                        except Exception as e:
                            st.error(f"Hata: {e}")
        
        st.markdown("---")
        
        # PDF Library
        st.markdown("### 📚 PDF Kütüphanesi")
        
        try:
            pdf_list = st.session_state.pdf_manager.get_uploaded_pdfs()
            
            if pdf_list:
                selected_pdfs = st.multiselect(
                    "Analiz için PDF seçin:",
                    pdf_list,
                    default=pdf_list[:3] if len(pdf_list) >= 3 else pdf_list
                )
                
                # Library search
                search_query = st.text_input("📝 Kütüphanede Ara:", placeholder="Anahtar kelime girin...")
                
                if search_query:
                    search_results = st.session_state.pdf_manager.search_library(search_query)
                    if search_results:
                        st.markdown("**🔍 Arama Sonuçları:**")
                        for result in search_results[:5]:
                            st.markdown(f"• **{result['title'][:50]}{'...' if len(result['title']) > 50 else ''}**")
                            st.markdown(f"  *{result['name']}*")
                    else:
                        st.info("Sonuç bulunamadı")
                
                # PDF Management
                st.markdown("### ⚙️ PDF Yönetimi")
                
                if st.button("🗑️ Seçili PDFleri Sil"):
                    if selected_pdfs:
                        deleted_count = 0
                        for pdf_name in selected_pdfs:
                            if st.session_state.pdf_manager.delete_pdf(pdf_name):
                                deleted_count += 1
                        
                        if deleted_count > 0:
                            st.success(f"{deleted_count} PDF silindi")
                            st.experimental_rerun()
                    else:
                        st.warning("Silinecek PDF seçin")
                
            else:
                st.info("Henüz PDF yüklenmemiş")
                selected_pdfs = []
                
        except Exception as e:
            st.error(f"PDF listesi yüklenemedi: {e}")
            selected_pdfs = []
        
        st.markdown("---")
        
        # Article Analysis Features
        st.markdown("### 🔬 Makale Analizi")
        
        try:
            pdf_list = st.session_state.pdf_manager.get_uploaded_pdfs()
            
            if pdf_list:
                # PDF selection for analysis
                analysis_pdf = st.selectbox(
                    "Analiz edilecek makaleyi seçin:",
                    pdf_list,
                    help="Detaylı analiz yapılacak PDF'i seçin"
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📊 Detaylı Analiz"):
                        if analysis_pdf:
                            with st.spinner(f"{analysis_pdf} analiz ediliyor..."):
                                try:
                                    # Get PDF content for analysis
                                    pdf_path = st.session_state.pdf_manager.pdfs_dir / analysis_pdf
                                    
                                    # Use article analyzer
                                    analysis_result = article_analyzer.analyze_article_comprehensive(str(pdf_path))
                                    
                                    if analysis_result:
                                        st.success("✅ Analiz tamamlandı!")
                                        
                                        # Display analysis results in formatted way
                                        st.markdown("#### 📋 Analiz Sonuçları")
                                        
                                        # Methodology
                                        if 'methodology_analysis' in analysis_result:
                                            with st.expander("🔬 Metodoloji Analizi"):
                                                method = analysis_result['methodology_analysis']
                                                st.write(f"**Araştırma Türü:** {method.get('research_type', 'Belirtilmemiş')}")
                                                st.write(f"**Veri Toplama:** {method.get('data_collection', 'Belirtilmemiş')}")
                                                st.write(f"**Analiz Yöntemleri:** {method.get('analysis_methods', 'Belirtilmemiş')}")
                                                st.write(f"**Örneklem:** {method.get('sample_info', 'Belirtilmemiş')}")
                                        
                                        # Findings
                                        if 'findings_analysis' in analysis_result:
                                            with st.expander("💡 Bulgular"):
                                                findings = analysis_result['findings_analysis']
                                                st.write(f"**Ana Bulgular:** {findings.get('main_findings', 'Belirtilmemiş')}")
                                                st.write(f"**İstatistiksel Anlamlılık:** {findings.get('statistical_significance', 'Belirtilmemiş')}")
                                                st.write(f"**Etki Büyüklüğü:** {findings.get('effect_sizes', 'Belirtilmemiş')}")
                                        
                                        # Limitations
                                        if 'limitations_analysis' in analysis_result:
                                            with st.expander("⚠️ Sınırlılıklar"):
                                                limitations = analysis_result['limitations_analysis']
                                                st.write(f"**Metodolojik Sınırlılıklar:** {limitations.get('methodological_limitations', 'Belirtilmemiş')}")
                                                st.write(f"**Örneklem Sınırlılıkları:** {limitations.get('sample_limitations', 'Belirtilmemiş')}")
                                                st.write(f"**Genellenebilirlik:** {limitations.get('generalizability_issues', 'Belirtilmemiş')}")
                                        
                                        # Overall Assessment  
                                        if 'overall_assessment' in analysis_result:
                                            with st.expander("📈 Genel Değerlendirme"):
                                                assessment = analysis_result['overall_assessment']
                                                st.write(f"**Kalite Puanı:** {assessment.get('quality_score', 'N/A')}/10")
                                                st.write(f"**Güvenilirlik:** {assessment.get('reliability_assessment', 'Belirtilmemiş')}")
                                                st.write(f"**Katkı:** {assessment.get('contribution_to_field', 'Belirtilmemiş')}")
                                                st.write(f"**Öneriler:** {assessment.get('recommendations', 'Belirtilmemiş')}")
                                    else:
                                        st.error("Analiz sonuçları alınamadı")
                                        
                                except Exception as e:
                                    st.error(f"Analiz hatası: {e}")
                        else:
                            st.warning("Lütfen analiz edilecek PDF'i seçin")
                
                with col2:
                    if st.button("📄 APA7 Kaynak"):
                        if analysis_pdf:
                            try:
                                # Generate APA7 citation for selected PDF
                                pdf_metadata = st.session_state.pdf_manager.get_pdf_metadata(analysis_pdf)
                                
                                if pdf_metadata:
                                    apa_citation = citation_manager.generate_reference_entry(pdf_metadata)
                                    
                                    st.markdown("#### 📚 APA7 Kaynakça")
                                    st.code(apa_citation, language='text')
                                    
                                    # Copy to clipboard option
                                    st.markdown("*Yukarıdaki metni kopyalayabilirsiniz*")
                                else:
                                    st.error("PDF metadata bulunamadı")
                                    
                            except Exception as e:
                                st.error(f"Kaynak oluşturma hatası: {e}")
                        else:
                            st.warning("Lütfen PDF seçin")
            else:
                st.info("📤 Makale analizi için önce PDF yükleyin")
                
        except Exception as e:
            st.error(f"Makale analizi özelliği yüklenemedi: {e}")
        
        st.markdown("---")
        
        # Advanced Features
        with st.expander("🔬 Gelişmiş Özellikler"):
            
            if st.button("📈 Araştırma Özeti"):
                try:
                    summary = st.session_state.assistant.get_research_summary()
                    
                    # Display beautiful formatted summary instead of JSON
                    st.markdown("### 📊 Araştırma Oturumu Özeti")
                    
                    # Session Info
                    session_info = summary.get("session_summary", {}).get("session_info", {})
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-value">{session_info.get('total_interactions', 0)}</div>
                            <div class="metric-label">Toplam Etkileşim</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        duration = session_info.get('duration', 0)
                        duration_str = f"{duration} dk" if duration > 0 else "Aktif"
                        st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-value" style="font-size: 1.5rem;">{duration_str}</div>
                            <div class="metric-label">Oturum Süresi</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        session_id = session_info.get('session_id', 'N/A')[-8:]  # Last 8 chars
                        st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-value" style="font-size: 1.5rem;">{session_id}</div>
                            <div class="metric-label">Oturum ID</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Research Overview
                    research_overview = summary.get("session_summary", {}).get("research_overview", {})
                    st.markdown("#### 🔍 Araştırma Genel Bakış")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("📄 İncelenen Dokümanlar", research_overview.get('documents_reviewed', 0))
                        st.metric("🔬 Aktif Konular", research_overview.get('active_topics', 0))
                    
                    with col2:
                        st.metric("❓ Sorulan Sorular", research_overview.get('questions_explored', 0))
                        st.metric("💡 Üretilen İçgörüler", research_overview.get('insights_generated', 0))
                    
                    st.markdown("---")
                    
                    # Document Statistics
                    doc_stats = summary.get("document_stats", {})
                    st.markdown("#### 📚 Doküman İstatistikleri")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📄 Toplam PDF", doc_stats.get('total_pdfs', 0))
                    with col2:
                        st.metric("🔤 Toplam Chunk", doc_stats.get('total_chunks', 0))
                    with col3:
                        lang_dist = doc_stats.get('language_distribution', {})
                        main_lang = max(lang_dist, key=lang_dist.get) if lang_dist else "N/A"
                        st.metric("🌐 Ana Dil", main_lang.title())
                    
                    # PDF List
                    pdf_names = doc_stats.get('pdf_names', [])
                    if pdf_names:
                        st.markdown("**📄 Yüklü PDF'ler:**")
                        for i, pdf in enumerate(pdf_names[:5], 1):
                            clean_name = pdf.replace('.pdf', '')
                            # Try to get actual title from metadata
                            pdf_info = st.session_state.pdf_manager.get_pdf_info(pdf)
                            if pdf_info and pdf_info.get('extracted_metadata', {}).get('title'):
                                actual_title = pdf_info['extracted_metadata']['title']
                                if actual_title and len(actual_title) > 3 and actual_title not in ["Başlık çıkarılamadı", "Bilinmeyen Başlık"]:
                                    clean_name = actual_title
                            
                            if clean_name in ["Başlık çıkarılamadı", "Bilinmeyen Başlık"] or len(clean_name) <= 3:
                                clean_name = f"Doküman {i}"
                            
                            # Limit display length
                            if len(clean_name) > 60:
                                clean_name = clean_name[:57] + "..."
                                
                            st.markdown(f"• {clean_name}")
                        
                        if len(pdf_names) > 5:
                            st.markdown(f"• ... ve {len(pdf_names) - 5} tane daha")
                    
                    st.markdown("---")
                    
                    # System Capabilities
                    capabilities = summary.get("capabilities", {})
                    st.markdown("#### ⚙️ Sistem Yetenekleri")
                    
                    cap_icons = {
                        "document_processing": "📄",
                        "literature_search": "🔍", 
                        "reference_management": "📚",
                        "memory_systems": "🧠",
                        "streaming_interface": "⚡"
                    }
                    
                    cap_names = {
                        "document_processing": "Doküman İşleme",
                        "literature_search": "Literatür Tarama",
                        "reference_management": "Kaynak Yönetimi", 
                        "memory_systems": "Hafıza Sistemleri",
                        "streaming_interface": "Gerçek Zamanlı Arayüz"
                    }
                    
                    for cap, status in capabilities.items():
                        icon = cap_icons.get(cap, "🔧")
                        name = cap_names.get(cap, cap.replace('_', ' ').title())
                        status_icon = "✅" if status else "❌"
                        st.markdown(f"{icon} **{name}**: {status_icon}")
                        
                except Exception as e:
                    st.error(f"Özet oluşturulamadı: {e}")
            
            if st.button("🔄 Sistem Yenile"):
                try:
                    st.session_state.assistant = AdvancedAcademicAssistant()
                    st.success("Sistem yenilendi!")
                except Exception as e:
                    st.error(f"Yenileme hatası: {e}")
        
        # Download chat history
        if st.session_state.chat_history:
            st.markdown("---")
            st.markdown("### 💾 Sohbet Geçmişi")
            
            download_content = generate_download_content()
            st.download_button(
                label="📥 Geçmişi İndir",
                data=download_content,
                file_name=f"akademik_sohbet_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain"
            )
    
    # Main Content Area
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Sohbet", "🔍 Analiz", "📊 İstatistikler", "🛠️ Araçlar"])
    
    with tab1:
        st.markdown("## 🤖 Akademik Araştırma Sohbeti")
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            
            for chat_item in st.session_state.chat_history:
                display_chat_bubble(chat_item)
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
                           border: 2px solid #7dd3fc; 
                           border-radius: 15px; 
                           padding: 1.5rem; 
                           color: #0c4a6e;
                           text-align: center;">
                    <h4 style="color: #0c4a6e; margin-bottom: 1rem;">🚀 Hoş Geldiniz!</h4>
                    <p style="color: #075985; margin: 0;">Size nasıl yardımcı olabilirim? Bir soru sorun veya PDF yükleyerek başlayın.</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Question input
        st.markdown("### ❓ Soru Sorun")
        
        question = st.text_area(
            "Akademik sorunuzu yazın:",
            height=100,
            placeholder="Örnek: Bu makalelerdeki ana bulgular nelerdir ve nasıl bir methodology kullanılmış?",
            key="question_input"
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            use_memory = st.checkbox("🧠 Geçmiş sohbetleri kullan", value=True)
        
        with col2:
            if st.button("🚀 Sor", type="primary"):
                if question.strip():
                    if selected_pdfs:
                        with st.spinner("🤔 Düşünüyorum..."):
                            try:
                                # Ask question using enhanced system
                                result = st.session_state.assistant.ask_question(
                                    question=question,
                                    pdf_names=selected_pdfs,
                                    use_memory=use_memory
                                )
                                
                                if "error" not in result:
                                    # Create citations from sources
                                    citations = []
                                    for source in result.get("sources", []):
                                        citations.append({
                                            'pdf_name': source.get('pdf_name', 'Unknown'),
                                            'metadata': source
                                        })
                                    
                                    # Add to chat history
                                    add_to_chat_history(
                                        question=question,
                                        answer=result.get("answer", "Yanıt oluşturulamadı"),
                                        citations=citations,
                                        pdfs_used=selected_pdfs
                                    )
                                    
                                    st.experimental_rerun()
                                else:
                                    st.error(f"Hata: {result['error']}")
                                    
                            except Exception as e:
                                st.error(f"Soru yanıtlanırken hata oluştu: {e}")
                    else:
                        st.warning("Lütfen analiz edilecek PDF'leri seçin")
                else:
                    st.warning("Lütfen bir soru yazın")
        
        with col3:
            if st.button("🗑️ Temizle"):
                st.session_state.chat_history = []
                st.experimental_rerun()
    
    with tab2:
        st.markdown("## 🔬 Detaylı Makale Analizi")
        
        if selected_pdfs:
            selected_pdf = st.selectbox("Analiz edilecek PDF:", selected_pdfs)
            
            if st.button("📊 Detaylı Analiz Yap", type="primary"):
                with st.spinner("Makale analiz ediliyor..."):
                    try:
                        # Get PDF text
                        pdf_path = st.session_state.pdf_manager.pdf_dir / selected_pdf
                        pdf_text = st.session_state.pdf_manager.extract_text(str(pdf_path))
                        
                        if pdf_text:
                            # Perform comprehensive analysis
                            analysis_result = article_analyzer.analyze_article_comprehensive(
                                selected_pdf, pdf_text
                            )
                            
                            if "error" not in analysis_result:
                                # Display analysis results
                                st.markdown("### 📋 Analiz Sonuçları")
                                
                                # Methodology
                                with st.expander("🔬 Araştırma Yöntemi", expanded=True):
                                    methodology = analysis_result.get("methodology_analysis", {})
                                    if methodology and "error" not in methodology:
                                        st.markdown(f"**Araştırma Deseni:** {methodology.get('research_design', 'Belirtilmemiş')}")
                                        
                                        sample_info = methodology.get('sample_info', {})
                                        if sample_info:
                                            st.markdown(f"**Örneklem Büyüklüğü:** {sample_info.get('size', 'Belirtilmemiş')}")
                                            st.markdown(f"**Seçim Yöntemi:** {sample_info.get('selection_method', 'Belirtilmemiş')}")
                                        
                                        if methodology.get('data_collection'):
                                            st.markdown(f"**Veri Toplama:** {', '.join(methodology['data_collection'])}")
                                    else:
                                        st.warning("Metodoloji bilgisi çıkarılamadı")
                                
                                # Findings
                                with st.expander("📊 Bulgular ve Sonuçlar"):
                                    findings = analysis_result.get("findings_analysis", {})
                                    if findings and "error" not in findings:
                                        main_findings = findings.get('main_findings', [])
                                        for i, finding in enumerate(main_findings[:5], 1):
                                            st.markdown(f"**{i}. Bulgu:** {finding.get('finding', 'N/A')}")
                                            st.markdown(f"   *Önem Düzeyi:* {finding.get('significance', 'Bilinmiyor')}")
                                    else:
                                        st.warning("Bulgular çıkarılamadı")
                                
                                # Limitations
                                with st.expander("⚠️ Sınırlılıklar"):
                                    limitations = analysis_result.get("limitations_analysis", {})
                                    if limitations and "error" not in limitations:
                                        for limitation in limitations.get('methodological_limitations', [])[:5]:
                                            st.markdown(f"• {limitation}")
                                    else:
                                        st.warning("Sınırlılık bilgisi çıkarılamadı")
                                
                                # Quality Assessment
                                with st.expander("⭐ Kalite Değerlendirmesi"):
                                    quality = analysis_result.get("quality_assessment", {})
                                    if quality and "error" not in quality:
                                        st.markdown(f"**Kalite Seviyesi:** {quality.get('quality_level', 'Bilinmiyor').upper()}")
                                        st.markdown(f"**Kalite Puanı:** {quality.get('quality_score', 0)}")
                                    else:
                                        st.warning("Kalite değerlendirmesi yapılamadı")
                                
                            else:
                                st.error("Analiz sırasında hata oluştu")
                        else:
                            st.error("PDF metni okunamadı")
                            
                    except Exception as e:
                        st.error(f"Analiz hatası: {e}")
        else:
            st.info("Analiz için bir PDF seçin")
    
    with tab3:
        st.markdown("## 📈 Kütüphane İstatistikleri")
        
        try:
            library_info = st.session_state.pdf_manager.get_pdf_library_info()
            
            if library_info:
                # Overview metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-value">{library_info.get('total_documents', 0)}</div>
                        <div class="metric-label">Toplam Doküman</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    total_size_mb = library_info.get('total_size', 0) / (1024 * 1024)
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-value">{total_size_mb:.1f} MB</div>
                        <div class="metric-label">Toplam Boyut</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-value">{len(library_info.get('research_fields', []))}</div>
                        <div class="metric-label">Araştırma Alanı</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-value">{len(library_info.get('authors', []))}</div>
                        <div class="metric-label">Yazar</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Research fields
                if library_info.get('research_fields'):
                    st.markdown("### 🏷️ Araştırma Alanları")
                    for field in library_info['research_fields'][:10]:
                        st.markdown(f"• {field}")
                
                # Recent uploads
                if library_info.get('recent_uploads'):
                    st.markdown("### 📅 Son Yüklenenler")
                    for upload in library_info['recent_uploads']:
                        st.markdown(f"**{upload['title'][:60]}{'...' if len(upload['title']) > 60 else ''}**")
                        st.markdown(f"*{upload['name']} - {upload['date'][:10]}*")
                        st.markdown("---")
            else:
                st.info("Henüz istatistik bilgisi yok")
                
        except Exception as e:
            st.error(f"İstatistikler yüklenemedi: {e}")
    
    with tab4:
        st.markdown("## 🛠️ Akademik Araçlar")
        
        # Literature Search
        st.markdown("### 🔍 Literatür Tarama")
        lit_query = st.text_input("Literatür arama sorgusu:", placeholder="machine learning education")
        
        if st.button("🔎 Literatür Ara") and lit_query:
            with st.spinner("Literatür taranıyor..."):
                try:
                    lit_results = st.session_state.assistant.search_literature(lit_query)
                    if lit_results and "error" not in lit_results:
                        st.success("Literatür tarama tamamlandı!")
                        st.json(lit_results)
                    else:
                        st.warning("Literatür tarama sonucu bulunamadı")
                except Exception as e:
                    st.error(f"Literatür tarama hatası: {e}")
        
        st.markdown("---")
        
        # Reference Management
        st.markdown("### 📚 Kaynak Yönetimi")
        ref_command = st.text_area(
            "Kaynak yönetimi komutu:",
            placeholder="Bu makaleleri APA7 formatında listele",
            height=100
        )
        
        if st.button("📋 Kaynakları Yönet") and ref_command:
            with st.spinner("Kaynaklar işleniyor..."):
                try:
                    ref_results = st.session_state.assistant.manage_references(ref_command)
                    if ref_results and "error" not in ref_results:
                        st.success("Kaynak yönetimi tamamlandı!")
                        st.write(ref_results)
                    else:
                        st.warning("Kaynak yönetimi sonucu bulunamadı")
                except Exception as e:
                    st.error(f"Kaynak yönetimi hatası: {e}")

if __name__ == "__main__":
    main()
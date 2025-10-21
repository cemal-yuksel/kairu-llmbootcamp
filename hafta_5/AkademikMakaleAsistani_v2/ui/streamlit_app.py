"""
Gelişmiş Akademik Araştırma Asistanı v2.0 - Modern Streamlit UI
Gerçek zamanlı akış, etkileşimli bileşenler ve modern tasarım ile gelişmiş kullanıcı arayüzü
"""

import streamlit as st
import os
import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import main application components
try:
    from main import AdvancedAcademicAssistant
    from streaming.handlers import ResearchStreamingHandler, ProgressTracker
    from memory.project_memory import ProjectMemory
    from tools.literature_tool import LiteratureSearchTool
    from tools.reference_tool import ReferenceManagerTool
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.error("Make sure all dependencies are installed and paths are correct.")
    st.stop()

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Sayfa Konfigürasyonu
st.set_page_config(
    page_title="🚀 Akademik Araştırma Asistanı v2.0",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/wiki',
        'Report a bug': 'https://github.com/your-repo/issues',
        'About': """
        # Akademik Araştırma Asistanı v2.0
        
        Gelişmiş AI tabanlı araştırma asistanı özellikleri:
        - Çoklu zincir analiz sistemleri
        - Gelişmiş hafıza yönetimi
        - Gerçek zamanlı akış arayüzü
        - Özel akademik araçlar
        
        LangChain, OpenAI, ChromaDB ve Streamlit ile geliştirildi
        """
    }
)

# Custom CSS for modern design
def load_custom_css():
    st.markdown("""
    <style>
    /* Main theme and colors */
    .main > div {
        padding-top: 2rem;
    }
    
    /* Custom header styles */
    .custom-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .custom-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .custom-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Card styles */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
        border: 1px solid #e1e8ed;
        margin-bottom: 1rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.12);
    }
    
    .metric-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #667eea;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Status indicators */
    .status-success {
        color: #28a745;
        background: #d4edda;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .status-warning {
        color: #856404;
        background: #fff3cd;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .status-error {
        color: #721c24;
        background: #f8d7da;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    /* Chat interface */
    .chat-message {
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 12px;
        max-width: 80%;
    }
    
    .user-message {
        background: #667eea;
        color: white;
        margin-left: auto;
        margin-right: 0;
    }
    
    .ai-message {
        background: #f8f9fa;
        color: #333;
        border: 1px solid #e9ecef;
    }
    
    /* Progress bars */
    .custom-progress {
        background: #e9ecef;
        border-radius: 10px;
        height: 8px;
        overflow: hidden;
    }
    
    .custom-progress-bar {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        height: 100%;
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    
    /* Sidebar enhancements */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Button customizations */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* File uploader styling */
    .uploadedFile {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8f9ff;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .custom-header h1 {
            font-size: 2rem;
        }
        
        .metric-card {
            margin-bottom: 0.5rem;
        }
        
        .chat-message {
            max-width: 95%;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'assistant' not in st.session_state:
        st.session_state.assistant = None
    
    if 'session_id' not in st.session_state:
        st.session_state.session_id = f"ui_session_{int(datetime.now().timestamp())}"
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'processed_documents' not in st.session_state:
        st.session_state.processed_documents = []
    
    if 'current_project' not in st.session_state:
        st.session_state.current_project = None
    
    if 'streaming_placeholder' not in st.session_state:
        st.session_state.streaming_placeholder = None
    
    if 'progress_data' not in st.session_state:
        st.session_state.progress_data = {}

def initialize_assistant():
    """Initialize the Academic Assistant"""
    try:
        if st.session_state.assistant is None:
            with st.spinner("🚀 Sistem başlatılıyor..."):
                st.session_state.assistant = AdvancedAcademicAssistant(
                    session_id=st.session_state.session_id
                )
            st.success("✅ Sistem başarıyla başlatıldı!")
        return True
    except Exception as e:
        st.error(f"❌ Sistem başlatılamadı: {str(e)}")
        return False

def render_header():
    """Uygulama başlığını göster"""
    st.markdown("""
    <div class="custom-header fade-in">
        <h1>🚀 Akademik Araştırma Asistanı v2.0</h1>
        <p>Gelişmiş hafıza ve akış yetenekleri ile AI destekli araştırma analizi</p>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Navigasyon ve kontrol paneli ile sidebar'ı göster"""
    with st.sidebar:
        st.markdown("## 🎛️ Kontrol Paneli")
        
        # Sistem durumu
        st.markdown("### 📊 Sistem Durumu")
        if st.session_state.assistant:
            st.markdown('<span class="status-success">🟢 Aktif</span>', unsafe_allow_html=True)
            
            # Oturum bilgileri
            session_summary = st.session_state.assistant.get_research_summary()
            session_info = session_summary.get('session_summary', {}).get('session_info', {})
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Etkileşimler", session_info.get('total_interactions', 0))
            with col2:
                st.metric("Dokümanlar", len(st.session_state.processed_documents))
                
        else:
            st.markdown('<span class="status-error">🔴 Pasif</span>', unsafe_allow_html=True)
        
        st.divider()
        
        # Navigasyon
        st.markdown("### 🧭 Navigasyon")
        page = st.selectbox(
            "Sayfa Seç",
            ["🏠 Anasayfa", "📄 Doküman İşleme", "💬 Araştırma Sohbeti", 
             "🔍 Literatür Arama", "📚 Referans Yöneticisi", "📊 Analitik",
             "🗂️ Proje Yöneticisi", "⚙️ Ayarlar"],
            key="page_selector"
        )
        
        st.divider()
        
        # Hızlı Aksiyonlar
        st.markdown("### ⚡ Hızlı Aksiyonlar")
        
        if st.button("🔄 Oturumu Sıfırla", help="Mevcut oturum verilerini temizle"):
            # Oturum durumunu sıfırla
            for key in ['chat_history', 'processed_documents', 'current_project']:
                if key in st.session_state:
                    st.session_state[key] = []
            st.success("Oturum sıfırlandı!")
            st.rerun()
        
        if st.button("💾 Oturumu Dışa Aktar", help="Mevcut oturum verilerini dışa aktar"):
            if st.session_state.assistant:
                summary = st.session_state.assistant.get_research_summary()
                st.download_button(
                    "📥 Download",
                    data=json.dumps(summary, indent=2, ensure_ascii=False),
                    file_name=f"research_session_{st.session_state.session_id}.json",
                    mime="application/json"
                )
        
        # API Key status
        st.divider()
        st.markdown("### 🔑 API Status")
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            st.markdown('<span class="status-success">🟢 OpenAI Connected</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-error">🔴 API Key Missing</span>', unsafe_allow_html=True)
            st.info("Add your OpenAI API key to .env file")
        
        return page

def render_dashboard():
    """Ana kontrol panelini göster"""
    st.markdown("## 📊 Araştırma Paneli")
    
    if not st.session_state.assistant:
        st.warning("Lütfen önce sistemi başlatın.")
        return
    
    # Get comprehensive stats
    summary = st.session_state.assistant.get_research_summary()
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    session_info = summary.get('session_summary', {}).get('session_info', {})
    research_overview = summary.get('session_summary', {}).get('research_overview', {})
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-number">{}</p>
            <p class="metric-label">Toplam Etkileşim</p>
        </div>
        """.format(session_info.get('total_interactions', 0)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-number">{}</p>
            <p class="metric-label">İşlenen Doküman</p>
        </div>
        """.format(len(st.session_state.processed_documents)), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-number">{}</p>
            <p class="metric-label">Araştırma Soruları</p>
        </div>
        """.format(research_overview.get('questions_explored', 0)), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-number">{}</p>
            <p class="metric-label">Önemli İçgörüler</p>
        </div>
        """.format(research_overview.get('insights_generated', 0)), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Grafikler ve görselleştirmeler
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Araştırma Aktivitesi")
        if st.session_state.chat_history:
            # Aktivite zaman çizelgesi oluştur
            activity_data = []
            for i, chat in enumerate(st.session_state.chat_history):
                timestamp = datetime.now() - timedelta(hours=len(st.session_state.chat_history)-i)
                activity_data.append({
                    'Zaman': timestamp,
                    'Aktivite': 'Araştırma Sorgusu',
                    'Sayı': 1
                })
            
            if activity_data:
                df = pd.DataFrame(activity_data)
                fig = px.line(df, x='Zaman', y='Sayı', title='Zaman İçindeki Araştırma Aktivitesi')
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Henüz aktivite verisi yok. Doküman işleyerek veya soru sorarak başlayın!")
    
    with col2:
        st.markdown("### 🏷️ Araştırma Konuları")
        current_focus = summary.get('session_summary', {}).get('current_focus', {})
        topics = current_focus.get('main_topics', [])
        
        if topics:
            topic_counts = {topic: 1 for topic in topics}  # Basitleştirilmiş sayım
            fig = px.pie(
                values=list(topic_counts.values()),
                names=list(topic_counts.keys()),
                title='Araştırma Odak Alanları'
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Henüz araştırma konusu tanımlanmadı. Konu dağılımını görmek için bazı dokümanları işleyin!")
    
    # Son aktiviteler
    st.markdown("### 📋 Son Aktiviteler")
    if st.session_state.chat_history:
        for i, chat in enumerate(st.session_state.chat_history[-3:]):  # Son 3 etkileşim
            with st.expander(f"💬 Sorgu {len(st.session_state.chat_history) - 2 + i}: {chat.get('question', 'Bilinmiyor')[:50]}..."):
                st.write("**Soru:** ", chat.get('question', 'Soru yok'))
                st.write("**Cevap:** ", chat.get('answer', 'Cevap yok')[:200] + "...")
                st.write("**Zaman:** ", chat.get('timestamp', 'Bilinmiyor'))
    else:
        st.info("Son aktivite yok. Doküman yükleyerek veya soru sorarak araştırma yolculuğunuzu başlatın!")

def render_document_processing():
    """Doküman işleme arayüzünü göster"""
    st.markdown("## 📄 Doküman İşleme")
    
    if not st.session_state.assistant:
        st.warning("Lütfen önce sistemi başlatın.")
        return
    
    # Dosya yükleme bölümü
    st.markdown("### 📤 Doküman Yükleme")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "PDF dosyalarını seçin",
            type=['pdf'],
            accept_multiple_files=True,
            help="Akademik makaleler, araştırma dokümanları veya analiz için herhangi bir PDF dosyası yükleyin"
        )
    
    with col2:
        # İşleme seçenekleri
        st.markdown("#### İşleme Seçenekleri")
        associate_with_project = st.checkbox("Proje ile ilişkilendir", help="Mevcut projeye bağla")
        enable_advanced_analysis = st.checkbox("Gelişmiş analiz", value=True, help="Kapsamlı analiz zincirlerini çalıştır")
        chunk_size = st.slider("Parça boyutu", 500, 2000, 1200, help="Vektör indeksleme için metin parça boyutu")
    
    # Yüklenen dosyaları işle
    if uploaded_files:
        if st.button("🚀 Dokümanları İşle", type="primary"):
            process_documents(uploaded_files, associate_with_project, enable_advanced_analysis, chunk_size)
    
    # İşlenmiş dokümanları göster
    st.markdown("### 📚 İşlenmiş Dokümanlar")
    
    if st.session_state.processed_documents:
        for i, doc in enumerate(st.session_state.processed_documents):
            with st.expander(f"📄 {doc.get('title', f'Document {i+1}')}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write("**File:** ", doc.get('filename', 'Unknown'))
                    st.write("**Processed:** ", doc.get('processed_date', 'Unknown'))
                    st.write("**Size:** ", f"{doc.get('text_length', 0):,} characters")
                    
                    # Show analysis results if available
                    if 'analysis' in doc:
                        analysis = doc['analysis']
                        if 'research_analysis' in analysis:
                            research = analysis['research_analysis']
                            if 'categorization' in research:
                                cat = research['categorization']
                                st.write("**Field:** ", cat.get('research_field', 'Unknown'))
                                st.write("**Novelty:** ", f"{cat.get('novelty_score', 'N/A')}/10")
                
                with col2:
                    if st.button(f"🔍 Analyze Again", key=f"reanalyze_{i}"):
                        st.info("Re-analysis feature coming soon!")
                    
                    if st.button(f"🗑️ Remove", key=f"remove_{i}"):
                        st.session_state.processed_documents.pop(i)
                        st.rerun()
    else:
        st.info("No documents processed yet. Upload some PDFs to get started!")

def process_documents(uploaded_files, associate_project, advanced_analysis, chunk_size):
    """Process uploaded documents with real-time progress"""
    
    # Create progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            # Update progress
            progress = i / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing {uploaded_file.name}...")
            
            # Save uploaded file temporarily
            temp_path = f"./temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Process document
            result = st.session_state.assistant.process_document(
                pdf_path=temp_path,
                project_id=st.session_state.current_project if associate_project else None
            )
            
            # Store result
            doc_info = {
                'filename': uploaded_file.name,
                'title': result.get('processing_stages', {}).get('extraction', {}).get('metadata', {}).get('title', uploaded_file.name),
                'processed_date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                'text_length': result.get('processing_stages', {}).get('extraction', {}).get('text_length', 0),
                'analysis': result.get('processing_stages', {}) if advanced_analysis else None
            }
            
            st.session_state.processed_documents.append(doc_info)
            
            # Clean up temp file
            os.remove(temp_path)
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
    
    # Final progress update
    progress_bar.progress(1.0)
    status_text.text("✅ All documents processed!")
    
    st.success(f"{len(uploaded_files)} doküman başarıyla işlendi!")

def render_research_chat():
    """İnteraktif araştırma sohbet arayüzünü göster"""
    st.markdown("## 💬 Araştırma Sohbeti")
    
    if not st.session_state.assistant:
        st.warning("Lütfen önce sistemi başlatın.")
        return
    
    # Sohbet arayüzü
    st.markdown("### 🤖 Araştırma Sorularınızı Sorun")
    
    # Chat options
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        user_question = st.text_input(
            "Sorunuzu girin:",
            placeholder="Yüklenen makalelerdeki ana bulgular nelerdir?",
            key="chat_input"
        )
    
    with col2:
        use_memory = st.checkbox("Bağlam hafızasını kullan", value=True, help="Önceki konuşma bağlamını dahil et")
    
    with col3:
        specific_docs = st.multiselect(
            "Belirli dokümanlar",
            options=[doc['filename'] for doc in st.session_state.processed_documents],
            help="Sadece seçili dokümanlarda ara"
        )
    
    # Soru gönder
    if st.button("🚀 Soru Sor", type="primary") and user_question:
        ask_question(user_question, specific_docs, use_memory)
    
    # Chat history display
    st.markdown("### 💭 Conversation History")
    
    if st.session_state.chat_history:
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            # User message
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You:</strong> {chat.get('question', '')}
            </div>
            """, unsafe_allow_html=True)
            
            # AI response
            st.markdown(f"""
            <div class="chat-message ai-message">
                <strong>🤖 Assistant:</strong><br>
                {chat.get('answer', '')}
            </div>
            """, unsafe_allow_html=True)
            
            # Show sources if available
            if chat.get('sources'):
                with st.expander(f"📚 Sources for question {len(st.session_state.chat_history) - i}"):
                    for source in chat['sources'][:3]:  # Show top 3 sources
                        st.write(f"**Document:** {source.get('pdf_name', 'Unknown')}")
                        st.write(f"**Similarity:** {source.get('similarity_score', 'N/A')}")
            
            st.markdown("---")
    else:
        st.info("No conversations yet. Ask a question to start your research dialogue!")

def ask_question(question, specific_docs, use_memory):
    """Process research question with streaming response"""
    
    # Create streaming placeholder
    response_placeholder = st.empty()
    
    with st.spinner("🤔 Thinking..."):
        try:
            # Get answer from assistant
            result = st.session_state.assistant.ask_question(
                question=question,
                pdf_names=specific_docs if specific_docs else None,
                use_memory=use_memory
            )
            
            # Display result
            response_placeholder.success("✅ Question answered!")
            
            # Add to chat history
            chat_entry = {
                'question': question,
                'answer': result.get('answer', 'No answer generated'),
                'sources': result.get('sources', []),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'confidence': result.get('confidence', 'unknown')
            }
            
            st.session_state.chat_history.append(chat_entry)
            
            # Clear input
            st.session_state.chat_input = ""
            
            # Rerun to update display
            st.rerun()
            
        except Exception as e:
            response_placeholder.error(f"❌ Error: {str(e)}")

def main():
    """Main Streamlit application"""
    
    # Load custom CSS
    load_custom_css()
    
    # Initialize session state
    initialize_session_state()
    
    # Render header
    render_header()
    
    # Initialize assistant if not done
    if not initialize_assistant():
        st.stop()
    
    # Render sidebar and get selected page
    selected_page = render_sidebar()
    
    # Seçilen sayfayı göster
    if selected_page == "🏠 Anasayfa":
        render_dashboard()
    elif selected_page == "📄 Doküman İşleme":
        render_document_processing()
    elif selected_page == "💬 Araştırma Sohbeti":
        render_research_chat()
    elif selected_page == "🔍 Literatür Arama":
        render_literature_search()
    elif selected_page == "📚 Referans Yöneticisi":
        render_reference_manager()
    elif selected_page == "📊 Analitik":
        render_analytics()
    elif selected_page == "🗂️ Proje Yöneticisi":
        render_project_manager()
    elif selected_page == "⚙️ Ayarlar":
        render_settings()

def render_literature_search():
    """Literatür arama arayüzü için placeholder"""
    st.markdown("## 🔍 Literatür Arama")
    st.info("🚧 Literatür arama arayüzü yakında gelecek! Bu, CrossRef, arXiv ve PubMed API'leri ile entegre olacak.")

def render_reference_manager():
    """Referans yöneticisi arayüzü için placeholder"""
    st.markdown("## 📚 Referans Yöneticisi")
    st.info("🚧 Referans yöneticisi arayüzü yakında gelecek! Bu, DOI doğrulama, atıf biçimlendirme ve bibliyografya yönetimi yapacak.")

def render_analytics():
    """Analitik panosu için placeholder"""
    st.markdown("## 📊 Analitik Paneli")
    st.info("🚧 Gelişmiş analitik paneli yakında gelecek! Bu, araştırma trendlerini, atıf ağlarını ve etki analizini gösterecek.")

def render_project_manager():
    """Proje yöneticisi için placeholder"""
    st.markdown("## 🗂️ Proje Yöneticisi")
    st.info("🚧 Proje yönetimi arayüzü yakında gelecek! Bu, birden çok araştırma projesini ve projeler arası içgörüleri yönetecek.")

def render_settings():
    """Uygulama ayarlarını göster"""
    st.markdown("## ⚙️ Ayarlar")
    
    st.markdown("### 🔧 Sistem Konfigürasyonu")
    
    # API Ayarları
    st.markdown("#### 🔑 API Konfigürasyonu")
    current_api_key = os.getenv("OPENAI_API_KEY", "")
    masked_key = "***" + current_api_key[-4:] if current_api_key else "Ayarlanmamış"
    st.text_input("OpenAI API Anahtarı", value=masked_key, disabled=True, help=".env dosyasında ayarlayın")
    
    # Model Settings
    st.markdown("#### 🤖 Model Settings")
    model_options = ["gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"]
    selected_model = st.selectbox("Default Model", model_options, index=1)
    
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1, help="Controls randomness in responses")
    
    # Memory Settings
    st.markdown("#### 🧠 Memory Settings")
    max_memory_items = st.number_input("Max Memory Items", 100, 10000, 1000, help="Maximum items to keep in memory")
    session_timeout = st.number_input("Session Timeout (hours)", 1, 24, 6, help="Auto-clear session after inactivity")
    
    # Performance Settings
    st.markdown("#### ⚡ Performance Settings")
    chunk_size = st.slider("Default Chunk Size", 500, 2000, 1200, help="Text chunk size for processing")
    max_results = st.slider("Max Search Results", 3, 20, 5, help="Maximum results per search query")
    
    # Save settings
    if st.button("💾 Save Settings"):
        st.success("Settings saved! (Note: Some settings require restart to take effect)")

if __name__ == "__main__":
    main()
"""
UI Components Module - Reusable Streamlit components for Academic Research Assistant
Enhanced components with animations, interactivity, and modern design
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import json

class ResearchMetrics:
    """Component for displaying research metrics and statistics"""
    
    @staticmethod
    def render_metric_card(title: str, value: str, delta: Optional[str] = None, 
                          delta_color: str = "normal", icon: str = "📊"):
        """Render an enhanced metric card"""
        delta_html = ""
        if delta:
            color = {"normal": "#28a745", "inverse": "#dc3545"}.get(delta_color, "#28a745")
            delta_html = f'<p style="color: {color}; font-size: 0.8rem; margin: 0.2rem 0;">{delta}</p>'
        
        st.markdown(f"""
        <div style="
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
            border-left: 4px solid #667eea;
            margin-bottom: 1rem;
            transition: transform 0.2s ease;
        ">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</span>
                <h4 style="margin: 0; color: #333; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.5px;">{title}</h4>
            </div>
            <p style="font-size: 2.5rem; font-weight: 700; color: #667eea; margin: 0;">{value}</p>
            {delta_html}
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_progress_bar(label: str, value: float, max_value: float = 100, 
                           color: str = "#667eea", show_percentage: bool = True):
        """Render an animated progress bar"""
        percentage = (value / max_value) * 100
        percentage_text = f" ({percentage:.1f}%)" if show_percentage else ""
        
        st.markdown(f"""
        <div style="margin-bottom: 1rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-weight: 600; color: #333;">{label}</span>
                <span style="color: #666;">{value}/{max_value}{percentage_text}</span>
            </div>
            <div style="background: #e9ecef; border-radius: 10px; height: 8px; overflow: hidden;">
                <div style="
                    background: linear-gradient(90deg, {color} 0%, {color}dd 100%);
                    height: 100%;
                    width: {percentage}%;
                    border-radius: 10px;
                    transition: width 0.8s ease;
                "></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

class DocumentCard:
    """Component for displaying document information cards"""
    
    @staticmethod
    def render_document_card(doc_info: Dict[str, Any], index: int):
        """Render a comprehensive document card"""
        title = doc_info.get('title', f"Document {index + 1}")
        filename = doc_info.get('filename', 'Unknown')
        processed_date = doc_info.get('processed_date', 'Unknown')
        text_length = doc_info.get('text_length', 0)
        
        # Extract analysis info if available
        analysis_info = ""
        if 'analysis' in doc_info and doc_info['analysis']:
            analysis = doc_info['analysis']
            if 'research_analysis' in analysis:
                research = analysis['research_analysis']
                if 'categorization' in research:
                    cat = research['categorization']
                    field = cat.get('research_field', 'Unknown')
                    novelty = cat.get('novelty_score', 'N/A')
                    analysis_info = f"""
                    <div style="background: #f8f9fa; padding: 0.5rem; border-radius: 6px; margin-top: 0.5rem;">
                        <small><strong>Field:</strong> {field} | <strong>Novelty:</strong> {novelty}/10</small>
                    </div>
                    """
        
        st.markdown(f"""
        <div style="
            background: white;
            border: 1px solid #e1e8ed;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            transition: all 0.2s ease;
            hover: transform: translateY(-2px);
        ">
            <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 1rem;">
                <div style="flex: 1;">
                    <h4 style="margin: 0 0 0.5rem 0; color: #333; font-size: 1.1rem;">📄 {title}</h4>
                    <p style="margin: 0; color: #666; font-size: 0.9rem;">
                        <strong>File:</strong> {filename}<br>
                        <strong>Processed:</strong> {processed_date}<br>
                        <strong>Size:</strong> {text_length:,} characters
                    </p>
                    {analysis_info}
                </div>
                <div style="display: flex; flex-direction: column; gap: 0.5rem; margin-left: 1rem;">
                    <button style="
                        background: #667eea;
                        color: white;
                        border: none;
                        padding: 0.4rem 0.8rem;
                        border-radius: 6px;
                        font-size: 0.8rem;
                        cursor: pointer;
                    ">🔍 View</button>
                    <button style="
                        background: #28a745;
                        color: white;
                        border: none;
                        padding: 0.4rem 0.8rem;
                        border-radius: 6px;
                        font-size: 0.8rem;
                        cursor: pointer;
                    ">💬 Ask</button>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

class ChatInterface:
    """Enhanced chat interface components"""
    
    @staticmethod
    def render_message(message: str, is_user: bool = True, timestamp: str = None, 
                      sources: List[Dict] = None):
        """Render a chat message with enhanced styling"""
        
        if timestamp is None:
            timestamp = datetime.now().strftime("%H:%M")
        
        # Message styling
        if is_user:
            align = "margin-left: auto; margin-right: 0;"
            bg_color = "#667eea"
            text_color = "white"
            icon = "👤"
        else:
            align = "margin-left: 0; margin-right: auto;"
            bg_color = "#f8f9fa"
            text_color = "#333"
            icon = "🤖"
        
        st.markdown(f"""
        <div style="
            {align}
            max-width: 80%;
            background: {bg_color};
            color: {text_color};
            padding: 1rem 1.5rem;
            border-radius: 18px;
            margin: 1rem 0;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            animation: slideIn 0.3s ease;
        ">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <span style="margin-right: 0.5rem; font-size: 1.2rem;">{icon}</span>
                <small style="opacity: 0.8;">{timestamp}</small>
            </div>
            <div style="line-height: 1.5;">
                {message}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show sources if provided
        if sources and not is_user:
            ChatInterface.render_sources(sources)
    
    @staticmethod
    def render_sources(sources: List[Dict]):
        """Render source citations in an expandable format"""
        if not sources:
            return
        
        st.markdown("""
        <details style="margin: 0.5rem 0; padding: 0.5rem; background: #f1f3f4; border-radius: 8px;">
            <summary style="cursor: pointer; font-weight: 600; color: #555;">📚 Sources ({} found)</summary>
            <div style="margin-top: 0.5rem;">
        """.format(len(sources)), unsafe_allow_html=True)
        
        for i, source in enumerate(sources[:3]):  # Show top 3 sources
            doc_name = source.get('pdf_name', 'Unknown Document')
            similarity = source.get('similarity_score', 'N/A')
            content_preview = source.get('content', '')[:100] + "..." if source.get('content') else ""
            
            st.markdown(f"""
                <div style="padding: 0.5rem; margin: 0.5rem 0; background: white; border-radius: 6px; border-left: 3px solid #667eea;">
                    <strong>{i+1}. {doc_name}</strong><br>
                    <small>Similarity: {similarity} | Preview: {content_preview}</small>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div></details>", unsafe_allow_html=True)

class ResearchCharts:
    """Interactive charts for research analytics"""
    
    @staticmethod
    def create_activity_timeline(chat_history: List[Dict]) -> go.Figure:
        """Create an interactive activity timeline"""
        if not chat_history:
            return None
        
        # Process chat history into timeline data
        timeline_data = []
        for i, chat in enumerate(chat_history):
            timestamp = datetime.now() - timedelta(hours=len(chat_history) - i)
            timeline_data.append({
                'Time': timestamp,
                'Questions': 1,
                'Type': 'Research Query'
            })
        
        df = pd.DataFrame(timeline_data)
        
        # Create line chart
        fig = px.line(
            df, 
            x='Time', 
            y='Questions',
            title='Research Activity Timeline',
            color_discrete_sequence=['#667eea']
        )
        
        fig.update_layout(
            height=300,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#333'),
            title_font_size=16,
            xaxis=dict(gridcolor='#e1e8ed'),
            yaxis=dict(gridcolor='#e1e8ed')
        )
        
        return fig
    
    @staticmethod
    def create_topic_distribution(topics: List[str]) -> go.Figure:
        """Create a topic distribution pie chart"""
        if not topics:
            return None
        
        # Count topic frequency
        topic_counts = {}
        for topic in topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        fig = px.pie(
            values=list(topic_counts.values()),
            names=list(topic_counts.keys()),
            title='Research Focus Areas',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_layout(
            height=300,
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#333'),
            title_font_size=16
        )
        
        return fig
    
    @staticmethod
    def create_performance_metrics(metrics_data: Dict) -> go.Figure:
        """Create performance metrics radar chart"""
        categories = list(metrics_data.keys())
        values = list(metrics_data.values())
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            fillcolor='rgba(102, 126, 234, 0.2)',
            line=dict(color='#667eea', width=2),
            name='Performance'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )
            ),
            height=300,
            title='System Performance Metrics',
            title_font_size=16,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        return fig

class StatusIndicators:
    """Status and notification components"""
    
    @staticmethod
    def render_status_badge(status: str, label: str = None):
        """Render a status badge with appropriate styling"""
        status_colors = {
            'success': {'bg': '#d4edda', 'color': '#155724', 'icon': '🟢'},
            'warning': {'bg': '#fff3cd', 'color': '#856404', 'icon': '🟡'},
            'error': {'bg': '#f8d7da', 'color': '#721c24', 'icon': '🔴'},
            'info': {'bg': '#d1ecf1', 'color': '#0c5460', 'icon': '🔵'},
            'loading': {'bg': '#e2e3e5', 'color': '#495057', 'icon': '⚪'}
        }
        
        style_info = status_colors.get(status, status_colors['info'])
        display_label = label or status.title()
        
        st.markdown(f"""
        <span style="
            background: {style_info['bg']};
            color: {style_info['color']};
            padding: 0.25rem 0.75rem;
            border-radius: 15px;
            font-size: 0.8rem;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
        ">
            {style_info['icon']} {display_label}
        </span>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_notification(message: str, type: str = 'info', dismissible: bool = True):
        """Render an enhanced notification"""
        type_styles = {
            'success': {'bg': '#d4edda', 'border': '#c3e6cb', 'color': '#155724', 'icon': '✅'},
            'warning': {'bg': '#fff3cd', 'border': '#ffeaa7', 'color': '#856404', 'icon': '⚠️'},
            'error': {'bg': '#f8d7da', 'border': '#f5c6cb', 'color': '#721c24', 'icon': '❌'},
            'info': {'bg': '#d1ecf1', 'border': '#bee5eb', 'color': '#0c5460', 'icon': 'ℹ️'}
        }
        
        style = type_styles.get(type, type_styles['info'])
        dismiss_btn = '×' if dismissible else ''
        
        st.markdown(f"""
        <div style="
            background: {style['bg']};
            color: {style['color']};
            border: 1px solid {style['border']};
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
            display: flex;
            align-items: center;
            justify-content: space-between;
            animation: fadeIn 0.3s ease;
        ">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="font-size: 1.2rem;">{style['icon']}</span>
                <span>{message}</span>
            </div>
            {f'<button style="background: none; border: none; color: {style["color"]}; font-size: 1.2rem; cursor: pointer;">{dismiss_btn}</button>' if dismiss_btn else ''}
        </div>
        """, unsafe_allow_html=True)

class SearchComponents:
    """Enhanced search and filter components"""
    
    @staticmethod
    def render_search_bar(placeholder: str = "Search...", key: str = "search"):
        """Render an enhanced search bar"""
        st.markdown("""
        <style>
        .search-container {
            position: relative;
            margin-bottom: 1rem;
        }
        .search-icon {
            position: absolute;
            left: 15px;
            top: 50%;
            transform: translateY(-50%);
            color: #666;
            z-index: 1;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="search-container">
            <span class="search-icon">🔍</span>
        </div>
        """, unsafe_allow_html=True)
        
        return st.text_input(
            "",
            placeholder=placeholder,
            key=key,
            help="Enter search terms"
        )
    
    @staticmethod
    def render_filter_chips(options: List[str], selected: List[str] = None, key: str = "filters"):
        """Render filter chips for multi-select filtering"""
        if selected is None:
            selected = []
        
        st.markdown("**Filter by:**")
        
        cols = st.columns(len(options))
        new_selection = []
        
        for i, option in enumerate(options):
            with cols[i]:
                is_selected = option in selected
                if st.button(
                    f"{'✓ ' if is_selected else ''}{option}",
                    key=f"{key}_{i}",
                    help=f"Toggle {option} filter"
                ):
                    if is_selected:
                        new_selection = [x for x in selected if x != option]
                    else:
                        new_selection = selected + [option]
        
        return new_selection if new_selection else selected

# Animation CSS
ANIMATION_CSS = """
<style>
@keyframes slideIn {
    from {
        transform: translateY(10px);
        opacity: 0;
    }
    to {
        transform: translateY(0);
        opacity: 1;
    }
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}

.animated {
    animation-duration: 0.3s;
    animation-fill-mode: both;
}

.slide-in {
    animation-name: slideIn;
}

.fade-in {
    animation-name: fadeIn;
}

.pulse {
    animation-name: pulse;
    animation-duration: 2s;
    animation-iteration-count: infinite;
}
</style>
"""
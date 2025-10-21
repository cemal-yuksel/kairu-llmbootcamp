"""
UI Utils Module - Utility functions for the Streamlit interface
Data processing, validation, and helper functions for enhanced user experience
"""

import streamlit as st
import json
import io
import base64
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import re
from pathlib import Path
import hashlib

class SessionManager:
    """Manage Streamlit session state and persistence"""
    
    @staticmethod
    def init_session_defaults():
        """Initialize default session state values"""
        defaults = {
            'assistant': None,
            'session_id': f"ui_session_{int(datetime.now().timestamp())}",
            'chat_history': [],
            'processed_documents': [],
            'current_project': None,
            'user_preferences': {
                'theme': 'light',
                'language': 'en',
                'auto_save': True,
                'notifications': True
            },
            'search_filters': {
                'date_range': None,
                'document_types': [],
                'research_fields': []
            },
            'ui_state': {
                'sidebar_expanded': True,
                'last_page': '🏠 Dashboard',
                'view_mode': 'cards'
            }
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    @staticmethod
    def save_session_to_file() -> str:
        """Save current session to JSON file"""
        session_data = {
            'session_id': st.session_state.get('session_id'),
            'timestamp': datetime.now().isoformat(),
            'chat_history': st.session_state.get('chat_history', []),
            'processed_documents': st.session_state.get('processed_documents', []),
            'user_preferences': st.session_state.get('user_preferences', {}),
            'ui_state': st.session_state.get('ui_state', {})
        }
        
        filename = f"session_{st.session_state.session_id}.json"
        return json.dumps(session_data, indent=2, ensure_ascii=False)
    
    @staticmethod
    def load_session_from_file(session_data: str) -> bool:
        """Load session from JSON data"""
        try:
            data = json.loads(session_data)
            
            # Update session state
            st.session_state.session_id = data.get('session_id')
            st.session_state.chat_history = data.get('chat_history', [])
            st.session_state.processed_documents = data.get('processed_documents', [])
            st.session_state.user_preferences = data.get('user_preferences', {})
            st.session_state.ui_state = data.get('ui_state', {})
            
            return True
        except Exception as e:
            st.error(f"Error loading session: {str(e)}")
            return False

class DataProcessor:
    """Process and validate data for UI components"""
    
    @staticmethod
    def validate_file_upload(uploaded_file) -> Tuple[bool, str]:
        """Validate uploaded file"""
        if not uploaded_file:
            return False, "No file uploaded"
        
        # Check file size (max 50MB)
        if uploaded_file.size > 50 * 1024 * 1024:
            return False, "File size exceeds 50MB limit"
        
        # Check file type
        allowed_types = ['.pdf', '.txt', '.docx']
        file_extension = Path(uploaded_file.name).suffix.lower()
        if file_extension not in allowed_types:
            return False, f"File type {file_extension} not supported. Allowed types: {', '.join(allowed_types)}"
        
        return True, "File is valid"
    
    @staticmethod
    def process_research_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Process research data for display"""
        processed = {
            'summary': {
                'total_documents': len(data.get('documents', [])),
                'total_questions': len(data.get('chat_history', [])),
                'research_fields': set(),
                'date_range': None,
                'key_topics': []
            },
            'metrics': {
                'engagement_score': 0,
                'diversity_score': 0,
                'completion_rate': 0,
                'accuracy_score': 0
            },
            'trends': {
                'daily_activity': [],
                'topic_evolution': [],
                'question_patterns': []
            }
        }
        
        # Process documents
        documents = data.get('documents', [])
        if documents:
            fields = set()
            for doc in documents:
                if 'analysis' in doc:
                    analysis = doc['analysis']
                    if 'research_analysis' in analysis:
                        research = analysis['research_analysis']
                        if 'categorization' in research:
                            field = research['categorization'].get('research_field')
                            if field:
                                fields.add(field)
            
            processed['summary']['research_fields'] = list(fields)
        
        # Calculate engagement metrics
        chat_history = data.get('chat_history', [])
        if chat_history:
            # Simple engagement calculation
            avg_response_length = sum(len(chat.get('answer', '')) for chat in chat_history) / len(chat_history)
            processed['metrics']['engagement_score'] = min(10, avg_response_length / 100)
        
        return processed
    
    @staticmethod
    def create_download_link(data: str, filename: str, mime_type: str = "text/plain") -> str:
        """Create a download link for data"""
        b64_data = base64.b64encode(data.encode()).decode()
        href = f'<a href="data:{mime_type};base64,{b64_data}" download="{filename}">📥 Download {filename}</a>'
        return href
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Format file size in human readable format"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"

class TextProcessor:
    """Text processing utilities for better display"""
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 100, add_ellipsis: bool = True) -> str:
        """Truncate text with optional ellipsis"""
        if len(text) <= max_length:
            return text
        
        truncated = text[:max_length]
        if add_ellipsis:
            truncated += "..."
        
        return truncated
    
    @staticmethod
    def highlight_search_terms(text: str, search_terms: List[str]) -> str:
        """Highlight search terms in text"""
        if not search_terms:
            return text
        
        highlighted = text
        for term in search_terms:
            if term.strip():
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                highlighted = pattern.sub(f'<mark style="background-color: yellow; padding: 0 2px;">{term}</mark>', highlighted)
        
        return highlighted
    
    @staticmethod
    def extract_keywords(text: str, max_keywords: int = 5) -> List[str]:
        """Extract keywords from text (simple implementation)"""
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must', 'this', 'that', 'these', 'those'}
        
        # Simple word extraction and counting
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        word_freq = {}
        
        for word in words:
            if word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:max_keywords]]
    
    @staticmethod
    def format_citation(source_info: Dict[str, Any]) -> str:
        """Format source information as citation"""
        title = source_info.get('title', 'Unknown Title')
        authors = source_info.get('authors', ['Unknown Author'])
        year = source_info.get('year', 'Unknown Year')
        
        if isinstance(authors, list):
            author_str = ', '.join(authors[:3])  # Show max 3 authors
            if len(authors) > 3:
                author_str += ' et al.'
        else:
            author_str = str(authors)
        
        return f"{author_str} ({year}). {title}"

class UIState:
    """Manage UI state and preferences"""
    
    @staticmethod
    def get_theme_colors() -> Dict[str, str]:
        """Get current theme colors"""
        is_dark = st.session_state.get('user_preferences', {}).get('theme') == 'dark'
        
        if is_dark:
            return {
                'primary': '#667eea',
                'secondary': '#764ba2',
                'background': '#1e1e1e',
                'surface': '#2d2d2d',
                'text_primary': '#ffffff',
                'text_secondary': '#cccccc',
                'border': '#404040'
            }
        else:
            return {
                'primary': '#667eea',
                'secondary': '#764ba2',
                'background': '#ffffff',
                'surface': '#f8f9fa',
                'text_primary': '#333333',
                'text_secondary': '#666666',
                'border': '#e1e8ed'
            }
    
    @staticmethod
    def toggle_sidebar():
        """Toggle sidebar state"""
        current_state = st.session_state.get('ui_state', {}).get('sidebar_expanded', True)
        if 'ui_state' not in st.session_state:
            st.session_state.ui_state = {}
        st.session_state.ui_state['sidebar_expanded'] = not current_state
    
    @staticmethod
    def set_page(page_name: str):
        """Set current page and update UI state"""
        if 'ui_state' not in st.session_state:
            st.session_state.ui_state = {}
        st.session_state.ui_state['last_page'] = page_name

class PerformanceMonitor:
    """Monitor and optimize UI performance"""
    
    @staticmethod
    def measure_load_time(func):
        """Decorator to measure function execution time"""
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            result = func(*args, **kwargs)
            end_time = datetime.now()
            
            execution_time = (end_time - start_time).total_seconds()
            
            # Store performance data
            if 'performance_data' not in st.session_state:
                st.session_state.performance_data = []
            
            st.session_state.performance_data.append({
                'function': func.__name__,
                'execution_time': execution_time,
                'timestamp': start_time.isoformat()
            })
            
            # Show warning for slow operations
            if execution_time > 5:
                st.warning(f"Operation '{func.__name__}' took {execution_time:.2f} seconds")
            
            return result
        
        return wrapper
    
    @staticmethod
    def get_performance_summary() -> Dict[str, Any]:
        """Get performance summary"""
        perf_data = st.session_state.get('performance_data', [])
        
        if not perf_data:
            return {'total_operations': 0, 'avg_time': 0, 'slowest_operation': None}
        
        total_ops = len(perf_data)
        avg_time = sum(op['execution_time'] for op in perf_data) / total_ops
        slowest_op = max(perf_data, key=lambda x: x['execution_time'])
        
        return {
            'total_operations': total_ops,
            'avg_time': avg_time,
            'slowest_operation': slowest_op,
            'operations_per_hour': len([op for op in perf_data if datetime.fromisoformat(op['timestamp']) > datetime.now() - timedelta(hours=1)])
        }

class ErrorHandler:
    """Enhanced error handling for UI"""
    
    @staticmethod
    def handle_api_error(error: Exception):
        """Handle API-related errors with user-friendly messages"""
        error_msg = str(error).lower()
        
        if 'api key' in error_msg or 'authentication' in error_msg:
            st.error("🔑 API Key Error: Please check your OpenAI API key configuration.")
            st.info("Make sure your API key is set in the .env file and has sufficient credits.")
        elif 'rate limit' in error_msg or 'quota' in error_msg:
            st.error("⚠️ Rate Limit Exceeded: Too many requests. Please wait a moment and try again.")
            st.info("Consider upgrading your API plan for higher rate limits.")
        elif 'timeout' in error_msg:
            st.error("⏱️ Request Timeout: The request took too long. Please try again.")
            st.info("This might be due to network issues or high server load.")
        else:
            st.error(f"❌ API Error: {str(error)}")
            st.info("Please check your internet connection and try again.")
    
    @staticmethod
    def handle_file_error(error: Exception, filename: str = ""):
        """Handle file-related errors"""
        error_msg = str(error).lower()
        
        if 'permission' in error_msg or 'access' in error_msg:
            st.error(f"🔒 Permission Error: Cannot access file {filename}")
            st.info("Please check file permissions and try again.")
        elif 'not found' in error_msg:
            st.error(f"📄 File Not Found: {filename} could not be located")
            st.info("Please verify the file path and try again.")
        elif 'corrupt' in error_msg or 'invalid' in error_msg:
            st.error(f"💥 Corrupt File: {filename} appears to be damaged")
            st.info("Please try uploading the file again or use a different file.")
        else:
            st.error(f"📁 File Error: {str(error)}")
    
    @staticmethod
    def show_error_with_details(title: str, error: Exception, show_traceback: bool = False):
        """Show detailed error information"""
        st.error(f"❌ {title}")
        
        with st.expander("Error Details", expanded=False):
            st.write(f"**Error Type:** {type(error).__name__}")
            st.write(f"**Error Message:** {str(error)}")
            
            if show_traceback:
                import traceback
                st.code(traceback.format_exc(), language="python")

class NotificationManager:
    """Manage user notifications and alerts"""
    
    @staticmethod
    def show_success(message: str, duration: int = 3):
        """Show success notification"""
        success_placeholder = st.empty()
        success_placeholder.success(f"✅ {message}")
        
        # Auto-dismiss after duration (simplified)
        if 'notifications' not in st.session_state:
            st.session_state.notifications = []
        
        st.session_state.notifications.append({
            'type': 'success',
            'message': message,
            'timestamp': datetime.now(),
            'duration': duration
        })
    
    @staticmethod
    def show_info(message: str, persistent: bool = False):
        """Show info notification"""
        st.info(f"ℹ️ {message}")
        
        if persistent:
            if 'persistent_notifications' not in st.session_state:
                st.session_state.persistent_notifications = []
            
            st.session_state.persistent_notifications.append({
                'type': 'info',
                'message': message,
                'timestamp': datetime.now()
            })
    
    @staticmethod
    def clear_notifications():
        """Clear all notifications"""
        if 'notifications' in st.session_state:
            st.session_state.notifications = []
        if 'persistent_notifications' in st.session_state:
            st.session_state.persistent_notifications = []

# Utility functions
def safe_get(dictionary: Dict, key_path: str, default=None):
    """Safely get nested dictionary values"""
    keys = key_path.split('.')
    current = dictionary
    
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    
    return current

def format_timestamp(timestamp, format_str: str = "%Y-%m-%d %H:%M:%S"):
    """Format timestamp for display"""
    if isinstance(timestamp, str):
        try:
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except:
            return timestamp
    
    if isinstance(timestamp, datetime):
        return timestamp.strftime(format_str)
    
    return str(timestamp)

def generate_unique_id(prefix: str = "item") -> str:
    """Generate unique ID for UI elements"""
    timestamp = str(int(datetime.now().timestamp() * 1000))
    return f"{prefix}_{timestamp}"
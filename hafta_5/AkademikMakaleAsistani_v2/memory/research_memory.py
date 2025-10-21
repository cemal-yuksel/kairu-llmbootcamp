"""
Research Session Memory - Advanced memory management for academic research sessions
Manages conversation history, research context, and user preferences across sessions
"""
import logging
import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
from langchain.memory import ConversationSummaryBufferMemory, ConversationBufferWindowMemory
from langchain.memory.chat_message_histories import FileChatMessageHistory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.chat_models import ChatOpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResearchSessionMemory:
    """
    Advanced memory system for research sessions that:
    - Maintains conversation context across sessions
    - Learns user research patterns and preferences
    - Stores research findings and insights
    - Manages multiple research projects simultaneously
    """
    
    def __init__(self, session_id: str, memory_dir: str = None, llm_model: str = "gpt-4o-mini"):
        self.session_id = session_id
        self.memory_dir = Path(memory_dir) if memory_dir else Path(__file__).parent.parent / 'data' / 'memory'
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # LLM for memory summarization
        self.llm = ChatOpenAI(model_name=llm_model, temperature=0.1)
        
        # File paths
        self.session_file = self.memory_dir / f"session_{session_id}.json"
        self.chat_history_file = self.memory_dir / f"chat_{session_id}.json"
        self.preferences_file = self.memory_dir / "user_preferences.json"
        
        # Initialize memory components
        self._setup_memory_components()
        self._load_session_data()
        
        logger.info(f"Research Session Memory initialized for session: {session_id}")
    
    def _setup_memory_components(self):
        """Setup LangChain memory components"""
        # File-based chat history
        self.chat_history = FileChatMessageHistory(str(self.chat_history_file))
        
        # Summary buffer memory for long conversations
        self.summary_memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            chat_memory=self.chat_history,
            max_token_limit=2000,
            return_messages=True
        )
        
        # Window memory for recent context
        self.window_memory = ConversationBufferWindowMemory(
            chat_memory=self.chat_history,
            k=5,  # Keep last 5 exchanges
            return_messages=True
        )
        
        # Research context storage
        self.research_context = {
            "current_topics": [],
            "active_documents": [],
            "research_questions": [],
            "findings": [],
            "insights": [],
            "connections": []
        }
    
    def _load_session_data(self):
        """Load existing session data"""
        try:
            if self.session_file.exists():
                with open(self.session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.research_context.update(data.get('research_context', {}))
                    self.session_metadata = data.get('metadata', {})
            else:
                self.session_metadata = {
                    'created_at': datetime.now().isoformat(),
                    'last_accessed': datetime.now().isoformat(),
                    'total_interactions': 0,
                    'research_duration': 0
                }
        except Exception as e:
            logger.error(f"Error loading session data: {e}")
    
    def _save_session_data(self):
        """Save current session data"""
        try:
            session_data = {
                'session_id': self.session_id,
                'research_context': self.research_context,
                'metadata': {
                    **self.session_metadata,
                    'last_accessed': datetime.now().isoformat()
                }
            }
            
            with open(self.session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Error saving session data: {e}")
    
    def add_interaction(self, user_input: str, ai_response: str, 
                       context_data: Dict[str, Any] = None):
        """Add a new interaction to memory"""
        try:
            # Add to chat history
            self.chat_history.add_user_message(user_input)
            self.chat_history.add_ai_message(ai_response)
            
            # Update research context
            if context_data:
                self._update_research_context(context_data)
            
            # Update metadata
            self.session_metadata['total_interactions'] += 1
            self.session_metadata['last_accessed'] = datetime.now().isoformat()
            
            # Save session data
            self._save_session_data()
            
            logger.info(f"Added interaction to memory for session: {self.session_id}")
            
        except Exception as e:
            logger.error(f"Error adding interaction: {e}")
    
    def _update_research_context(self, context_data: Dict[str, Any]):
        """Update research context with new information"""
        try:
            # Extract and update research topics
            if 'topics' in context_data:
                for topic in context_data['topics']:
                    if topic not in self.research_context['current_topics']:
                        self.research_context['current_topics'].append(topic)
            
            # Update active documents
            if 'document' in context_data:
                doc_info = {
                    'name': context_data['document'],
                    'accessed_at': datetime.now().isoformat()
                }
                self.research_context['active_documents'].append(doc_info)
            
            # Update research questions
            if 'question' in context_data:
                question_info = {
                    'question': context_data['question'],
                    'timestamp': datetime.now().isoformat()
                }
                self.research_context['research_questions'].append(question_info)
            
            # Update findings
            if 'finding' in context_data:
                finding_info = {
                    'finding': context_data['finding'],
                    'source': context_data.get('source', 'unknown'),
                    'confidence': context_data.get('confidence', 'medium'),
                    'timestamp': datetime.now().isoformat()
                }
                self.research_context['findings'].append(finding_info)
            
            # Update insights
            if 'insight' in context_data:
                insight_info = {
                    'insight': context_data['insight'],
                    'related_topics': context_data.get('related_topics', []),
                    'timestamp': datetime.now().isoformat()
                }
                self.research_context['insights'].append(insight_info)
                
        except Exception as e:
            logger.error(f"Error updating research context: {e}")
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation"""
        try:
            return self.summary_memory.predict_new_summary(
                self.summary_memory.chat_memory.messages,
                ""
            )
        except Exception as e:
            logger.error(f"Error getting conversation summary: {e}")
            return "Özet oluşturulamadı"
    
    def get_recent_context(self, k: int = 3) -> List[BaseMessage]:
        """Get recent conversation messages"""
        try:
            messages = self.window_memory.chat_memory.messages
            return messages[-k*2:] if messages else []  # k exchanges = k*2 messages
        except Exception as e:
            logger.error(f"Error getting recent context: {e}")
            return []
    
    def get_research_summary(self) -> Dict[str, Any]:
        """Get a comprehensive research summary"""
        try:
            summary = {
                "session_info": {
                    "session_id": self.session_id,
                    "total_interactions": self.session_metadata.get('total_interactions', 0),
                    "last_accessed": self.session_metadata.get('last_accessed'),
                    "duration": self._calculate_session_duration()
                },
                "research_overview": {
                    "active_topics": len(self.research_context.get('current_topics', [])),
                    "documents_reviewed": len(self.research_context.get('active_documents', [])),
                    "questions_explored": len(self.research_context.get('research_questions', [])),
                    "findings_collected": len(self.research_context.get('findings', [])),
                    "insights_generated": len(self.research_context.get('insights', []))
                },
                "current_focus": {
                    "main_topics": self.research_context.get('current_topics', [])[-3:],
                    "recent_documents": [doc['name'] for doc in self.research_context.get('active_documents', [])[-3:]],
                    "latest_questions": [q['question'] for q in self.research_context.get('research_questions', [])[-3:]]
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating research summary: {e}")
            return {"error": str(e)}
    
    def _calculate_session_duration(self) -> int:
        """Calculate session duration in minutes"""
        try:
            created = datetime.fromisoformat(self.session_metadata.get('created_at', datetime.now().isoformat()))
            last_access = datetime.fromisoformat(self.session_metadata.get('last_accessed', datetime.now().isoformat()))
            duration = (last_access - created).total_seconds() / 60
            return int(duration)
        except:
            return 0
    
    def search_memory(self, query: str, search_type: str = "all") -> List[Dict[str, Any]]:
        """
        Search through memory for relevant information
        
        Args:
            query: Search query
            search_type: Type of search (all, findings, insights, questions)
        """
        try:
            results = []
            query_lower = query.lower()
            
            if search_type in ["all", "findings"]:
                # Search findings
                for finding in self.research_context.get('findings', []):
                    if query_lower in finding['finding'].lower():
                        results.append({
                            "type": "finding",
                            "content": finding['finding'],
                            "source": finding.get('source'),
                            "timestamp": finding['timestamp']
                        })
            
            if search_type in ["all", "insights"]:
                # Search insights
                for insight in self.research_context.get('insights', []):
                    if query_lower in insight['insight'].lower():
                        results.append({
                            "type": "insight",
                            "content": insight['insight'],
                            "related_topics": insight.get('related_topics', []),
                            "timestamp": insight['timestamp']
                        })
            
            if search_type in ["all", "questions"]:
                # Search questions
                for question in self.research_context.get('research_questions', []):
                    if query_lower in question['question'].lower():
                        results.append({
                            "type": "question",
                            "content": question['question'],
                            "timestamp": question['timestamp']
                        })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching memory: {e}")
            return []
    
    def clear_session(self):
        """Clear current session data"""
        try:
            # Clear chat history
            self.chat_history.clear()
            
            # Reset research context
            self.research_context = {
                "current_topics": [],
                "active_documents": [],
                "research_questions": [],
                "findings": [],
                "insights": [],
                "connections": []
            }
            
            # Reset metadata
            self.session_metadata = {
                'created_at': datetime.now().isoformat(),
                'last_accessed': datetime.now().isoformat(),
                'total_interactions': 0,
                'research_duration': 0
            }
            
            # Save cleared state
            self._save_session_data()
            
            logger.info(f"Cleared session: {self.session_id}")
            
        except Exception as e:
            logger.error(f"Error clearing session: {e}")
    
    def get_contextual_prompt_addition(self) -> str:
        """Get additional context to add to prompts"""
        try:
            # Get recent topics and findings for context
            recent_topics = self.research_context.get('current_topics', [])[-3:]
            recent_findings = [f['finding'] for f in self.research_context.get('findings', [])][-2:]
            
            context_addition = ""
            
            if recent_topics:
                context_addition += f"\\n\\nAraştırma Bağlamı - Aktif Konular: {', '.join(recent_topics)}"
            
            if recent_findings:
                context_addition += f"\\n\\nÖnceki Bulgular: {' | '.join(recent_findings)}"
            
            return context_addition
            
        except Exception as e:
            logger.error(f"Error generating contextual prompt addition: {e}")
            return ""
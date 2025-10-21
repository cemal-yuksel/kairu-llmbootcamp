"""
Streaming Handlers - Real-time streaming interface for academic research assistant
Provides progressive response display and progress tracking
"""
import logging
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResearchStreamingHandler(BaseCallbackHandler):
    """
    Advanced streaming handler for academic research operations that:
    - Provides real-time token streaming
    - Shows progress indicators for long operations
    - Handles multi-step chain execution display
    - Manages error reporting and recovery
    """
    
    def __init__(self, session_id: str = None, ui_callback=None):
        self.session_id = session_id or f"stream_{int(time.time())}"
        self.ui_callback = ui_callback  # Function to update UI
        self.current_step = ""
        self.total_steps = 0
        self.completed_steps = 0
        self.start_time = None
        self.tokens_streamed = 0
        self.current_response = ""
        
        # Stream buffer
        self.stream_buffer = []
        self.buffer_size = 50  # tokens
        
        logger.info(f"Research Streaming Handler initialized for session: {self.session_id}")
    
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Called when LLM starts generating"""
        self.start_time = datetime.now()
        self.tokens_streamed = 0
        self.current_response = ""
        
        if self.ui_callback:
            self.ui_callback({
                "type": "llm_start",
                "message": "🤖 AI düşünüyor...",
                "timestamp": self.start_time.isoformat()
            })
        
        logger.info(f"LLM generation started for session: {self.session_id}")
    
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Called for each new token generated"""
        self.tokens_streamed += 1
        self.current_response += token
        self.stream_buffer.append(token)
        
        # Flush buffer when full or at sentence boundaries
        should_flush = (
            len(self.stream_buffer) >= self.buffer_size or
            token in ['.', '!', '?', '\n'] or
            self.tokens_streamed % 10 == 0  # Periodic flush
        )
        
        if should_flush:
            self._flush_buffer()
    
    def _flush_buffer(self):
        """Flush the token buffer to UI"""
        if self.stream_buffer and self.ui_callback:
            buffer_text = ''.join(self.stream_buffer)
            
            self.ui_callback({
                "type": "token_stream",
                "content": buffer_text,
                "tokens_count": self.tokens_streamed,
                "timestamp": datetime.now().isoformat()
            })
            
            self.stream_buffer = []
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Called when LLM finishes generating"""
        # Flush any remaining tokens
        if self.stream_buffer:
            self._flush_buffer()
        
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds() if self.start_time else 0
        
        if self.ui_callback:
            self.ui_callback({
                "type": "llm_end",
                "message": "✅ Yanıt tamamlandı",
                "duration": duration,
                "tokens_generated": self.tokens_streamed,
                "timestamp": end_time.isoformat()
            })
        
        logger.info(f"LLM generation completed. Tokens: {self.tokens_streamed}, Duration: {duration:.2f}s")
    
    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        """Called when LLM encounters an error"""
        if self.ui_callback:
            self.ui_callback({
                "type": "llm_error",
                "message": f"❌ Hata oluştu: {str(error)}",
                "error": str(error),
                "timestamp": datetime.now().isoformat()
            })
        
        logger.error(f"LLM error in session {self.session_id}: {error}")
    
    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Called when a chain starts execution"""
        chain_name = serialized.get("name", "Unknown Chain")
        self.current_step = chain_name
        
        if self.ui_callback:
            self.ui_callback({
                "type": "chain_start",
                "message": f"🔗 {chain_name} başlatılıyor...",
                "chain_name": chain_name,
                "inputs": inputs,
                "timestamp": datetime.now().isoformat()
            })
        
        logger.info(f"Chain started: {chain_name}")
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Called when a chain ends execution"""
        self.completed_steps += 1
        
        if self.ui_callback:
            progress = (self.completed_steps / max(self.total_steps, 1)) * 100
            
            self.ui_callback({
                "type": "chain_end",
                "message": f"✅ {self.current_step} tamamlandı",
                "progress": progress,
                "completed_steps": self.completed_steps,
                "total_steps": self.total_steps,
                "timestamp": datetime.now().isoformat()
            })
        
        logger.info(f"Chain completed: {self.current_step} ({self.completed_steps}/{self.total_steps})")
    
    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Called when a chain encounters an error"""
        if self.ui_callback:
            self.ui_callback({
                "type": "chain_error",
                "message": f"❌ {self.current_step} hatası: {str(error)}",
                "error": str(error),
                "chain_name": self.current_step,
                "timestamp": datetime.now().isoformat()
            })
        
        logger.error(f"Chain error in {self.current_step}: {error}")
    
    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Called when a tool starts execution"""
        tool_name = serialized.get("name", "Unknown Tool")
        
        if self.ui_callback:
            self.ui_callback({
                "type": "tool_start",
                "message": f"🛠️ {tool_name} kullanılıyor...",
                "tool_name": tool_name,
                "input": input_str,
                "timestamp": datetime.now().isoformat()
            })
        
        logger.info(f"Tool started: {tool_name}")
    
    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Called when a tool ends execution"""
        if self.ui_callback:
            self.ui_callback({
                "type": "tool_end",
                "message": "✅ Tool tamamlandı",
                "output_preview": output[:100] + "..." if len(output) > 100 else output,
                "timestamp": datetime.now().isoformat()
            })
        
        logger.info("Tool execution completed")
    
    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Called when a tool encounters an error"""
        if self.ui_callback:
            self.ui_callback({
                "type": "tool_error",
                "message": f"❌ Tool hatası: {str(error)}",
                "error": str(error),
                "timestamp": datetime.now().isoformat()
            })
        
        logger.error(f"Tool error: {error}")
    
    def on_text(self, text: str, **kwargs: Any) -> None:
        """Called on arbitrary text"""
        if self.ui_callback:
            self.ui_callback({
                "type": "text_update",
                "content": text,
                "timestamp": datetime.now().isoformat()
            })
    
    def set_total_steps(self, total: int):
        """Set the total number of expected steps"""
        self.total_steps = total
        self.completed_steps = 0
    
    def update_progress(self, step_name: str, progress: float = None):
        """Manually update progress"""
        self.current_step = step_name
        
        if self.ui_callback:
            self.ui_callback({
                "type": "progress_update",
                "message": f"📊 {step_name}",
                "progress": progress,
                "step_name": step_name,
                "timestamp": datetime.now().isoformat()
            })
    
    def get_stream_stats(self) -> Dict[str, Any]:
        """Get streaming statistics"""
        return {
            "session_id": self.session_id,
            "tokens_streamed": self.tokens_streamed,
            "completed_steps": self.completed_steps,
            "total_steps": self.total_steps,
            "current_step": self.current_step,
            "start_time": self.start_time.isoformat() if self.start_time else None
        }

class ProgressTracker:
    """
    Progress tracking system for long-running research operations
    """
    
    def __init__(self, operation_name: str = "Research Operation"):
        self.operation_name = operation_name
        self.start_time = datetime.now()
        self.stages = []
        self.current_stage = None
        self.progress_callbacks = []
    
    def add_stage(self, stage_name: str, estimated_duration: int = None):
        """Add a stage to the operation"""
        stage = {
            "name": stage_name,
            "status": "pending",
            "start_time": None,
            "end_time": None,
            "estimated_duration": estimated_duration,
            "progress": 0
        }
        self.stages.append(stage)
    
    def start_stage(self, stage_name: str):
        """Start a specific stage"""
        for stage in self.stages:
            if stage["name"] == stage_name:
                stage["status"] = "running"
                stage["start_time"] = datetime.now()
                self.current_stage = stage
                self._notify_progress()
                break
    
    def complete_stage(self, stage_name: str):
        """Complete a specific stage"""
        for stage in self.stages:
            if stage["name"] == stage_name:
                stage["status"] = "completed"
                stage["end_time"] = datetime.now()
                stage["progress"] = 100
                self._notify_progress()
                break
    
    def update_stage_progress(self, stage_name: str, progress: float):
        """Update progress of a specific stage"""
        for stage in self.stages:
            if stage["name"] == stage_name:
                stage["progress"] = min(100, max(0, progress))
                self._notify_progress()
                break
    
    def add_progress_callback(self, callback):
        """Add a callback function for progress updates"""
        self.progress_callbacks.append(callback)
    
    def _notify_progress(self):
        """Notify all progress callbacks"""
        progress_data = self.get_progress_summary()
        
        for callback in self.progress_callbacks:
            try:
                callback(progress_data)
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get comprehensive progress summary"""
        completed_stages = len([s for s in self.stages if s["status"] == "completed"])
        total_stages = len(self.stages)
        overall_progress = (completed_stages / total_stages * 100) if total_stages > 0 else 0
        
        # Calculate estimated time remaining
        if self.current_stage and self.current_stage.get("estimated_duration"):
            elapsed = (datetime.now() - self.current_stage["start_time"]).total_seconds()
            estimated_total = self.current_stage["estimated_duration"]
            progress_ratio = self.current_stage["progress"] / 100
            
            if progress_ratio > 0:
                estimated_remaining = (estimated_total - elapsed) * (1 - progress_ratio)
            else:
                estimated_remaining = estimated_total
        else:
            estimated_remaining = None
        
        return {
            "operation_name": self.operation_name,
            "overall_progress": overall_progress,
            "completed_stages": completed_stages,
            "total_stages": total_stages,
            "current_stage": self.current_stage["name"] if self.current_stage else None,
            "estimated_remaining_seconds": estimated_remaining,
            "stages": self.stages,
            "start_time": self.start_time.isoformat(),
            "elapsed_seconds": (datetime.now() - self.start_time).total_seconds()
        }
    
    def is_complete(self) -> bool:
        """Check if all stages are completed"""
        return all(stage["status"] == "completed" for stage in self.stages)
    
    def has_errors(self) -> bool:
        """Check if any stages have errors"""
        return any(stage["status"] == "error" for stage in self.stages)
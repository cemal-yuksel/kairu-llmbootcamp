"""
Advanced Academic Research Assistant v2.0
Enhanced version with LangChain integration, memory systems, and streaming capabilities

Main application file that integrates all components:
- Multi-chain research analysis
- Advanced memory management  
- Custom academic tools
- Real-time streaming interface
- Project management
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import json

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / 'logs' / 'app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Create logs directory
(project_root / 'logs').mkdir(exist_ok=True)

logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import project components
try:
    from chains.research_chains import ResearchAnalysisChain
    from chains.writing_chains import AcademicWritingChain
    from chains.analysis_chains import DocumentAnalysisChain
    
    from memory.research_memory import ResearchSessionMemory
    from memory.project_memory import ProjectMemory
    
    from tools.pdf_manager import EnhancedPDFManager
    from tools.vector_db import EnhancedVectorDB
    from tools.literature_tool import LiteratureSearchTool, CitationManagerTool
    from tools.reference_tool import ReferenceManagerTool
    from tools.citation_manager import citation_manager
    from tools.article_analyzer import article_analyzer
    
    from streaming.handlers import ResearchStreamingHandler, ProgressTracker
    
except ImportError as e:
    logger.error(f"Error importing components: {e}")
    print(f"Import Error: {e}")
    print("Make sure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

class AdvancedAcademicAssistant:
    """
    Main class that orchestrates the advanced academic research assistant
    """
    
    def __init__(self, session_id: str = None):
        self.session_id = session_id or f"session_{int(datetime.now().timestamp())}"
        
        logger.info(f"Initializing Advanced Academic Assistant - Session: {self.session_id}")
        
        # Initialize core components
        self._initialize_components()
        
        # Setup streaming
        self.streaming_handler = ResearchStreamingHandler(
            session_id=self.session_id,
            ui_callback=self._handle_stream_update
        )
        
        logger.info("Advanced Academic Assistant initialized successfully")
    
    def _initialize_components(self):
        """Initialize all system components"""
        try:
            # Memory systems
            self.research_memory = ResearchSessionMemory(self.session_id)
            self.project_memory = ProjectMemory()
            
            # Document processing
            self.pdf_manager = EnhancedPDFManager()
            self.vector_db = EnhancedVectorDB()
            
            # Analysis chains
            self.research_chain = ResearchAnalysisChain()
            self.writing_chain = AcademicWritingChain()
            self.analysis_chain = DocumentAnalysisChain()
            
            # Academic tools
            self.literature_tool = LiteratureSearchTool()
            self.citation_tool = CitationManagerTool()
            self.reference_tool = ReferenceManagerTool()
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def _handle_stream_update(self, update: Dict[str, Any]):
        """Handle streaming updates (can be overridden by UI)"""
        timestamp = update.get('timestamp', datetime.now().isoformat())
        message = update.get('message', '')
        
        # Simple console output - will be replaced by Streamlit UI
        print(f"[{timestamp[:19]}] {message}")
    
    def process_document(self, pdf_path: str, project_id: str = None) -> Dict[str, Any]:
        """
        Process a PDF document through the complete analysis pipeline
        
        Args:
            pdf_path: Path to PDF file
            project_id: Optional project ID to associate document with
            
        Returns:
            Comprehensive processing results
        """
        try:
            logger.info(f"Processing document: {pdf_path}")
            
            # Setup progress tracking
            tracker = ProgressTracker("Document Processing")
            tracker.add_stage("PDF Text Extraction", 10)
            tracker.add_stage("Vector Indexing", 15)
            tracker.add_stage("Research Analysis", 30)
            tracker.add_stage("Quality Analysis", 25)
            tracker.add_stage("Memory Integration", 10)
            
            results = {"document_path": pdf_path, "processing_stages": {}}
            
            # Stage 1: Extract text and metadata
            tracker.start_stage("PDF Text Extraction")
            text, metadata = self.pdf_manager.process_pdf(pdf_path)
            
            if not text:
                return {"error": "Failed to extract text from PDF"}
            
            results["processing_stages"]["extraction"] = {
                "success": True,
                "text_length": len(text),
                "metadata": metadata
            }
            tracker.complete_stage("PDF Text Extraction")
            
            # Stage 2: Add to vector database
            tracker.start_stage("Vector Indexing")
            pdf_name = Path(pdf_path).name
            self.vector_db.add_document(text, metadata, pdf_name)
            
            results["processing_stages"]["indexing"] = {
                "success": True,
                "pdf_name": pdf_name
            }
            tracker.complete_stage("Vector Indexing")
            
            # Stage 3: Research analysis
            tracker.start_stage("Research Analysis")
            research_analysis = self.research_chain.analyze_document(
                text, 
                metadata.get('title', 'Unknown Title'),
                callbacks=[self.streaming_handler]
            )
            
            results["processing_stages"]["research_analysis"] = research_analysis
            tracker.complete_stage("Research Analysis")
            
            # Stage 4: Quality analysis
            tracker.start_stage("Quality Analysis")
            document_data = {
                "text": text,
                "title": metadata.get('title', 'Unknown Title'),
                "abstract": "",  # Could extract from text
                "research_type": research_analysis.get("categorization", {}).get("research_type", "Unknown")
            }
            
            quality_analysis = self.analysis_chain.analyze_document(
                document_data,
                callbacks=[self.streaming_handler]
            )
            
            results["processing_stages"]["quality_analysis"] = quality_analysis
            tracker.complete_stage("Quality Analysis")
            
            # Stage 5: Update memory systems
            tracker.start_stage("Memory Integration")
            
            # Add to research memory
            context_data = {
                "document": pdf_name,
                "topics": [research_analysis.get("categorization", {}).get("research_field", "")],
                "finding": research_analysis.get("findings_analysis", {}).get("main_findings", [{}])[0] if research_analysis.get("findings_analysis", {}).get("main_findings") else ""
            }
            
            self.research_memory.add_interaction(
                user_input=f"Processed document: {pdf_name}",
                ai_response="Document processed successfully",
                context_data=context_data
            )
            
            # Add to project if specified
            if project_id:
                self.project_memory.add_resource(
                    project_id=project_id,
                    name=metadata.get('title', pdf_name),
                    resource_type="pdf",
                    path=pdf_path,
                    summary=research_analysis.get("categorization", {}).get("research_field", "")
                )
            
            results["processing_stages"]["memory_integration"] = {
                "success": True,
                "project_associated": bool(project_id)
            }
            tracker.complete_stage("Memory Integration")
            
            # Generate summary
            results["summary"] = self._generate_processing_summary(results)
            results["processing_complete"] = True
            
            logger.info(f"Document processing completed: {pdf_name}")
            return results
            
        except Exception as e:
            error_msg = f"Error processing document: {e}"
            logger.error(error_msg)
            return {"error": error_msg, "processing_complete": False}
    
    def ask_question(self, question: str, pdf_names: List[str] = None, 
                    use_memory: bool = True) -> Dict[str, Any]:
        """
        Ask a research question with context from documents and memory
        
        Args:
            question: Research question to ask
            pdf_names: Specific PDFs to search (optional)
            use_memory: Whether to use research memory context
            
        Returns:
            Comprehensive answer with sources and analysis
        """
        try:
            logger.info(f"Processing question: {question}")
            
            # Search relevant documents
            documents, metadatas = self.vector_db.search_documents(
                query=question,
                pdf_names=pdf_names,
                top_k=5
            )
            
            if not documents:
                return {
                    "answer": "İlgili dokumanlarda bu soruya yanıt bulunamadı.",
                    "sources": [],
                    "confidence": "low"
                }
            
            # Get memory context if requested
            memory_context = ""
            if use_memory:
                memory_context = self.research_memory.get_contextual_prompt_addition()
            
            # Enhanced prompt with context
            enhanced_question = question + memory_context
            
            # Generate enhanced answer with citations
            answer, citations = self._generate_enhanced_answer(
                question=enhanced_question,
                documents=documents,
                metadatas=metadatas
            )
            
            # Add to memory
            if use_memory:
                context_data = {
                    "question": question,
                    "finding": answer[:200],
                    "topics": [question.split()[0] if question.split() else "general"]
                }
                
                self.research_memory.add_interaction(
                    user_input=question,
                    ai_response=answer,
                    context_data=context_data
                )
            
            result = {
                "answer": answer,
                "sources": citations,
                "documents_searched": len(documents),
                "confidence": "high",
                "memory_used": use_memory,
                "question_timestamp": datetime.now().isoformat(),
                "citations": citations
            }
            
            logger.info(f"Question answered successfully")
            return result
            
        except Exception as e:
            error_msg = f"Error answering question: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def search_literature(self, query: str) -> Dict[str, Any]:
        """Search academic literature using the literature tool"""
        try:
            logger.info(f"Literature search: {query}")
            
            results = self.literature_tool._run(
                query=query,
                run_manager=None
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Literature search error: {e}")
            return {"error": str(e)}
    
    def manage_references(self, command: str) -> Dict[str, Any]:
        """Manage references using the reference tool"""
        try:
            logger.info(f"Reference management: {command}")
            
            results = self.reference_tool._run(
                query=command,
                run_manager=None
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Reference management error: {e}")
            return {"error": str(e)}
    
    def get_research_summary(self) -> Dict[str, Any]:
        """Get comprehensive research session summary"""
        try:
            research_summary = self.research_memory.get_research_summary()
            project_analytics = self.project_memory.get_global_analytics()
            vector_stats = self.vector_db.get_document_stats()
            
            summary = {
                "session_summary": research_summary,
                "project_analytics": project_analytics,
                "document_stats": vector_stats,
                "capabilities": {
                    "document_processing": True,
                    "literature_search": True,
                    "reference_management": True,
                    "memory_systems": True,
                    "streaming_interface": True
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating research summary: {e}")
            return {"error": str(e)}
    
    def _generate_enhanced_answer(self, question: str, documents: List[str], 
                                metadatas: List[Dict]) -> tuple[str, List[Dict]]:
        """
        Generate enhanced answer with proper APA7 citations
        """
        try:
            from openai import OpenAI
            
            # Prepare context
            context = "\n\n".join(documents)
            
            # Create in-text citations
            in_text_citations = []
            citations_data = []
            
            for meta in metadatas:
                pdf_name = meta.get('pdf_name', 'Unknown')
                if pdf_name != 'Unknown':
                    citation = citation_manager.create_in_text_citation(pdf_name, meta)
                    in_text_citations.append(citation)
                    citations_data.append({
                        'pdf_name': pdf_name,
                        'metadata': meta
                    })
            
            citations_text = ", ".join(set(in_text_citations)) if in_text_citations else ""
            
            # Enhanced academic prompt
            prompt = f"""Sen bir akademik araştırma uzmanısın. Aşağıdaki soru ve bağlam bilgisine göre MUTLAKA ŞU KURALLARA UYARAK detaylı bir akademik yanıt oluştur:

KURALLAR:
1. YANIT UZUNLUĞU: Minimum 3 paragraf, 12 cümle olmalıdır. Her paragraf en az 4 cümle içermelidir.
2. AKADEMİK DİL: Profesyonel, bilimsel ve objektif bir dil kullan.
3. DETAYLI AÇIKLAMA: Her noktayı örneklerle ve detaylılarıyla açıkla.
4. YAPILANDIRMA: Giriş, gelişme ve sonuç paragrafları oluştur.
5. APA7 ALINTI: Metin içinde mutlaka şu alıntıları kullan: {citations_text}
6. ALINTI KULLANIMI: Her paragrafta en az bir alıntı kullan.
7. İKİNCİL KAYNAK: Eğer bir makalede başka bir makaleye atıf varsa, "(Özgün Yazar, Yıl, Aktaran Makale'de belirtildiği gibi)" formatında yaz.

Soru: {question}

Bağlam (Akademik Makalelerden):
{context}

DETAYLI AKADEMİK YANIT (Minimum 3 paragraf, 12 cümle + APA7 metin içi alıntılar):"""

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system", 
                        "content": """Sen bir akademik araştırma uzmanısın. Detaylı, kapsamlı ve bilimsel yanıtlar üretirsin. 
                        Her yanıt en az 3 paragraf ve 12 cümle içermelidir. 
                        APA7 formatında metin içi alıntıları mutlaka kullanmalısın.
                        İkincil kaynaklar için '(Özgün Yazar, Yıl, Aktaran Makale'de belirtildiği gibi)' formatını kullan."""
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            answer = response.choices[0].message.content
            
            return answer, citations_data
            
        except Exception as e:
            logger.error(f"Error generating enhanced answer: {e}")
            
            # Fallback answer
            fallback_answer = f"""
Bu soruya ilişkin bulgular akademik kaynaklarda incelenmiştir. 

Araştırmalar, konuyla ilgili önemli bulgular ortaya koymaktadır. 

Daha detaylı bilgi için kaynak metinleri inceleyebilirsiniz.

Kaynak: {metadatas[0].get('pdf_name', 'Bilinmeyen kaynak') if metadatas else 'Kaynak bulunamadı'}
"""
            
            return fallback_answer, citations_data if 'citations_data' in locals() else []
    
    def _generate_processing_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of document processing results"""
        try:
            summary = {
                "document": results.get("document_path", "Unknown"),
                "success": results.get("processing_complete", False),
                "stages_completed": len([
                    stage for stage in results.get("processing_stages", {}).values()
                    if stage.get("success", False)
                ]),
                "key_insights": [],
                "recommendations": []
            }
            
            # Extract key insights from analysis
            research_analysis = results.get("processing_stages", {}).get("research_analysis", {})
            if "categorization" in research_analysis:
                cat = research_analysis["categorization"]
                summary["key_insights"].append(
                    f"Research Field: {cat.get('research_field', 'Unknown')}"
                )
                summary["key_insights"].append(
                    f"Novelty Score: {cat.get('novelty_score', 'N/A')}/10"
                )
            
            # Add recommendations
            quality_analysis = results.get("processing_stages", {}).get("quality_analysis", {})
            if "overall_analysis" in quality_analysis:
                overall = quality_analysis["overall_analysis"]
                summary["recommendations"].append(
                    overall.get("recommendation", "No specific recommendations")
                )
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating processing summary: {e}")
            return {"error": str(e)}

def main():
    """Main function for command-line testing"""
    print("🚀 Advanced Academic Research Assistant v2.0")
    print("=" * 50)
    
    try:
        # Initialize assistant
        assistant = AdvancedAcademicAssistant()
        
        # Test basic functionality
        print("\n📊 System Status:")
        summary = assistant.get_research_summary()
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        
        # Interactive mode
        print("\n💬 Interactive Mode (type 'quit' to exit)")
        print("Commands:")
        print("- process <pdf_path> : Process a PDF document")
        print("- ask <question> : Ask a research question") 
        print("- search <query> : Search literature")
        print("- summary : Get research summary")
        
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                elif user_input.startswith('process '):
                    pdf_path = user_input[8:].strip()
                    if os.path.exists(pdf_path):
                        print("📄 Processing document...")
                        result = assistant.process_document(pdf_path)
                        print(json.dumps(result, indent=2, ensure_ascii=False))
                    else:
                        print(f"❌ File not found: {pdf_path}")
                
                elif user_input.startswith('ask '):
                    question = user_input[4:].strip()
                    print("🤔 Answering question...")
                    result = assistant.ask_question(question)
                    print(json.dumps(result, indent=2, ensure_ascii=False))
                
                elif user_input.startswith('search '):
                    query = user_input[7:].strip()
                    print("🔍 Searching literature...")
                    result = assistant.search_literature(query)
                    print(json.dumps(result, indent=2, ensure_ascii=False))
                
                elif user_input == 'summary':
                    print("📈 Generating summary...")
                    result = assistant.get_research_summary()
                    print(json.dumps(result, indent=2, ensure_ascii=False))
                
                else:
                    print("❓ Unknown command. Type 'quit' to exit.")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("\n👋 Session ended. Thank you for using Advanced Academic Research Assistant!")
        
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
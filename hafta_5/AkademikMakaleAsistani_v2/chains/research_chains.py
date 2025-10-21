"""
Research Analysis Chain - Advanced academic research analysis system
Multi-step chain for comprehensive document analysis and research insights
"""
import logging
from typing import Dict, List, Optional, Any
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResearchAnalysisOutputParser(BaseOutputParser):
    """Custom parser for research analysis output"""
    
    def parse(self, text: str) -> Dict[str, Any]:
        try:
            # Try to parse as JSON first
            if text.strip().startswith('{'):
                return json.loads(text)
            
            # Fallback: parse structured text
            lines = text.strip().split('\n')
            result = {}
            
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    result[key.strip().lower().replace(' ', '_')] = value.strip()
            
            return result
        except Exception as e:
            logger.warning(f"Error parsing output: {e}")
            return {"raw_output": text}

class ResearchAnalysisChain:
    """
    Multi-step research analysis chain that processes academic documents
    and provides comprehensive analysis including:
    - Document categorization
    - Methodology extraction
    - Key findings identification
    - Research gap analysis
    """
    
    def __init__(self, llm_model: str = "gpt-4o", temperature: float = 0.3):
        self.llm = ChatOpenAI(model_name=llm_model, temperature=temperature)
        self.output_parser = ResearchAnalysisOutputParser()
        
        # Define analysis steps
        self._setup_analysis_chains()
        self._setup_sequential_chain()
        
        logger.info("Research Analysis Chain initialized")
    
    def _setup_analysis_chains(self):
        """Setup individual analysis chains"""
        
        # 1. Document Categorization Chain
        categorization_prompt = PromptTemplate(
            input_variables=["document_text", "title"],
            template="""
            Aşağıdaki akademik makaleyi analiz ederek detaylı kategorizasyon yap:

            Başlık: {title}
            
            Makale Metni: {document_text}

            Lütfen aşağıdaki formatta JSON çıktısı ver:
            {{
                "research_field": "Ana araştırma alanı",
                "sub_discipline": "Alt disiplin",
                "research_type": "Teorik/Ampirik/Meta-analiz/Derleme",
                "methodology": "Kullanılan metodoloji",
                "target_audience": "Hedef kitle",
                "academic_level": "Lisans/Yüksek Lisans/Doktora/Post-doc",
                "novelty_score": "1-10 arası yenilik skoru",
                "complexity_level": "Başlangıç/Orta/İleri/Uzman"
            }}
            """
        )
        
        self.categorization_chain = LLMChain(
            llm=self.llm,
            prompt=categorization_prompt,
            output_key="categorization",
            output_parser=self.output_parser
        )
        
        # 2. Methodology Extraction Chain
        methodology_prompt = PromptTemplate(
            input_variables=["document_text"],
            template="""
            Bu akademik makalenin metodoloji bölümünü analiz et ve aşağıdaki bilgileri çıkar:

            Makale: {document_text}

            JSON formatında çıktı ver:
            {{
                "research_design": "Araştırma deseni",
                "data_collection": "Veri toplama yöntemi",
                "sample_size": "Örneklem büyüklüğü",
                "analysis_methods": ["Analiz yöntemleri listesi"],
                "tools_used": ["Kullanılan araçlar"],
                "limitations": ["Metodolojik sınırlılıklar"],
                "validity_measures": ["Geçerlilik önlemleri"],
                "ethical_considerations": "Etik hususlar"
            }}
            """
        )
        
        self.methodology_chain = LLMChain(
            llm=self.llm,
            prompt=methodology_prompt,
            output_key="methodology_analysis",
            output_parser=self.output_parser
        )
        
        # 3. Key Findings Extraction Chain
        findings_prompt = PromptTemplate(
            input_variables=["document_text", "categorization"],
            template="""
            Bu makalenin ana bulgularını ve katkılarını analiz et:

            Makale: {document_text}
            Kategori Bilgisi: {categorization}

            JSON formatında detaylı analiz:
            {{
                "main_findings": ["Ana bulgular listesi"],
                "statistical_results": ["İstatistiksel sonuçlar"],
                "theoretical_contributions": ["Teorik katkılar"],
                "practical_implications": ["Pratik uygulamalar"],
                "future_research": ["Gelecek araştırma önerileri"],
                "significance_level": "Düşük/Orta/Yüksek/Çok Yüksek",
                "citation_potential": "1-10 arası atıf potansiyeli",
                "reproducibility": "Tekrarlanabilirlik değerlendirmesi"
            }}
            """
        )
        
        self.findings_chain = LLMChain(
            llm=self.llm,
            prompt=findings_prompt,
            output_key="findings_analysis",
            output_parser=self.output_parser
        )
        
        # 4. Research Gap Analysis Chain
        gap_analysis_prompt = PromptTemplate(
            input_variables=["document_text", "categorization", "findings_analysis"],
            template="""
            Bu makaleyi temel alarak literatürdeki boşlukları ve gelecek araştırma fırsatlarını analiz et:

            Makale: {document_text}
            Kategori: {categorization}
            Bulgular: {findings_analysis}

            JSON formatında gap analizi:
            {{
                "identified_gaps": ["Tespit edilen literatür boşlukları"],
                "unexplored_areas": ["Keşfedilmemiş alanlar"],
                "methodological_gaps": ["Metodolojik boşluklar"],
                "theoretical_gaps": ["Teorik boşluklar"],
                "future_directions": ["Gelecek araştırma yönleri"],
                "collaboration_opportunities": ["İşbirliği fırsatları"],
                "interdisciplinary_potential": ["Disiplinlerarası potansiyel"],
                "innovation_score": "1-10 arası yenilik potansiyeli"
            }}
            """
        )
        
        self.gap_analysis_chain = LLMChain(
            llm=self.llm,
            prompt=gap_analysis_prompt,
            output_key="gap_analysis",
            output_parser=self.output_parser
        )
    
    def _setup_sequential_chain(self):
        """Setup the main sequential chain"""
        self.sequential_chain = SequentialChain(
            chains=[
                self.categorization_chain,
                self.methodology_chain,
                self.findings_chain,
                self.gap_analysis_chain
            ],
            input_variables=["document_text", "title"],
            output_variables=["categorization", "methodology_analysis", "findings_analysis", "gap_analysis"],
            verbose=True
        )
    
    def analyze_document(self, document_text: str, title: str, 
                        callbacks: List[BaseCallbackHandler] = None) -> Dict[str, Any]:
        """
        Run complete research analysis on a document
        
        Args:
            document_text: Full text of the academic document
            title: Document title
            callbacks: Optional callback handlers for streaming
            
        Returns:
            Comprehensive analysis results
        """
        try:
            logger.info(f"Starting research analysis for: {title}")
            
            # Run the sequential chain
            results = self.sequential_chain({
                "document_text": document_text[:8000],  # Limit text length
                "title": title
            }, callbacks=callbacks)
            
            # Add metadata
            results["analysis_metadata"] = {
                "document_title": title,
                "text_length": len(document_text),
                "analysis_timestamp": str(logger.handlers[0].formatter.formatTime if logger.handlers else "unknown"),
                "model_used": self.llm.model_name
            }
            
            logger.info("Research analysis completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in research analysis: {e}")
            return {
                "error": str(e),
                "analysis_metadata": {
                    "document_title": title,
                    "failed_at": "research_analysis"
                }
            }
    
    def analyze_multiple_documents(self, documents: List[Dict[str, str]], 
                                 callbacks: List[BaseCallbackHandler] = None) -> List[Dict[str, Any]]:
        """
        Analyze multiple documents in batch
        
        Args:
            documents: List of {"text": str, "title": str} dictionaries
            callbacks: Optional callback handlers
            
        Returns:
            List of analysis results
        """
        results = []
        
        for i, doc in enumerate(documents):
            logger.info(f"Analyzing document {i+1}/{len(documents)}: {doc['title']}")
            
            analysis = self.analyze_document(
                document_text=doc["text"],
                title=doc["title"],
                callbacks=callbacks
            )
            
            results.append(analysis)
        
        return results
    
    def get_analysis_summary(self, analysis_results: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of analysis results
        """
        try:
            summary = f"## Araştırma Analizi Özeti\n\n"
            
            # Categorization summary
            if "categorization" in analysis_results:
                cat = analysis_results["categorization"]
                summary += f"**Araştırma Alanı:** {cat.get('research_field', 'Bilinmiyor')}\n"
                summary += f"**Metodoloji:** {cat.get('methodology', 'Belirtilmemiş')}\n"
                summary += f"**Yenilik Skoru:** {cat.get('novelty_score', 'N/A')}/10\n\n"
            
            # Key findings
            if "findings_analysis" in analysis_results:
                findings = analysis_results["findings_analysis"]
                summary += "**Ana Bulgular:**\n"
                for finding in findings.get("main_findings", [])[:3]:
                    summary += f"• {finding}\n"
                summary += "\n"
            
            # Research gaps
            if "gap_analysis" in analysis_results:
                gaps = analysis_results["gap_analysis"]
                summary += "**Araştırma Fırsatları:**\n"
                for gap in gaps.get("future_directions", [])[:3]:
                    summary += f"• {gap}\n"
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating analysis summary: {e}")
            return "Özet oluşturulurken hata oluştu."
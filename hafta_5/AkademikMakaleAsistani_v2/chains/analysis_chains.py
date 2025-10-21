"""
Analysis Chain - Advanced document analysis and quality assessment system
Multi-step chain for comprehensive document evaluation and feedback
"""
import logging
from typing import Dict, List, Optional, Any
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
import json
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalysisOutputParser(BaseOutputParser):
    """Custom parser for analysis output"""
    
    def parse(self, text: str) -> Dict[str, Any]:
        try:
            if text.strip().startswith('{'):
                return json.loads(text)
            
            # Parse structured analysis
            analysis = {}
            current_section = None
            
            for line in text.strip().split('\n'):
                if any(keyword in line.lower() for keyword in ['analiz', 'değerlendirme', 'skor', 'öneri']):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        analysis[key.strip().lower().replace(' ', '_')] = value.strip()
                
            return analysis if analysis else {"analysis": text}
            
        except Exception as e:
            logger.warning(f"Error parsing analysis output: {e}")
            return {"raw_analysis": text}

class DocumentAnalysisChain:
    """
    Comprehensive document analysis chain that evaluates:
    - Content quality and depth
    - Academic rigor and methodology
    - Citation quality and completeness
    - Structure and organization
    - Language and clarity
    - Originality and contribution
    """
    
    def __init__(self, llm_model: str = "gpt-4o", temperature: float = 0.2):
        self.llm = ChatOpenAI(model_name=llm_model, temperature=temperature)
        self.output_parser = AnalysisOutputParser()
        
        self._setup_analysis_chains()
        self._setup_sequential_chain()
        
        logger.info("Document Analysis Chain initialized")
    
    def _setup_analysis_chains(self):
        """Setup individual analysis chains"""
        
        # 1. Content Quality Analysis Chain
        content_quality_prompt = PromptTemplate(
            input_variables=["document_text", "title", "abstract"],
            template="""
            Bu akademik dokümanın içerik kalitesini kapsamlı olarak analiz et:

            Başlık: {title}
            Özet: {abstract}
            Dokuman: {document_text}

            JSON formatında detaylı kalite analizi:
            {{
                "content_depth": {{
                    "theoretical_foundation": "Teorik temelin gücü (1-10)",
                    "conceptual_clarity": "Kavramsal netlik (1-10)", 
                    "complexity_level": "Karmaşıklık seviyesi",
                    "comprehensive_coverage": "Kapsam genişliği (1-10)"
                }},
                "academic_rigor": {{
                    "evidence_quality": "Kanıt kalitesi (1-10)",
                    "logical_reasoning": "Mantıksal akıl yürütme (1-10)",
                    "critical_analysis": "Eleştirel analiz seviyesi (1-10)",
                    "objectivity": "Objektiflik derecesi (1-10)"
                }},
                "originality": {{
                    "novelty_score": "Yenilik skoru (1-10)",
                    "unique_contribution": "Özgün katkı açıklaması",
                    "innovation_level": "İnovasyon seviyesi",
                    "field_advancement": "Alana katkı potansiyeli (1-10)"
                }},
                "overall_quality_score": "Genel kalite skoru (1-10)",
                "strengths": ["Güçlü yönler"],
                "weaknesses": ["Zayıf yönler"],
                "improvement_suggestions": ["İyileştirme önerileri"]
            }}
            """
        )
        
        self.content_quality_chain = LLMChain(
            llm=self.llm,
            prompt=content_quality_prompt,
            output_key="content_quality_analysis",
            output_parser=self.output_parser
        )
        
        # 2. Methodology Analysis Chain
        methodology_analysis_prompt = PromptTemplate(
            input_variables=["document_text", "research_type"],
            template="""
            Bu çalışmanın metodolojisini detaylı analiz et:

            Dokuman: {document_text}
            Araştırma Türü: {research_type}

            JSON formatında metodoloji analizi:
            {{
                "methodology_evaluation": {{
                    "research_design_quality": "Araştırma deseni kalitesi (1-10)",
                    "data_collection_appropriateness": "Veri toplama uygunluğu (1-10)",
                    "sample_adequacy": "Örneklem yeterliliği (1-10)",
                    "analysis_rigor": "Analiz titizliği (1-10)"
                }},
                "validity_reliability": {{
                    "internal_validity": "İç geçerlilik (1-10)",
                    "external_validity": "Dış geçerlilik (1-10)", 
                    "construct_validity": "Yapı geçerliliği (1-10)",
                    "reliability_measures": "Güvenilirlik önlemleri"
                }},
                "ethical_considerations": {{
                    "ethical_approval": "Etik onay durumu",
                    "participant_protection": "Katılımcı koruması",
                    "bias_mitigation": "Yanlılık önlemleri"
                }},
                "methodology_strengths": ["Metodolojik güçlü yönler"],
                "methodology_limitations": ["Metodolojik sınırlılıklar"],
                "replication_feasibility": "Tekrarlanabilirlik değerlendirmesi"
            }}
            """
        )
        
        self.methodology_analysis_chain = LLMChain(
            llm=self.llm,
            prompt=methodology_analysis_prompt,
            output_key="methodology_evaluation",
            output_parser=self.output_parser
        )
        
        # 3. Citation and Reference Analysis Chain
        citation_analysis_prompt = PromptTemplate(
            input_variables=["document_text", "references"],
            template="""
            Bu dokumandaki atıf ve kaynak kullanımını analiz et:

            Dokuman: {document_text}
            Kaynaklar: {references}

            JSON formatında atıf analizi:
            {{
                "citation_quality": {{
                    "source_diversity": "Kaynak çeşitliliği (1-10)",
                    "recency_score": "Güncellik skoru (1-10)",
                    "authority_level": "Kaynak otoritesi (1-10)",
                    "citation_density": "Atıf yoğunluğu değerlendirmesi"
                }},
                "reference_analysis": {{
                    "total_references": "Toplam kaynak sayısı",
                    "primary_sources": "Birincil kaynak sayısı",
                    "secondary_sources": "İkincil kaynak sayısı",
                    "recent_sources_5yr": "Son 5 yıldaki kaynak sayısı",
                    "seminal_works": "Temel eser sayısı"
                }},
                "citation_patterns": {{
                    "over_citation": "Aşırı atıf problemi",
                    "under_citation": "Yetersiz atıf problemi", 
                    "self_citation_rate": "Özatıf oranı",
                    "citation_balance": "Atıf dengesi değerlendirmesi"
                }},
                "improvement_recommendations": ["Atıf iyileştirme önerileri"]
            }}
            """
        )
        
        self.citation_analysis_chain = LLMChain(
            llm=self.llm,
            prompt=citation_analysis_prompt,
            output_key="citation_evaluation",
            output_parser=self.output_parser
        )
        
        # 4. Structure and Language Analysis Chain
        structure_language_prompt = PromptTemplate(
            input_variables=["document_text", "target_audience"],
            template="""
            Bu dokumandaki yapı ve dil kullanımını analiz et:

            Dokuman: {document_text}
            Hedef Kitle: {target_audience}

            JSON formatında yapı ve dil analizi:
            {{
                "structure_analysis": {{
                    "logical_flow": "Mantıksal akış (1-10)",
                    "section_organization": "Bölüm organizasyonu (1-10)",
                    "paragraph_coherence": "Paragraf tutarlılığı (1-10)",
                    "transition_quality": "Geçiş kalitesi (1-10)"
                }},
                "language_quality": {{
                    "clarity": "Netlik (1-10)",
                    "precision": "Kesinlik (1-10)",
                    "academic_tone": "Akademik ton (1-10)",
                    "readability": "Okunabilirlik (1-10)"
                }},
                "technical_writing": {{
                    "terminology_consistency": "Terminoloji tutarlılığı",
                    "sentence_structure": "Cümle yapısı kalitesi",
                    "grammar_accuracy": "Dilbilgisi doğruluğu",
                    "style_appropriateness": "Stil uygunluğu"
                }},
                "audience_alignment": {{
                    "complexity_match": "Karmaşıklık uyumu",
                    "background_assumptions": "Arka plan varsayımları",
                    "accessibility": "Erişilebilirlik"
                }},
                "language_improvements": ["Dil iyileştirme önerileri"]
            }}
            """
        )
        
        self.structure_language_chain = LLMChain(
            llm=self.llm,
            prompt=structure_language_prompt,
            output_key="structure_language_analysis",
            output_parser=self.output_parser
        )
    
    def _setup_sequential_chain(self):
        """Setup the main sequential analysis chain"""
        self.sequential_chain = SequentialChain(
            chains=[
                self.content_quality_chain,
                self.methodology_analysis_chain,
                self.citation_analysis_chain,
                self.structure_language_chain
            ],
            input_variables=["document_text", "title", "abstract", "research_type", "references", "target_audience"],
            output_variables=["content_quality_analysis", "methodology_evaluation", "citation_evaluation", "structure_language_analysis"],
            verbose=True
        )
    
    def analyze_document(self, document_data: Dict[str, Any], 
                        callbacks: List[BaseCallbackHandler] = None) -> Dict[str, Any]:
        """
        Run comprehensive document analysis
        
        Args:
            document_data: Document information including text, title, etc.
            callbacks: Optional callback handlers for streaming
            
        Returns:
            Comprehensive analysis results
        """
        try:
            logger.info(f"Starting comprehensive analysis for: {document_data.get('title', 'Untitled')}")
            
            # Prepare input parameters with defaults
            analysis_input = {
                "document_text": document_data.get("text", "")[:10000],  # Limit text length
                "title": document_data.get("title", "Başlık belirtilmemiş"),
                "abstract": document_data.get("abstract", "Özet mevcut değil"),
                "research_type": document_data.get("research_type", "Genel araştırma"),
                "references": document_data.get("references", "Kaynaklar belirtilmemiş"),
                "target_audience": document_data.get("target_audience", "Akademik topluluk")
            }
            
            # Run the sequential chain
            results = self.sequential_chain(analysis_input, callbacks=callbacks)
            
            # Calculate overall scores
            overall_analysis = self._calculate_overall_scores(results)
            results["overall_analysis"] = overall_analysis
            
            # Add metadata
            results["analysis_metadata"] = {
                "document_title": document_data.get("title"),
                "analysis_timestamp": "2024-10-21",  # Would use actual timestamp
                "model_used": self.llm.model_name,
                "analysis_type": "comprehensive"
            }
            
            logger.info("Comprehensive analysis completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            return {
                "error": str(e),
                "analysis_metadata": {
                    "document_title": document_data.get("title"),
                    "failed_at": "comprehensive_analysis"
                }
            }
    
    def _calculate_overall_scores(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall quality scores from individual analyses"""
        try:
            scores = []
            
            # Extract numeric scores from each analysis
            for analysis_key in analysis_results:
                if isinstance(analysis_results[analysis_key], dict):
                    analysis = analysis_results[analysis_key]
                    
                    # Look for numeric scores in nested dictionaries
                    for section_key, section_value in analysis.items():
                        if isinstance(section_value, dict):
                            for metric_key, metric_value in section_value.items():
                                # Extract numeric score
                                if isinstance(metric_value, str) and '(' in metric_value:
                                    score_match = re.search(r'\((\d+)-\d+\)', metric_value)
                                    if score_match:
                                        try:
                                            score = int(score_match.group(1))
                                            scores.append(score)
                                        except:
                                            continue
            
            # Calculate overall metrics
            if scores:
                average_score = sum(scores) / len(scores)
                quality_rating = self._get_quality_rating(average_score)
            else:
                average_score = 0
                quality_rating = "Değerlendirilemedi"
            
            return {
                "overall_score": round(average_score, 1),
                "quality_rating": quality_rating,
                "total_metrics_evaluated": len(scores),
                "recommendation": self._get_overall_recommendation(average_score)
            }
            
        except Exception as e:
            logger.error(f"Error calculating overall scores: {e}")
            return {
                "overall_score": 0,
                "quality_rating": "Hesaplanamadı",
                "error": str(e)
            }
    
    def _get_quality_rating(self, score: float) -> str:
        """Convert numeric score to quality rating"""
        if score >= 8.5:
            return "Mükemmel"
        elif score >= 7.5:
            return "İyi"
        elif score >= 6.5:
            return "Orta"
        elif score >= 5.0:
            return "Zayıf"
        else:
            return "Yetersiz"
    
    def _get_overall_recommendation(self, score: float) -> str:
        """Get overall recommendation based on score"""
        if score >= 8.5:
            return "Bu dokuman yayın kalitesindedir. Küçük düzeltmelerle yayınlanabilir."
        elif score >= 7.5:
            return "İyi kalitede bir çalışma. Bazı iyileştirmeler yapılarak güçlendirilebilir."
        elif score >= 6.5:
            return "Orta kalitede. Önemli revizyonlar gerekli."
        elif score >= 5.0:
            return "Zayıf kalite. Kapsamlı revizyonlar gerekli."
        else:
            return "Yetersiz kalite. Temel düzeyde yeniden yazım önerilir."
    
    def quick_analysis(self, text: str, focus_area: str = "general") -> Dict[str, Any]:
        """
        Perform a quick focused analysis
        
        Args:
            text: Text to analyze
            focus_area: Specific area to focus on (general, methodology, writing, citations)
            
        Returns:
            Quick analysis results
        """
        try:
            focus_prompts = {
                "general": "Bu metni genel kalite açısından hızlıca değerlendir.",
                "methodology": "Bu metnin metodolojisini analiz et.",
                "writing": "Bu metnin yazım kalitesini değerlendir.",
                "citations": "Bu metindeki atıfları analiz et."
            }
            
            prompt = f"""
            {focus_prompts.get(focus_area, focus_prompts["general"])}
            
            Metin: {text[:3000]}
            
            Kısa ve net bir analiz sağla.
            """
            
            response = self.llm.predict(prompt)
            
            return {
                "quick_analysis": response,
                "focus_area": focus_area,
                "text_length": len(text)
            }
            
        except Exception as e:
            logger.error(f"Error in quick analysis: {e}")
            return {"error": str(e)}
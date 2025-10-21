"""
Writing Assistant Chain - Academic writing and structure improvement system
Multi-step chain for helping with academic writing tasks
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

class WritingOutputParser(BaseOutputParser):
    """Custom parser for writing assistance output"""
    
    def parse(self, text: str) -> Dict[str, Any]:
        try:
            if text.strip().startswith('{'):
                return json.loads(text)
            
            # Parse structured text
            sections = {}
            current_section = None
            current_content = []
            
            for line in text.strip().split('\n'):
                if line.startswith('##') or line.startswith('**'):
                    if current_section:
                        sections[current_section] = '\n'.join(current_content)
                    current_section = line.strip('# *:')
                    current_content = []
                else:
                    current_content.append(line)
            
            if current_section:
                sections[current_section] = '\n'.join(current_content)
            
            return sections if sections else {"content": text}
            
        except Exception as e:
            logger.warning(f"Error parsing writing output: {e}")
            return {"raw_output": text}

class AcademicWritingChain:
    """
    Academic writing assistance chain that helps with:
    - Argument structure development
    - Literature review organization
    - Academic style improvement
    - Citation integration
    - Conclusion formulation
    """
    
    def __init__(self, llm_model: str = "gpt-4o", temperature: float = 0.3):
        self.llm = ChatOpenAI(model_name=llm_model, temperature=temperature)
        self.output_parser = WritingOutputParser()
        
        self._setup_writing_chains()
        self._setup_sequential_chain()
        
        logger.info("Academic Writing Chain initialized")
    
    def _setup_writing_chains(self):
        """Setup individual writing assistance chains"""
        
        # 1. Argument Structure Chain
        argument_prompt = PromptTemplate(
            input_variables=["research_topic", "main_points", "evidence"],
            template="""
            Aşağıdaki araştırma konusu için güçlü bir akademik argüman yapısı oluştur:

            Araştırma Konusu: {research_topic}
            Ana Noktalar: {main_points}  
            Kanıtlar: {evidence}

            JSON formatında argüman yapısı:
            {{
                "thesis_statement": "Ana tez cümlesi",
                "argument_structure": {{
                    "introduction": "Giriş paragrafı önerisi",
                    "body_paragraphs": [
                        {{"topic": "Paragraf konusu", "supporting_evidence": "Destekleyici kanıt", "analysis": "Analiz"}},
                        {{"topic": "Paragraf konusu", "supporting_evidence": "Destekleyici kanıt", "analysis": "Analiz"}}
                    ],
                    "conclusion": "Sonuç paragrafı önerisi"
                }},
                "logical_flow": "Argümanın mantıksal akışı",
                "counterarguments": ["Karşı argümanlar"],
                "refutations": ["Karşı argümanlara cevaplar"],
                "strength_score": "1-10 arası argüman gücü skoru"
            }}
            """
        )
        
        self.argument_chain = LLMChain(
            llm=self.llm,
            prompt=argument_prompt,
            output_key="argument_structure",
            output_parser=self.output_parser
        )
        
        # 2. Literature Review Organization Chain
        literature_prompt = PromptTemplate(
            input_variables=["research_topic", "sources", "themes"],
            template="""
            Bu araştırma konusu için sistematik literatür taraması organizasyonu oluştur:

            Araştırma Konusu: {research_topic}
            Kaynaklar: {sources}
            Temalar: {themes}

            JSON formatında literatür organizasyonu:
            {{
                "thematic_organization": [
                    {{
                        "theme": "Tema adı",
                        "sources": ["İlgili kaynaklar"],
                        "key_findings": ["Ana bulgular"],
                        "synthesis": "Tema sentezi"
                    }}
                ],
                "chronological_development": "Kronolojik gelişim",
                "theoretical_frameworks": ["Teorik çerçeveler"],
                "research_gaps": ["Literatür boşlukları"],
                "synthesis_paragraph": "Genel sentez paragrafı",
                "transition_to_research": "Araştırmaya geçiş cümlesi"
            }}
            """
        )
        
        self.literature_chain = LLMChain(
            llm=self.llm,
            prompt=literature_prompt,
            output_key="literature_organization",
            output_parser=self.output_parser
        )
        
        # 3. Academic Style Improvement Chain
        style_prompt = PromptTemplate(
            input_variables=["text_sample", "target_style", "academic_level"],
            template="""
            Aşağıdaki metni akademik yazım standartlarına göre iyileştir:

            Metin: {text_sample}
            Hedef Stil: {target_style}
            Akademik Seviye: {academic_level}

            JSON formatında stil iyileştirmesi:
            {{
                "improved_text": "İyileştirilmiş metin",
                "changes_made": [
                    {{"original": "Orijinal ifade", "improved": "İyileştirilmiş ifade", "reason": "Değişiklik sebebi"}}
                ],
                "style_suggestions": [
                    {{"category": "Kategori", "suggestion": "Öneri", "example": "Örnek"}}
                ],
                "vocabulary_enhancements": ["Kelime dağarcığı iyileştirmeleri"],
                "sentence_structure": "Cümle yapısı önerileri",
                "academic_tone": "Akademik ton değerlendirmesi",
                "clarity_score": "1-10 arası netlik skoru"
            }}
            """
        )
        
        self.style_chain = LLMChain(
            llm=self.llm,
            prompt=style_prompt,
            output_key="style_improvement",
            output_parser=self.output_parser
        )
        
        # 4. Citation Integration Chain
        citation_prompt = PromptTemplate(
            input_variables=["text", "sources", "citation_style"],
            template="""
            Bu metne doğru akademik atıfları entegre et:

            Metin: {text}
            Kaynaklar: {sources}
            Atıf Stili: {citation_style}

            JSON formatında atıf entegrasyonu:
            {{
                "text_with_citations": "Atıflarla birlikte metin",
                "citation_points": [
                    {{"location": "Atıf yeri", "source": "Kaynak", "reason": "Atıf sebebi"}}
                ],
                "in_text_citations": ["Metin içi atıflar"],
                "reference_list": ["Kaynakça listesi"],
                "citation_analysis": {{
                    "primary_sources": "Birincil kaynak sayısı",
                    "secondary_sources": "İkincil kaynak sayısı", 
                    "recent_sources": "Son 5 yıldaki kaynak sayısı",
                    "diversity_score": "Kaynak çeşitliliği skoru"
                }}
            }}
            """
        )
        
        self.citation_chain = LLMChain(
            llm=self.llm,
            prompt=citation_prompt,
            output_key="citation_integration",
            output_parser=self.output_parser
        )
    
    def _setup_sequential_chain(self):
        """Setup sequential writing assistance chain"""
        self.sequential_chain = SequentialChain(
            chains=[
                self.argument_chain,
                self.literature_chain,
                self.style_chain,
                self.citation_chain
            ],
            input_variables=["research_topic", "main_points", "evidence", "sources", "themes", "text_sample", "target_style", "academic_level", "citation_style", "text"],
            output_variables=["argument_structure", "literature_organization", "style_improvement", "citation_integration"],
            verbose=True
        )
    
    def assist_writing(self, writing_request: Dict[str, Any], 
                      callbacks: List[BaseCallbackHandler] = None) -> Dict[str, Any]:
        """
        Provide comprehensive writing assistance
        
        Args:
            writing_request: Dictionary containing writing parameters
            callbacks: Optional callback handlers for streaming
            
        Returns:
            Writing assistance results
        """
        try:
            logger.info(f"Starting writing assistance for: {writing_request.get('research_topic', 'Unknown topic')}")
            
            # Set defaults for missing parameters
            default_params = {
                "research_topic": "Genel araştırma konusu",
                "main_points": "Ana noktalar belirtilmemiş",
                "evidence": "Kanıtlar belirtilmemiş", 
                "sources": "Kaynaklar belirtilmemiş",
                "themes": "Temalar belirtilmemiş",
                "text_sample": writing_request.get("text_sample", "Örnek metin yok"),
                "text": writing_request.get("text", "Metin belirtilmemiş"),
                "target_style": "Akademik makale",
                "academic_level": "Yüksek lisans",
                "citation_style": "APA 7"
            }
            
            # Update with provided parameters
            default_params.update(writing_request)
            
            # Run the sequential chain
            results = self.sequential_chain(default_params, callbacks=callbacks)
            
            # Add metadata
            results["writing_metadata"] = {
                "topic": writing_request.get("research_topic"),
                "assistance_type": "comprehensive",
                "model_used": self.llm.model_name
            }
            
            logger.info("Writing assistance completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in writing assistance: {e}")
            return {
                "error": str(e),
                "writing_metadata": {
                    "topic": writing_request.get("research_topic"),
                    "failed_at": "writing_assistance"
                }
            }
    
    def improve_paragraph(self, paragraph: str, context: str = "", 
                         target_style: str = "academic") -> Dict[str, Any]:
        """
        Improve a single paragraph
        """
        try:
            improvement_prompt = f"""
            Aşağıdaki paragrafı akademik yazım standartlarına göre iyileştir:

            Bağlam: {context}
            
            Paragraf: {paragraph}
            
            Hedef Stil: {target_style}

            İyileştirilmiş paragrafı ve değişiklikleri açıkla.
            """
            
            response = self.llm.predict(improvement_prompt)
            return {"improved_paragraph": response}
            
        except Exception as e:
            logger.error(f"Error improving paragraph: {e}")
            return {"error": str(e)}
    
    def generate_outline(self, research_topic: str, main_arguments: List[str]) -> Dict[str, Any]:
        """
        Generate a structured outline for academic writing
        """
        try:
            outline_prompt = f"""
            Aşağıdaki araştırma konusu ve ana argümanlar için detaylı bir akademik makale outline'ı oluştur:

            Araştırma Konusu: {research_topic}
            Ana Argümanlar: {', '.join(main_arguments)}

            Şu bölümleri içeren kapsamlı bir outline oluştur:
            1. Giriş
            2. Literatür Taraması  
            3. Ana argümanlar (alt başlıklar ile)
            4. Sonuç
            5. Öneriler

            Her bölüm için ana noktaları ve alt başlıkları belirt.
            """
            
            response = self.llm.predict(outline_prompt)
            return {"outline": response, "topic": research_topic}
            
        except Exception as e:
            logger.error(f"Error generating outline: {e}")
            return {"error": str(e)}
    
    def check_coherence(self, text: str) -> Dict[str, Any]:
        """
        Check text coherence and provide improvement suggestions
        """
        try:
            coherence_prompt = f"""
            Aşağıdaki metni tutarlılık (coherence) açısından analiz et:

            Metin: {text}

            Analiz et:
            1. Paragraflar arası geçişler
            2. Mantıksal akış  
            3. Argüman tutarlılığı
            4. Tema birliği
            5. İyileştirme önerileri

            Detaylı analiz ve öneriler ver.
            """
            
            response = self.llm.predict(coherence_prompt)
            return {"coherence_analysis": response}
            
        except Exception as e:
            logger.error(f"Error checking coherence: {e}")
            return {"error": str(e)}
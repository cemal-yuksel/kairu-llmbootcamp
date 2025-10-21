"""
Enhanced Article Analysis Module for Academic Research
Provides comprehensive analysis of academic papers including methodology, findings, and limitations
"""

import openai
import os
from dotenv import load_dotenv
from typing import Dict, List, Optional, Any
import logging
import json
import re
from datetime import datetime

load_dotenv()
logger = logging.getLogger(__name__)

class EnhancedArticleAnalyzer:
    """
    Advanced article analyzer that performs comprehensive analysis of academic papers
    """
    
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        
        self.client = openai.OpenAI(api_key=self.openai_api_key)
        self.analysis_cache = {}
    
    def analyze_article_comprehensive(self, pdf_name: str, pdf_text: str, 
                                    metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Performs comprehensive analysis of an academic article
        
        Args:
            pdf_name: Name of the PDF file
            pdf_text: Full text content of the article
            metadata: Optional metadata about the article
            
        Returns:
            Comprehensive analysis results
        """
        try:
            logger.info(f"Starting comprehensive analysis for: {pdf_name}")
            
            # Check cache first
            cache_key = f"{pdf_name}_{hash(pdf_text[:1000])}"
            if cache_key in self.analysis_cache:
                logger.info("Returning cached analysis")
                return self.analysis_cache[cache_key]
            
            # Analyze different components
            analysis_results = {
                "pdf_name": pdf_name,
                "analysis_timestamp": datetime.now().isoformat(),
                "metadata": metadata or {},
                "methodology_analysis": self._analyze_methodology(pdf_text),
                "findings_analysis": self._analyze_findings(pdf_text),
                "limitations_analysis": self._analyze_limitations(pdf_text),
                "contribution_analysis": self._analyze_contributions(pdf_text),
                "quality_assessment": self._assess_research_quality(pdf_text),
                "citation_potential": self._assess_citation_potential(pdf_text),
                "research_gaps": self._identify_research_gaps(pdf_text)
            }
            
            # Add overall summary
            analysis_results["overall_summary"] = self._generate_overall_summary(analysis_results)
            
            # Cache results
            self.analysis_cache[cache_key] = analysis_results
            
            logger.info(f"Comprehensive analysis completed for: {pdf_name}")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            return {
                "error": str(e),
                "pdf_name": pdf_name,
                "analysis_timestamp": datetime.now().isoformat()
            }
    
    def _analyze_methodology(self, text: str) -> Dict[str, Any]:
        """Analyze research methodology"""
        try:
            prompt = f"""
Aşağıdaki akademik makale metnini analiz ederek ARAŞTIRMA YÖNTEMİ hakkında detaylı bilgi çıkar:

ÇIKARMAN GEREKEN BİLGİLER:
1. Araştırma Deseni (Nicel, Nitel, Karma Yöntem)
2. Örneklem Bilgileri (Büyüklük, seçim yöntemi, özellikleri)
3. Veri Toplama Yöntemleri (Anket, mülakat, gözlem, vb.)
4. Veri Analiz Teknikleri (İstatistiksel yöntemler, analiz araçları)
5. Kullanılan Araçlar ve Ölçekler
6. Geçerlik ve Güvenirlik Bilgileri

Makale Metni (İlk 8000 karakter):
{text[:8000]}

Lütfen JSON formatında yanıt ver:
{{
    "research_design": "string",
    "sample_info": {{
        "size": "string",
        "selection_method": "string",
        "characteristics": "string"
    }},
    "data_collection": ["method1", "method2"],
    "data_analysis": ["technique1", "technique2"],
    "instruments": ["instrument1", "instrument2"],
    "validity_reliability": "string",
    "methodology_strengths": ["strength1", "strength2"],
    "methodology_limitations": ["limitation1", "limitation2"]
}}
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system", 
                        "content": "Sen akademik makaleleri analiz eden bir uzmansın. Araştırma yöntemlerini detaylıca analiz edersin ve JSON formatında yanıt verirsin."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500,
                timeout=90
            )
            
            # Parse JSON response
            response_text = response.choices[0].message.content.strip()
            
            try:
                # Clean response text for JSON parsing
                if response_text.startswith("```json"):
                    response_text = response_text.replace("```json", "").replace("```", "").strip()
                elif response_text.startswith("```"):
                    response_text = response_text.replace("```", "").strip()
                
                methodology_data = json.loads(response_text)
                
                # Validate that we have meaningful content
                if methodology_data.get("research_design") and "çıkarılamadı" not in methodology_data.get("research_design", ""):
                    return methodology_data
                else:
                    # Try alternative extraction
                    return self._extract_methodology_alternative(text, response_text)
                    
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing failed: {e}")
                # Alternative extraction when JSON fails
                return self._extract_methodology_alternative(text, response_text)
            
            return methodology_data
            
        except Exception as e:
            logger.error(f"Methodology analysis error: {e}")
            return {"error": str(e)}
    
    def _extract_methodology_alternative(self, text: str, response_text: str) -> Dict[str, Any]:
        """Alternative methodology extraction when JSON parsing fails"""
        try:
            # Use a simpler prompt for more reliable extraction
            prompt = f"""
Lütfen bu akademik makale metnindeki araştırma metodolojisini analiz et ve kısa yanıtlar ver:

1. Araştırma türü nedir? (nicel/nitel/karma)
2. Örneklem büyüklüğü nedir?
3. Hangi veri toplama yöntemi kullanılmış?
4. Hangi analiz yöntemi kullanılmış?

Makale metni: {text[:5000]}

Kısa ve net yanıtlar ver:
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Use faster model for backup
                messages=[
                    {
                        "role": "system", 
                        "content": "Sen akademik makaleleri analiz eden bir asistansın. Kısa ve net yanıtlar verirsin."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800,
                timeout=30
            )
            
            analysis_text = response.choices[0].message.content
            
            # Structure the response
            return {
                "research_design": self._extract_info_from_text(analysis_text, "araştırma türü", "Belirtilmemiş"),
                "sample_info": {
                    "size": self._extract_info_from_text(analysis_text, "örneklem", "Belirtilmemiş"),
                    "selection_method": "Belirtilmemiş",
                    "characteristics": "Belirtilmemiş"
                },
                "data_collection": [self._extract_info_from_text(analysis_text, "veri toplama", "Belirtilmemiş")],
                "data_analysis": [self._extract_info_from_text(analysis_text, "analiz", "Belirtilmemiş")],
                "methodology_summary": analysis_text,
                "extraction_method": "alternative_parsing"
            }
            
        except Exception as e:
            logger.error(f"Alternative methodology extraction failed: {e}")
            return {
                "research_design": "Metodoloji bilgisi çıkarılamadı - lütfen makaleyi manuel olarak inceleyin",
                "extraction_method": "failed",
                "error": str(e)
            }
    
    def _extract_info_from_text(self, text: str, keyword: str, default: str = "Belirtilmemiş") -> str:
        """Extract specific information from analysis text"""
        try:
            # Look for lines containing the keyword
            lines = text.lower().split('\n')
            for line in lines:
                if keyword.lower() in line:
                    # Extract the part after the keyword
                    parts = line.split(':')
                    if len(parts) > 1:
                        return parts[1].strip()
            return default
        except:
            return default
    
    def _analyze_findings(self, text: str) -> Dict[str, Any]:
        """Analyze research findings and results"""
        try:
            prompt = f"""
Aşağıdaki akademik makale metnini analiz ederek BULGULAR VE SONUÇLAR hakkında detaylı bilgi çıkar:

ÇIKARMAN GEREKEN BİLGİLER:
1. Ana Bulgular (En önemli sonuçlar)
2. İstatistiksel Sonuçlar (p değerleri, korelasyonlar, vb.)
3. Hipotez Durumları (Kabul/Ret edilenler)
4. Beklenmeyen Bulgular
5. Bulguların Teorik Katkısı
6. Pratik Uygulamalar

Makale Metni (Ortası 8000 karakter):
{text[len(text)//3:len(text)//3+8000]}

Lütfen JSON formatında yanıt ver:
{{
    "main_findings": [
        {{
            "finding": "string",
            "significance": "high/medium/low",
            "evidence": "string"
        }}
    ],
    "statistical_results": [
        {{
            "test": "string",
            "result": "string",
            "p_value": "string"
        }}
    ],
    "hypothesis_outcomes": [
        {{
            "hypothesis": "string",
            "outcome": "accepted/rejected/partially_supported",
            "evidence": "string"
        }}
    ],
    "unexpected_findings": ["finding1", "finding2"],
    "theoretical_contributions": ["contribution1", "contribution2"],
    "practical_implications": ["implication1", "implication2"]
}}
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system", 
                        "content": "Sen akademik araştırma bulgularını analiz eden bir uzmansın. Bulguları kategorize eder ve önemini değerlendirirsin."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500,
                timeout=90
            )
            
            response_text = response.choices[0].message.content.strip()
            
            try:
                # Clean response for JSON parsing
                if response_text.startswith("```json"):
                    response_text = response_text.replace("```json", "").replace("```", "").strip()
                elif response_text.startswith("```"):
                    response_text = response_text.replace("```", "").strip()
                
                findings_data = json.loads(response_text)
                return findings_data
                
            except json.JSONDecodeError:
                # Alternative extraction for findings
                return self._extract_findings_alternative(text, response_text)
            
            return findings_data
            
        except Exception as e:
            logger.error(f"Findings analysis error: {e}")
            return {"error": str(e)}
    
    def _extract_findings_alternative(self, text: str, response_text: str) -> Dict[str, Any]:
        """Alternative findings extraction when JSON parsing fails"""
        try:
            # Use simpler approach for findings
            prompt = f"""
Bu akademik makale metnindeki ana bulguları listele:

1. En önemli 3 bulgu nedir?
2. Hangi istatistiksel sonuçlar var?
3. Hangi hipotezler desteklendi?

Makale metni: {text[len(text)//2:len(text)//2+4000]}

Kısa listeler halinde yanıt ver:
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Sen araştırma bulgularını özetleyen bir asistansın."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800,
                timeout=30
            )
            
            findings_text = response.choices[0].message.content
            
            return {
                "main_findings": [
                    {
                        "finding": "Ana bulgular metin analizi ile çıkarıldı",
                        "significance": "medium",
                        "evidence": findings_text[:500]
                    }
                ],
                "findings_summary": findings_text,
                "extraction_method": "alternative_parsing"
            }
            
        except Exception as e:
            logger.error(f"Alternative findings extraction failed: {e}")
            return {
                "main_findings": [{"finding": "Bulgular analiz edilemedi", "significance": "unknown", "evidence": "Hata oluştu"}],
                "error": str(e)
            }
    
    def _analyze_limitations(self, text: str) -> Dict[str, Any]:
        """Analyze research limitations and constraints"""
        try:
            prompt = f"""
Aşağıdaki akademik makale metnini analiz ederek ARAŞTIRMANIN SINIRLILIKLARI hakkında bilgi çıkar:

ÇIKARMAN GEREKEN BİLGİLER:
1. Metodolojik Sınırlılıklar
2. Örneklem Sınırlılıkları  
3. Veri Toplama Sınırları
4. Genelleme Sınırları
5. Zaman/Mekan Sınırları
6. Gelecek Araştırma Önerileri

Makale Metni (Son 8000 karakter):
{text[-8000:]}

Lütfen JSON formatında yanıt ver:
{{
    "methodological_limitations": ["limitation1", "limitation2"],
    "sample_limitations": ["limitation1", "limitation2"],
    "data_collection_limits": ["limit1", "limit2"],
    "generalization_limits": ["limit1", "limit2"],
    "temporal_spatial_limits": ["limit1", "limit2"],
    "future_research_suggestions": ["suggestion1", "suggestion2"],
    "overall_limitation_assessment": "high/medium/low"
}}
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system", 
                        "content": "Sen akademik araştırmaların sınırlılıklarını değerlendiren bir uzmansın. Objektif ve yapıcı analiz yaparsın."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1200
            )
            
            response_text = response.choices[0].message.content
            try:
                limitations_data = json.loads(response_text)
            except json.JSONDecodeError:
                limitations_data = {
                    "methodological_limitations": ["Sınırlılıklar metin analizi ile çıkarılamadı"],
                    "analysis_raw": response_text
                }
            
            return limitations_data
            
        except Exception as e:
            logger.error(f"Limitations analysis error: {e}")
            return {"error": str(e)}
    
    def _analyze_contributions(self, text: str) -> Dict[str, Any]:
        """Analyze theoretical and practical contributions"""
        try:
            # Extract key sections that typically contain contributions
            contribution_indicators = [
                "katkı", "contribution", "özgün", "novel", "yenilik",
                "teorik", "theoretical", "pratik", "practical", "sonuç", "conclusion"
            ]
            
            relevant_sections = []
            text_lower = text.lower()
            
            # Find sections with contribution indicators
            sentences = text.split('.')
            for sentence in sentences:
                if any(indicator in sentence.lower() for indicator in contribution_indicators):
                    relevant_sections.append(sentence.strip())
            
            contributions_text = '. '.join(relevant_sections[:10])  # Limit to first 10 relevant sentences
            
            prompt = f"""
Aşağıdaki metin bölümlerinden araştırmanın TEORİK VE PRATİK KATKILARINI çıkar:

Metin:
{contributions_text}

Lütfen şu kategorilerde katkıları belirle:
1. Teorik Katkılar (Bilime katkı)
2. Metodolojik Katkılar  
3. Pratik Katkılar (Uygulama alanları)
4. Politika Önerileri
5. Özgünlük Değerlendirmesi

JSON formatında yanıt ver:
{{
    "theoretical_contributions": ["katkı1", "katkı2"],
    "methodological_contributions": ["katkı1", "katkı2"],
    "practical_contributions": ["katkı1", "katkı2"],
    "policy_recommendations": ["öneri1", "öneri2"],
    "originality_assessment": "high/medium/low",
    "contribution_strength": "high/medium/low"
}}
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system", 
                        "content": "Sen akademik araştırmaların katkılarını değerlendiren bir uzmansın. Teorik ve pratik katkıları ayırt edersin."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            response_text = response.choices[0].message.content
            try:
                contributions_data = json.loads(response_text)
            except json.JSONDecodeError:
                contributions_data = {
                    "theoretical_contributions": ["Katkılar metin analizi ile çıkarılamadı"],
                    "analysis_raw": response_text
                }
            
            return contributions_data
            
        except Exception as e:
            logger.error(f"Contributions analysis error: {e}")
            return {"error": str(e)}
    
    def _assess_research_quality(self, text: str) -> Dict[str, Any]:
        """Assess overall research quality"""
        try:
            # Simple quality assessment based on text analysis
            quality_indicators = {
                "has_methodology": any(term in text.lower() for term in ["method", "yöntem", "analiz", "analysis"]),
                "has_results": any(term in text.lower() for term in ["result", "sonuç", "bulgu", "finding"]),
                "has_references": any(term in text.lower() for term in ["reference", "kaynak", "citation", "atıf"]),
                "has_discussion": any(term in text.lower() for term in ["discussion", "tartışma", "sonuç", "conclusion"]),
                "text_length_adequate": len(text) > 5000,
                "has_statistical_terms": any(term in text.lower() for term in ["p<", "r=", "significant", "anlamlı", "korelasyon"])
            }
            
            quality_score = sum(quality_indicators.values()) / len(quality_indicators)
            
            if quality_score >= 0.8:
                quality_level = "high"
            elif quality_score >= 0.6:
                quality_level = "medium"
            else:
                quality_level = "low"
            
            return {
                "quality_indicators": quality_indicators,
                "quality_score": round(quality_score, 2),
                "quality_level": quality_level,
                "assessment_date": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Quality assessment error: {e}")
            return {"error": str(e)}
    
    def _assess_citation_potential(self, text: str) -> Dict[str, Any]:
        """Assess citation potential of the research"""
        try:
            # Factors that indicate high citation potential
            high_impact_indicators = [
                "novel", "first", "comprehensive", "significant", "important",
                "yeni", "ilk", "kapsamlı", "önemli", "anlamlı"
            ]
            
            methodology_strength = [
                "rigorous", "systematic", "randomized", "controlled",
                "titiz", "sistematik", "randomize", "kontrollü"
            ]
            
            practical_relevance = [
                "practical", "application", "implementation", "policy",
                "pratik", "uygulama", "politika", "gerçek"
            ]
            
            text_lower = text.lower()
            
            impact_score = sum(1 for indicator in high_impact_indicators if indicator in text_lower)
            method_score = sum(1 for indicator in methodology_strength if indicator in text_lower)
            relevance_score = sum(1 for indicator in practical_relevance if indicator in text_lower)
            
            total_score = impact_score + method_score + relevance_score
            
            if total_score >= 8:
                citation_potential = "high"
            elif total_score >= 4:
                citation_potential = "medium"
            else:
                citation_potential = "low"
            
            return {
                "impact_indicators_found": impact_score,
                "methodology_strength_indicators": method_score,
                "practical_relevance_indicators": relevance_score,
                "total_score": total_score,
                "citation_potential": citation_potential
            }
            
        except Exception as e:
            logger.error(f"Citation potential assessment error: {e}")
            return {"error": str(e)}
    
    def _identify_research_gaps(self, text: str) -> List[str]:
        """Identify research gaps mentioned in the paper"""
        try:
            gap_indicators = [
                "future research", "further study", "limitation", "gap",
                "gelecek araştırma", "ileri çalışma", "sınırlılık", "boşluk",
                "needs to be", "should be studied", "requires investigation"
            ]
            
            gap_sentences = []
            sentences = text.split('.')
            
            for sentence in sentences:
                if any(indicator in sentence.lower() for indicator in gap_indicators):
                    gap_sentences.append(sentence.strip())
            
            # Limit to most relevant gaps
            return gap_sentences[:5]
            
        except Exception as e:
            logger.error(f"Research gaps identification error: {e}")
            return []
    
    def _generate_overall_summary(self, analysis_results: Dict[str, Any]) -> str:
        """Generate overall summary of the analysis"""
        try:
            methodology = analysis_results.get("methodology_analysis", {})
            findings = analysis_results.get("findings_analysis", {})
            quality = analysis_results.get("quality_assessment", {})
            
            summary = f"""
BU ARAŞTIRMANIN GENEL DEĞERLENDİRMESİ:

🔬 YÖNTEM: {methodology.get('research_design', 'Belirtilmemiş')}
📊 KALİTE SEVİYESİ: {quality.get('quality_level', 'Bilinmiyor').upper()}
📈 ATIF POTANSİYELİ: {analysis_results.get('citation_potential', {}).get('citation_potential', 'Bilinmiyor').upper()}

Ana bulgular ve katkıları içeren kapsamlı bir akademik çalışma.
"""
            
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Summary generation error: {e}")
            return "Özet oluşturulurken hata oluştu."

# Global analyzer instance
article_analyzer = EnhancedArticleAnalyzer()

# Legacy function for backward compatibility
def analyze_article(pdf_name, pdf_text):
    """Legacy function wrapper"""
    result = article_analyzer.analyze_article_comprehensive(pdf_name, pdf_text)
    # Return only methodology analysis for compatibility
    methodology = result.get("methodology_analysis", {})
    findings = result.get("findings_analysis", {})
    limitations = result.get("limitations_analysis", {})
    contributions = result.get("contribution_analysis", {})
    
    # Format as expected by the original function
    formatted_result = f"""
1. ARAŞTIRMA YÖNTEMİ
   - Araştırma Deseni: {methodology.get('research_design', 'Belirtilmemiş')}
   - Örneklem: {methodology.get('sample_info', {}).get('characteristics', 'Belirtilmemiş')}
   - Veri Toplama: {', '.join(methodology.get('data_collection', ['Belirtilmemiş']))}

2. ANA BULGULAR
   {chr(10).join([f"   - {finding.get('finding', 'N/A')}" for finding in findings.get('main_findings', [])[:3]])}

3. SONUÇLAR VE ÖNERİLER
   {chr(10).join([f"   - {contrib}" for contrib in contributions.get('theoretical_contributions', ['Belirtilmemiş'])[:3]])}

4. ARAŞTIRMANIN SINIRLILIKLARI
   {chr(10).join([f"   - {limit}" for limit in limitations.get('methodological_limitations', ['Belirtilmemiş'])[:3]])}
"""
    
    return formatted_result
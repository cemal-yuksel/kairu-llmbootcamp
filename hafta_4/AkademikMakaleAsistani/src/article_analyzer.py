import openai
import os
from dotenv import load_dotenv
from .pdf_manager import extract_text

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def analyze_article(pdf_name, pdf_text):
    """
    Makaleyi LLM ile analiz eder: Yöntem, Bulgular, Sonuçlar, Sınırlılıklar
    """
    
    prompt = f"""
Aşağıdaki akademik makaleyi detaylı bir şekilde analiz et ve şu başlıklar altında özet çıkar:

1. ARAŞTIRMA YÖNTEMİ
   - Kullanılan araştırma deseni
   - Örneklem ve veri toplama yöntemleri
   - Veri analiz teknikleri

2. ANA BULGULAR
   - En önemli bulgular
   - İstatistiksel sonuçlar (varsa)
   - Öne çıkan veriler

3. SONUÇLAR VE ÖNERİLER
   - Araştırmanın ana sonuçları
   - Teorik ve pratik katkılar
   - Öneriler

4. ARAŞTIRMANIN SINIRLILIKLARI
   - Metodolojik sınırlılıklar
   - Örneklem sınırlılıkları
   - Gelecek araştırmalar için öneriler

Makale Metni:
{pdf_text[:8000]}  # İlk 8000 karakter (token limiti için)

Lütfen her başlık altında detaylı, akademik ve profesyonel bir analiz sun.
"""

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Sen akademik makaleleri analiz eden bir uzmansın. Detaylı, bilimsel ve objektif analizler yaparsın."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=2000
    )
    
    analysis = response.choices[0].message.content
    return analysis

import openai
import os
from dotenv import load_dotenv
from . import citation_manager

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Soru ve bağlam ile LLM'den yanıt al
def answer_with_citations(question, context_chunks, metadata):
    context = "\n\n".join(context_chunks)
    
    # Metadata'dan metin içi alıntıları oluştur
    in_text_citations = []
    if isinstance(metadata, list):
        for meta in metadata:
            if isinstance(meta, dict):
                pdf_name = meta.get('pdf_name', 'Makale')
                citation = citation_manager.create_in_text_citation(pdf_name, meta)
                in_text_citations.append(citation)
    
    citations_info = ", ".join(set(in_text_citations)) if in_text_citations else ""
    
    prompt = f"""Sen bir akademik araştırma asistanısın. Aşağıdaki soru ve bağlam bilgisine göre MUTLAKA ŞU KURALLARA UYARAK detaylı bir akademik yanıt oluştur:

KURALLAR:
1. YANIT UZUNLUĞU: Minimum 3 paragraf, 12 cümle olmalıdır. Her paragraf en az 4 cümle içermelidir.
2. AKADEMİK DİL: Profesyonel, bilimsel ve objektif bir dil kullan.
3. DETAYLI AÇIKLAMA: Her noktayı örneklerle ve detaylılarıyla açıkla.
4. YAPILANDIRMA: Giriş, gelişme ve sonuç paragrafları oluştur.
5. APA7 ALINTI: Metin içinde mutlaka şu alıntıları kullan: {citations_info}
6. ALINTI KULLANIMI: Her paragrafta en az bir alıntı kullan.
7. İKİNCİL KAYNAK: Eğer bir makalede başka bir makaleye atıf varsa, "(Özgün Yazar, Yıl, Aktaran Makale'de belirtildiği gibi)" formatında yaz.

Soru: {question}

Bağlam (Akademik Makalelerden):
{context}

DETAYLI AKADEMİK YANIT (Minimum 3 paragraf, 12 cümle + APA7 metin içi alıntılar):
"""
    
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
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
    
    # Metadata'dan citation bilgilerini oluştur
    citations = []
    if metadata:
        for item in metadata:
            if isinstance(item, dict):
                # PDF ismi varsa ekle
                pdf_name = item.get('pdf_name', 'Unknown')
                if pdf_name != 'Unknown' and pdf_name not in [c.get('pdf_name', '') for c in citations]:
                    citations.append({
                        'pdf_name': pdf_name,
                        'metadata': item
                    })
            else:
                # Eğer dict değilse, string olarak ekle
                citations.append({'pdf_name': str(item), 'metadata': {}})
    
    return answer, citations

def generate_outline(prompt):
    """Makale anahatı oluşturur"""
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system", 
                "content": """Sen bir akademik yazım uzmanısın. Verilen makale fikrini kapsamlı bir akademik makale anahatına dönüştürürsün. 
                Anahatlar detaylı, akademik standartlara uygun ve uygulanabilir olmalıdır.
                Her bölüm için açıklayıcı alt başlıklar ve temel noktalar ekle.
                Türkçe akademik dilin gerekliliklerine uygun şekilde yaz."""
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=2000
    )
    return response.choices[0].message.content

"""
Basit RAG (Retrieval-Augmented Generation) Demo
==============================================

Bu dosya, RAG (Retrieval-Augmented Generation) sisteminin temel adımlarını basit bir örnekle gösterir.
Her adım ve fonksiyon için Türkçe açıklamalar eklenmiştir.

RAG Süreci:
1. 📚 3 kısa belgeyi Chroma Vector DB'ye ekle
2. 🔍 Sorgu ile Vector DB'de arama yap
3. 📄 En yakın belgeyi bul ve bağlam olarak kullan
4. 🤖 LLM (Dil Modeli) ile yanıt üret

RAG Pipeline:
Query → Vector DB Search → Retrieve → Augment → Generate
"""


# Gerekli kütüphaneler
import numpy as np  # Sayısal işlemler için
from sentence_transformers import SentenceTransformer  # Metinleri vektöre dönüştürmek için
import chromadb  # Vektör veritabanı
import os  # Ortam değişkenleri ve dosya işlemleri
from dotenv import load_dotenv  # .env dosyasından anahtarları yüklemek için


# .env dosyasını yükle (API anahtarları vs.)
load_dotenv()


# OpenAI kütüphanesini yüklemeye çalış (LLM için)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


print("🤖 Basit RAG Demo - 4 Adımda RAG")
print("="*40)


# ADIM 1: 📚 Chroma Vector DB Kurulumu ve Belge Yükleme
print("\n📚 ADIM 1: Vector Database Setup")
print("-" * 30)


# ChromaDB istemcisi oluşturuluyor
client = chromadb.Client()

# Koleksiyon (collection) oluşturuluyor. Eğer önceden varsa silinir.
collection_name = "simple_rag_demo"
try:
    client.delete_collection(collection_name)  # Önceki koleksiyonu sil
except:
    pass

# Yeni koleksiyon oluşturuluyor (vektörler cosine benzerliği ile karşılaştırılacak)
collection = client.create_collection(
    name=collection_name,
    metadata={"hnsw:space": "cosine"}
)


# 3 kısa belge örneği (id, metin, kategori)
documents = [
    {
        "id": "doc_1",
        "text": "Python kolay öğrenilebilen, güçlü bir programlama dilidir. Web uygulamaları, veri analizi ve yapay zeka projelerinde kullanılır.",
        "category": "programming"
    },
    {
        "id": "doc_2", 
        "text": "JavaScript web tarayıcılarında çalışan dinamik bir dildir. Frontend ve backend geliştirme için Node.js ile birlikte kullanılabilir.",
        "category": "programming"
    },
    {
        "id": "doc_3",
        "text": "Machine learning algoritmaları verilerden pattern öğrenir. Supervised learning etiketli verilerle, unsupervised learning etiketsiz verilerle çalışır.",
        "category": "ai"
    }
]


# Belgeler hazırlandı, ekrana özet bilgi yazdırılıyor
print(f"✅ {len(documents)} belge hazırlandı:")
for i, doc in enumerate(documents, 1):
    print(f"{i}. [{doc['category']}] {doc['text'][:50]}...")


# ADIM 2: 🗄️ Belgeleri Vector DB'ye Yükleme
print("\n🗄️ ADIM 2: Vector DB'ye Yükleme")
print("-" * 30)


# Embedding (vektörleştirme) modeli yükleniyor
# Chroma kendi embedding yapabiliyor, burada kontrol amaçlı yükleniyor
model = SentenceTransformer('all-MiniLM-L6-v2')
print(f"✅ Embedding model: {model.get_sentence_embedding_dimension()}D")


# Belgeleri ChromaDB'ye eklemek için metin, id ve metadata listeleri hazırlanıyor
texts = [doc["text"] for doc in documents]
ids = [doc["id"] for doc in documents] 
metadatas = [{"category": doc["category"]} for doc in documents]

# Belgeler veritabanına ekleniyor
collection.add(
    documents=texts,
    metadatas=metadatas,
    ids=ids
)


print(f"✅ {collection.count()} belge Vector DB'de saklandı")
print(f"📊 Kullanılan similarity: cosine")


# ADIM 3: 🔍 Vector DB'de Arama
print("\n🔍 ADIM 3: Vector DB'de Arama")
print("-" * 30)


# Sorgu ile en yakın belgeyi bulmak için fonksiyon
def search_vector_db(query, top_k=1):
    """
    Vector DB'de arama yapar.
    - query: Kullanıcının sorgusu
    - top_k: Kaç en yakın belge dönecek (default: 1)
    """
    print(f"🔤 Sorgu: '{query}'")
    
    # ChromaDB'de arama yapılıyor
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    
    print(f"📊 Vector DB araması tamamlandı")
    print(f"🎯 Benzerlik skorları: {[f'{d:.3f}' for d in results['distances'][0]]}")
    
    # En iyi sonucu al
    best_doc_id = results['ids'][0][0]
    best_score = results['distances'][0][0]
    best_document = results['documents'][0][0]
    best_metadata = results['metadatas'][0][0]
    
    print(f"🏆 En yakın belge: {best_doc_id} (Skor: {best_score:.3f})")
    print(f"📂 Kategori: {best_metadata['category']}")
    print(f"📄 İçerik: {best_document[:100]}...")
    
    return {
        'id': best_doc_id,
        'text': best_document,
        'score': best_score,
        'metadata': best_metadata
    }


# OpenAI ile yanıt üretmek için fonksiyon
def answer_with_openai(prompt):
    """
    OpenAI GPT ile yanıt üretir.
    - prompt: LLM'e gönderilecek metin
    """
    if not OPENAI_AVAILABLE:
        return "❌ OpenAI kütüphanesi yüklü değil"
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return "❌ OPENAI_API_KEY environment variable bulunamadı"
    
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Sen yardımcı bir AI asistanısın. Verilen bağlam bilgisini kullanarak soruları yanıtla."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ OpenAI API hatası: {str(e)}"


# Test sorgusu (örnek bir arama)
query = "Python hakkında bilgi istiyorum"
retrieved_doc = search_vector_db(query)


# ADIM 4: 🤖 RAG Pipeline - Prompt + LLM
print("\n🤖 ADIM 4: RAG Pipeline - Prompt + LLM")
print("-" * 30)


# RAG için prompt (LLM'e gönderilecek metin) oluşturma fonksiyonu
def create_rag_prompt(query, context):
    """
    RAG prompt oluşturur.
    - query: Kullanıcı sorusu
    - context: Vector DB'den dönen en yakın belge
    """
    prompt = f"""Aşağıdaki bağlam bilgisini kullanarak soruyu yanıtla:

BAĞLAM: {context['text']}

SORU: {query}

YANIT:"""
    return prompt


# Tüm RAG sürecini birleştiren fonksiyon
def rag_pipeline(query, use_openai=False):
    """
    Tam RAG pipeline'ı çalıştırır.
    - query: Kullanıcı sorgusu
    - use_openai: Gerçek LLM mi kullanılacak (True) yoksa mock mu (False)
    """
    print(f"🚀 RAG Pipeline başlatılıyor...")
    
    # 1. Vector DB'de arama
    context = search_vector_db(query)
    

    # 2. Prompt oluştur
    prompt = create_rag_prompt(query, context)
    print(f"\n📝 RAG Prompt oluşturuldu ({len(prompt)} karakter)")
    
    # 3. LLM ile yanıt al
    if use_openai:
        print(f"🤖 OpenAI GPT ile yanıt alınıyor...")
        response = answer_with_openai(prompt)
    else:
        print(f"🎭 Mock yanıt oluşturuluyor...")
        response = f"""✅ RAG Pipeline Demo Tamamlandı!

🔍 Retrieval Sonucu:
- Belge: {context['id']} ({context['metadata']['category']})
- Similarity Score: {context['score']:.3f}

📝 Bağlam: "{context['text'][:100]}..."

🤖 Gerçek LLM için .env dosyasında OPENAI_API_KEY ayarlayın.

Bu demo Vector DB + Context + LLM akışını gösterir."""
    
    return {
        'query': query,
        'context': context,
        'prompt': prompt,
        'response': response
    }


# RAG pipeline'ı test etmek için ana akış
print(f"\n" + "="*50)
print(f"🎯 RAG PİPELİNE TEST")
print(f"="*50)

# OpenAI API anahtarı varsa gerçek LLM kullanılır, yoksa mock yanıt döner
api_key = os.getenv('OPENAI_API_KEY')
use_real_llm = api_key is not None and OPENAI_AVAILABLE

result = rag_pipeline(query, use_openai=use_real_llm)

print(f"\n🎭 RAG SONUCU:")
print("-" * 30)
print(result['response'])


# BONUS: Farklı sorgularla Vector DB testleri
print(f"\n🧪 BONUS: Farklı Sorgular ile Vector DB Test")
print("-" * 40)

test_queries = [
    "JavaScript nedir?",
    "Machine learning nasıl çalışır?", 
    "Web geliştirme için hangi dil?",
    "Yapay zeka algoritmaları"
]

for i, test_query in enumerate(test_queries, 1):
    print(f"\n🔍 Test {i}: {test_query}")
    context = search_vector_db(test_query)
    print(f"   ➡️ {context['id']} seçildi (Skor: {context['score']:.3f})")


# RAG sistemi özet bilgisi
print(f"\n📋 MODERN RAG SİSTEMİ ÖZETİ")
print("="*50)

summary = f"""
🔄 RAG PIPELINE:
1. 📚 Documents → Vector Database (Chroma)
2. 🔍 Query → Vector Search (Similarity)
3. 🎯 Retrieve (En yakın belgeler)
4. ✍️ Augment (Prompt + Context)
5. 🤖 Generate (LLM Response)

💡 KULLANILAN TEKNOLOJİLER:
• Vector DB: ChromaDB (cosine similarity)
• Embeddings: Sentence Transformers
• LLM: OpenAI GPT-3.5-turbo
• Framework: Python + dotenv

🚀 PRODUCTION HAZIR:
• ✅ Vector Database entegrasyonu
• ✅ API key yönetimi (.env)
• ✅ Error handling
• ✅ Metadata desteği

📊 DEMO İSTATİSTİKLER:
• Toplam belge: {collection.count()}
• Vector DB: ChromaDB
• Embedding boyutu: {model.get_sentence_embedding_dimension()}D
• LLM: {"OpenAI GPT" if use_real_llm else "Mock Response"}
"""

print(summary)

print("✅ Modern RAG demo tamamlandı!")
print("🔗 Daha kapsamlı version için: rag_system.py")
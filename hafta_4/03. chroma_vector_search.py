"""
Chroma DB ile Vektör Arama
==========================

Bu kod Chroma DB kullanarak:
1. Vektör koleksiyonu oluşturma
2. Metadata ile birlikte vektör saklama
3. Semantik arama yapma

Chroma DB Avantajları:
- Kolay kullanım (high-level API)
- Metadata desteği
- Otomatik persistence
- Gömülü veritabanı

Gerekli Kütüphaneler:
pip install chromadb numpy
"""

# Gerekli kütüphaneleri import ediyoruz
import chromadb        # Chroma vektör veritabanı kütüphanesi
import numpy as np     # Sayısal işlemler ve vektör operasyonları için
import time           # Performans ölçümü için zaman takibi
from typing import List, Dict  # Tip tanımlamaları için

# Kullanıcıya ne yapılacağını bildirelim
print("🎨 Chroma DB ile Vektör Arama Öğreticisi")
print("="*50)

# Adım 1: Chroma Client ve Collection oluşturma
print("\n📚 1. Chroma Client ve Collection Oluşturma")
print("-" * 40)

# Chroma DB client'ını başlatıyoruz (bellek içinde çalışacak)
# Bu client veritabanı ile iletişim kurmamızı sağlar
client = chromadb.Client()

# Koleksiyon adını belirliyoruz - koleksiyon vektörlerin saklandığı tabloya benzer
collection_name = "vector_search_demo"

try:
    # Eğer aynı isimde koleksiyon varsa siliyoruz (temiz başlangıç için)
    client.delete_collection(collection_name)
except:
    # Koleksiyon yoksa hata almayalım diye pass
    pass

# Yeni koleksiyon oluşturuyoruz
collection = client.create_collection(
    name=collection_name,
    metadata={"hnsw:space": "cosine"}  # Benzerlik ölçümü olarak cosine similarity kullanılacak
)

print(f"✅ Collection oluşturuldu: {collection_name}")
print(f"📊 Similarity metric: cosine")  # Cosine similarity: vektörler arası açıyı ölçer (0-2 arası)

# Adım 2: Örnek vektörler ve metadata oluşturma
print("\n🗂️  2. Vektörler ve Metadata Hazırlama")
print("-" * 40)

# Vektör boyutunu ve sayısını belirliyoruz
dimension = 512     # Her vektör 512 boyutlu olacak (tipik embedding boyutu)
n_vectors = 1000    # Toplamda 1000 vektör oluşturacağız

# Rastgele vektörler oluşturuyoruz (gerçek projede bu embeddings olur)
# Seed kullanarak her çalıştırma da aynı sonuçları alıyoruz
np.random.seed(42)
vectors = np.random.random((n_vectors, dimension)).astype('float32').tolist()

# Her vektör için benzersiz ID oluşturuyoruz
# ID'ler vektörleri tanımlamak için kullanılır
ids = [f"vec_{i}" for i in range(n_vectors)]

# Metadata oluşturma: Her vektörle ilişkili ek bilgiler
# Bu bilgiler filtreleme ve sıralama için kullanılır
categories = ["teknoloji", "spor", "sanat", "bilim", "müzik"]
metadatas = []  # Her vektörün metadata'sını saklayacak liste
documents = []  # Her vektörün temsil ettiği belgeyi saklayacak liste

for i in range(n_vectors):
    # Her vektör için sırayla kategori atıyoruz (döngüsel olarak)
    category = categories[i % len(categories)]
    
    # Her vektör için metadata objesi oluşturuyoruz
    metadatas.append({
        "category": category,           # Vektörün kategorisi
        "index": i,                    # Vektörün sıra numarası
        "group": f"group_{i // 100}"   # 100'lük gruplar (group_0, group_1, vs.)
    })
    
    # Her vektör için örnek belge metni oluşturuyoruz
    documents.append(f"Bu {category} kategorisinden örnek belge {i}")

print(f"✅ {n_vectors} vektör hazırlandı")
print(f"📝 Kategoriler: {categories}")
print(f"🏷️  Her vektör için metadata ve doküman oluşturuldu")

# Adım 3: Vektörleri Chroma'ya ekleme
print("\n💾 3. Vektörleri Chroma'ya Ekleme")
print("-" * 40)

# Ekleme işleminin ne kadar sürdüğünü ölçmek için başlangıç zamanını kaydet
start_time = time.time()

# Performans için vektörleri küçük gruplar halinde (batch) ekleyeceğiz
# Tek seferde çok vektör eklemek bellek sorunlarına yol açabilir
batch_size = 100

for i in range(0, n_vectors, batch_size):
    # Batch'in bitiş index'ini hesapla (son batch daha küçük olabilir)
    end_idx = min(i + batch_size, n_vectors)
    
    # Bu batch'teki vektörleri koleksiyona ekle
    collection.add(
        embeddings=vectors[i:end_idx],     # Vektörlerin kendisi
        metadatas=metadatas[i:end_idx],    # Her vektörün metadata'sı
        documents=documents[i:end_idx],     # Her vektörün temsil ettiği belge
        ids=ids[i:end_idx]                 # Her vektörün benzersiz ID'si
    )

# Ekleme işleminin toplam süresini hesapla
add_time = time.time() - start_time
print(f"✅ {collection.count()} vektör eklendi")
print(f"⏱️  Ekleme süresi: {add_time:.4f} saniye")

# Adım 4: Basit vektör arama
print("\n🔍 4. Vektör Arama İşlemleri")
print("-" * 40)

# Arama yapmak için sorgu vektörü oluşturuyoruz
# Gerçek uygulamada bu kullanıcının sorusunun embedding'i olur
query_vector = np.random.random(dimension).astype('float32').tolist()

# Arama işleminin ne kadar sürdüğünü ölçmek için başlangıç zamanını kaydet
start_time = time.time()

# Sorgu vektörüne en yakın 5 vektörü bul
# Chroma otomatik olarak cosine similarity kullanarak en yakınları bulur
results = collection.query(
    query_embeddings=[query_vector],  # Aranacak vektör (liste şeklinde verilmeli)
    n_results=5                       # Kaç sonuç istediğimiz
)

# Arama işleminin süresini hesapla
search_time = time.time() - start_time

print(f"🎯 Arama süresi: {search_time:.4f} saniye")
print(f"📊 Bulunan sonuç sayısı: {len(results['ids'][0])}")

print("\n📋 Arama Sonuçları:")
# Bulunan her sonuç için detayları yazdır
for i, (doc_id, distance, metadata, document) in enumerate(zip(
    results['ids'][0],          # Bulunan vektörlerin ID'leri
    results['distances'][0],    # Sorgu vektörüne olan mesafeleri
    results['metadatas'][0],    # Bulunan vektörlerin metadata'ları
    results['documents'][0]     # Bulunan vektörlerin belgeleri
)):
    print(f"  {i+1}. ID: {doc_id}")
    print(f"     Mesafe: {distance:.4f}")                    # Düşük mesafe = yüksek benzerlik
    print(f"     Kategori: {metadata['category']}")
    print(f"     Grup: {metadata['group']}")
    print(f"     Doküman: {document[:50]}...")              # İlk 50 karakteri göster
    print()

# Adım 5: Metadata ile filtreleme
print("\n🎛️  5. Metadata ile Filtreleme")
print("-" * 40)

# Sadece belirli kriterleri karşılayan vektörler arasında arama yapabiliriz
# Bu özellik Chroma DB'nin en güçlü yanlarından biri

# Örnek 1: Sadece "teknoloji" kategorisindeki vektörler arasında ara
tech_results = collection.query(
    query_embeddings=[query_vector],  # Aynı sorgu vektörünü kullan
    n_results=3,                      # 3 sonuç iste
    where={"category": "teknoloji"}   # Sadece teknoloji kategorisinde ara
)

print("🔬 Sadece 'teknoloji' kategorisindeki sonuçlar:")
for i, (doc_id, distance, metadata) in enumerate(zip(
    tech_results['ids'][0],
    tech_results['distances'][0],
    tech_results['metadatas'][0]
)):
    print(f"  {i+1}. ID: {doc_id}, Mesafe: {distance:.4f}, Kategori: {metadata['category']}")

# Örnek 2: Karmaşık filtreler - birden fazla koşulu birleştirme
# $and: VE operatörü, $in: İÇERİR operatörü, $gte: BÜYÜK EŞİT operatörü
complex_results = collection.query(
    query_embeddings=[query_vector],
    n_results=3,
    where={
        "$and": [                                          # VE operatörü: her iki koşul da sağlanmalı
            {"category": {"$in": ["teknoloji", "bilim"]}}, # Kategori teknoloji VEYA bilim olmalı
            {"index": {"$gte": 100}}                       # Index 100'den büyük veya eşit olmalı
        ]
    }
)

print("\n🧪 Karmaşık filtre (teknoloji VEYA bilim VE index >= 100):")
for i, (doc_id, distance, metadata) in enumerate(zip(
    complex_results['ids'][0],
    complex_results['distances'][0],
    complex_results['metadatas'][0]
)):
    print(f"  {i+1}. ID: {doc_id}, Mesafe: {distance:.4f}")
    print(f"     Kategori: {metadata['category']}, Index: {metadata['index']}")

# Adım 6: Koleksiyon istatistikleri
print("\n📈 6. Koleksiyon İstatistikleri")
print("-" * 40)

# Koleksiyondaki toplam vektör sayısını öğren
print(f"📊 Toplam vektör sayısı: {collection.count()}")

# Kategorilere göre dağılım analizi yapalım
# Bunun için tüm metadata'ları çekiyoruz
category_counts = {}  # Her kategoriden kaç tane olduğunu saklayacak sözlük

# Koleksiyondaki tüm metadata'ları getir (sadece metadata kısmını)
all_metadatas = collection.get(include=['metadatas'])['metadatas']

# Her metadata'yı kontrol et ve kategori sayılarını say
for metadata in all_metadatas:
    category = metadata['category']
    # Eğer bu kategori daha önce görülmediyse 0'dan başla, yoksa 1 artır
    category_counts[category] = category_counts.get(category, 0) + 1

print("\n📈 Kategori Dağılımı:")
for category, count in category_counts.items():
    print(f"  {category}: {count} vektör")

# Adım 7: FAISS vs Chroma karşılaştırması
print("\n⚖️  7. FAISS vs Chroma DB Karşılaştırması")
print("-" * 40)

# İki popüler vektör veritabanının detaylı karşılaştırması
comparison = """
🏃‍♂️ HIZ KARŞILAŞTIRMASI:
• FAISS: Çok hızlı (C++ ile yazılmış, düşük seviye optimizasyonlar)
• Chroma: Orta hızlı (Python tabanlı, kullanım kolaylığı odaklı)

🛠️  KULLANIM KOLAYLIĞI:
• FAISS: Düşük seviye API, vektör indeksleme bilgisi gerekli
• Chroma: Yüksek seviye API, hızlı başlangıç, Python-friendly

📏 ÖLÇEKLENEBİLİRLİK:
• FAISS: Milyarlarca vektör destekler, büyük ölçekli sistemler için
• Chroma: Milyonlarca vektör, orta ölçekli uygulamalar için ideal

🎯 ÖZELLİK KARŞILAŞTIRMASI:
• FAISS: Sadece vektör benzerlik arama, saf matematik odaklı
• Chroma: Metadata filtreleme, otomatik persistence, REST API

💾 BELLEK YÖNETİMİ:
• FAISS: Manuel bellek yönetimi, optimize edilmiş veri yapıları
• Chroma: Otomatik bellek yönetimi, kolay kullanım

🔧 GPU DESTEĞİ:
• FAISS: Mükemmel GPU desteği, CUDA optimizasyonları
• Chroma: Sınırlı GPU desteği, ağırlıklı CPU tabanlı

✅ HANGI DURUMDA HANGİSİNİ KULLANMALI:

🚀 FAISS KULLAN:
• Çok büyük veri setleri (>10 milyon vektör)
• Maksimum hız kritik (milisaniye seviyesi)
• GPU kullanımı gerekli
• Bare-metal performans istiyorsan
• Özel indeksleme algoritmaları gerekli

🎨 CHROMA KULLAN:
• Hızlı prototipleme ve geliştirme
• Metadata ile zengin arama özellikleri
• Kolay deployment ve yönetim
• Web uygulamaları ve API'lar
• Ekip geliştirme (öğrenme eğrisi düşük)
"""

print(comparison)

print("\n✅ Chroma DB öğreticisi tamamlandı!")
print(f"🗄️  Koleksiyon: {collection.count()} vektör içeriyor")
print("\n💡 ÖZETİ Bu kodda öğrendiğiniz konular:")
print("   • Chroma DB client oluşturma ve koleksiyon yönetimi")
print("   • Vektör + metadata + belge ekleme işlemleri")
print("   • Semantik benzerlik araması (cosine similarity)")
print("   • Metadata tabanlı filtreleme ve karmaşık sorgular")
print("   • Performans ölçümü ve batch işleme")
print("   • FAISS vs Chroma DB karşılaştırması")
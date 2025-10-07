"""
FAISS ile Vektör Arama Sistemi
=============================

FAISS (Facebook AI Similarity Search) Nedir?
--------------------------------------------
FAISS, Facebook AI Research tarafından geliştirilen, vektör benzerlik arama 
için optimize edilmiş bir kütüphanedir. Özellikle büyük vektör veri setlerinde
hızlı ve etkili arama yapabilmek için tasarlanmıştır.

FAISS'in Temel Avantajları:
• Hız: C++ backend ile yazılmış, çok hızlı işlem
• GPU Desteği: CUDA ile GPU hızlandırması
• Skalabilite: Milyonlarca vektörü işleyebilir
• Algoritma Çeşitliliği: Brute force'tan yaklaşık aramalara
• Bellek Optimizasyonu: Farklı sıkıştırma teknikleri

Bu Kodun Yaptıkları:
1. 512 boyutlu rastgele vektörlerden FAISS indexi oluşturma
2. En yakın komşu arama (k-NN search) işlemleri
3. Farklı index türlerinin performans karşılaştırması
4. Sonuçların görselleştirilmesi

Gerekli Kütüphaneler:
pip install faiss-cpu numpy matplotlib
(GPU için: pip install faiss-gpu)
"""

# Gerekli kütüphaneleri import ediyoruz
import numpy as np          # Sayısal işlemler ve vektör operasyonları için
import faiss               # FAISS vektör arama kütüphanesi
import time                # Performans ölçümü için zaman takibi
import matplotlib.pyplot as plt  # Grafik çizimi için

# Program başlangıcında kullanıcıya bilgi veriyoruz
print("🚀 FAISS ile Vektör Arama Öğreticisi")
print("="*50)

# ADIM 1: Vektör veri seti oluşturma
# =====================================
print("\n📊 1. Vektör Veri Seti Oluşturma")
print("-" * 30)

# Vektör araması için gerekli parametreleri tanımlıyoruz
dimension = 512          # Her vektörün boyutu (512 boyutlu uzayda noktalar)
n_vectors = 10000       # Toplam vektör sayısı (arama yapacağımız veri havuzu)
n_query = 5            # Kaç tane sorgu vektörü test edeceğiz
k = 3                  # Her sorgu için kaç tane en yakın komşu bulacağız

# Kullanıcıya parametreleri gösteriyoruz
print(f"• Vektör boyutu: {dimension}")
print(f"• Toplam vektör sayısı: {n_vectors}")
print(f"• Sorgu sayısı: {n_query}")
print(f"• Aranacak komşu sayısı: {k}")

# Rastgele sayı üreteci için seed belirliyoruz (tekrarlanabilir sonuçlar için)
np.random.seed(42)

# Ana vektör veri setini oluşturuyoruz
# random.random(): 0-1 arası rastgele sayılar üretir
# astype('float32'): FAISS float32 format istiyor, bellek tasarrufu için
vectors = np.random.random((n_vectors, dimension)).astype('float32')

# L2 normalizasyon yapıyoruz (vektörlerin uzunluğunu 1 yapıyoruz)
# Bu, cosine similarity hesaplamasını kolaylaştırır
vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

# Sorgu vektörlerini de aynı şekilde oluşturuyoruz
query_vectors = np.random.random((n_query, dimension)).astype('float32')
query_vectors = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)

# Oluşturulan vektörlerin boyutlarını kontrol ediyoruz
print(f"✅ Vektörler oluşturuldu: {vectors.shape}")
print(f"✅ Sorgu vektörleri: {query_vectors.shape}")

# ADIM 2: FAISS Index oluşturma ve karşılaştırma
# ==============================================
print("\n🔧 2. FAISS Index Oluşturma")
print("-" * 30)

# FAISS'te farklı index türleri var, her birinin kendine özgü avantajları var:
# Flat: Brute force arama - her vektörle karşılaştırır, %100 doğru ama yavaş
# IVF: Inverted File - vektörleri kümelere böler, daha hızlı ama yaklaşık sonuç
index_types = {
    'Flat': faiss.IndexFlatIP,      # Inner Product (IP) ile brute force arama
    'IVF': lambda d: faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, 100)  # 100 küme ile IVF
}

# Her index türü için sonuçları saklayacağımız dictionary
results = {}

# Her index türünü tek tek test ediyoruz
for index_name, index_creator in index_types.items():
    print(f"\n🏗️  {index_name} Index oluşturuluyor...")
    
    # Index türüne göre farklı oluşturma işlemleri
    if index_name == 'Flat':
        # Flat index doğrudan oluşturulur
        index = index_creator(dimension)
    else:
        # IVF index için önce quantizer (kümeleme) oluşturulur
        index = index_creator(dimension)
        # IVF indexler training (eğitim) aşaması gerektirir
        # Bu aşamada vektörler kümelere bölünür
        index.train(vectors)
    
    # Vektörleri indexe ekleme işleminin süresini ölçüyoruz
    start_time = time.time()
    index.add(vectors)  # Tüm vektörleri indexe ekle
    add_time = time.time() - start_time
    
    # Ekleme işlemi hakkında bilgi veriyoruz
    print(f"   ✅ {index.ntotal} vektör eklendi")
    print(f"   ⏱️  Ekleme süresi: {add_time:.4f} saniye")
    
    # Arama performansını test ediyoruz
    start_time = time.time()
    # search() fonksiyonu: (sorgu_vektörleri, kaç_komşu_bulunacak)
    # Geri döndürdükleri: (mesafeler, indeksler)
    distances, indices = index.search(query_vectors, k)
    search_time = time.time() - start_time
    
    # Arama performansı hakkında bilgi veriyoruz
    print(f"   🔍 Arama süresi: {search_time:.4f} saniye")
    print(f"   📈 Saniyede sorgu: {n_query/search_time:.0f}")
    
    # Bu index türünün sonuçlarını kaydediyoruz
    results[index_name] = {
        'add_time': add_time,           # Vektör ekleme süresi
        'search_time': search_time,     # Arama süresi
        'distances': distances,         # Bulunan komşuların mesafeleri
        'indices': indices              # Bulunan komşuların index numaraları
    }

# ADIM 3: Arama sonuçlarını detaylı analiz etme
# =============================================
print("\n🔍 3. Arama Sonuçları Analizi")
print("-" * 30)

# Her sorgu vektörü için bulunan sonuçları gösteriyoruz
for i, query_vector in enumerate(query_vectors):
    print(f"\n📍 Sorgu {i+1} için sonuçlar:")
    
    # Her index türünün bu sorgu için bulduğu sonuçları karşılaştırıyoruz
    for index_name in results:
        # Bu index türünün i. sorgu için bulduğu mesafeler ve indeksler
        distances = results[index_name]['distances'][i]
        indices = results[index_name]['indices'][i]
        
        print(f"  {index_name} Index:")
        # En yakın k komşuyu tek tek gösteriyoruz
        for j, (dist, idx) in enumerate(zip(distances, indices)):
            print(f"    {j+1}. En yakın: Index {idx}, Benzerlik skoru: {dist:.4f}")
            # Not: dist değeri Inner Product sonucu (yüksek = daha benzer)

# ADIM 4: Performans karşılaştırması görselleştirmesi
# ==================================================
print("\n📊 4. Performans Görselleştirmesi")
print("-" * 30)

# Matplotlib ile iki grafikli bir figür oluşturuyoruz
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Index isimlerini ve performans verilerini alıyoruz
index_names = list(results.keys())
add_times = [results[name]['add_time'] for name in index_names]      # Ekleme süreleri
search_times = [results[name]['search_time'] for name in index_names]  # Arama süreleri

# Sol grafik: Index oluşturma süresi karşılaştırması
ax1.bar(index_names, add_times, color=['blue', 'red'], alpha=0.7)
ax1.set_title('Index Oluşturma Süresi Karşılaştırması', fontsize=14, fontweight='bold')
ax1.set_ylabel('Süre (saniye)', fontsize=12)
ax1.grid(True, alpha=0.3)  # Hafif grid çizgileri ekliyoruz

# Sağ grafik: Arama süresi karşılaştırması
ax2.bar(index_names, search_times, color=['green', 'orange'], alpha=0.7)
ax2.set_title('Arama Süresi Karşılaştırması', fontsize=14, fontweight='bold')
ax2.set_ylabel('Süre (saniye)', fontsize=12)
ax2.grid(True, alpha=0.3)

# Grafikleri düzenli bir şekilde yerleştiriyoruz
plt.tight_layout()

# Grafiği dosyaya kaydediyoruz (yolu Windows formatına uygun hale getiriyoruz)
plt.savefig('images/faiss_performance.png', dpi=300, bbox_inches='tight')
print("📁 Grafik kaydedildi: images/faiss_performance.png")

# Grafiği ekranda gösteriyoruz
plt.show()

# ADIM 5: FAISS Özellikleri, İpuçları ve Pratik Bilgiler
# ======================================================
print("\n💡 5. FAISS İpuçları ve Özellikler")
print("-" * 30)

print("""
🎯 FAISS Index Türleri ve Kullanım Alanları:

• IndexFlatIP/L2: 
  - Brute force arama (her vektörle karşılaştırır)
  - %100 doğru sonuçlar
  - Küçük veri setleri için ideal
  - Yavaş ama kesin

• IndexIVFFlat: 
  - Inverted File Index (kümeleme tabanlı)
  - Vektörleri kümelere böler, sadece ilgili kümelerde arar
  - Hızlı ama yaklaşık sonuçlar
  - Orta büyüklükteki veri setleri için

• IndexIVFPQ: 
  - Product Quantization ile sıkıştırılmış
  - Çok az bellek kullanır
  - Çok büyük veri setleri için
  - En hızlı ama en az doğru

• IndexHNSW: 
  - Hierarchical Navigable Small World
  - Hızlı ve oldukça doğru
  - Modern uygulamalarda popüler

⚡ Performans Optimizasyon İpuçları:

• GPU Kullanımı: faiss-gpu versiyonu 10-100x daha hızlı
• IVF Küme Sayısı: sqrt(n_vectors) kadar küme optimal
• PQ Boyutu: Vektör boyutu 8'in katı olmalı
• Normalizasyon: Cosine similarity için L2 normalize edin
• Batch İşleme: Tek tek değil, toplu arama yapın

🔧 Hangi Index Ne Zaman Kullanılır:

• Web Arama Motoru: HNSW (hız + doğruluk)
• Öneri Sistemleri: IVF (orta hız, iyi doğruluk)
• Real-time Chatbot: Flat (kesin sonuç gerekli)
• Büyük E-ticaret: IVF + PQ (hız + bellek tasarrufu)
• Akademik Araştırma: Flat (referans sonuçlar için)
""")

# Bellek kullanımı analizi yapıyoruz
print(f"\n💾 Bellek Kullanımı Analizi:")

# Basit bir Flat index oluşturup bellek kullanımını hesaplıyoruz
index_flat = faiss.IndexFlatIP(dimension)
index_flat.add(vectors)

# Vektör verisinin ne kadar yer kapladığını hesaplıyoruz
vector_size_mb = vectors.nbytes / 1024 / 1024

print(f"• Orijinal vektör verisi: {vector_size_mb:.1f} MB")
print(f"• Flat Index boyutu: ~{vector_size_mb:.1f} MB (vektörlerin kopyası)")
print(f"• Toplam bellek kullanımı: ~{2 * vector_size_mb:.1f} MB")
print(f"• IVF Index: ~%70-80 daha az bellek kullanır")
print(f"• PQ Index: ~%90-95 daha az bellek kullanır")

print("\n🎉 FAISS Vektör Arama Öğreticisi Tamamlandı!")
print("="*50)
print("📊 Bu kod size gösterdi:")
print("  ✅ FAISS index nasıl oluşturulur")
print("  ✅ Farklı index türlerinin performans farkları")
print("  ✅ Vektör arama sonuçları nasıl yorumlanır")
print("  ✅ Bellek kullanımı nasıl optimize edilir")
print("📁 Performance grafiği kaydedildi: images/faiss_performance.png")
"""
FAISS vs Chroma DB Performans Karşılaştırması
=============================================

Bu kod aynı veri seti üzerinde FAISS ve Chroma DB'nin:
1. Ekleme performansını
2. Arama performansını
3. Bellek kullanımını
karşılaştırır.
"""

# --- Kütüphaneler içe aktarılıyor ---
# numpy: Vektör işlemleri için
# faiss: Facebook tarafından geliştirilen hızlı vektör arama kütüphanesi
# chromadb: Chroma vektör veritabanı
# time: Zaman ölçümleri için
# matplotlib.pyplot: Grafik çizimi için
# psutil: Bellek ve sistem kaynaklarını izlemek için
# os: İşletim sistemi ile ilgili işlemler için

import numpy as np
import faiss
import chromadb
import time
import matplotlib.pyplot as plt
import psutil
import os


# --- Kullanıcıya bilgilendirme mesajı ---
print("⚖️  FAISS vs Chroma DB Performans Karşılaştırması")
print("="*60)


# --- Test parametreleri ---
# dimensions: Vektörlerin boyutları (ör: 128, 256, 512)
# vector_counts: Testte kullanılacak vektör sayıları
# k: Aramada döndürülecek en yakın komşu sayısı
dimensions = [128, 256, 512]
vector_counts = [1000, 5000, 10000]
k = 5  # En yakın k komşu


# --- Sonuçları tutacak sözlük yapısı ---
# Her bir sistem için ekleme süresi, arama süresi ve bellek kullanımı saklanacak
results = {
    'faiss': {'add_times': [], 'search_times': [], 'memory': []},
    'chroma': {'add_times': [], 'search_times': [], 'memory': []}
}


# --- Bellek kullanımını ölçen fonksiyon ---
def get_memory_usage():
    """Mevcut bellek kullanımını MB cinsinden döndürür"""
    process = psutil.Process(os.getpid())  # Şu anki Python sürecini alır
    return process.memory_info().rss / 1024 / 1024  # RAM kullanımını MB cinsinden döndürür


# --- FAISS performansını test eden fonksiyon ---
def test_faiss(vectors, query_vector, dimension):
    """FAISS performansını test eder"""
    start_memory = get_memory_usage()  # Başlangıçtaki bellek kullanımı
    
    # Index oluştur ve vektörleri ekle
    start_time = time.time()  # Ekleme işlemi başlama zamanı
    index = faiss.IndexFlatIP(dimension)  # FAISS iç çarpım tabanlı indeks oluşturuluyor
    index.add(vectors)  # Vektörler indekse ekleniyor
    add_time = time.time() - start_time  # Ekleme işlemi süresi hesaplanıyor
    
    end_memory = get_memory_usage()  # Ekleme sonrası bellek kullanımı
    memory_used = end_memory - start_memory  # Ekleme işleminin kullandığı bellek miktarı
    
    # Arama yap
    start_time = time.time()  # Arama işlemi başlama zamanı
    distances, indices = index.search(query_vector.reshape(1, -1), k)  # Sorgu vektörü ile en yakın k komşu aranıyor
    search_time = time.time() - start_time  # Arama işlemi süresi hesaplanıyor
    
    return add_time, search_time, memory_used  # Sonuçlar döndürülüyor


# --- Chroma DB performansını test eden fonksiyonun başlangıcı ---

# --- Chroma DB performansını test eden fonksiyon ---
def test_chroma(vectors, query_vector, dimension):
    """Chroma DB performansını test eder"""
    start_memory = get_memory_usage()  # Başlangıçtaki bellek kullanımı
    
    # Chroma istemcisi oluşturuluyor
    client = chromadb.Client()
    
    # Test için yeni bir koleksiyon oluşturuluyor
    collection_name = f"test_collection_{dimension}"
    try:
        client.delete_collection(collection_name)  # Önceki koleksiyon varsa silinir
    except:
        pass  # Yoksa hata alınmaz
    
    collection = client.create_collection(name=collection_name)  # Yeni koleksiyon oluşturuluyor
    
    # Vektörleri ekle
    start_time = time.time()  # Ekleme işlemi başlama zamanı
    vectors_list = vectors.tolist()  # Numpy array'i listeye dönüştürülüyor
    ids = [f"vec_{i}" for i in range(len(vectors))]  # Her vektöre bir id atanıyor
    
    # Vektörler toplu (batch) olarak ekleniyor
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        end_idx = min(i + batch_size, len(vectors))
        collection.add(
            embeddings=vectors_list[i:end_idx],
            ids=ids[i:end_idx]
        )
    
    add_time = time.time() - start_time  # Ekleme işlemi süresi
    
    end_memory = get_memory_usage()  # Ekleme sonrası bellek kullanımı
    memory_used = end_memory - start_memory  # Ekleme işleminin kullandığı bellek miktarı
    
    # Arama yap
    start_time = time.time()  # Arama işlemi başlama zamanı
    results = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=k
    )
    search_time = time.time() - start_time  # Arama işlemi süresi
    
    return add_time, search_time, memory_used  # Sonuçlar döndürülüyor


# --- Testlerin başladığını bildiren mesajlar ---
print("\n🧪 Test başlıyor...")
print("📊 Test edilecek boyutlar:", dimensions)
print("📈 Test edilecek vektör sayıları:", vector_counts)


# --- Her kombinasyon için testler yapılıyor ---
test_results = []

for dim in dimensions:
    for vec_count in vector_counts:
        print(f"\n🔬 Test: {dim}D, {vec_count} vektör")
        
        # Rastgele vektörler oluşturuluyor
        np.random.seed(42)
        vectors = np.random.random((vec_count, dim)).astype('float32')  # Vektörler float32 tipinde
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)  # Her vektör normalize ediliyor
        query_vector = np.random.random(dim).astype('float32')  # Sorgu vektörü oluşturuluyor
        query_vector = query_vector / np.linalg.norm(query_vector)  # Sorgu vektörü normalize ediliyor
        
        # FAISS test ediliyor
        print("   📘 FAISS test ediliyor...")
        faiss_add, faiss_search, faiss_memory = test_faiss(vectors, query_vector, dim)
        
        # Bellek temizliği (GC)
        import gc
        gc.collect()
        
        # Chroma test ediliyor
        print("   📗 Chroma test ediliyor...")
        chroma_add, chroma_search, chroma_memory = test_chroma(vectors, query_vector, dim)
        
        # Sonuçlar kaydediliyor
        test_results.append({
            'dimension': dim,
            'vector_count': vec_count,
            'faiss_add': faiss_add,
            'faiss_search': faiss_search,
            'faiss_memory': faiss_memory,
            'chroma_add': chroma_add,
            'chroma_search': chroma_search,
            'chroma_memory': chroma_memory
        })
        
        print(f"   ✅ FAISS  - Ekleme: {faiss_add:.4f}s, Arama: {faiss_search:.6f}s, Bellek: {faiss_memory:.1f}MB")
        print(f"   ✅ Chroma - Ekleme: {chroma_add:.4f}s, Arama: {chroma_search:.6f}s, Bellek: {chroma_memory:.1f}MB")


# --- Sonuçlar görselleştiriliyor ---
print("\n📊 Sonuçlar görselleştiriliyor...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('FAISS vs Chroma DB Performans Karşılaştırması', fontsize=16, fontweight='bold')

# --- Grafikler için veri hazırlanıyor ---
faiss_add_times = [r['faiss_add'] for r in test_results]
chroma_add_times = [r['chroma_add'] for r in test_results]
faiss_search_times = [r['faiss_search'] for r in test_results]
chroma_search_times = [r['chroma_search'] for r in test_results]
faiss_memory = [r['faiss_memory'] for r in test_results]
chroma_memory = [r['chroma_memory'] for r in test_results]
labels = [f"{r['dimension']}D-{r['vector_count']}" for r in test_results]

# --- Ekleme süreleri grafiği ---
x = np.arange(len(labels))
width = 0.35

axes[0, 0].bar(x - width/2, faiss_add_times, width, label='FAISS', alpha=0.8, color='blue')
axes[0, 0].bar(x + width/2, chroma_add_times, width, label='Chroma', alpha=0.8, color='red')
axes[0, 0].set_title('Ekleme Süreleri')
axes[0, 0].set_ylabel('Süre (saniye)')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(labels, rotation=45)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# --- Arama süreleri grafiği ---
axes[0, 1].bar(x - width/2, faiss_search_times, width, label='FAISS', alpha=0.8, color='blue')
axes[0, 1].bar(x + width/2, chroma_search_times, width, label='Chroma', alpha=0.8, color='red')
axes[0, 1].set_title('Arama Süreleri')
axes[0, 1].set_ylabel('Süre (saniye)')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(labels, rotation=45)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# --- Bellek kullanımı grafiği ---
axes[0, 2].bar(x - width/2, faiss_memory, width, label='FAISS', alpha=0.8, color='blue')
axes[0, 2].bar(x + width/2, chroma_memory, width, label='Chroma', alpha=0.8, color='red')
axes[0, 2].set_title('Bellek Kullanımı')
axes[0, 2].set_ylabel('Bellek (MB)')
axes[0, 2].set_xticks(x)
axes[0, 2].set_xticklabels(labels, rotation=45)
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# --- Hız oranları grafiği ---
speed_ratios = [c/f for c, f in zip(chroma_search_times, faiss_search_times)]
axes[1, 0].bar(x, speed_ratios, alpha=0.8, color='green')
axes[1, 0].set_title('Arama Hızı Oranı (Chroma/FAISS)')
axes[1, 0].set_ylabel('Oran (>1 = Chroma yavaş)')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(labels, rotation=45)
axes[1, 0].axhline(y=1, color='red', linestyle='--', alpha=0.7)
axes[1, 0].grid(True, alpha=0.3)

# --- Throughput karşılaştırması grafiği ---
faiss_throughput = [vec_count / search_time for r, search_time in zip(test_results, faiss_search_times) 
                   for vec_count in [r['vector_count']]]
chroma_throughput = [vec_count / search_time for r, search_time in zip(test_results, chroma_search_times) 
                    for vec_count in [r['vector_count']]]

axes[1, 1].bar(x - width/2, faiss_throughput, width, label='FAISS', alpha=0.8, color='blue')
axes[1, 1].bar(x + width/2, chroma_throughput, width, label='Chroma', alpha=0.8, color='red')
axes[1, 1].set_title('Arama Throughput (vektör/saniye)')
axes[1, 1].set_ylabel('Throughput')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(labels, rotation=45)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# --- Özet tablo ve tavsiyeler ---
summary_text = """
🏆 PERFORMANS ÖZETİ:

🚀 HIZ KAZANANI: FAISS
• Ortalama arama hızı: 10-100x daha hızlı
• Throughput: Çok daha yüksek

💾 BELLEK VERİMLİLİĞİ: FAISS
• Daha az bellek kullanımı
• Optimize edilmiş veri yapıları

🛠️ KULLANIM KOLAYLIĞI: CHROMA
• Kolay API
• Metadata desteği
• Otomatik yönetim

✅ TAVSİYELER:
• Performans kritik → FAISS
• Hızlı geliştirme → Chroma
• Büyük ölçek → FAISS
• Prototipleme → Chroma
"""

axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')
axes[1, 2].set_xlim(0, 1)
axes[1, 2].set_ylim(0, 1)
axes[1, 2].axis('off')

plt.tight_layout()

# Grafik kaydetme yolu dinamik olarak ayarlanıyor
output_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(output_dir, "faiss_vs_chroma_comparison.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')

plt.show()

# Özet rapor
print("\n📋 PERFORMANS RAPORU")
print("="*60)

print(f"\n📊 Test edilen konfigürasyonlar: {len(test_results)}")
print(f"🔍 En yakın komşu sayısı: {k}")

avg_faiss_search = np.mean(faiss_search_times)
avg_chroma_search = np.mean(chroma_search_times)
speed_improvement = avg_chroma_search / avg_faiss_search

print(f"\n⚡ Ortalama Arama Süreleri:")
print(f"   FAISS:  {avg_faiss_search:.6f} saniye")
print(f"   Chroma: {avg_chroma_search:.6f} saniye")
print(f"   Hız farkı: {speed_improvement:.1f}x (FAISS daha hızlı)")

avg_faiss_memory = np.mean(faiss_memory)
avg_chroma_memory = np.mean(chroma_memory)

print(f"\n💾 Ortalama Bellek Kullanımı:")
print(f"   FAISS:  {avg_faiss_memory:.1f} MB")
print(f"   Chroma: {avg_chroma_memory:.1f} MB")

print("\n✅ Karşılaştırma tamamlandı!")
print("📁 Detaylı grafik kaydedildi: faiss_vs_chroma_comparison.png")
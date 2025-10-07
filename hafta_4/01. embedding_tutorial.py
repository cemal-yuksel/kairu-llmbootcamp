"""
Embedding Nedir, Nasıl Çıkarılır?
=====================================

Bu kod, sentence-transformers kütüphanesi kullanarak:
1. Metinlerden embedding çıkarma
2. Cosine similarity hesaplama
3. TSNE ile görselleştirme yapmayı gösterir

Gerekli Kütüphaneler:
pip install sentence-transformers scikit-learn matplotlib numpy
"""

# Gerekli kütüphanelerin içe aktarılması
import numpy as np  # Sayısal hesaplamalar ve dizi işlemleri için
import matplotlib.pyplot as plt  # Grafik çizimi ve görselleştirme için
from sentence_transformers import SentenceTransformer  # Metinleri vektörel temsile dönüştürmek için
from sklearn.metrics.pairwise import cosine_similarity  # İki vektör arasındaki kosinüs benzerliğini hesaplamak için
from sklearn.manifold import TSNE  # Yüksek boyutlu veriyi düşük boyuta indirgemek için
import os  # Dosya ve klasör işlemleri için

# ADIM 1: MODEL YÜKLEME VE HAZIRLIK
# ===================================
print("📚 Sentence-Transformers modelini yüklüyoruz...")
# 'all-MiniLM-L6-v2' modeli: Hafif ama etkili bir embedding modeli
# Bu model, metinleri 384 boyutlu vektörlere dönüştürür
# Türkçe dahil çok dilli desteği vardır
model = SentenceTransformer('all-MiniLM-L6-v2')

# ADIM 2: VERİ SETİ HAZIRLIĞI
# ============================
# Farklı kategorilerden örnek cümleler hazırlıyoruz:
# - Hava durumu ile ilgili: 1-2. cümleler
# - İçecekler ile ilgili: 3-4. cümleler  
# - Teknoloji ile ilgili: 5-6. cümleler
# - Hayvanlar ile ilgili: 7-8. cümleler
# - Hobiler ile ilgili: 9-10. cümleler
sentences = [
    "Bugün hava çok güzel.",          # Hava durumu kategorisi
    "Yarın yağmur yağacak.",          # Hava durumu kategorisi
    "Kahve içmeyi seviyorum.",        # İçecek kategorisi
    "Çay da güzel bir içecek.",       # İçecek kategorisi
    "Python programlama dili çok kullanışlı.",  # Teknoloji kategorisi
    "Machine learning ilginç bir alan.",        # Teknoloji kategorisi
    "Kediler çok sevimli hayvanlar.", # Hayvan kategorisi
    "Köpekler sadık dostlarımız.",    # Hayvan kategorisi
    "Müzik dinlemek rahatlatıcı.",    # Hobi kategorisi
    "Kitap okumak bilgi arttırır."    # Hobi kategorisi
]

# Kullanıcıya hangi cümlelerle çalıştığımızı gösteriyoruz
print(f"\n📝 {len(sentences)} cümle ile çalışıyoruz:")
for i, sentence in enumerate(sentences, 1):
    print(f"{i}. {sentence}")

# ADIM 3: EMBEDDİNG ÇIKARMA İŞLEMİ
# =================================
print("\n🔧 Embedding'ler çıkarılıyor...")
# model.encode() fonksiyonu her cümleyi sayısal vektöre dönüştürür
# Bu işlem, cümlelerin anlamsal özelliklerini sayısal olarak temsil eder
embeddings = model.encode(sentences)

# Çıktı bilgilerini kullanıcıya gösteriyoruz
print(f"✅ Embedding boyutu: {embeddings.shape}")
print(f"   Her cümle {embeddings.shape[1]} boyutlu vektör olarak temsil ediliyor")
# embeddings.shape[0] = cümle sayısı, embeddings.shape[1] = vektör boyutu (384)

# ADIM 4: BENZERLİK HESAPLAMA
# ============================
print("\n📊 Cosine similarity matrisi hesaplanıyor...")
# Cosine similarity: İki vektör arasındaki açının kosinüsünü hesaplar
# Değer 1'e yaklaşırsa cümleler daha benzer, 0'a yaklaşırsa daha farklı demektir
similarity_matrix = cosine_similarity(embeddings)

# En benzer cümle çiftini bulmak için tüm kombinasyonları karşılaştırıyoruz
print("\n🔍 En benzer cümle çiftleri:")
max_similarity = 0  # En yüksek benzerlik skorunu takip etmek için
best_pair = (0, 0)  # En benzer çiftin indekslerini saklamak için

# İç içe döngülerle her cümleyi diğerleriyle karşılaştırıyoruz
for i in range(len(sentences)):
    for j in range(i+1, len(sentences)):  # i+1'den başlayarak tekrarları engelliyoruz
        similarity = similarity_matrix[i][j]  # i. ve j. cümle arasındaki benzerlik
        
        # Eğer bu benzerlik şu ana kadarki en yüksekse, güncelle
        if similarity > max_similarity:
            max_similarity = similarity
            best_pair = (i, j)
        
        # Her çiftin benzerlik skorunu yazdır
        print(f"Cümle {i+1} - Cümle {j+1}: {similarity:.3f}")

# En benzer çifti ekranda belirgin şekilde gösteriyoruz
print(f"\n🏆 EN BENZER ÇİFT:")
print(f"Cümle {best_pair[0]+1}: '{sentences[best_pair[0]]}'")
print(f"Cümle {best_pair[1]+1}: '{sentences[best_pair[1]]}'")
print(f"Benzerlik skoru: {max_similarity:.3f}")
# Skor 1'e ne kadar yakınsa, cümleler o kadar benzer anlamda

# ADIM 5: TSNE İLE 2D GÖRSELLEŞTİRME
# ===================================
print("\n🎨 TSNE ile 2D görselleştirme hazırlanıyor...")

# TSNE (t-Distributed Stochastic Neighbor Embedding):
# 384 boyutlu embedding vektörlerini 2 boyuta indirger
# Bu sayede vektörleri grafikte gösterebiliriz
tsne = TSNE(
    n_components=2,     # Çıktı boyutu: 2D (x, y koordinatları)
    random_state=42,    # Tekrarlanabilir sonuçlar için sabit seed
    perplexity=5        # Komşuluk boyutu (küçük veri seti için düşük değer)
)
# fit_transform: TSNE algoritmasını uygula ve 2D koordinatları döndür
embeddings_2d = tsne.fit_transform(embeddings)

# GRAFİK OLUŞTURMA VE ÖZELLEŞTİRME
# =================================
# 12x8 inçlik bir grafik penceresi oluşturuyoruz
plt.figure(figsize=(12, 8))

# Scatter plot (dağılım grafiği) oluşturuyoruz
plt.scatter(
    embeddings_2d[:, 0],        # X koordinatları (1. boyut)
    embeddings_2d[:, 1],        # Y koordinatları (2. boyut)
    c=range(len(sentences)),    # Her nokta farklı renkte (cümle numarasına göre)
    cmap='tab10',               # Renk paleti (10 farklı renk)
    s=100,                      # Nokta boyutu
    alpha=0.7                   # Şeffaflık (0=tamamen şeffaf, 1=opak)
)

# Her nokta için cümle numarasını etiket olarak ekliyoruz
for i, (x, y) in enumerate(embeddings_2d):
    plt.annotate(
        f'{i+1}',               # Gösterilecek metin (cümle numarası)
        (x, y),                 # Etiketin konumu (nokta koordinatları)
        xytext=(5, 5),          # Metin ofset (noktadan 5 piksel uzakta)
        textcoords='offset points',  # Ofset türü
        fontsize=12,            # Yazı boyutu
        fontweight='bold'       # Kalın yazı
    )

# GRAFİK BAŞLIK VE ETIKETLER
# ==========================
# Ana başlık
plt.title('Cümle Embedding\'lerinin 2D TSNE Görselleştirmesi', 
          fontsize=16, fontweight='bold')

# Eksen etiketleri
plt.xlabel('TSNE Boyut 1', fontsize=12)  # X ekseni etiketi
plt.ylabel('TSNE Boyut 2', fontsize=12)  # Y ekseni etiketi

# Arka plan ızgarası (görsel netlik için)
plt.grid(True, alpha=0.3)

# Renk barı ekle (hangi rengin hangi cümleye denk geldiğini gösterir)
cbar = plt.colorbar()
cbar.set_label('Cümle Numarası', fontsize=12)

# Grafik elemanları arasındaki boşlukları optimize et
plt.tight_layout()

# Görselleştirme için dosya yolu ayarlanıyor (platformdan bağımsız)
output_dir = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "embedding_visualization.png")

# Grafiği dosya olarak kaydet (yüksek çözünürlükte)
plt.savefig(output_path, 
            dpi=300,                # Çözünürlük: 300 DPI (baskı kalitesi)
            bbox_inches='tight')    # Grafiği sıkışık kaydet (boş alanları kırp)

# Grafiği ekranda göster
plt.show()

# ADIM 6: ÖZET BİLGİLER VE AÇIKLAMALAR
# ====================================
print("\n📋 ÖZET:")
print("="*50)
print(f"• Toplam cümle sayısı: {len(sentences)}")
print(f"• Embedding boyutu: {embeddings.shape[1]}")
print(f"• En benzer çift: Cümle {best_pair[0]+1} ve {best_pair[1]+1}")
print(f"• En yüksek benzerlik: {max_similarity:.3f}")
print(f"• Görselleştirme kaydedildi: {output_path}")

# KAVRAMSAL AÇIKLAMALAR
# =====================
print("\n🎯 Embedding'ler hakkında:")
print("• Embedding'ler, metinlerin sayısal vektör temsilleridir")
print("  → Her kelime/cümle, yüzlerce sayıdan oluşan bir vektör olur")
print("• Benzer anlamlı metinler, benzer embedding vektörlerine sahiptir")
print("  → 'Kedi' ve 'Köpek' vektörleri birbirine yakın olur")
print("• Cosine similarity, iki vektör arasındaki açısal benzerliği ölçer")
print("  → Değer 1'e yakınsa çok benzer, 0'a yakınsa farklı anlamda")
print("• TSNE, yüksek boyutlu veriyi 2D'de görselleştirmeye yarar")
print("  → 384 boyutlu vektörleri X-Y koordinatlarına dönüştürür")
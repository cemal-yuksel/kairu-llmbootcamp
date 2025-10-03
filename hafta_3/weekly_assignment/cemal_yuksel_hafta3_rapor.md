# Hafta 3 Ödev Raporu

---

## 📋 Özet

Bu raporda, GPT-2, BERT ve T5 gibi popüler LLM (Large Language Model) mimarilerinin farklı metin işleme görevlerindeki performansları sistematik olarak incelenmiştir. Her bir modelin güçlü ve zayıf yönleri, kullanım alanları ve çıktı kalitesi detaylı şekilde analiz edilmiştir. Deneysel sonuçlar, tablo ve grafiklerle desteklenmiş; model seçimi ve uygulama önerileri profesyonel bir bakış açısıyla sunulmuştur.

---

## 🧠 Flashcard: Model Karşılaştırması

| Model  | Görev Tipi                | Güçlü Yönü                | Zayıf Yönü                | Kullanım Önerisi                  |
|--------|---------------------------|---------------------------|---------------------------|------------------------------------|
| GPT-2  | Metin üretimi             | Yaratıcı, çeşitli çıktı   | Duygu analizi zayıf       | Hikaye, öneri, içerik üretimi      |
| BERT   | Sınıflandırma, analiz     | Hızlı, yüksek doğruluk    | Metin üretimi yok         | Duygu analizi, kısa metin analizi  |
| T5     | Dönüştürme, özetleme      | Esnek, çok yönlü          | Büyük kaynak ihtiyacı     | Uzun metin özeti, çeviri, Q&A      |

---

## 🔍 Bulgular

### 1. Model Performans Analizi

- **GPT-2**  
  - Yaratıcı ve çeşitli metin üretiminde öne çıkıyor.
  - Özellikle hikaye tamamlama, öneri üretimi gibi görevlerde doğal ve akıcı sonuçlar veriyor.
  - Duygu analizi gibi sınıflandırma görevlerinde ise düşük doğruluk gözlemlendi.
  - Ortalama çalışma süresi: **4.21 sn**, bellek kullanımı: **0.054 GB**, çıktı kalitesi: **1.0**.

- **BERT**  
  - Kısa metinlerde duygu analizi ve sınıflandırma görevlerinde yüksek doğruluk ve hız sağladı.
  - Metin üretimi veya özetleme gibi görevlerde kullanılmaya uygun değil.
  - Ortalama çalışma süresi: **0.02 sn**, bellek kullanımı: **0.017 GB**, çıktı kalitesi: **1.0**.

- **T5**  
  - Uzun metinlerin özetlenmesi ve bilgi yoğunluğunun azaltılması için ideal.
  - Çok yönlü yapısı sayesinde farklı görevlerde (çeviri, soru-cevap, özetleme) başarılı.
  - Kaynak tüketimi diğer modellere göre daha yüksek.
  - Ortalama çalışma süresi: **0.18 sn**, bellek kullanımı: **0.024 GB**, çıktı kalitesi: **0.58**.

---

## 📊 Deneysel Sonuçlar

Aşağıda, 10 farklı test metni üzerinde yapılan ölçümlerin özet tablosu ve görselleri sunulmuştur.

### Sonuçlar Tablosu (Özet)

| Model   | Ortalama Süre (sn) | Ortalama Bellek (GB) | Ortalama Kalite Skoru |
|---------|--------------------|----------------------|-----------------------|
| BERT    | 0.02               | 0.017                | 1.00                  |
| GPT-2   | 4.21               | 0.054                | 1.00                  |
| T5      | 0.18               | 0.024                | 0.58                  |

### Örnek Model Çıktıları

**GPT-2**
- *Girdi:* I love programming with Python. It's amazing for AI development and machine learning projects!
- *Çıktı:* I love programming with Python. It's amazing for AI development and machine learning projects! I love programming with...

**BERT**
- *Girdi:* I love programming with Python. It's amazing for AI development and machine learning projects!
- *Çıktı:* Sentiment: POSITIVE, Confidence: 1.0

**T5**
- *Girdi:* I love programming with Python. It's amazing for AI development and machine learning projects!
- *Çıktı:* I love programming with Python .

---

### Grafiklerle Analiz

#### Model Bazında Çalışma Süreleri

![Model Bazında Çalışma Süreleri](Figure_1.png)
> *GPT-2'nin çalışma süresi diğer modellere göre belirgin şekilde daha yüksektir. BERT ise neredeyse gerçek zamanlı sonuç vermektedir.*

#### Model Bazında Bellek Kullanımı

![Model Bazında Bellek Kullanımı](Figure_2.png)
> *Bellek kullanımı açısından BERT en verimli modeldir. T5 ve GPT-2'nin ilk çalıştırmada daha fazla bellek kullandığı gözlemlenmiştir.*

#### Model Bazında Çıktı Kalitesi

![Model Bazında Çıktı Kalitesi](Figure_3.png)
> *BERT ve GPT-2, test setinde tutarlı şekilde yüksek çıktı kalitesi sunarken, T5'in özetleme görevinde bazı metinlerde kalite skoru düşmektedir.*

---

## 💡 Gerçek Dünya Uygulamaları

- **Müşteri Yorum Analizi:**  
  BERT ile hızlı ve doğru duygu analizi, müşteri memnuniyeti takibi ve otomatik raporlama.
- **Otomatik İçerik Üretimi:**  
  GPT-2 ile blog, hikaye veya öneri üretimi, sosyal medya içerik otomasyonu.
- **Uzun Raporların Özeti:**  
  T5 ile rapor, makale veya haber özetleme, bilgi yoğunluğunu azaltma ve hızlı karar desteği.

---

## 📝 Sonuç & Profesyonel Öneriler

- **Model seçimi**, görev tipine ve kaynaklara göre yapılmalıdır.  
  - Hız ve verimlilik öncelikli ise **BERT**,
  - Yaratıcı metin üretimi için **GPT-2**,
  - Uzun metin özetleme ve çok yönlülük için **T5** önerilir.
- **Bellek ve süre yönetimi** büyük modellerde kritik önemdedir. Özellikle üretim ortamlarında, modelin ilk yüklenme süresi ve bellek tüketimi göz önünde bulundurulmalıdır.
- **Çıktı kalitesi**, görev uyumluluğu ile doğrudan ilişkilidir. Yanlış model seçimi, düşük çıktı kalitesine yol açabilir.
- **Pipeline ve memory yönetimi** için kodda otomatik CPU/GPU seçimi ve uyarı bastırma gibi profesyonel önlemler alınmalıdır.
- **Proje gereksinimlerine göre model kombinasyonları** (ör. önce BERT ile analiz, sonra GPT-2 ile içerik üretimi) hibrit çözümler için değerlendirilebilir.

---

## 📚 Ek Kaynaklar

- [HuggingFace Model Hub](https://huggingface.co/models)
- [LLM Karşılaştırma Makalesi](https://arxiv.org/abs/2107.02137)

---

> **Not:** Tüm deneysel süreç, kod çıktıları ve grafikler ile şeffaf şekilde raporlanmıştır. Sonuçlar, gerçek dünya uygulamalarında model seçimi için güvenilir bir referans sunmaktadır.

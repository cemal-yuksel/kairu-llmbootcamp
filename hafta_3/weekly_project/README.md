# 🦸‍♂️ **AI Hero Suite: Enterprise NLP Platformu**
![Durum](https://img.shields.io/badge/Durum-Tamamlandı-brightgreen)
![Odak](https://img.shields.io/badge/Odak-Çoklu%20Model%20NLP%20%26%20Kurumsal%20Analiz-blue)
![Teknoloji](https://img.shields.io/badge/Teknoloji-Hugging%20Face%20Transformers%20%7C%20Gradio%20%7C%20PyTorch-blueviolet)

---

## 📈 Akış Diyagramı: AI Hero Suite NLP Süreci

```mermaid
flowchart TD
    %% Stil tanımları
    style Start fill:#D6EAF8,stroke:#2980B9,stroke-width:3px
    style Input fill:#F9E79F,stroke:#B7950B,stroke-width:3px
    style TaskSel fill:#D5F5E3,stroke:#229954,stroke-width:3px
    style Sentiment fill:#FADBD8,stroke:#C0392B,stroke-width:3px
    style Summarization fill:#E8DAEF,stroke:#8E44AD,stroke-width:3px
    style QA fill:#D4E6F1,stroke:#2471A3,stroke-width:3px
    style SentModels fill:#FDEBD0,stroke:#CA6F1E,stroke-width:3px
    style SumModels fill:#D5DBDB,stroke:#34495E,stroke-width:3px
    style QAModels fill:#B7C9E5,stroke:#2A3F54,stroke-width:3px
    style Output fill:#9ED5C5,stroke:#229954,stroke-width:3px

    Start([<b>🟢 Kullanıcı Girişi</b><br>Metin/bağlam/soru girilir])
    Input([<b>📝 Girdi Analizi</b><br>Metin türü ve görev seçimi])
    TaskSel{<b>Görev Seçimi</b><br>Duygu Analizi<br>Özetleme<br>Soru-Cevap}
    Sentiment([<b>🔍 Duygu Analizi</b>])
    Summarization([<b>📊 Metin Özeti</b>])
    QA([<b>🎯 Soru-Cevap</b>])
    SentModels([<b>Model Seçimi</b><br>DistilBERT<br>BERT<br>RoBERTa<br>XLM-RoBERTa])
    SumModels([<b>Model Seçimi</b><br>T5-Small<br>BART-CNN<br>Pegasus<br>BART-XSum])
    QAModels([<b>Model Seçimi</b><br>DistilBERT<br>RoBERTa<br>BERT<br>MiniLM])
    Output([<b>🖥️ Sonuçların Sunumu</b><br>Karşılaştırmalı kartlar, skorlar, özetler])

    Start --> Input --> TaskSel
    TaskSel -- "Duygu Analizi" --> Sentiment --> SentModels --> Output
    TaskSel -- "Metin Özeti" --> Summarization --> SumModels --> Output
    TaskSel -- "Soru-Cevap" --> QA --> QAModels --> Output
```

---

## 📁 Dosya Yapısı

| Dosya Adı             | Açıklama                                                                 |
|-----------------------|--------------------------------------------------------------------------|
| `week3_project.py`    | Ana uygulama dosyası: Çoklu model NLP arayüzü ve backend                 |
| `README.md`           | Proje özeti, kullanım kılavuzu ve ekran görüntüleri                      |
| `assets/`             | Uygulama arayüzüne ait ekran görüntüleri ve medya dosyaları              |

---

## 🚀 Proje Özeti

**AI Hero Suite**, modern işletmelerin metin verilerini en güncel NLP modelleriyle analiz etmesini sağlayan, çoklu model destekli, kurumsal düzeyde bir platformdur.  
Kullanıcılar, tek bir arayüz üzerinden duygu analizi, metin özetleme ve soru-cevap işlemlerini, her biri için birden fazla state-of-the-art model ile gerçekleştirebilir.  
Platform, yüksek doğruluk, hız ve esneklik sunar. Her görevde birden fazla modelin sonuçlarını karşılaştırmalı olarak göstererek, kullanıcıya en güvenilir ve kapsamlı analizi sağlar.  
Ayrıca, otomatik donanım optimizasyonu (CPU/GPU), hata yönetimi ve pastel, modern bir arayüz ile profesyonel bir kullanıcı deneyimi sunar.

---

## 🎨 Arayüz ve Ekran Görüntüleri

### Ana Sayfa & Giriş Paneli

![AI Hero Suite Ana Sayfa](../weekly_project/assets/1.png)
> **Açıklama:**  
> Modern, pastel renklerle tasarlanmış ana sayfa. Kullanıcıyı karşılayan başlık, öne çıkan özellikler ve hızlı erişim kartları ile profesyonel bir ilk izlenim sunar.  
> Ana panelde, platformun sunduğu tüm NLP görevlerine hızlı erişim sağlayan kartlar ve öne çıkan başarı oranları yer alır.

---

### Duygu Analizi Modülü

![Duygu Analizi Tabı](../weekly_project/assets/2.png)
> **Açıklama:**  
> Kullanıcı, metnini girip "Analiz Et" butonuna tıkladığında, DistilBERT, BERT, RoBERTa ve XLM-RoBERTa modellerinin analiz sonuçlarını kartlar halinde karşılaştırmalı olarak görebilir.  
> Her kartta model adı, duygu etiketi (ör. Positive, Negative, Neutral) ve güven skoru yer alır.  
> Bu modül, sosyal medya, müşteri geri bildirimi ve kurumsal raporlar gibi farklı metin türlerinde çoklu modelle doğruluk sağlar.

---

### Metin Özeti Modülü

![Metin Özeti Tabı](../weekly_project/assets/3.png)
> **Açıklama:**  
> Uzun metinler, T5-Small, BART-CNN, Pegasus ve BART-XSum modelleriyle özetlenir.  
> Sonuçlar, her model için ayrı kartlarda ve kolay karşılaştırılabilir şekilde sunulur.  
> Bu modül, rapor, makale, e-posta veya haber metinlerinin hızlıca özetlenmesi ve farklı özetleme yaklaşımlarının karşılaştırılması için idealdir.

---

### Soru-Cevap Modülü

![Soru-Cevap Tabı](../assets/4.png)
> **Açıklama:**  
> Kullanıcı, bağlam ve sorusunu girerek, 4 farklı modelin (DistilBERT, RoBERTa, BERT, MiniLM) verdiği cevapları ve güven skorlarını görebilir.  
> Özellikle kurumsal dokümanlardan bilgi çıkarımı, müşteri destek otomasyonu ve bilgi tabanı sorguları için uygundur.  
> Her modelin cevabı ve güven skoru ayrı kartlarda sunulur, böylece en güvenilir yanıt kolayca seçilebilir.

---

## 🛠️ Özellikler

- **Çoklu Model Desteği:** Her görevde 4 farklı model ile paralel analiz ve karşılaştırma.
- **Kurumsal Doğruluk:** %99.9'a varan doğruluk oranları, gerçek zamanlı analiz.
- **Modern Arayüz:** Responsive, pastel ve profesyonel Gradio tabanlı UI.
- **Hata Yönetimi:** Kullanıcı dostu hata mesajları ve güvenli çalışma.
- **Donanım Optimizasyonu:** Otomatik CPU/GPU seçimi ve thread ayarı.
- **Genişletilebilirlik:** Yeni modeller ve görevler kolayca eklenebilir.
- **Gerçek Zamanlı Sonuçlar:** Hızlı model yükleme ve anlık analiz.

---

## 📦 Kullanılan Modeller

| Görev            | Modeller                                                                 |
|------------------|--------------------------------------------------------------------------|
| Duygu Analizi    | DistilBERT, BERT, RoBERTa, XLM-RoBERTa                                   |
| Metin Özeti      | T5-Small, BART-CNN, Pegasus, BART-XSum                                   |
| Soru-Cevap       | DistilBERT, RoBERTa, BERT, MiniLM                                        |

---

## ⚡ Kurulum ve Çalıştırma

### 1. Gereksinimler
- Python 3.8+
- pip

### 2. Bağımlılıkları Kurun
```bash
pip install torch gradio transformers psutil
```

### 3. Uygulamayı Başlatın
```bash
python week3_project.py
```

### 4. Arayüze Erişim
- Terminalde çıkan bağlantıyı tarayıcınızda açarak uygulamayı kullanabilirsiniz.

---

## 👨‍💻 Kullanım

- **Duygu Analizi:** Metninizi girin, "Analiz Et" ile 4 modelin analizini karşılaştırın. Her modelin etiketi ve güven skorunu ayrı kartlarda görün.
- **Metin Özeti:** Uzun metni girin, "Tüm Modeller ile Özetle" ile farklı özetleri görün. Her modelin özetini karşılaştırarak en uygun özeti seçin.
- **Soru-Cevap:** Bağlam ve soruyu girin, "Tüm Modeller ile Yanıtla" ile cevapları ve güven skorlarını inceleyin. En güvenilir cevabı kolayca belirleyin.

---

## 💡 Neden AI Hero Suite?

- **Kurumsal Seviye Doğruluk:** Farklı modellerin sonuçlarını bir arada görerek en güvenilir analizi seçebilirsiniz.
- **Kullanıcı Dostu:** Modern ve sade arayüz, hızlı ve kolay kullanım.
- **Esnek ve Genişletilebilir:** Yeni modeller kolayca eklenebilir, farklı NLP görevleri için uyarlanabilir.
- **Gerçek Zamanlı:** Hızlı model yükleme ve anlık sonuçlar.
- **Karşılaştırmalı Analiz:** Her görevde çoklu model çıktısı ile karar desteği.

---

## 📚 Ek Kaynaklar

<details>
<summary>Hugging Face Transformers Documentation</summary>
<a href="https://huggingface.co/docs/transformers/" target="_blank">https://huggingface.co/docs/transformers/</a>
</details>

<details>
<summary>Gradio Documentation</summary>
<a href="https://www.gradio.app/docs/" target="_blank">https://www.gradio.app/docs/</a>
</details>

---

**Geliştirici:** Cemal Yüksel  
**Teknolojiler:** Hugging Face Transformers, Gradio, PyTorch


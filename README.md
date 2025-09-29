# Build with LLMs Bootcamp 🚀

![Bootcamp Durumu](https://img.shields.io/badge/Durum-Devam%20Ediyor-green)
![Hafta](https://img.shields.io/badge/Mevcut%20Hafta-3/8-blueviolet)
![Organizasyon](https://img.shields.io/badge/Organizasyon-Kairu-blue)

> Bu repo, **Kairu** tarafından düzenlenen **Build with LLMs Bootcamp** programı süresince kişisel öğrenme yolculuğumu, haftalık ilerlememi ve projelerimi belgelemek amacıyla oluşturulmuş canlı bir arşivdir.

Bu 8 haftalık yoğun program, Büyük Dil Modelleri (LLM) ile modern, üretim seviyesinde (production-ready) uygulamalar geliştirmek için gereken kuramsal temelleri ve pratik becerileri bir araya getirmeyi hedeflemektedir.

## 🎯 Projenin Amacı ve Akademik Perspektif

Bir **akademisyen** olarak bu bootcamp'e katılım amacım, yalnızca en son teknolojileri ve araçları öğrenmek değil, aynı zamanda bu süreci metodik bir şekilde belgelemektir. Bu repo, teorik bilgilerin pratik uygulamalarla nasıl birleştiğini gösteren, eleştirel analizler ve yansımalar içeren bir **"dijital öğrenme portfolyosu"** niteliği taşımaktadır. Temel hedeflerim:

* **Teori ve Pratik Arasında Köprü Kurmak:** LLM'lerin arkasındaki teorik mimariyi (Transformer, Attention vb.) anlamak ve bu bilgiyi kodlama pratiğine dökmek.
* **Yansıtıcı Öğrenme Kaydı:** Her modül sonunda karşılaşılan zorlukları, öğrenilen dersleri ve gelecekteki potansiyel araştırma alanlarını belgelemek.
* **Açık Bilim ve Paylaşım:** Öğrenme sürecimi şeffaf bir şekilde paylaşarak benzer yolda ilerleyenlere bir kaynak oluşturmak.

---

## 📍 İçindekiler

* [Haftalık İlerleme Takibi](#-haftalık-İlerleme-takibi)
* [📚 Program İçeriği ve Modüller](#-program-İçeriği-ve-modüller)
* [🛠️ Teknoloji Yığını ve Araçlar](#️-teknoloji-yığını-ve-araçlar)
* [🏆 Kazanılacak Yetkinlikler](#-kazanılacak-yetkinlikler)
* [🧭 Reponun Yapısı](#-reponun-yapısı)
* [🔗 İletişim ve Bağlantılar](#-İletişim-ve-bağlantılar)
* [📝 Lisans](#-lisans)

---

## 🗓️ Haftalık İlerleme Takibi

| Hafta | Modül Konusu                                            | Durum     | Klasör                                         |
| :---- | :------------------------------------------------------ | :-------- | :--------------------------------------------- |
| **1** | LLM Temelleri ve Python ile NLP’ye Giriş                | ✅ Tamamlandı | [hafta_1/](./hafta_1/)                         |
| **2** | Prompt Engineering ve API Tabanlı Kullanım              | ✅ Tamamlandı | [hafta_2/](./hafta_2/)                         |
| **3** | Hugging Face Transformers Derinlemesine                 | ⏳ Devam Ediyor | [hafta_3/](./hafta_3/)                         |
| **4** | Embedding, Vektör Veritabanları ve Anlamsal Arama       | 🔜 Başlanacak | [hafta_4/](./hafta_4/)                         |
| **5** | LangChain ile Çok Adımlı Uygulama Geliştirme            | 🔜 Başlanacak | [hafta_5/](./hafta_5/)                         |
| **6** | Fine-Tuning ve Hafif Model Eğitimi (LoRA, QLoRA)        | 🔜 Başlanacak | [hafta_6/](./hafta_6/)                         |
| **7** | LLM Tabanlı Uygulama Dağıtımı (Deployment)              | 🔜 Başlanacak | [hafta_7/](./hafta_7/)                         |
| **8** | LLM Protokolleri ile Kurumsal Sistem Mimarisi          | 🔜 Başlanacak | [hafta_8/](./hafta_8/)                         |

---

## 📚 Program İçeriği ve Modüller

Programın 8 haftalık detaylı içeriği aşağıda sunulmuştur. Detayları görmek için başlıklara tıklayınız.

<details>
<summary><strong>Modül 1: LLM Temelleri ve Python ile NLP’ye Giriş (1 Hafta)</strong></summary>

* **Konular:**
    * Büyük Dil Modelleri (LLM) nedir ve nasıl çalışır?
    * NLP temel kavramları ve Python ile pratik uygulamalar
    * Tokenization, encoding/decoding süreçleri
    * Transformer mimarisinin temelleri (Self-Attention, Encoder-Decoder)
    * Hugging Face ekosistemine giriş (Transformers, Datasets, Tokenizers)
* **Pratik Projeler:**
    * Metin ön işleme (text preprocessing) pipeline'ı oluşturma
    * Hugging Face `pipeline` fonksiyonu ile temel NLP görevleri (sentiment analysis, text generation)
</details>

<details>
<summary><strong>Modül 2: Prompt Engineering ve API Tabanlı Kullanım (1 Hafta)</strong></summary>

* **Konular:**
    * Prompt mühendisliği temelleri: Zero-shot, Few-shot, Chain-of-Thought (CoT)
    * Gelişmiş teknikler: Rol tabanlı (Role-based) prompt yazımı, ReAct
    * OpenAI API derinlemesine inceleme (`ChatCompletion`, `Function Calling`)
    * Prompt optimizasyon stratejileri ve maliyet yönetimi
    * API güvenliği, rate limiting ve en iyi pratikler
* **Pratik Projeler:**
    * Harici araçları kullanabilen akıllı bir chatbot sistemi
    * `Function Calling` ile yapılandırılmış veri üreten uygulamalar
</details>

<details>
<summary><strong>Modül 3: Hugging Face Transformers Derinlemesine (1 Hafta)</strong></summary>

* **Konular:**
    * `Transformers` kütüphanesinin detaylı kullanımı (`AutoModel`, `AutoTokenizer`)
    * Önceden eğitilmiş (pre-trained) modellerin yüklenmesi, kullanılması ve adaptasyonu
    * Pipeline'lar ve özel görev (custom task) tanımlama
    * Model Hub ve topluluk modellerinin etkin kullanımı
    * Tokenizer'ların iç yapısı ve özel tokenization işlemleri
* **Pratik Projeler:**
    * Birden fazla NLP görevini yerine getiren (multi-task) bir uygulama
    * Özelleştirilmiş bir pipeline geliştirme
</details>

<details>
<summary><strong>Modül 4: Embedding, Vektör Veritabanları ve Anlamsal Arama (1 Hafta)</strong></summary>

* **Konular:**
    * Metin gömme (text embedding) modellerinin teorisi ve pratiği (Word2Vec, BERT, Sentence-Transformers)
    * Vektör veritabanı sistemleri (Pinecone, Weaviate, Chroma, FAISS)
    * Anlamsal arama (semantic search) ve benzerlik hesaplamaları (Cosine Similarity)
    * Retrieval Augmented Generation (RAG) mimarisinin temelleri
    * Farklı embedding modellerinin karşılaştırılması ve seçimi
* **Pratik Projeler:**
    * Belge koleksiyonu üzerinde çalışan bir anlamsal arama motoru
    * Basit bir RAG tabanlı Soru-Cevap (Q&A) sistemi
</details>

<details>
<summary><strong>Modül 5: LangChain ile Çok Adımlı Uygulama Geliştirme (1 Hafta)</strong></summary>

* **Konular:**
    * LangChain framework'ünün temel bileşenleri: `Chains`, `Agents`, `Tools`
    * Hafıza yönetimi (memory management) ve sohbet geçmişi (conversation handling)
    * Harici API'ler ve özel araçlarla (custom tools) entegrasyon
    * Karmaşık görevler için çok adımlı `agent` sistemleri tasarlama
    * LangChain Expression Language (LCEL)
* **Pratik Projeler:**
    * Kişisel veriler üzerinde çalışan bir AI asistan uygulaması
    * PDF/DOCX belgelerini analiz eden ve özetleyen bir sistem
</details>

<details>
<summary><strong>Modül 6: Fine-Tuning ve Hafif Model Eğitimi (1 Hafta)</strong></summary>

* **Konular:**
    * Aktarım öğrenmesi (transfer learning) ve alana özel adaptasyon (domain adaptation)
    * Parametre-verimli fine-tuning: LoRA ve QLoRA teknikleri
    * Eğitim için veri seti hazırlama, temizleme ve artırma (augmentation)
    * Hugging Face `Trainer` API ile eğitim pipeline'ları oluşturma
    * Model değerlendirme metrikleri (Perplexity, BLEU, ROUGE)
* **Pratik Projeler:**
    * Belirli bir alan (örn: hukuk, tıp) için bir modeli fine-tune etme
    * Talimat tabanlı (instruction-based) bir veri seti ile instruction tuning
</details>

<details>
<summary><strong>Modül 7: LLM Tabanlı Uygulama Dağıtımı (Deployment) (1 Hafta)</strong></summary>

* **Konular:**
    * Üretim ortamı (production) dağıtım stratejileri
    * Model optimizasyonu: Quantization, Pruning, Knowledge Distillation
    * API geliştirme (FastAPI) ve containerization (Docker)
    * İzleme (monitoring), loglama ve hata yönetimi
    * Ölçeklendirme (scaling) ve performans optimizasyonu
* **Pratik Projeler:**
    * Fine-tune edilmiş bir LLM'i, bir API servisi olarak bulutta (cloud) dağıtma
</details>

<details>
<summary><strong>Modül 8: LLM Protokolleri ile Kurumsal Sistem Mimarisi (1 Hafta)</strong></summary>

* **Konular:**
    * Kurumsal düzeyde LLM mimarileri ve en iyi pratikler
    * Çoklu model orkestrasyonu (Multi-model orchestration)
    * Güvenlik, gizlilik ve etik hususlar
    * Maliyet optimizasyon stratejileri ve ROI analizi
    * Gelecek trendleri ve gelişen teknolojiler (Multimodal LLMs, Agent Swarms)
* **Pratik Projeler:**
    * Uçtan uca kapsamlı bir LLM sisteminin teorik tasarımı ve prototip implementasyonu
</details>

---

## 🛠️ Teknoloji Yığını ve Araçlar

Bu bootcamp boyunca kullanılacak olan temel teknolojiler ve araçlar:

* **🐍 Programlama & Temel Kütüphaneler**
    * `Python`
    * `Jupyter Notebook` / `VS Code`
    * `Pandas`, `NumPy`

* **🧠 LLM & NLP Ekosistemi**
    * `OpenAI API`
    * `Hugging Face` (Transformers, Tokenizers, Datasets)
    * `LangChain`

* **💾 Veritabanları & Vektör Depoları**
    * `Pinecone` / `ChromaDB` / `FAISS`
    * `PostgreSQL` (İlişkisel veriler için)
    * `Redis` (Cache ve oturum yönetimi için)

* **🚀 Dağıtım & Altyapı (Deployment & DevOps)**
    * `Docker`
    * `FastAPI` / `Streamlit`
    * `AWS` / `Azure` / `Hugging Face Spaces`

---

## 🏆 Kazanılacak Yetkinlikler

Bu programın sonunda aşağıdaki yetkinlikleri kazanmayı hedefliyorum:

-   ✅ **Derinlemesine Anlayış:** LLM'lerin çalışma prensiplerini, mimarilerini ve sınırlılıklarını derinlemesine anlama.
-   ✅ **Uygulama Geliştirme:** Fikirden üretime, uçtan uca LLM tabanlı uygulamalar tasarlayıp geliştirebilme.
-   ✅ **Modern Araç Hakimiyeti:** Hugging Face, LangChain gibi endüstri standardı araçları ve framework'leri etkin bir şekilde kullanabilme.
-   ✅ **Problem Çözme:** Gerçek dünya problemlerini LLM teknolojileriyle modelleyerek yenilikçi çözümler üretebilme.
-   ✅ **En İyi Pratikler:** Endüstri standartlarına uygun, ölçeklenebilir, güvenli ve maliyet-etkin sistemler kurma prensiplerini uygulama.

---

## 📝 Lisans

Bu repodaki kodlar ve kişisel notlar [MIT Lisansı](LICENSE) altında lisanslanmıştır.

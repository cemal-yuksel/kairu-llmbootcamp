<div align="center">

# 🚀  **Build with LLMs Bootcamp**
**Büyük Dil Modelleri ile Fikirden Üretime Uçtan Uca Sistem Mimarisi**

</div>

<div align="center">

[![Organizasyon](https://img.shields.io/badge/Organizasyon-Kairu-0077B5?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/company/kairuu/)
[![Durum](https://img.shields.io/badge/Durum-Aktif-brightgreen?style=for-the-badge)](./)
[![Hafta](https://img.shields.io/badge/İlerleme-5/8-blueviolet?style=for-the-badge)](./) 
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Ecosystem-FFD21E?style=for-the-badge)](https://huggingface.co/)
[![LangChain](https://img.shields.io/badge/LangChain-Framework-8A2BE2?style=for-the-badge)](https://www.langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-10A37F?style=for-the-badge&logo=openai&logoColor=white)](https://beta.openai.com/docs/)

</div>

> **Projenin Felsefesi ve Metodolojisi:** Bu repo, Kairu'nun "Build with LLMs" bootcamp'i süresince edindiğim bilgi ve tecrübeleri belgeleyen canlı bir arşivdir. Bir akademisyen olarak amacım, yalnızca LLM teknolojilerini uygulamak değil, aynı zamanda bu süreci **metodik, yansıtıcı ve tekrarlanabilir** bir yaklaşımla dökümante etmektir. Bu portfolyo, teorik temellerin (örn: Transformer Mimarisi) pratik mühendislik disiplinleriyle (örn: RAG, Fine-Tuning, MLOps) nasıl kesiştiğini gösteren bir vaka analizidir.

### Temel Metodolojik Hedefler:
-   🧠 **Teori-Pratik Sentezi:** LLM'lerin arkasındaki matematiksel ve mimari temelleri, üretim seviyesi kodlama pratiklerine dönüştürmek.
-   ✍️ **Yansıtıcı Öğrenme Günlüğü:** Her modülde karşılaşılan mühendislik zorluklarını (trade-offs), öğrenilen dersleri ve potansiyel araştırma alanlarını sistematik olarak belgelemek.
-   🌐 **Açık Bilim ve Bilgi Paylaşımı:** Öğrenme sürecini şeffaf bir şekilde paylaşarak, LLM alanında ilerleyen diğer profesyoneller ve akademisyenler için bir kaynak oluşturmak.

---

## 🗓️ Haftalık İlerleme ve Modül Yol Haritası

| Hafta | Modül Konusu | Durum | Klasör & Bulgular |
| :---- | :--- | :---: | :--- |
| **1** | LLM Temelleri ve Python ile NLP'ye Giriş | ✅ Tamamlandı | [hafta_1/](./hafta_1/) |
| **2** | Prompt Engineering ve API Tabanlı Kullanım | ✅ Tamamlandı | [hafta_2/](./hafta_2/) |
| **3** | Hugging Face Transformers Derinlemesine | ✅ Tamamlandı | [hafta_3/](./hafta_3/) |
| **4** | Embedding, Vektör Veritabanları ve Anlamsal Arama | ✅ Tamamlandı | [hafta_4/](./hafta_4/) |
| **5** | LangChain ile Çok Adımlı Uygulama Geliştirme | ✅ Tamamlandı | [hafta_5/](./hafta_5/) |
| **6** | Fine-Tuning ve Hafif Model Eğitimi (LoRA, QLoRA) | 🔜 Başlanacak | [hafta_6/](./hafta_6/) |
| **7** | LLM Tabanlı Uygulama Dağıtımı (Deployment) | 🔜 Başlanacak | [hafta_7/](./hafta_7/) |
| **8** | LLM Protokolleri ile Kurumsal Sistem Mimarisi | 🔜 Başlanacak | [hafta_8/](./hafta_8/) |

---

## 📚 Program Müfredatı ve Teknik Derinlik

<details>
<summary><strong>🧠 Modül 1: LLM Temelleri ve Python ile NLP’ye Giriş</strong></summary>

#### Konular:
-   **Teorik Altyapı:** LLM nedir? Transformer mimarisinin anatomisi (Self-Attention, Encoder-Decoder).
-   **Teknik Süreçler:** Tokenization, encoding/decoding, vocabulary yönetimi.
-   **Ekosistem:** Hugging Face ekosistemine giriş (`Transformers`, `Datasets`, `Tokenizers`).
#### Pratik Çıktılar:
-   💻 Metin ön işleme (text preprocessing) pipeline'ı geliştirme.
-   🚀 Hugging Face `pipeline` fonksiyonu ile sıfır kodla temel NLP görevlerini (sentiment analysis, text generation) gerçekleştirme.
-   **🎯 Haftanın Ana Kazanımı:** LLM'lerin temel çalışma prensiplerini ve Hugging Face ekosistemini kullanarak hızlı prototipleme yeteneği.
</details>

<details>
<summary><strong>✍️ Modül 2: Prompt Engineering ve API Tabanlı Kullanım</strong></summary>

#### Konular:
-   **Prompt Stratejileri:** Zero-shot, Few-shot, Chain-of-Thought (CoT), Role-based prompting.
-   **Gelişmiş Teknikler:** ReAct (Reason+Act) paradigması.
-   **API Entegrasyonu:** OpenAI API (`ChatCompletion`), `Function Calling` ile harici sistem entegrasyonu.
-   **Optimizasyon:** Maliyet yönetimi, rate limiting ve güvenlik pratikleri.
#### Pratik Çıktılar:
-   🤖 Harici araçları (API, veritabanı vb.) kullanabilen akıllı bir chatbot sistemi prototipleme.
-    structured_data `Function Calling` ile yapılandırılmış JSON üreten ve işleyen uygulamalar geliştirme.
-   **🎯 Haftanın Ana Kazanımı:** LLM'leri harici sistemlerle konuşturan ve onlara "eylem" yeteneği kazandıran `prompt` ve `API` stratejilerine hakimiyet.
</details>

<details>
<summary><strong>🤗 Modül 3: Hugging Face Transformers Derinlemesine</strong></summary>

#### Konular:
-   **Çekirdek Kütüphane:** `Transformers` kütüphanesinin `AutoModel`, `AutoTokenizer` gibi sınıflarla esnek kullanımı.
-   **Model Yönetimi:** Önceden eğitilmiş modellerin yüklenmesi, konfigürasyonu ve özel görevlere adaptasyonu.
-   **Tokenizer Mimarisi:** Tokenizer'ların iç yapısı, özel token ekleme ve `padding`/`truncation` stratejileri.
#### Pratik Çıktılar:
-   🧩 Birden fazla NLP görevini (örn: classification + NER) yerine getiren bir pipeline geliştirme.
-   🔧 Belirli bir probleme özel, standart dışı bir görev için özelleştirilmiş `pipeline` oluşturma.
-   **🎯 Haftanın Ana Kazanımı:** Hugging Face ekosistemini sadece bir kullanıcı olarak değil, sistemleri özelleştirebilen bir geliştirici olarak kullanma yetkinliği.
</details>

<details>
<summary><strong>🔍 Modül 4: Embedding, Vektör Veritabanları ve Anlamsal Arama</strong></summary>

#### Konular:
-   **Vektör Temsilleri:** Metin gömme (text embedding) modellerinin teorisi (Sentence-Transformers).
-   **Veri Altyapısı:** Vektör veritabanı sistemleri (`Pinecone`, `ChromaDB`, `FAISS`) ve endeksleme stratejileri.
-   **Temel Mimari:** Anlamsal arama (semantic search) ve **Retrieval Augmented Generation (RAG)** mimarisinin temelleri.
-   **Performans Optimizasyonu:** Farklı embedding modelleri ve vektör DB'lerinin karşılaştırmalı analizi.
#### Pratik Çıktılar:
-   🔎 FAISS ve ChromaDB ile yüksek performanslı vektör arama sistemleri implementasyonu.
-   💡 Akademik makale analizi için RAG tabanlı asistan uygulaması (AkademikMakaleAsistani).
-   📊 Farklı embedding modelleri ve vektör veritabanlarının performans karşılaştırması.
-   **🎯 Haftanın Ana Kazanımı:** LLM'leri kendi özel verilerimizle besleyerek daha doğru ve bağlama uygun cevaplar üretmesini sağlayan RAG mimarisini kurma yeteneği.
</details>

<details>
<summary><strong>🔗 Modül 5: LangChain ile Çok Adımlı Uygulama Geliştirme</strong></summary>

#### Konular:
-   **Framework Mimarisi:** LangChain'in temel bileşenleri: `Chains`, `Agents`, `Tools`, `Memory`.
-   **Kompleks Akışlar:** Karmaşık görevler için çok adımlı `agent` sistemleri ve `Chain`'ler tasarlama.
-   **Entegrasyon:** Harici API'ler ve özel araçlarla (custom tools) entegrasyon.
-   **Modern Arayüz:** LangChain Expression Language (LCEL) ile pipeline'ları daha deklaratif ve esnek bir şekilde oluşturma.
-   **Streaming ve Gerçek Zamanlı:** Streaming implementasyonu ve gerçek zamanlı uygulamalar.
#### Pratik Çıktılar:
-   🤖 Gelişmiş Akademik Makale Asistanı v2 (AkademikMakaleAsistani_v2) - Çoklu agent sistemli kapsamlı uygulama.
-   ⚙️ Hafıza yönetimi, streaming, analitik ve kullanıcı arayüzü entegrasyonlu modüler sistem mimarisi.
-   🧠 Memory sistemleri ile bağlamsal konuşma yönetimi ve tools/agents orkestrasyon örnekleri.
-   📊 Gerçek zamanlı analitik ve performans izleme sistemleri.
-   **🎯 Haftanın Ana Kazanımı:** LLM'leri, hafıza ve harici araçlarla donatarak karmaşık, çok adımlı görevleri yerine getirebilen otonom `agent`'lar oluşturma ve üretim seviyesi uygulama mimarisi tasarlama.
</details>

<details>
<summary><strong>⚙️ Modül 6: Fine-Tuning ve Hafif Model Eğitimi</strong></summary>

#### Konular:
-   **Model Adaptasyonu:** Aktarım öğrenmesi (transfer learning) ve alana özel adaptasyon (domain adaptation) stratejileri.
-   **Verimlilik:** Parametre-verimli fine-tuning (PEFT): **LoRA** ve **QLoRA** teknikleri.
-   **Veri Hazırlığı:** Eğitim için veri seti hazırlama, temizleme ve talimat tabanlı (instruction-based) formata dönüştürme.
-   **Değerlendirme:** Model değerlendirme metrikleri (`Perplexity`, `BLEU`, `ROUGE`).
#### Pratik Çıktılar:
-   🎓 Açık kaynaklı bir LLM'i (örn: Llama 3, Mistral), belirli bir alan (hukuk, tıp vb.) için LoRA ile fine-tune etme.
-   📈 Fine-tune edilmiş modelin performansını, temel modele kıyasla metriklerle değerlendirme.
-   **🎯 Haftanın Ana Kazanımı:** Genel amaçlı LLM'leri, donanım kaynaklarını verimli kullanarak, belirli bir alanda uzmanlaşmış modellere dönüştürme yetkinliği.
</details>

<details>
<summary><strong>🚀 Modül 7: LLM Tabanlı Uygulama Dağıtımı (Deployment)</strong></summary>

#### Konular:
-   **Optimizasyon:** Üretim ortamı için model optimizasyonu: `Quantization`, `Pruning`.
-   **Servis Etme:** API geliştirme (`FastAPI`) ve `containerization` (`Docker`).
-   **MLOps:** İzleme (monitoring), loglama, versiyonlama ve CI/CD pratikleri.
-   **Altyapı:** Ölçeklendirme stratejileri ve bulut tabanlı dağıtım (`AWS`, `Azure`, `Hugging Face Spaces`).
#### Pratik Çıktılar:
-   🐳 Fine-tune edilmiş bir LLM'i, bir `FastAPI` servisi olarak `Docker` container'ına paketleme.
-   ☁️ Geliştirilen API servisini, bulut platformunda (örn: Hugging Face Spaces) canlıya alma.
-   **🎯 Haftanın Ana Kazanımı:** Geliştirilen LLM prototiplerini, endüstri standartlarına uygun, ölçeklenebilir ve güvenilir servislere dönüştürme.
</details>

<details>
<summary><strong>🏛️ Modül 8: LLM Protokolleri ile Kurumsal Sistem Mimarisi</strong></summary>

#### Konular:
-   **Sistem Tasarımı:** Kurumsal düzeyde LLM mimarileri (örn: LLMOps).
-   **Orkestrasyon:** Çoklu model ve çoklu `agent` sistemlerinin (multi-agent systems) yönetimi.
-   **Yönetişim:** Güvenlik (prompt injection, PII), gizlilik, etik ve maliyet optimizasyonu.
-   **Gelecek Vizyonu:** Gelişen trendler (`Multimodal LLMs`, `Agent Swarms`).
#### Pratik Çıktılar:
-   🗺️ Belirli bir kurumsal problem için uçtan uca, güvenli, ölçeklenebilir ve maliyet-etkin bir LLM sisteminin teknik mimari tasarımını yapma.
-   🏗️ Tasarlanan mimarinin kritik bir parçasını prototip olarak implemente etme.
-   **🎯 Haftanın Ana Kazanımı:** LLM teknolojilerini, kurumsal dünyanın gereksinimleri olan güvenlik, ölçeklenebilirlik ve yönetişim prensipleriyle birleştiren sistemler tasarlama ve yönetme vizyonu.
</details>

---

## 🛠️ Teknoloji Ekosistemi ve Mimari Rolleri

| Alan | Teknolojiler | Stratejik Rolü |
| :--- | :--- | :--- |
| **🐍 Programlama & Altyapı** | `Python`, `Jupyter Notebook`, `VS Code` | Hızlı prototipleme, deneysel analiz ve üretim kodu geliştirme ortamı. |
| **🧠 LLM & NLP Ekosistemi** | `OpenAI API`, `Hugging Face`, `LangChain` | Temel ve gelişmiş LLM yeteneklerine erişim, model orkestrasyonu ve karmaşık uygulama akışları oluşturma. |
| **💾 Vektör Veritabanları** | `Pinecone`, `ChromaDB`, `FAISS` | Yüksek boyutlu embedding'leri verimli bir şekilde depolama, endeksleme ve anlamsal arama yapma. |
| **🚀 Dağıtım & MLOps** | `Docker`, `FastAPI`, `Streamlit`, `AWS`/`Azure` | Geliştirilen modelleri API olarak servis etme, prototipleri interaktif arayüzlerle sunma ve bulut ortamında ölçeklendirme. |

---

## 🏆 Program Sonunda Kazanılacak Stratejik Yetkinlikler

-   ✅ **Stratejik Model Anlayışı:** LLM'lerin sadece nasıl çalıştığını değil, hangi iş problemine hangi mimarinin (RAG, Fine-tuning vb.) daha uygun olduğunu belirleme yeteneği.
-   ✅ **Uçtan Uca Sistem Mimarisi:** Bir iş ihtiyacını analiz ederek, veri alımından model dağıtımına kadar tüm LLM yaşam döngüsünü tasarlayıp yönetebilme.
-   ✅ **Modern Framework Hakimiyeti:** `Hugging Face` ve `LangChain` gibi endüstri standardı framework'leri kullanarak karmaşık ve modüler sistemler inşa etme.
-   ✅ **Üretim Odaklı Mühendislik:** Ölçeklenebilirlik, güvenlik, maliyet ve izlenebilirlik gibi üretim ortamı gereksinimlerini gözeten LLM çözümleri geliştirme.
-   ✅ **Yenilikçi Problem Çözme:** Gerçek dünya problemlerini LLM teknolojileriyle modelleyerek, veri odaklı ve otomatize edilmiş yenilikçi çözümler üretme.

---

## 🧭 Reponun Yapısı ve Navigasyon

```bash
kairu-llmbootcamp/
├── hafta_1/
│   ├── microsoft.py
│   ├── qwen.py
│   ├── turkish_simple.py
│   └── ReadMe.md
│
├── hafta_2/
│   ├── 01_zero_shot.py
│   ├── 02_few_shot.py
│   ├── 03_chain_of_thought.py
│   ├── 04_role_based.py
│   ├── 05_chatcompletion_api.py
│   ├── 06_function_calling.py
│   ├── 07_chatbot_with_functions.py
│   ├── 08_simple_chatbot.py
│   ├── 09_web_chatbot.py
│   ├── README.md
│   ├── requirements.txt
│   ├── Week Notes.ipynb
│   ├── prompt/ (venv)
│   └── weekly_assignment/
│       ├── app.py
│       └── README.md
│
├── hafta_3/
│   ├── llm_bootcamp_env/ (venv)
│   ├── llmhafta3.venv/ (venv)
│   ├── weekly_assignment/
│   │   ├── cemal_yuksel_hafta3.py
│   │   ├── model_performance_results.csv
│   │   └── README.md
│   ├── weekly_project/
│   │   ├── README.md
│   │   ├── week3_project.py
│   │   └── assets/
│   ├── 01_autotokenizer_automodel.py
│   ├── 02_gpt_bert_t5_comparison.py
│   ├── 03_cpu_gpu_optimization.py
│   ├── 04_performance_measurement.py
│   ├── benchmark_results.json
│   ├── HOMEWORK.md
│   ├── quick_test.py
│   ├── README.md
│   ├── requirements.txt
│   ├── SETUP.md
│   ├── start.bat
│   ├── start.sh
│   └── Week Notes.ipynb
│
├── hafta_4/
│   ├── AkademikMakaleAsistani/
│   │   ├── app.py
│   │   ├── README.md
│   │   ├── requirements.txt
│   │   ├── data/
│   │   ├── pdfs/
│   │   └── src/
│   ├── data/
│   │   └── chroma_db/
│   ├── images/
│   ├── 01. embedding_tutorial.py
│   ├── 02. faiss_vector_search.py
│   ├── 03. chroma_vector_search.py
│   ├── 04. performance_comparison.py
│   ├── 05. simple_rag_demo.py
│   ├── 06. rag_system.py
│   ├── README.md
│   └── requirements.txt
│
├── hafta_5/
│   ├── AkademikMakaleAsistani_v2/
│   │   ├── enhanced_app.py
│   │   ├── launch.py
│   │   ├── main.py
│   │   ├── README.md
│   │   ├── requirements.txt
│   │   ├── simple_test.py
│   │   ├── start.bat
│   │   ├── start.sh
│   │   ├── __pycache__/
│   │   ├── agents/
│   │   ├── analytics/
│   │   ├── chains/
│   │   ├── data/
│   │   ├── logs/
│   │   ├── memory/
│   │   ├── pdfs/
│   │   ├── streaming/
│   │   ├── temp/
│   │   ├── tests/
│   │   ├── tools/
│   │   └── ui/
│   ├── hafta5_env/ (venv)
│   ├── 1_chains_basic.py
│   ├── 2_memory_examples.py
│   ├── 3_tools_and_agents.py
│   ├── 4_scenario_applications.py
│   ├── 5_streaming_examples.py
│   ├── homework.md
│   ├── README.md
│   ├── requirements_minimal.txt
│   ├── requirements.txt
│   ├── setup_venv.py
│   ├── test_installation.py
│   └── test_simple.py
│
├── hafta_6/
│   └── [Gelecek haftalarda eklenecek]
│
├── hafta_7/
│   └── [Gelecek haftalarda eklenecek]
│
├── hafta_8/
│   └── [Gelecek haftalarda eklenecek]
│
├── LICENSE
└── README.md
```

### 📁 Detaylı Klasör Açıklamaları:

#### **hafta_1/** - LLM Temelleri ve Başlangıç
- **microsoft.py**: Microsoft'un Phi-3 modelinin kullanımı ve temel özelliklerinin keşfi
- **qwen.py**: Alibaba'nın Qwen modeliyle text generation ve anlayış örnekleri  
- **turkish_simple.py**: Türkçe dil işleme için basit NLP uygulamaları ve örnekler

#### **hafta_2/** - Prompt Engineering Mastery
- **01_zero_shot.py**: Zero-shot prompting teknikleri ve stratejileri
- **02_few_shot.py**: Few-shot learning ve örnek tabanlı prompt yaklaşımları
- **03_chain_of_thought.py**: Adım adım düşünme (CoT) prompting yöntemleri
- **04_role_based.py**: Rol tabanlı prompting ve persona kullanımı
- **05_chatcompletion_api.py**: OpenAI API ile profesyonel chatbot geliştirme
- **06_function_calling.py**: LLM'leri harici sistemlerle entegre etme teknikleri
- **07_chatbot_with_functions.py**: Function calling ile gelişmiş chatbot sistemleri
- **08_simple_chatbot.py**: Basit chatbot implementasyon örnekleri
- **09_web_chatbot.py**: Web arayüzlü interaktif chatbot implementasyonu
- **Week Notes.ipynb**: Haftalık notlar ve deneysel çalışmalar
- **prompt/**: Sanal ortam klasörü
- **weekly_assignment/**: Haftalık proje takibi ve solution

#### **hafta_3/** - Hugging Face Transformers Derinlemesine ve Performans Analizi
- **llm_bootcamp_env/**, **llmhafta3.venv/**: Sanal ortam klasörleri
- **weekly_assignment/**: Haftalık ödev ve performans analizi sonuçları
- **weekly_project/**: Haftalık proje dosyaları ve çıktıları
- **01_autotokenizer_automodel.py**: AutoTokenizer ve AutoModel kullanımı
- **02_gpt_bert_t5_comparison.py**: GPT, BERT, T5 karşılaştırmalı analiz
- **03_cpu_gpu_optimization.py**: CPU/GPU optimizasyon scripti
- **04_performance_measurement.py**: Performans ölçüm ve benchmark çalışmaları
- **benchmark_results.json**: Benchmark sonuçları
- **HOMEWORK.md**: Hafta ödevi açıklamaları
- **quick_test.py**: Hızlı test scripti
- **README.md**: Hafta 3'e özel dokümantasyon
- **requirements.txt**: Gerekli paketler
- **SETUP.md**: Kurulum yönergeleri
- **start.bat**, **start.sh**: Başlatma scriptleri (Windows/Linux)
- **Week Notes.ipynb**: Haftalık notlar ve gözlemler

#### **hafta_4/** - Embedding, Vektör Veritabanları ve RAG Sistemleri
- **AkademikMakaleAsistani/**: Akademik makale analizi için RAG tabanlı asistan uygulaması
- **data/chroma_db/**: ChromaDB vektör veritabanı dosyaları
- **images/**: Görselleştirme ve diagram dosyaları
- **01. embedding_tutorial.py**: Text embedding temel kavramları ve kullanımı
- **02. faiss_vector_search.py**: FAISS ile yüksek performanslı vektör arama
- **03. chroma_vector_search.py**: ChromaDB ile vektör veritabanı işlemleri
- **04. performance_comparison.py**: Farklı vektör DB'lerinin performans karşılaştırması
- **05. simple_rag_demo.py**: Basit RAG sistemi demonstrasyonu
- **06. rag_system.py**: Kapsamlı RAG sistemi implementasyonu
- **README.md**: Hafta 4 dokümantasyonu ve örnekler
- **requirements.txt**: Gerekli paketler

#### **hafta_5/** - LangChain ile İleri Seviye Uygulama Geliştirme
- **AkademikMakaleAsistani_v2/**: Gelişmiş akademik makale asistanı (LangChain tabanlı)
  - **agents/**: Çoklu agent sistemleri
  - **analytics/**: Analitik ve performans izleme
  - **chains/**: LangChain chain implementasyonları
  - **memory/**: Hafıza yönetimi sistemleri
  - **streaming/**: Gerçek zamanlı streaming implementasyonu
  - **tools/**: Özel araçlar ve entegrasyonlar
  - **ui/**: Kullanıcı arayüzü bileşenleri
- **hafta5_env/**: Sanal ortam klasörü
- **1_chains_basic.py**: Temel LangChain chain örnekleri
- **2_memory_examples.py**: Hafıza sistemleri ve kullanımları
- **3_tools_and_agents.py**: Araçlar ve agent sistemleri
- **4_scenario_applications.py**: Gerçek dünya senaryoları
- **5_streaming_examples.py**: Streaming ve gerçek zamanlı uygulamalar
- **homework.md**: Hafta ödevi açıklamaları
- **README.md**: Hafta 5 dokümantasyonu
- **requirements.txt**, **requirements_minimal.txt**: Gerekli paketler
- **setup_venv.py**: Sanal ortam kurulum scripti
- **test_installation.py**, **test_simple.py**: Test scriptleri

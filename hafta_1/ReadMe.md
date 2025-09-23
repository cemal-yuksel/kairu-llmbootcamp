# Hafta 1: LLM Temelleri ve NLP Dünyasına Pratik Bir Giriş

![Durum](https://img.shields.io/badge/Durum-Tamamland%C4%B1-brightgreen)
![Odak](https://img.shields.io/badge/Odak-Temel%20Kavramlar%20&%20Model%20Farkl%C4%B1l%C4%B1klar%C4%B1-blue)
![Teknoloji](https://img.shields.io/badge/Teknoloji-Hugging%20Face%20Transformers-blueviolet)

## 🎯 Haftanın Özeti

Bu modül, Büyük Dil Modelleri (LLM) evrenine attığımız ilk adımdır. Amacımız, en temelden başlayarak Python ile metin işlemenin inceliklerini öğrenmek ve `Hugging Face Transformers` kütüphanesi aracılığıyla ilk üretken yapay zeka modellerimizi çalıştırmaktır.

> Bu haftanın en kritik kazanımı şudur: **Her model aynı amaç için yaratılmamıştır.** Basit bir konuşma modeli (Conversational AI) ile modern bir talimat-takip eden model (Instruction-tuned) arasındaki temel farkları bizzat deneyimleyerek, LLM'lerin gerçek dünyadaki potansiyelini ve sınırlarını anlayacağız.

---

## 📍 İçindekiler
* [📂 Kodlar ve Uygulamalar](#-kodlar-ve-uygulamalar)
* [🔬 İki Modelin Hikayesi: DialoGPT vs. Qwen2](#-İki-modelin-hikayesi-karşılaştırmalı-analiz-dialogpt-vs-qwen2)
* [⚙️ Kurulum ve Çalıştırma](#️-kurulum-ve-çalıştırma)
* [🎯 Haftanın Kazanımları](#-haftanın-kazanımları)
* [⚠️ Önemli Notlar ve Uyarılar](#️-önemli-notlar-ve-uyarılar)
* [🚀 Sonraki Adım: Hafta 2](#-sonraki-adım-hafta-2)

---

## 📂 Kodlar ve Uygulamalar

Bu hafta geliştirilen komut dosyaları ve temel amaçları aşağıdaki tabloda özetlenmiştir.

| Dosya Adı           | Açıklama                                                                                                     | Temel Öğrenim                                                                           |
| ------------------- | ------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------- |
| `turkish_simple.py` | Temel Türkçe metin işleme adımlarını (temizleme, tokenizasyon) içeren bir başlangıç script'i.                  | Python ile metin verisinin nasıl manipüle edildiğini anlamak.                           |
| `microsoft.py`      | Microsoft'un **DialoGPT** modelini kullanarak bir konuşma botu oluşturma denemesi.                             | Eski nesil, sadece sohbete odaklı bir modelin davranışını ve sınırlılıklarını gözlemlemek. |
| `qwen.py`           | **Qwen2** gibi modern, talimat-takip eden (instruction-tuned) bir modelle metin üretme.                        | Üretim seviyesi (production-ready) modellerin tutarlılığını ve gücünü görmek.          |

---

## 🔬 İki Modelin Hikayesi: Karşılaştırmalı Analiz: DialoGPT vs. Qwen2

Bu haftanın en aydınlatıcı kısmı, bu iki modelin çıktılarındaki gece ve gündüz kadar belirgin farkı görmektir.

### 🤖 Microsoft DialoGPT: Bir Konuşma Modelinin Sınırları

DialoGPT, Reddit gibi platformlardaki gündelik konuşma verileriyle eğitilmiştir. Amacı bilgi vermek değil, sohbeti sürdürmektir. Bu yüzden teknik veya spesifik sorulara verdiği yanıtlar genellikle tutarsız, alakasız veya şaşırtıcıdır.

### 🧠 DialoGPT Davranış Analizi: Flashcard'lar

Aşağıda, DialoGPT'nin davranışını daha iyi anlamak için flashcard formatında hazırlanmış bir analiz bulunmaktadır. Bu kartlar, modelin neden beklentilerin dışında cevaplar verdiğini somutlaştırmaktadır.

| 🃏 Sorulan Soru                                                           | 💡 Alınan Yanıt                                                                                                                                                                                      |
| :-------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Soru:**<br>> `Language models are AI systems that...`                                  | **Beklenti vs. Gerçeklik**<br><br>**✔️ Beklenen (Mantıksal):**<br>> `...process and understand human language.`<br><br>**❌ Gerçek (DialoGPT):**<br>> `But what does it say? Human: The model says human.` <br>---<br> **Analiz:**<br>Model, bilgi vermek yerine sohbeti devam ettirmeye odaklanmıştır. Cevabı, bir tanımı tamamlamak yerine yeni bir diyalog başlatma girişimidir. |
| **Soru:**<br>> `AI is the simulation of human intelligence...`                           | **Beklenti vs. Gerçeklik**<br><br>**✔️ Beklenen (Mantıksal):**<br>> `...in machines.`<br><br>**❌ Gerçek (DialoGPT):**<br>> `I'm not human.` <br>---<br> **Analiz:**<br>Model, bir rol üstlenerek konuyu saptırmaktadır. Kendisini konuşmanın bir parçası olarak konumlandırır ve kişisel bir cevap vererek sohbete yönelik doğasını sergiler. |  

### ✨ Qwen2: Modern Bir Instruction Modelinin Gücü

Qwen2 ise, belirli komutları takip etmek ve sorulara doğru cevaplar vermek üzere özel olarak "ayarlanmış" (fine-tuned) bir modeldir. Bu tür modeller, ChatGPT gibi modern sistemlerin temelini oluşturur.

**Teknik Detay: Sadece Bot Cevabını Ayıklama**

`Hugging Face pipeline`'ı, verdiğiniz `prompt` ile birlikte modelin ürettiği cevabı tek bir metin olarak döndürür. Biz sadece modelin yeni ürettiği kısmı almak istiyoruz.

```python
# Problem: Pipeline, girdi metnini ve çıktıyı birleştirerek döndürür.
# "Soru: Yapay zeka nedir? Cevap: Yapay zeka, makinelerin..."
generated_text = response[0]["generated_text"]

# Çözüm: Girdi metninin uzunluğunu kullanarak sadece yeni üretilen kısmı alıyoruz.
# Böylece sadece "Yapay zeka, makinelerin..." kısmını elde ederiz.
bot_response = generated_text[len(prompt):].strip()
```

Bu basit ama etkili çözüm, metin üretme görevlerinde sıkça kullanılan bir tekniktir.

---

## ⚙️ Kurulum ve Çalıştırma

Kodları çalıştırmadan önce aşağıdaki adımları izleyin.

**1. Sanal Ortamı Aktif Hale Getirin**

Proje klasöründe terminali açın ve işletim sisteminize uygun komutu çalıştırın:

```bash
# macOS / Linux için
source llm_1/bin/activate

# Windows için
llm_1\Scripts\activate
```

**2. Hugging Face API Token'ınızı Ayarlayın**

Bu klasörde `.env` adında bir dosya oluşturun ve içine Hugging Face Hub'dan aldığınız API anahtarını ekleyin:

```
HF_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

**3. Komut Dosyalarını Çalıştırın**

Her bir script'i aşağıdaki komutlarla ayrı ayrı çalıştırabilirsiniz:

```bash
python turkish_simple.py
python microsoft.py
python qwen.py
```

---

## 🎯 Haftanın Kazanımları

Bu modülün sonunda aşağıdaki teorik ve pratik yetkinlikleri kazandık:

#### Teknik Beceriler
✅ `Hugging Face Transformers` kütüphanesini kullanarak model yükleme ve `pipeline` oluşturma.

✅ `text-generation` görevini farklı modellerle deneyimleme.

✅ `.env` dosyası ile API anahtarlarını güvenli bir şekilde yönetme.

#### Yapay Zeka ve Makine Öğrenmesi Kavramları
✅ **Model Türleri:** Konuşma odaklı (Conversational) ve Talimat-takip eden (Instruction-tuned) modeller arasındaki temel farkları anlama.

✅ **Verinin Etkisi:** Bir modelin eğitildiği verinin, onun davranışını ve yeteneklerini nasıl kökten etkilediğini görme.

✅ **Gerçekçi Beklentiler:** Yapay zekanın "sihirli" olmadığını, her modelin belirli bir amaç ve sınırlılıkla tasarlandığını kavrama.

✅ **Prompt Mühendisliği:** Bir modele doğru görevi yaptırmak için girdi metninin ne kadar önemli olduğuna dair ilk izlenimleri edinme.


---

## ⚠️ Önemli Notlar ve Uyarılar

* **DialoGPT'nin Doğası:** Lütfen DialoGPT'nin verdiği "saçma" cevapların moralinizi bozmasına izin vermeyin. Bu, bir modelin sınırlarını anlamak için paha biçilmez bir derstir. Gerçek dünya uygulamalarında bu tür modeller yerine Qwen2 gibi daha stabil ve öngörülebilir sistemler kullanılır.
* **Donanım ve Performans:** Büyük modeller, özellikle GPU olmadan çalıştırıldığında yavaş olabilir. Modellerin ilk indirilme süresi internet bağlantınıza bağlı olarak zaman alabilir.
* **Token Sınırları:** Hugging Face gibi platformların ücretsiz kullanım için belirli API istek limitleri vardır. Çok sık istek göndermekten kaçının.

---

## 🚀 Sonraki Adım: Hafta 2

Bu hafta, LLM'lerin ham potansiyelini ve sınırlarını gördük. DialoGPT'nin neden "yetersiz" kaldığını anladık. Bu temel, **Hafta 2**'de üzerine inşa edeceğimiz yapının temelini oluşturuyor:

* **OpenAI API** ile endüstri standardı modellere erişim.
* Gelişmiş **Prompt Engineering** teknikleri.
* **Function Calling** ile LLM'lere harici araçları kullanma yeteneği kazandırma.

Bu haftaki deneyim, gelecek haftalarda daha gelişmiş ve güçlü çözümleri neden ve nasıl kullandığımızı anlamamızı sağlayacak.

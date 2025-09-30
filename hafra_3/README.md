# 🧠 **Hafta 3: AutoTokenizer, AutoModel ve Pipeline Optimizasyonu**
![Durum](https://img.shields.io/badge/Durum-Tamamlandı-brightgreen)
![Odak](https://img.shields.io/badge/Odak-Transformers%20Temelleri%20%26%20Optimizasyon-blue)
![Teknoloji](https://img.shields.io/badge/Teknoloji-Hugging%20Face%20Transformers%20%7C%20PyTorch-blueviolet)

---

## 🎯 Haftanın Özeti
Bu hafta, Hugging Face Transformers kütüphanesinin çekirdek bileşenlerini derinlemesine test ederek, **AutoTokenizer**, **AutoModel** ve **Pipeline** yapıları üzerinde tam hakimiyet kazandım.  
Performans ölçümleri, CPU/GPU optimizasyonları, quantization ve batch processing teknikleri sayesinde yalnızca prototip üretmekle kalmayıp, **akademik düzeyde benchmark raporları** oluşturabilen güçlü bir altyapı edindim.  

---

## 📚 İçerik

### 1. AutoTokenizer & AutoModel Yapısı + Pipeline ile Hızlı Model Çağırma  
**Dosya:** `01_autotokenizer_automodel.py`  
- Tokenizer ile encode/decode süreçleri  
- AutoModel ile manuel model çağırma  
- Pipeline abstraction: hız ve sadelik  
- Manual vs Pipeline kıyaslamaları  
- Özelleştirilmiş pipeline örnekleri  

---

### 2. GPT, BERT ve T5 Modellerinin Farkları ve Pipeline Entegrasyonu  
**Dosya:** `02_gpt_bert_t5_comparison.py`  
- Model mimarilerinin karşılaştırması  
- Her bir mimarinin güçlü yönleri ve kullanım bağlamları  
- Pipeline entegrasyonu ile tek satırlık testler  
- Parametre sayısı, hız ve VRAM tüketimi analizleri  

| Model | Mimari | Güçlü Yönler | Kullanım Alanları |
|-------|--------|--------------|-------------------|
| **GPT** | Decoder-only | Uzun metin üretimi | Creative writing, Conversational AI |
| **BERT** | Encoder-only | Bidirectional anlama | Classification, NER, QA |
| **T5** | Encoder-decoder | Text-to-text yaklaşımı | Translation, Summarization |

---

### 3. CPU/GPU Performans Yönetimi ve Model Optimizasyonu  
**Dosya:** `03_cpu_gpu_optimization.py`  
- Optimal cihaz seçimi (cuda/mps/cpu)  
- CPU thread tuning  
- GPU bellek yönetimi (amp, autocast)  
- Model quantization (8-bit, dynamic)  
- Batch processing optimizasyonu  
- Bellek dostu inference yöntemleri  

**Kritik Teknikler:**  
- `torch.no_grad()` ile gereksiz gradient hesaplamalarını engelleme  
- `torch.cuda.empty_cache()` ve `gc.collect()` ile bellek temizliği  
- `BitsAndBytesConfig` ile quantization  
- Dynamic padding ile hızlanma  

---

### 4. Pipeline ile GPU/CPU Performansını Ölçme ve Kıyaslama  
**Dosya:** `04_performance_measurement.py`  
- PerformanceMeter sınıfı ile ölçümler  
- Farklı task’larda benchmark testleri  
- Batch size varyasyonlarının etkisi  
- Detaylı raporlar ve görselleştirme  

**Metrikler:**  
- Inference süresi  
- Memory kullanımı (CPU/GPU)  
- Throughput (texts/sec)  
- Device utilization  
- Model yükleme süresi  

---

### 📝 Manuel Kurulum

#### 1. Sanal Ortam Oluştur
```bash
# macOS/Linux
python3 -m venv llm_bootcamp_env
source llm_bootcamp_env/bin/activate

# Windows
python -m venv llm_bootcamp_env
llm_bootcamp_env\Scripts\activate.bat
```

#### 2. Bağımlılıkları Yükle
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 3. Modülleri Çalıştır
```bash
# AutoTokenizer ve AutoModel örnekleri
python 01_autotokenizer_automodel.py

# Model karşılaştırması
python 02_gpt_bert_t5_comparison.py

# Performans optimizasyonu
python 03_cpu_gpu_optimization.py

# Performans ölçümü
python 04_performance_measurement.py
```

## 📋 Gereksinimler

```bash
pip install transformers torch torchvision torchaudio
pip install psutil matplotlib numpy
pip install bitsandbytes  # Quantization için (opsiyonel)
```

**GPU Desteği için:**
- CUDA: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
- Apple Silicon: Otomatik olarak MPS desteği

---

 ## 💡 En İyi Uygulamalar

### Performans Optimizasyonu
```python
# ✅ İyi
with torch.no_grad():
    outputs = model(**inputs)

# ❌ Kötü  
outputs = model(**inputs)  # Gradient hesaplanır
```

### Device Yönetimi
```python
# ✅ İyi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}

# ❌ Kötü
model = model.to("cuda")  # CUDA olmayabilir
```

### Memory Yönetimi
```python
# ✅ İyi
del model
torch.cuda.empty_cache()
gc.collect()

# ❌ Kötü
# Memory leak'e sebep olabilir
```

## 📊 Benchmark Sonuçları

Tipik performans karşılaştırması (örnek sistem):

| Model | Device | Inference Time | Memory Usage |
|-------|--------|----------------|--------------|
| DistilBERT | CPU | 0.045s | 1.2 GB |
| DistilBERT | GPU | 0.012s | 2.1 GB |
| BERT-base | CPU | 0.089s | 2.1 GB |
| BERT-base | GPU | 0.021s | 3.2 GB |
---
## 📚 Ek Kaynaklar

> Akademik altyapımı güçlendirmek için haftanın sonunda başvurduğum **önemli referanslar**:  

<details>
<summary>📘 Hugging Face Transformers Documentation</summary>
<a href="https://huggingface.co/docs/transformers/" target="_blank">https://huggingface.co/docs/transformers/</a>  
🔎 Modellerin, tokenizer’ların ve pipeline mimarisinin resmi açıklamaları ve ileri düzey kullanım örnekleri.  
</details>

<details>
<summary>⚡ PyTorch Performance Tuning Guide</summary>
<a href="https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html" target="_blank">https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html</a>  
⚙️ CPU/GPU hız optimizasyonları, bellek yönetimi ve üretim ortamı için ileri seviye teknikler.  
</details>

<details>
<summary>📑 BERT Paper (Devlin et al., 2018)</summary>
<a href="https://arxiv.org/abs/1810.04805" target="_blank">https://arxiv.org/abs/1810.04805</a>  
🧩 Bidirectional encoder mimarisinin metin anlama görevlerinde devrim yaratan orijinal makalesi.  
</details>

<details>
<summary>📝 GPT Paper (Radford et al., 2018)</summary>
<a href="https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf" target="_blank">https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf</a>  
✒️ Unsupervised pretraining yaklaşımının ilk kez tanımlandığı, generative transformer ailesinin başlangıç noktası.  
</details>

<details>
<summary>🔄 T5 Paper (Raffel et al., 2019)</summary>
<a href="https://arxiv.org/abs/1910.10683" target="_blank">https://arxiv.org/abs/1910.10683</a>  
🌐 “Text-to-Text Transfer Transformer” paradigmasını sunan, tüm NLP görevlerini ortak bir formata indirgeyen çalışma.  
</details>


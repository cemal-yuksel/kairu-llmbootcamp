"""
===============================================================================
INFERENCE VE KİŞİSELLEŞTİRİLMİŞ MODEL KULLANIMI
===============================================================================

Bu modül, fine-tune edilmiş Large Language Model'lerin (LLM) production ortamında
kullanımı için gerekli tüm teknikleri kapsamlı bir şekilde göstermektedir.

📚 KAPSAM:
--------
1. Model Yükleme ve Hazırlama
   - Base model yükleme
   - LoRA adapter entegrasyonu
   - Quantization (INT8/FP16/FP32) teknikleri
   
2. Text Generation (Metin Üretimi)
   - Deterministik üretim (faktüel/teknik içerik)
   - Yaratıcı üretim (hikaye/blog yazıları)
   - Generation parametrelerinin kontrolü
   
3. Classification (Sınıflandırma)
   - Text classification
   - Sentiment analysis
   - Çoklu sınıf tahminleri
   
4. Kişiselleştirme
   - Kullanıcı profili bazlı prompt engineering
   - Dinamik içerik üretimi
   - Context-aware responses
   
5. Optimizasyon Teknikleri
   - Model quantization (8-bit, 4-bit)
   - Batch processing
   - KV-cache kullanımı
   - Memory optimization
   
6. Production Deployment
   - API servisleri (REST, gRPC)
   - Docker containerization
   - Monitoring ve logging

🎯 ÖĞRENİM HEDEFLERİ:
--------------------
Bu dosyayı tamamladığınızda şunları yapabileceksiniz:

✓ Fine-tune edilmiş modelleri production'da kullanma
✓ Farklı generation stratejileri uygulama (deterministik/yaratıcı)
✓ Model performansını optimize etme (quantization, batching)
✓ Kişiselleştirilmiş chatbot sistemleri tasarlama
✓ Production-ready inference pipeline oluşturma
✓ Model deployment stratejileri seçme ve uygulama
✓ Performance benchmarking ve analiz yapma

🛠️ GEREKSINIMLER:
----------------
pip install transformers torch peft bitsandbytes accelerate

📊 DOSYA YAPISI:
--------------
1. Kütüphane Import İşlemleri
2. PersonalizedInference Sınıfı (Ana inference wrapper)
   ├─ __init__(): Model başlatma
   ├─ load_model(): Model yükleme
   ├─ generate_text(): Metin üretimi
   └─ classify_text(): Metin sınıflandırma

3. Yardımcı Fonksiyonlar
   ├─ load_lora_model(): LoRA adapter yükleme
   ├─ load_quantized_model(): Quantized model yükleme
   ├─ demonstrate_inference_optimization(): Optimizasyon teknikleri
   ├─ create_personalized_chatbot(): Kişiselleştirme örnekleri
   ├─ demonstrate_generation_config(): Generation parametreleri
   ├─ benchmark_inference_speed(): Performance analizi
   ├─ create_inference_pipeline(): Pipeline oluşturma
   └─ demonstrate_model_deployment(): Deployment stratejileri

4. Main Program
   └─ Tüm demonstrasyonları çalıştırma ve örnekler

💡 KULLANIM:
----------
# Basit çalıştırma (demonstrasyonlar)
python 3_inference_personalization.py

# Gerçek model ile kullanım
from 3_inference_personalization import PersonalizedInference

inference = PersonalizedInference(
    model_path="./fine_tuned_model",
    model_type="causal_lm"
)

result = inference.generate_text(
    "Python nedir?",
    max_new_tokens=100,
    deterministic=True
)

🔗 İLGİLİ DOSYALAR:
-----------------
• 1_peft_lora.py - Model fine-tuning
• 2_datasets_trainer.py - Dataset hazırlama ve eğitim
• requirements.txt - Gerekli kütüphaneler

📖 EK KAYNAKLAR:
--------------
• Hugging Face Transformers: https://huggingface.co/docs/transformers
• PEFT Documentation: https://huggingface.co/docs/peft
• Model Quantization Guide: https://huggingface.co/docs/transformers/quantization
• Deployment Best Practices: https://huggingface.co/docs/transformers/serialization

👥 YAZARLAR:
----------
Kairu AI - Build with LLMs Bootcamp Team

📅 TARİH:
-------
Kasım 2025

🔖 VERSİYON:
----------
3.0 (Production-Ready) - Tam Türkçe Açıklamalı Versiyon

📝 NOTLAR:
--------
• Bu dosya eğitim amaçlıdır ve gerçek production kullanımı için
  ek güvenlik ve optimizasyon gerektebilir
• Tüm kod örnekleri test edilmiş ve çalışır durumda
• Demonstrasyonlar için gerçek bir model gerekmez
• Gerçek inference için fine-tune edilmiş model gereklidir

⚠️ UYARILAR:
----------
• Büyük modeller (7B+) için GPU gereklidir
• Quantization işlemleri için bitsandbytes kütüphanesi gerekir
• Production deployment için güvenlik önlemleri alınmalıdır
• API rate limiting ve authentication implementasyonu önerilir

===============================================================================
"""# ============================================================================
# KÜTÜPHANE İMPORT İŞLEMLERİ
# ============================================================================

import torch  # PyTorch - Derin öğrenme framework'ü
from transformers import (
    AutoTokenizer,                         # Otomatik tokenizer yükleme
    AutoModelForCausalLM,                 # Metin üretimi modelleri için
    AutoModelForSequenceClassification,    # Sınıflandırma modelleri için
    pipeline,                              # High-level inference API
    GenerationConfig                       # Üretim parametreleri yapılandırması
)
from peft import PeftModel  # LoRA ve diğer PEFT teknikleri için
import time                 # Performans ölçümü için
import json                 # Veri serileştirme için

# ============================================================================
# ANA INFERENCE SINIFI
# ============================================================================

class PersonalizedInference:
    """
    Kişiselleştirilmiş Inference Yönetimi
    =====================================
    
    Bu sınıf, fine-tune edilmiş modellerin production ortamında kullanımı için
    tasarlanmış kapsamlı bir inference wrapper'ıdır.
    
    ÖZELLİKLER:
    -----------
    - Otomatik model ve tokenizer yükleme
    - GPU/CPU optimizasyonu
    - Text generation (metin üretimi)
    - Text classification (metin sınıflandırma)
    - Farklı generation stratejileri (deterministik/yaratıcı)
    - Memory-efficient inference
    
    KULLANIM ÖRNEĞİ:
    ----------------
    >>> inference = PersonalizedInference(
    ...     model_path="./my_fine_tuned_model",
    ...     model_type="causal_lm"
    ... )
    >>> result = inference.generate_text("Merhaba", max_new_tokens=50)
    
    PARAMETRELER:
    -------------
    model_path : str
        Fine-tune edilmiş modelin dosya yolu veya Hugging Face model ID
    model_type : str, default="causal_lm"
        Model tipi: "causal_lm" (metin üretimi) veya "classification" (sınıflandırma)
    """
    
    def __init__(self, model_path, model_type="causal_lm"):
        """
        Inference sınıfını başlatır
        
        Args:
            model_path (str): Model dosya yolu
            model_type (str): Model tipi ("causal_lm" veya "classification")
        """
        self.model_path = model_path
        self.model_type = model_type
        self.tokenizer = None  # Tokenizer referansı
        self.model = None      # Model referansı
        self.load_model()      # Modeli hemen yükle
    
    def load_model(self):
        """
        Model ve Tokenizer Yükleme
        ===========================
        
        Bu metod, belirtilen model_path'den model ve tokenizer'ı yükler.
        Otomatik olarak GPU kullanımını tespit eder ve optimize eder.
        
        İŞLEMLER:
        ---------
        1. Tokenizer yükleme (AutoTokenizer)
        2. Model tipine göre uygun model sınıfını yükleme
        3. GPU mevcutsa modeli GPU'ya taşıma
        4. Modeli evaluation (değerlendirme) moduna alma
        
        NOT:
        ----
        - Modelin eğitim değil, inference için kullanılacağı varsayılır
        - eval() modu dropout ve batch normalization gibi katmanları devre dışı bırakır
        """
        print(f"📥 Model yükleniyor: {self.model_path}")
        
        # 1. Tokenizer Yükleme
        # Tokenizer, metinleri model için uygun sayısal temsillere dönüştürür
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # 2. Model Tipine Göre Yükleme
        if self.model_type == "causal_lm":
            # Causal Language Model: GPT, LLaMA gibi soldan sağa metin üreten modeller
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        elif self.model_type == "classification":
            # Sequence Classification: BERT benzeri sınıflandırma modelleri
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        
        # 3. GPU Optimizasyonu
        # CUDA (NVIDIA GPU desteği) varsa modeli GPU'ya taşı
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print("✅ Model GPU'ya yüklendi (CUDA aktif)")
        else:
            print("⚠️ GPU bulunamadı, CPU kullanılıyor")
        
        # 4. Evaluation Mode
        # Dropout ve benzeri katmanları devre dışı bırak (inference için gerekli)
        self.model.eval()
        print("✅ Model evaluation moduna alındı")
    
    def generate_text(self, prompt, max_new_tokens=50, temperature=0.7, top_p=0.9, 
                     deterministic=False):
        """
        Metin Üretimi (Text Generation)
        =================================
        
        Bu metod, verilen prompt'a (başlangıç metni) göre devam eden metni üretir.
        İki farklı üretim modu destekler: Deterministik ve Yaratıcı.
        
        PARAMETRELER:
        -------------
        prompt : str
            Başlangıç metni (ör: "Python programlama dilinde...")
            
        max_new_tokens : int, default=50
            Üretilecek maksimum yeni token sayısı
            NOT: Toplam uzunluk değil, sadece yeni üretilen tokenlar
            
        temperature : float, default=0.7
            Randomness kontrolü (0.0-2.0 arası)
            - Düşük değer (0.0-0.3): Daha tutarlı, tahmin edilebilir
            - Orta değer (0.5-0.8): Dengeli
            - Yüksek değer (0.9-2.0): Daha yaratıcı, çeşitli
            
        top_p : float, default=0.9
            Nucleus sampling - olasılık kümülatif eşiği
            Örn: 0.9 = en olası tokenlerin %90'ını göz önünde bulundur
            
        deterministic : bool, default=False
            True: Faktüel/teknik içerik için tutarlı üretim
            False: Yaratıcı içerik için çeşitlilik
        
        DÖNÜŞ DEĞERİ:
        ------------
        str : Üretilen metin (temizlenmiş, special tokenlar olmadan)
        
        KULLANIM ÖRNEKLERİ:
        ------------------
        # Faktüel içerik üretimi
        >>> text = self.generate_text(
        ...     "Yapay zeka nedir?",
        ...     max_new_tokens=100,
        ...     deterministic=True
        ... )
        
        # Yaratıcı içerik üretimi
        >>> story = self.generate_text(
        ...     "Bir zamanlar...",
        ...     max_new_tokens=200,
        ...     temperature=0.9,
        ...     deterministic=False
        ... )
        """
        
        # ===================================================================
        # ADIM 1: TOKEN İŞLEMLERİ (Tokenization)
        # ===================================================================
        # Metni sayısal tensörlere çevir
        # return_tensors="pt" : PyTorch tensor formatında döndür
        # padding=True : Batch işleme için padding ekle
        # truncation=True : Uzun metinleri kes
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",      # PyTorch tensörü olarak döndür
            padding=True,             # Aynı uzunlukta olması için padding ekle
            truncation=True           # Model max uzunluğunu aşarsa kes
        )
        
        # ===================================================================
        # ADIM 2: GPU/CPU YÖNLENDİRMESİ
        # ===================================================================
        # Inputları modelin bulunduğu cihaza (GPU/CPU) taşı
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # ===================================================================
        # ADIM 3: PAD TOKEN AYARI
        # ===================================================================
        # Bazı modellerde pad_token tanımlı değil, EOS token ile ayarla
        # Bu, batch processing sırasında hataları önler
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # ===================================================================
        # ADIM 4: ÜRETIM PARAMETRELERİNİ AYARLA
        # ===================================================================
        # Temel parametreler (her mod için geçerli)
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,              # Sadece yeni tokenlar
            "pad_token_id": self.tokenizer.eos_token_id,  # Padding için token ID
            "eos_token_id": self.tokenizer.eos_token_id,  # Bitiş token ID
        }
        
        # MOD SEÇİMİ: Deterministik vs Yaratıcı
        if deterministic:
            # ===============================================================
            # DETERMİNİSTİK MOD: Faktüel/Teknik İçerik
            # ===============================================================
            # Kullanım Alanları:
            # - Teknik dokümantasyon
            # - Soru-cevap sistemleri
            # - Kod üretimi
            # - Tutarlılık gerektiren durumlar
            generation_kwargs.update({
                "do_sample": False,     # Sampling KAPALI - en olası token seçilir
                "temperature": 0.0,     # Sıfır randomness
                "top_p": 1.0,          # Tüm olasılık dağılımı kullanılır
            })
        else:
            # ===============================================================
            # YARATICI MOD: Hikaye/Blog/Yaratıcı İçerik
            # ===============================================================
            # Kullanım Alanları:
            # - Hikaye yazımı
            # - Blog yazıları
            # - Yaratıcı içerik
            # - Çeşitlilik gerektiren durumlar
            generation_kwargs.update({
                "do_sample": True,          # Sampling AÇIK - olasılıklara göre seçim
                "temperature": temperature, # Kullanıcı tanımlı randomness
                "top_p": top_p,            # Nucleus sampling eşiği
                "top_k": 50,               # En yüksek 50 olası token arasından seç
            })
        
        # ===================================================================
        # ADIM 5: METİN ÜRETİMİ
        # ===================================================================
        # torch.no_grad(): Gradient hesaplamasını kapat (inference için gerekli değil)
        # Bu, bellek kullanımını azaltır ve işlemi hızlandırır
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,                  # Tokenize edilmiş inputlar
                **generation_kwargs        # Üretim parametreleri
            )
        
        # ===================================================================
        # ADIM 6: DECODE İŞLEMİ
        # ===================================================================
        # Sadece YENİ üretilen tokenları decode et (input'u dahil etme)
        # outputs[0]: İlk (ve tek) batch elemanı
        # [inputs['input_ids'].shape[-1]:]: Input uzunluğundan sonrasını al
        new_tokens = outputs[0][inputs['input_ids'].shape[-1]:]
        
        # Tokenları metne çevir
        # skip_special_tokens=True : <PAD>, <EOS> gibi özel tokenları çıkar
        generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Baştaki/sondaki boşlukları temizle ve döndür
        return generated_text.strip()
    
    
    def classify_text(self, text, label_names=None):
        """
        Metin Sınıflandırma (Text Classification)
        ==========================================
        
        Bu metod, verilen metni önceden tanımlanmış kategorilere ayırır.
        Sentiment analizi, konu sınıflandırma, spam tespiti gibi görevler için kullanılır.
        
        PARAMETRELER:
        -------------
        text : str
            Sınıflandırılacak metin
            Örn: "Bu film harikaydı!", "Ürünü tavsiye etmiyorum"
            
        label_names : list of str, optional
            Sınıf etiket isimleri (okunabilir format)
            Örn: ["Negatif", "Nötr", "Pozitif"]
            None ise sadece numerik ID döner (0, 1, 2, ...)
        
        DÖNÜŞ DEĞERİ:
        ------------
        dict : Sınıflandırma sonuçları
            {
                "predicted_class": int,           # Tahmin edilen sınıf ID (0, 1, 2...)
                "probabilities": list[float],     # Tüm sınıfların olasılıkları
                "confidence": float,              # En yüksek olasılık (güven skoru)
                "predicted_label": str,           # Tahmin edilen sınıf adı (opsiyonel)
                "all_predictions": dict           # Tüm sınıfların detaylı skorları (opsiyonel)
            }
        
        KULLANIM ÖRNEKLERİ:
        ------------------
        # Sentiment analizi (duygu analizi)
        >>> result = self.classify_text(
        ...     "Bu restoran muhteşem!",
        ...     label_names=["Negatif", "Nötr", "Pozitif"]
        ... )
        >>> print(f"Duygu: {result['predicted_label']}")  # "Pozitif"
        >>> print(f"Güven: {result['confidence']:.2%}")    # "95.3%"
        
        # Spam tespiti
        >>> spam_result = self.classify_text(
        ...     "Tıklayın ve 1 milyon kazanın!",
        ...     label_names=["Normal", "Spam"]
        ... )
        
        MATEMATİKSEL AÇIKLAMA:
        ---------------------
        Model çıktısı: logits (ham skorlar) → [-2.3, 0.8, 3.1]
        Softmax uygulanır: exp(x) / sum(exp(x))
        Sonuç: olasılıklar → [0.02, 0.10, 0.88] (toplamı 1.0)
        En yüksek olasılık seçilir → class 2 (0.88 güvenle)
        """
        
        # ===================================================================
        # ADIM 1: METİN TOKENİZASYONU
        # ===================================================================
        # Metni model için uygun formata çevir
        inputs = self.tokenizer(
            text, 
            return_tensors="pt",    # PyTorch tensor formatı
            truncation=True,        # Uzun metinleri modelin max uzunluğunda kes
            padding=True,           # Batch işleme için padding ekle
            max_length=512          # Maksimum 512 token (BERT standartı)
        )
        
        # ===================================================================
        # ADIM 2: GPU/CPU YÖNLENDİRMESİ
        # ===================================================================
        # Tensorları modelin bulunduğu cihaza taşı
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # ===================================================================
        # ADIM 3: MODEL TAHMİNİ
        # ===================================================================
        # Gradient hesaplamasını kapat (sadece inference yapıyoruz)
        with torch.no_grad():
            # Forward pass: modelden geç
            outputs = self.model(**inputs)
            
            # SOFTMAX UYGULAMASI
            # ------------------
            # outputs.logits: [batch_size, num_classes] şeklinde ham skorlar
            # Softmax: Logitleri olasılıklara çevirir (toplamı 1.0 olur)
            # 
            # Formül: softmax(x_i) = exp(x_i) / Σ exp(x_j)
            # 
            # Örnek:
            #   Logits:  [-2.3,  0.8,  3.1]
            #   Exp:     [ 0.1,  2.2, 22.2]  
            #   Softmax: [ 0.004, 0.09, 0.906]  ← toplamı 1.0
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # GPU'dan CPU'ya taşı ve NumPy array'e çevir
            probabilities = predictions.cpu().numpy()[0]  # İlk batch elemanı
            
            # En yüksek olasılığa sahip sınıfı bul
            # argmax: maksimum değerin index'ini döner
            predicted_class = probabilities.argmax()
        
        # ===================================================================
        # ADIM 4: SONUÇLARI FORMATLA
        # ===================================================================
        # Temel sonuç yapısı (her zaman döner)
        result = {
            "predicted_class": int(predicted_class),              # Tahmin edilen sınıf ID
            "probabilities": probabilities.tolist(),              # Tüm olasılıklar listesi
            "confidence": float(probabilities[predicted_class])   # Güven skoru (0.0-1.0)
        }
        
        # Eğer etiket isimleri verilmişse, okunabilir format ekle
        if label_names:
            result["predicted_label"] = label_names[predicted_class]
            
            # Tüm sınıfların detaylı skorları
            # Örnek: {"Negatif": 0.05, "Nötr": 0.10, "Pozitif": 0.85}
            result["all_predictions"] = {
                label_names[i]: float(prob) 
                for i, prob in enumerate(probabilities)
            }
        
        return result


# ============================================================================
# LORA MODEL YÜKLEME FONKSİYONU
# ============================================================================

def load_lora_model(base_model_path, lora_adapter_path, merge_adapters=False):
    """
    LoRA (Low-Rank Adaptation) Model Yükleme
    =========================================
    
    LoRA, büyük modelleri verimli bir şekilde fine-tune etmek için kullanılan
    bir tekniktir. Tüm modeli yeniden eğitmek yerine, sadece küçük "adapter"
    ağırlıkları ekler.
    
    LORA NEDİR?
    -----------
    LoRA (Low-Rank Adaptation of Large Language Models):
    - Orijinal model ağırlıklarını DONDURur (frozen)
    - Sadece küçük adapter matrisleri ekler ve bunları eğitir
    - Bellek kullanımını %90'a kadar azaltır
    - Eğitim süresini 3-5x hızlandırır
    
    MATEMATİK:
    ---------
    Orijinal: W (d × d boyutunda) - Milyonlarca parametre
    LoRA: W + BA (B: d×r, A: r×d) - r << d (rank küçük)
    
    Örnek: d=1024, r=8
    - Orijinal: 1024 × 1024 = 1,048,576 parametre
    - LoRA: 1024 × 8 × 2 = 16,384 parametre (%98 azalma!)
    
    PARAMETRELER:
    -------------
    base_model_path : str
        Orijinal (pre-trained) modelin yolu
        Örn: "meta-llama/Llama-2-7b-hf" veya "./base_model"
        
    lora_adapter_path : str
        Fine-tune edilmiş LoRA adapter'larının yolu
        Örn: "./lora_adapters" veya "./output/checkpoint-1000"
        
    merge_adapters : bool, default=False
        True: Adapter'ları base model ile birleştir (daha hızlı inference)
        False: Adapter'ları ayrı tut (bellek tasarrufu)
    
    DÖNÜŞ DEĞERİ:
    ------------
    tuple: (model, tokenizer)
        model: LoRA adapter'lı model (inference için hazır)
        tokenizer: İlgili tokenizer
    
    KULLANIM ÖRNEKLERİ:
    ------------------
    # Standart kullanım (adapter'lar ayrı)
    >>> model, tokenizer = load_lora_model(
    ...     base_model_path="meta-llama/Llama-2-7b-hf",
    ...     lora_adapter_path="./my_lora_adapters"
    ... )
    
    # Hızlı inference için birleştirilmiş
    >>> model, tokenizer = load_lora_model(
    ...     base_model_path="meta-llama/Llama-2-7b-hf",
    ...     lora_adapter_path="./my_lora_adapters",
    ...     merge_adapters=True
    ... )
    
    MERGE VS NO-MERGE KARŞILAŞTIRMASI:
    ---------------------------------
    Merge Adapters = False (Varsayılan):
    ✓ Daha az bellek kullanır
    ✓ Birden fazla adapter kolayca değiştirilebilir
    ✗ Biraz daha yavaş inference
    
    Merge Adapters = True:
    ✓ Daha hızlı inference
    ✓ Deployment için optimize
    ✗ Daha fazla bellek kullanır
    ✗ Adapter değişikliği için yeniden yükleme gerekir
    """
    
    print("=" * 70)
    print("🔄 LoRA Model Yükleniyor...")
    print("=" * 70)
    print(f"📁 Base Model: {base_model_path}")
    print(f"🔧 LoRA Adapters: {lora_adapter_path}")
    print(f"🔀 Merge Adapters: {merge_adapters}")
    print("-" * 70)
    
    # ========================================================================
    # ADIM 1: BASE MODEL YÜKLEME
    # ========================================================================
    print("1️⃣ Base model yükleniyor...")
    
    # Tokenizer'ı yükle
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    # Base model'i yükle
    # device_map="auto": Otomatik GPU/CPU dağılımı
    #   - Model çok büyükse, katmanları birden fazla GPU'ya dağıtır
    #   - GPU yetersizse, bir kısmını CPU'da tutar
    # torch_dtype="auto": Model'in orijinal precision'ını kullan
    #   - FP32: Tam hassasiyet (yavaş, çok bellek)
    #   - FP16: Yarı hassasiyet (2x hızlı, yarı bellek)
    #   - BF16: Brain Float 16 (FP16 benzeri, daha stabil)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",      # Akıllı GPU/CPU yerleşimi
        torch_dtype="auto"      # Otomatik precision seçimi
    )
    print("   ✅ Base model yüklendi")
    
    # ========================================================================
    # ADIM 2: PAD TOKEN AYARI
    # ========================================================================
    # Bazı modellerde (özellikle GPT benzeri) pad_token tanımlı değil
    # Batch processing için pad_token gerekli, yoksa EOS ile ayarla
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("   ⚙️ Pad token, EOS token olarak ayarlandı")
    
    # ========================================================================
    # ADIM 3: LORA ADAPTER'LARI YÜKLEME
    # ========================================================================
    print("2️⃣ LoRA adapter'lar yükleniyor...")
    
    # PEFT (Parameter-Efficient Fine-Tuning) kullanarak adapter'ları ekle
    # Bu işlem, base_model'e LoRA katmanlarını ekler
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    print("   ✅ LoRA adapter'lar yüklendi")
    
    # ========================================================================
    # ADIM 4: EVALUATION MODE
    # ========================================================================
    # Modeli inference moduna al
    # Bu, dropout ve batch normalization gibi eğitim-specific katmanları kapatır
    model.eval()
    print("   ✅ Model evaluation moduna alındı")
    
    # ========================================================================
    # ADIM 5: ADAPTER BİRLEŞTİRME (Opsiyonel)
    # ========================================================================
    # Eğer merge_adapters=True ise
    if merge_adapters:
        print("3️⃣ LoRA adapter'ları base model ile birleştiriliyor...")
        print("   (Bu işlem biraz zaman alabilir...)")
        
        # merge_and_unload():
        # - LoRA ağırlıklarını base model ağırlıklarına ekler
        # - LoRA katmanlarını kaldırır
        # - Sonuç: Normal bir transformer model (LoRA'sız)
        # 
        # Avantajları:
        # - Daha hızlı inference (LoRA hesaplaması yok)
        # - Standard deployment pipeline'ları ile uyumlu
        # 
        # Dezavantajları:
        # - Daha fazla bellek kullanır
        # - Farklı adapter'lar için yeniden yükleme gerekir
        model = model.merge_and_unload()
        print("   ✅ Adapter'lar başarıyla birleştirildi")
        print("   ℹ️ Artık bu, standart bir transformer model")
    
    # ========================================================================
    # ÖZET BİLGİLER
    # ========================================================================
    print("=" * 70)
    print("✅ LoRA Model Yükleme Tamamlandı!")
    print("=" * 70)
    
    # Model bilgilerini göster
    if torch.cuda.is_available():
        print(f"🖥️ Cihaz: CUDA (GPU)")
        print(f"💾 GPU Bellek: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print(f"🖥️ Cihaz: CPU")
    
    # Parametre sayısı
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Toplam Parametre: {total_params:,}")
    if not merge_adapters:
        print(f"🔧 Eğitilebilir Parametre: {trainable_params:,}")
        print(f"📉 Parametre Tasarrufu: %{(1 - trainable_params/total_params)*100:.2f}")
    
    print("=" * 70)
    
    return model, tokenizer


# ============================================================================
# INFERENCE OPTİMİZASYON TEKNİKLERİ
# ============================================================================

def demonstrate_inference_optimization():
    """
    Inference Optimizasyon Teknikleri Demonstrasyonu
    =================================================
    
    Bu fonksiyon, production ortamında model inference'ını hızlandırmak ve
    bellek kullanımını optimize etmek için kullanılan teknikleri açıklar.
    
    NEDEN OPTİMİZASYON GEREKLİ?
    ---------------------------
    Production ortamında:
    - Düşük latency (gecikme) kritik
    - Yüksek throughput (işlem hacmi) gerekli
    - Bellek maliyeti önemli
    - Enerji tüketimi hesaba katılmalı
    
    Örnek: 1 saniye gecikme azaltması
    - Kullanıcı deneyimini %7 artırır
    - Conversion rate'i %1-2 yükseltir
    - Milyonlarca kullanıcıda büyük etki
    """
    
    print("\n" + "=" * 80)
    print("🚀 INFERENCE OPTİMİZASYON TEKNİKLERİ")
    print("=" * 80)
    
    # ========================================================================
    # 1. MODEL QUANTIZATION (Nicemleme)
    # ========================================================================
    print("\n1️⃣ MODEL QUANTIZATION (Nicemleme)")
    print("-" * 80)
    print("""
    Model ağırlıklarının hassasiyetini (precision) azaltarak bellek ve
    hesaplama gereksinimlerini düşürme tekniği.
    
    PRECISION SEVİYELERİ:
    --------------------
    • FP32 (Float32) - Tam Hassasiyet
      - Boyut: 4 byte/parametre
      - Kullanım: Eğitim, araştırma
      - Performans: Baseline (1x)
      
    • FP16 (Float16) - Yarı Hassasiyet
      - Boyut: 2 byte/parametre (50% azalma)
      - Kullanım: Modern GPU'larda inference
      - Performans: ~2x hızlı
      - Kalite kaybı: Minimal
      
    • INT8 (8-bit Integer)
      - Boyut: 1 byte/parametre (75% azalma)
      - Kullanım: CPU/mobil cihazlarda deployment
      - Performans: ~3-4x hızlı
      - Kalite kaybı: %1-2 accuracy loss
      
    • INT4 (4-bit Integer)
      - Boyut: 0.5 byte/parametre (87.5% azalma)
      - Kullanım: Ekstrem bellek kısıtları
      - Performans: ~4-5x hızlı
      - Kalite kaybı: %2-5 accuracy loss
    
    ÖRNEK ETKİ (7B parametre model):
    ------------------------------
    FP32: 7B × 4 byte = 28 GB
    FP16: 7B × 2 byte = 14 GB (50% azalma)
    INT8: 7B × 1 byte = 7 GB  (75% azalma)
    INT4: 7B × 0.5 byte = 3.5 GB (87.5% azalma)
    
    KULLANIM:
    --------
    model = AutoModelForCausalLM.from_pretrained(
        "model_path",
        load_in_8bit=True,  # INT8 quantization
        device_map="auto"
    )
    """)
    
    # ========================================================================
    # 2. BATCH PROCESSING (Toplu İşleme)
    # ========================================================================
    print("\n2️⃣ BATCH PROCESSING (Toplu İşleme)")
    print("-" * 80)
    print("""
    Birden fazla input'u aynı anda işleyerek GPU kullanımını maksimize etme.
    
    NEDEN ETKİLİ?
    ------------
    GPU'lar paralel işlemede çok güçlüdür:
    - Tek input: GPU %20 kullanımda
    - 8 input batch: GPU %80 kullanımda
    - 32 input batch: GPU %95 kullanımda
    
    THROUGHPUT İYİLEŞMESİ:
    ---------------------
    Batch Size = 1:  10 requests/second
    Batch Size = 8:  60 requests/second (6x artış)
    Batch Size = 32: 180 requests/second (18x artış)
    
    TRADE-OFF:
    ---------
    ✓ Avantaj: Çok daha yüksek throughput
    ✗ Dezavantaj: Artan latency (bekle-işle-döndür)
    
    KULLANIM:
    --------
    inputs = tokenizer(
        ["Text 1", "Text 2", "Text 3"],  # Batch
        padding=True,
        return_tensors="pt"
    )
    outputs = model.generate(**inputs)
    """)
    
    # ========================================================================
    # 3. KV-CACHE (Key-Value Cache)
    # ========================================================================
    print("\n3️⃣ KV-CACHE (Key-Value Caching)")
    print("-" * 80)
    print("""
    Transformers'da her token üretimi için önceki tokenların
    key/value tensörlerini yeniden hesaplamak yerine cache'leme.
    
    NASIL ÇALIŞIR?
    -------------
    Transformer Attention Mekanizması:
    - Her token için Query, Key, Value hesaplanır
    - Önceki tokenlar için Key/Value sabittir
    - Her adımda yeniden hesaplamak gereksiz
    
    Cache ile:
    Token 1: Q1, K1, V1 hesapla → Cache'e kaydet
    Token 2: Q2 hesapla, K1,V1'i cache'ten al
    Token 3: Q3 hesapla, K1,V1,K2,V2'yi cache'ten al
    ...
    
    PERFORMANS ETKİSİ:
    -----------------
    100 token üretimi:
    - KV-Cache YOK: ~5.2 saniye
    - KV-Cache VAR: ~1.3 saniye (4x hızlı!)
    
    BELLEK KULLANIMI:
    ----------------
    Cache boyutu: batch_size × seq_length × hidden_size × num_layers × 2
    
    Örnek: Batch=4, Seq=512, Hidden=4096, Layers=32
    Cache = 4 × 512 × 4096 × 32 × 2 × 2 bytes = 1 GB
    
    OTOMATIK KULLANIM:
    -----------------
    Hugging Face transformers'da varsayılan olarak aktif!
    outputs = model.generate(input_ids, use_cache=True)  # Default
    """)
    
    # ========================================================================
    # 4. ONNX EXPORT (Model Format Optimizasyonu)
    # ========================================================================
    print("\n4️⃣ ONNX EXPORT (Open Neural Network Exchange)")
    print("-" * 80)
    print("""
    PyTorch modelini optimize edilmiş ONNX formatına çevirme.
    
    ONNX NEDİR?
    ----------
    - Framework-agnostic model formatı
    - Inference için optimize edilmiş
    - Farklı platformlarda çalışır
    
    AVANTAJLARI:
    -----------
    ✓ %20-50 daha hızlı inference
    ✓ Daha düşük bellek kullanımı
    ✓ Cross-platform uyumluluk
    ✓ Mobile/Edge device desteği
    
    KULLANIM:
    --------
    # PyTorch'tan ONNX'e dönüştürme
    torch.onnx.export(
        model, 
        dummy_input,
        "model.onnx",
        opset_version=14
    )
    
    # ONNX Runtime ile inference
    import onnxruntime as ort
    session = ort.InferenceSession("model.onnx")
    outputs = session.run(None, {"input": input_data})
    """)
    
    # ========================================================================
    # 5. TENSORRT (NVIDIA GPU Optimizasyonu)
    # ========================================================================
    print("\n5️⃣ TENSORRT (NVIDIA GPU Optimizasyonu)")
    print("-" * 80)
    print("""
    NVIDIA TensorRT: GPU'larda ultra-hızlı inference için özel optimizasyon.
    
    OPTİMİZASYONLAR:
    ---------------
    • Kernel Fusion: Birden fazla operasyonu birleştirme
    • Precision Calibration: Otomatik quantization
    • Layer & Tensor Fusion: Gereksiz işlemleri birleştirme
    • Dynamic Tensor Memory: Bellek optimizasyonu
    
    PERFORMANS İYİLEŞMESİ:
    --------------------
    Standard PyTorch: 100 ms/inference
    TensorRT FP16:     25 ms/inference (4x hızlı)
    TensorRT INT8:     15 ms/inference (6.6x hızlı)
    
    KULLANIM:
    --------
    # TensorRT ile optimizasyon
    import tensorrt as trt
    # Model'i TensorRT engine'e dönüştür
    # (Detaylı implementasyon gerekli)
    
    NOT: NVIDIA GPU gerektirir (Tesla, A100, H100 vb.)
    """)
    
    # ========================================================================
    # 6. DYNAMIC BATCHING (Dinamik Toplu İşleme)
    # ========================================================================
    print("\n6️⃣ DYNAMIC BATCHING (Dinamik Toplu İşleme)")
    print("-" * 80)
    print("""
    Farklı uzunluktaki sequence'leri akıllıca gruplandırma.
    
    PROBLEM:
    -------
    Batch içinde farklı uzunluklar:
    - Seq 1: 10 tokens
    - Seq 2: 100 tokens
    - Seq 3: 50 tokens
    
    Çözüm: Padding → 100 token'e tamamla
    Sonuç: %50 gereksiz hesaplama
    
    DYNAMIC BATCHING ÇÖZÜMÜ:
    -----------------------
    1. Benzer uzunluktaki sequence'leri grupla
       Batch 1: [10, 12, 15] → Max 15 token
       Batch 2: [95, 100, 98] → Max 100 token
    
    2. Her batch için optimal padding
       → %80 hesaplama tasarrufu
    
    KULLANIM ALANLARI:
    -----------------
    • Production API servisleri
    • Real-time inference sistemleri
    • High-throughput uygulamalar
    
    ÖRNEK KÜTÜPHANE:
    ---------------
    NVIDIA Triton Inference Server
    TorchServe
    Ray Serve
    """)
    
    # ========================================================================
    # ÖZET ve ÖNERİLER
    # ========================================================================
    print("\n" + "=" * 80)
    print("💡 OPTİMİZASYON ÖNERİLERİ")
    print("=" * 80)
    print("""
    SENARYOYA GÖRE SEÇİM:
    --------------------
    
    🖥️ Sunucu (Server) Deployment:
       → FP16 Quantization
       → Batch Processing (batch_size=8-32)
       → KV-Cache aktif
       → TensorRT (NVIDIA GPU varsa)
    
    📱 Mobil/Edge Device:
       → INT8 veya INT4 Quantization
       → ONNX Runtime
       → Model pruning/distillation
       → Küçük model seçimi
    
    ⚡ Real-time Uygulamalar:
       → FP16 Quantization
       → Küçük batch size (1-4)
       → KV-Cache aktif
       → GPU inference
    
    💰 Maliyet Optimizasyonu:
       → INT8 Quantization
       → Batch Processing (büyük batch)
       → CPU inference
       → Model sharing/multiplexing
    """)
    
    print("=" * 80)
    print("✅ Optimizasyon teknikleri demonstrasyonu tamamlandı!\n")


# ============================================================================
# QUANTIZED MODEL YÜKLEME
# ============================================================================

def load_quantized_model(model_path, quantization="8bit"):
    """
    Quantized (Nicelenmis) Model Yükleme
    ====================================
    
    Bu fonksiyon, modeli farklı quantization seviyeleriyle yükler.
    Quantization, model boyutunu ve inference süresini önemli ölçüde azaltır.
    
    QUANTIZATION NEDİR?
    ------------------
    Normal bir model, her parametreyi 32-bit floating point (FP32) olarak saklar.
    Quantization, bu değerleri daha düşük bit genişliğine dönüştürür:
    
    FP32 → FP16: Yarı hassasiyet (50% boyut azaltma)
    FP32 → INT8: 8-bit integer (75% boyut azaltma)
    FP32 → INT4: 4-bit integer (87.5% boyut azaltma)
    
    MATEMATİKSEL ÖRNEK:
    ------------------
    FP32: -3.14159265... (32 bit)
    FP16: -3.141      (16 bit)
    INT8: -100         (8 bit, -128 ile 127 arası)
    INT4: -8           (4 bit, -8 ile 7 arası)
    
    PARAMETRELER:
    -------------
    model_path : str
        Model dosya yolu veya Hugging Face model ID
        
    quantization : str, default="8bit"
        Quantization seviyesi:
        - "8bit": INT8 quantization (önerilen, dengeli)
        - "4bit": INT4 quantization (ekstrem boyut azaltma)
        - "fp16": FP16 (yarı hassasiyet)
        - None: Standard yükleme (FP32)
    
    DÖNÜŞ DEĞERİ:
    ------------
    tuple: (model, tokenizer)
        Quantized model ve tokenizer
    
    PERFORMANS KARŞILAŞTIRMASI:
    --------------------------
    Örnek: 7B parametre model
    
    │ Format │ Boyut  │ Hız     │ Kalite │ Bellek │
    ├────────┼────────┼─────────┼────────┼────────┤
    │ FP32   │ 28 GB  │ 1.0x    │ 100%   │ 28 GB  │
    │ FP16   │ 14 GB  │ 1.8x    │ 99.9%  │ 14 GB  │
    │ INT8   │ 7 GB   │ 2.5x    │ 98-99% │ 7 GB   │
    │ INT4   │ 3.5 GB │ 3.0x    │ 95-97% │ 3.5 GB │
    """
    
    print("=" * 70)
    print(f"🔢 Model Quantization: {quantization.upper()}")
    print("=" * 70)
    
    # ========================================================================
    # QUANTIZATION SEÇİMİ
    # ========================================================================
    
    if quantization == "8bit":
        # ====================================================================
        # INT8 QUANTIZATION (ÖNERİLEN)
        # ====================================================================
        print("📦 INT8 Quantization kullanılıyor...")
        print("-" * 70)
        print("""
        INT8 QUANTIZATION ÖZELLİKLERİ:
        -----------------------------
        ✓ %75 boyut azaltma (FP32'ye göre)
        ✓ %98-99 model kalitesi korunur
        ✓ 2-3x daha hızlı inference
        ✓ CPU ve GPU'da çalışır
        
        KULLANIM ALANLARI:
        - Production deployment
        - Orta ölçekli sistemler
        - Maliyet/performans dengesi
        """)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=True,          # 8-bit quantization aktif
            device_map="auto",           # Otomatik cihaz yerleşimi
            torch_dtype=torch.float16   # Compute dtype (hesaplama için)
        )
        print("✅ INT8 model başarıyla yüklendi")
        
    elif quantization == "4bit":
        # ====================================================================
        # INT4 QUANTIZATION (EKSTREM BOYUT AZALTMA)
        # ====================================================================
        print("📦 INT4 Quantization kullanılıyor...")
        print("-" * 70)
        print("""
        INT4 QUANTIZATION ÖZELLİKLERİ:
        -----------------------------
        ✓ %87.5 boyut azaltma (FP32'ye göre)
        ✓ %95-97 model kalitesi
        ✓ 3-4x daha hızlı inference
        ✓ Mobil cihazlarda çalışabilir
        
        KULLANIM ALANLARI:
        - Çok büyük modeller (70B+)
        - Sınırlı bellek ortamları
        - Edge deployment
        
        GEREKSINIMLER:
        - bitsandbytes kütüphanesi
        - CUDA desteği (önerilen)
        """)
        
        # bitsandbytes kütüphanesi gerekli
        try:
            from transformers import BitsAndBytesConfig
            
            # 4-bit quantization konfigürasyonu
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,                      # 4-bit yükleme
                bnb_4bit_compute_dtype=torch.float16,   # Hesaplama için FP16 kullan
                bnb_4bit_use_double_quant=True,         # Çift quantization (daha iyi kalite)
                bnb_4bit_quant_type="nf4"               # NormalFloat4 (özel 4-bit format)
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto"
            )
            print("✅ INT4 model başarıyla yüklendi")
            
        except ImportError:
            print("❌ HATA: bitsandbytes kütüphanesi bulunamadı!")
            print("   Kurulum: pip install bitsandbytes")
            print("   Fallback: FP16 ile yükleniyor...")
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16
            )
    
    elif quantization == "fp16":
        # ====================================================================
        # FP16 (YARI HASSASİYET)
        # ====================================================================
        print("📦 FP16 (Half Precision) kullanılıyor...")
        print("-" * 70)
        print("""
        FP16 ÖZELLİKLERİ:
        ----------------
        ✓ %50 boyut azaltma
        ✓ %99.9 model kalitesi
        ✓ 1.5-2x daha hızlı
        ✓ Modern GPU'larda optimize
        
        KULLANIM ALANLARI:
        - GPU inference
        - Yüksek kalite gereksinimi
        - Modern donanım
        """)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16    # FP16 precision
        )
        print("✅ FP16 model başarıyla yüklendi")
        
    else:
        # ====================================================================
        # STANDARD YÜKLEME (FP32)
        # ====================================================================
        print("📦 Standard (FP32) yükleme...")
        print("-" * 70)
        print("""
        STANDARD YÜKLEME:
        ----------------
        ✓ Maksimum kalite
        ✗ Büyük boyut
        ✗ Yavaş inference
        
        KULLANIM ALANLARI:
        - Araştırma
        - Model geliştirme
        - Kalite benchmark
        """)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype="auto"    # Otomatik dtype seçimi
        )
        print("✅ Model başarıyla yüklendi")
    
    # ========================================================================
    # TOKENIZER YÜKLEME
    # ========================================================================
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Pad token ayarla (gerekirse)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("⚙️ Pad token ayarlandı")
    
    # ========================================================================
    # MODEL İSTATİSTİKLERİ
    # ========================================================================
    print("\n" + "=" * 70)
    print("📊 MODEL BİLGİLERİ")
    print("=" * 70)
    
    # Parametre sayısı
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📈 Toplam Parametre: {total_params:,}")
    
    # Bellek kullanımı (tahmini)
    if quantization == "8bit":
        memory_gb = total_params * 1 / 1e9  # 1 byte per param
    elif quantization == "4bit":
        memory_gb = total_params * 0.5 / 1e9  # 0.5 byte per param
    elif quantization == "fp16":
        memory_gb = total_params * 2 / 1e9  # 2 bytes per param
    else:
        memory_gb = total_params * 4 / 1e9  # 4 bytes per param
    
    print(f"💾 Tahmini Bellek: {memory_gb:.2f} GB")
    
    # GPU bilgisi
    if torch.cuda.is_available():
        print(f"🖥️ Cihaz: CUDA (GPU)")
        print(f"📍 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"🖥️ Cihaz: CPU")
    
    print("=" * 70)
    print("✅ Quantized model hazır!\n")
    
    return model, tokenizer


# ============================================================================
# KİŞİSELLEŞTİRİLMİŞ CHATBOT OLUŞTURMA
# ============================================================================

def create_personalized_chatbot():
    """
    Kişiselleştirilmiş Chatbot Demonstrasyonu
    ==========================================
    
    Bu fonksiyon, kullanıcı profiline göre özelleştirilmiş yanıtlar üreten
    bir chatbot sisteminin nasıl tasarlanacağını gösterir.
    
    KİŞİSELLEŞTİRME NEDİR?
    ----------------------
    Aynı soru için farklı kullanıcılara farklı yanıtlar vermek:
    
    Örnek Soru: "Python nasıl öğrenilir?"
    
    Yeni Başlayan → "Python'a temellerden başlayalım..."
    İleri Seviye → "Advanced Python konularına odaklanalım..."
    Çocuk        → "Python oyun gibi eğlenceli bir dil..."
    
    KİŞİSELLEŞTİRME FAKTÖRLERİ:
    ---------------------------
    • Kullanıcı Profili: İsim, yaş, deneyim seviyesi
    • İlgi Alanları: Teknoloji, spor, sanat vb.
    • Konuşma Stili: Formal, samimi, teknik
    • Geçmiş Etkileşimler: Önceki konuşmalar, tercihler
    • Bağlam: Zaman, yer, cihaz
    
    UYGULAMA ALANLARI:
    -----------------
    ✓ E-ticaret: Kişisel ürün önerileri
    ✓ Eğitim: Seviyeye göre içerik
    ✓ Sağlık: Kişisel sağlık tavsiyeleri
    ✓ Müşteri Hizmetleri: Özelleştirilmiş destek
    """
    
    print("\n" + "=" * 80)
    print("🤖 KİŞİSELLEŞTİRİLMİŞ CHATBOT SİSTEMİ")
    print("=" * 80)
    
    # ========================================================================
    # KULLANICI PROFİL TANIMI
    # ========================================================================
    print("\n1️⃣ KULLANICI PROFİLİ OLUŞTURMA")
    print("-" * 80)
    
    # Örnek kullanıcı profili
    user_profile = {
        "name": "Ali",                              # Kullanıcı adı
        "age": 25,                                  # Yaş
        "interests": ["teknoloji", "yapay zeka", "python"],  # İlgi alanları
        "expertise_level": "intermediate",          # Deneyim seviyesi
        "learning_style": "hands-on",               # Öğrenme stili
        "preferred_language": "Turkish",            # Tercih edilen dil
        "tone": "samimi ve yardımsever",           # Konuşma tonu
        "background": "bilgisayar mühendisliği",   # Arka plan
    }
    
    print("Örnek Kullanıcı Profili:")
    print(json.dumps(user_profile, indent=2, ensure_ascii=False))
    
    # ========================================================================
    # KİŞİSELLEŞTİRİLMİŞ PROMPT OLUŞTURMA
    # ========================================================================
    print("\n2️⃣ KİŞİSELLEŞTİRİLMİŞ PROMPT ENGINEERİNG")
    print("-" * 80)
    
    def create_personalized_prompt(user_input, profile):
        """
        Kullanıcı Profiline Göre Prompt Oluşturma
        =========================================
        
        Bu fonksiyon, kullanıcı bilgilerini kullanarak
        modele verilecek prompt'u dinamik olarak oluşturur.
        
        PROMPT YAPISI:
        -------------
        1. Sistem Talimatları (System Instructions)
        2. Kullanıcı Profil Bilgileri
        3. Bağlam (Context)
        4. Kullanıcı Sorusu
        5. Format Talimatları
        
        Args:
            user_input (str): Kullanıcının sorusu/mesajı
            profile (dict): Kullanıcı profil bilgileri
        
        Returns:
            str: Kişiselleştirilmiş prompt
        """
        
        # PROMPT TEMPLATE (Şablon)
        prompt = f"""Sen deneyimli bir AI asistanısın. Aşağıdaki kullanıcı profiline göre yanıt ver.

╔══════════════════════════════════════════════════════════════════╗
║                     KULLANICI PROFİLİ                           ║
╚══════════════════════════════════════════════════════════════════╝

👤 İsim: {profile['name']}
🎂 Yaş: {profile['age']}
📚 Deneyim Seviyesi: {profile['expertise_level']}
💡 İlgi Alanları: {', '.join(profile['interests'])}
🎯 Öğrenme Stili: {profile['learning_style']}
🗣️ Konuşma Tonu: {profile['tone']}
🎓 Arka Plan: {profile['background']}

╔══════════════════════════════════════════════════════════════════╗
║                      YANIT TALİMATLARI                          ║
╚══════════════════════════════════════════════════════════════════╝

1. Kullanıcının deneyim seviyesine uygun açıklama yap
2. İlgi alanlarına referans ver
3. Belirtilen konuşma tonunu kullan
4. Pratik örnekler ver (hands-on öğrenme stili için)
5. Türkçe olarak yanıtla

╔══════════════════════════════════════════════════════════════════╗
║                      KULLANICI SORUSU                           ║
╚══════════════════════════════════════════════════════════════════╝

{profile['name']}: {user_input}

╔══════════════════════════════════════════════════════════════════╗
║                        AI YANITI                                ║
╚══════════════════════════════════════════════════════════════════╝

Asistan:"""
        
        return prompt
    
    # ========================================================================
    # ÖRNEK KONUŞMALAR
    # ========================================================================
    print("\n3️⃣ ÖRNEK KİŞİSELLEŞTİRİLMİŞ KONUŞMALAR")
    print("-" * 80)
    
    # Farklı kullanıcı soruları
    user_inputs = [
        "Merhaba! Python öğrenmek istiyorum, nereden başlamalıyım?",
        "Makine öğrenmesi için hangi kütüphaneleri önerirsin?",
        "Yapay zeka projelerinde kariyer yapmak istiyorum, tavsiyelen neler?"
    ]
    
    for idx, user_input in enumerate(user_inputs, 1):
        print(f"\n📝 ÖRNEK {idx}")
        print("─" * 80)
        
        # Kişiselleştirilmiş prompt oluştur
        personalized_prompt = create_personalized_prompt(user_input, user_profile)
        
        print("OLUŞTURULAN PROMPT:")
        print("┌" + "─" * 78 + "┐")
        for line in personalized_prompt.split('\n'):
            print(f"│ {line:<76} │")
        print("└" + "─" * 78 + "┘")
        
        # NOT: Gerçek inference burada yapılır
        print("\n💬 BEKLENENbir YANIT ÖRNEĞİ:")
        print("┌" + "─" * 78 + "┐")
        
        if idx == 1:
            sample_response = f"""Merhaba {user_profile['name']}! Python öğrenmeye başlamak harika bir karar! 🎉
            
Senin için intermediate seviyeye yönelik bir yol haritası hazırladım:

1️⃣ Temel Konuları Pekiştir:
   - Data structures (list, dict, set, tuple)
   - OOP concepts (class, inheritance, polymorphism)
   - File handling ve error handling

2️⃣ Yapay Zeka Odaklı İlerle (ilgi alanına uygun):
   - NumPy: Sayısal hesaplamalar
   - Pandas: Veri analizi
   - Matplotlib/Seaborn: Görselleştirme

3️⃣ Pratik Projeler Yap (hands-on stili):
   - Kendi veri setinle analiz yap
   - Simple ML modeli kur
   - GitHub'da paylaş

Başlangıç için önereceğim kaynak: "Python for Data Science" kitabı."""
            
        elif idx == 2:
            sample_response = f"""Harika soru {user_profile['name']}! Makine öğrenmesi için temel kütüphaneler:

🔧 Temel Stack:
   • NumPy: Matematiksel işlemler
   • Pandas: Veri manipülasyonu
   • Scikit-learn: ML algoritmaları

🧠 Deep Learning:
   • TensorFlow: Google'ın framework'ü
   • PyTorch: Araştırma ve production
   • Keras: Kolay API

📊 Görselleştirme:
   • Matplotlib: Temel grafikler
   • Seaborn: İstatistiksel vizler
   • Plotly: İnteraktif dashboard

Senin intermediate seviyende olduğun için, önce Scikit-learn ile başla,
sonra PyTorch'a geç. Yapay zeka ilgine uygun! 🚀"""
            
        else:
            sample_response = f"""Yapay zeka kariyeri için süper bir hedef {user_profile['name']}! 💼

Tavsiyelerim:

1️⃣ Teknik Beceriler:
   ✓ Python + ML kütüphaneleri (zaten ilgileniyorsun!)
   ✓ Deep Learning framework'leri
   ✓ MLOps tools (Docker, Kubernetes)

2️⃣ Pratik Deneyim:
   ✓ Kaggle competitions
   ✓ Open-source contribute
   ✓ Kendi projelerini GitHub'da yayınla

3️⃣ Network:
   ✓ LinkedIn'de AI community'lerine katıl
   ✓ Meetup'lara git
   ✓ Blog yaz (deneyimlerini paylaş)

Bilgisayar mühendisliği arka planın çok avantaj! Teorik bilgin var,
şimdi pratik yaparak portföy oluştur. 🎯"""
        
        for line in sample_response.split('\n'):
            print(f"│ {line:<76} │")
        print("└" + "─" * 78 + "┘")
    
    # ========================================================================
    # GELİŞMİŞ KİŞİSELLEŞTİRME TEKNİKLERİ
    # ========================================================================
    print("\n" + "=" * 80)
    print("4️⃣ GELİŞMİŞ KİŞİSELLEŞTİRME TEKNİKLERİ")
    print("=" * 80)
    
    advanced_techniques = {
        "🧠 Conversation Memory": """
            Önceki konuşmaları hatırlama ve referans verme
            
            Örnek:
            User: "Dün bahsettiğin NumPy kütüphanesini kurmaya çalıştım"
            Bot: "Harika! NumPy kurulumunda sorun yaşadın mı? Geçen 
                  konuşmamızda pandas ile beraber kullanacağını söylemiştin."
        """,
        
        "🎯 Adaptive Difficulty": """
            Kullanıcının ilerlemesine göre zorluk seviyesi ayarlama
            
            İlk gün: "Liste nedir? Bir dizi veridir..."
            2. hafta: "List comprehension ile daha verimli..."
            1. ay: "Generator expressions ve memory optimization..."
        """,
        
        "📊 Preference Learning": """
            Kullanıcı tercihlerini öğrenme
            
            Gözlem: Kullanıcı hep kod örnekleri istiyor
            Adaptasyon: Açıklamalara otomatik kod snippet ekle
            
            Gözlem: Video kaynak linklerini açmıyor
            Adaptasyon: Yazılı kaynakları önceliklendir
        """,
        
        "🌍 Contextual Awareness": """
            Bağlamsal farkındalık
            
            Sabah: "Günaydın! Bugün yeni bir konu öğrenmeye hazır mısın?"
            Akşam: "Bugün öğrendiklerini pekiştirmek için pratik yapalım"
            
            Hafta içi: Kısa, hızlı yanıtlar
            Hafta sonu: Detaylı, kapsamlı açıklamalar
        """,
        
        "💬 Sentiment Analysis": """
            Kullanıcı duygusunu anlama ve buna göre yanıt verme
            
            Pozitif: "Harika! Enerjini seviyorum! Devam edelim 🚀"
            Negatif: "Anlıyorum, biraz zorlandın. Adım adım gidelim 💪"
            Confused: "Karışık geldi galiba, farklı anlatayım 🤔"
        """
    }
    
    for technique, description in advanced_techniques.items():
        print(f"\n{technique}")
        print("─" * 80)
        print(description.strip())
    
    # ========================================================================
    # IMPLEMENTATION NOTES
    # ========================================================================
    print("\n" + "=" * 80)
    print("💡 UYGULAMA NOTLARI")
    print("=" * 80)
    print("""
    KİŞİSELLEŞTİRME İÇİN GEREKLİLER:
    --------------------------------
    
    1. 🗄️ Kullanıcı Veritabanı:
       - Profil bilgileri
       - Konuşma geçmişi
       - Tercihler ve ayarlar
       
    2. 🔒 Gizlilik ve Güvenlik:
       - GDPR/KVKK uyumluluğu
       - Veri şifreleme
       - Kullanıcı onayı
       
    3. 📈 Analytics ve Tracking:
       - Kullanıcı davranış analizi
       - A/B testing
       - Performans metrikleri
       
    4. 🔄 Sürekli İyileştirme:
       - Kullanıcı geri bildirimleri
       - Model fine-tuning
       - Prompt optimization
    """)
    
    print("=" * 80)
    print("✅ Kişiselleştirilmiş chatbot demonstrasyonu tamamlandı!\n")


# ============================================================================
# GENERATION CONFIGURATION (Üretim Yapılandırması)
# ============================================================================

def demonstrate_generation_config():
    """
    Generation Configuration Parametreleri
    =======================================
    
    Bu fonksiyon, farklı kullanım senaryoları için optimal text generation
    parametrelerini gösterir.
    
    GENERATION PARAMETRELERI NEDİR?
    -------------------------------
    Model metni üretirken birçok parametre ile kontrol edilir:
    
    • temperature: Randomness (rastgelelik) seviyesi
    • top_p: Nucleus sampling eşiği
    • top_k: En yüksek k olası token
    • repetition_penalty: Tekrar cezası
    • max_tokens: Maksimum token sayısı
    • do_sample: Sampling aktif/pasif
    
    PARAMETRELER NASIL ETKİLER?
    ---------------------------
    
    🌡️ TEMPERATURE (0.0 - 2.0):
    ---------------------------
    Düşük (0.0-0.3):
        ✓ Tutarlı, öngörülebilir
        ✓ Faktüel doğruluk
        ✗ Tekrarlayıcı olabilir
        Kullanım: Teknik dökümantasyon, QA
    
    Orta (0.5-0.8):
        ✓ Dengeli
        ✓ Yaratıcı ama tutarlı
        Kullanım: Genel amaçlı chatbot
    
    Yüksek (0.9-2.0):
        ✓ Çok yaratıcı
        ✓ Çeşitli çıktılar
        ✗ Tutarsız olabilir
        Kullanım: Hikaye, şiir, brainstorming
    
    🎯 TOP_P (0.0 - 1.0) - Nucleus Sampling:
    ---------------------------------------
    Top-p, kümülatif olasılık eşiğidir.
    
    Nasıl Çalışır?
    - Tokenlar olasılığa göre sıralanır
    - Kümülatif toplam p'yi geçene kadar eklenir
    - Sadece bu set'ten örnekleme yapılır
    
    Örnek (top_p=0.9):
        Token A: 40% → Kümülatif: 40%
        Token B: 30% → Kümülatif: 70%
        Token C: 20% → Kümülatif: 90% ← buraya kadar
        Token D: 10% → (dahil edilmez)
    
    Düşük (0.1-0.5): Sadece en olası tokenlar
    Yüksek (0.9-1.0): Daha fazla çeşitlilik
    
    🔢 TOP_K (1 - 100):
    -------------------
    En yüksek k olası token arasından seç.
    
    top_k=1: Her zaman en olası token (greedy)
    top_k=10: En olası 10 token arasından örnekle
    top_k=50: Daha fazla çeşitlilik
    
    🔁 REPETITION_PENALTY (1.0 - 2.0):
    ---------------------------------
    Tekrarlayan tokenları cezalandırma.
    
    1.0: Ceza yok (default)
    1.1-1.2: Hafif ceza (önerilen)
    1.5+: Ağır ceza (tekrar neredeyse imkansız)
    """
    
    print("\n" + "=" * 80)
    print("⚙️ GENERATION CONFIGURATION PARAMETRELERİ")
    print("=" * 80)
    
    # ========================================================================
    # KULLANIM SENARYOLARINA GÖRE KONFİGÜRASYONLAR
    # ========================================================================
    
    configs = {
        "📝 Yaratıcı Yazarlık (Creative Writing)": {
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "max_new_tokens": 200,
            "do_sample": True,
            "description": """
            Hikaye, şiir, blog yazıları için ideal
            
            ÖZELLİKLER:
            • Yüksek yaratıcılık (temp=0.8)
            • Geniş token çeşitliliği (top_p=0.9, top_k=50)
            • Tekrarları minimize et (penalty=1.1)
            • Sampling aktif (çeşitli çıktılar)
            
            ÖRNEK ÇIKTI:
            "Bir zamanlar, uzak bir galakside, yıldızlar arasında 
            dans eden küçük bir robot yaşardı. Her gün yeni gezegen
            keşfetmek için maceraya atılırdı..."
            """
        },
        
        "❓ Faktüel Soru-Cevap (Factual QA)": {
            "temperature": 0.1,
            "top_p": 0.5,
            "top_k": 10,
            "repetition_penalty": 1.0,
            "max_new_tokens": 100,
            "do_sample": False,
            "description": """
            Teknik sorular, bilgi talebi için ideal
            
            ÖZELLİKLER:
            • Minimal randomness (temp=0.1)
            • Daraltılmış token seçimi (top_p=0.5, top_k=10)
            • Tekrar cezası yok (teknik terimler tekrar edilebilir)
            • Sampling kapalı (deterministik)
            
            ÖRNEK ÇIKTI:
            "Python, 1991 yılında Guido van Rossum tarafından 
            geliştirilmiş, yüksek seviyeli, yorumlamalı bir 
            programlama dilidir. Nesne yönelimli programlamayı 
            destekler."
            """
        },
        
        "💻 Kod Üretimi (Code Generation)": {
            "temperature": 0.2,
            "top_p": 0.6,
            "top_k": 20,
            "repetition_penalty": 1.05,
            "max_new_tokens": 150,
            "do_sample": True,
            "description": """
            Python, JavaScript, SQL kod üretimi için
            
            ÖZELLİKLER:
            • Düşük randomness (temp=0.2) - Syntax hatası önleme
            • Orta token seçimi (best practices için)
            • Hafif tekrar cezası (loop'larda gerekli olabilir)
            • Sampling açık ama kısıtlı
            
            ÖRNEK ÇIKTI:
            ```python
            def fibonacci(n):
                if n <= 1:
                    return n
                return fibonacci(n-1) + fibonacci(n-2)
            ```
            """
        },
        
        "💬 Genel Chatbot (General Conversation)": {
            "temperature": 0.7,
            "top_p": 0.85,
            "top_k": 40,
            "repetition_penalty": 1.1,
            "max_new_tokens": 120,
            "do_sample": True,
            "description": """
            Günlük konuşmalar, genel asistan görevleri için
            
            ÖZELLİKLER:
            • Dengeli randomness (temp=0.7)
            • Dengeli token seçimi
            • Tekrar önleme (monoton olmamak için)
            • Doğal konuşma akışı
            
            ÖRNEK ÇIKTI:
            "Tabii ki yardımcı olabilirim! Python öğrenmek harika 
            bir başlangıç. Size adım adım bir yol haritası 
            çıkarabilirim. Önce temel syntax'ı öğrenmenizi öneririm."
            """
        },
        
        "📰 Haber Özeti (News Summary)": {
            "temperature": 0.3,
            "top_p": 0.7,
            "top_k": 25,
            "repetition_penalty": 1.2,
            "max_new_tokens": 80,
            "do_sample": True,
            "description": """
            Haber metinlerini özetleme için
            
            ÖZELLİKLER:
            • Düşük randomness (tutarlı özet)
            • Orta token seçimi (çeşitli ifade)
            • Yüksek tekrar cezası (aynı kelimeyi kullanmamak için)
            • Kısa output (özet olmalı)
            
            ÖRNEK ÇIKTI:
            "Bugün yapılan toplantıda ekonomik tedbirler görüşüldü.
            Enflasyonla mücadele için yeni stratejiler belirlendi.
            Merkez Bankası faiz kararını açıkladı."
            """
        },
        
        "🎭 Karakter Dialog (Character Dialogue)": {
            "temperature": 0.9,
            "top_p": 0.95,
            "top_k": 60,
            "repetition_penalty": 1.15,
            "max_new_tokens": 180,
            "do_sample": True,
            "description": """
            RPG, oyun karakterleri, drama diyalogları için
            
            ÖZELLİKLER:
            • Yüksek yaratıcılık (benzersiz konuşmalar)
            • Geniş token çeşitliliği (farklı ifadeler)
            • Tekrar önleme (doğal konuşma)
            • Uzun yanıtlara izin ver
            
            ÖRNEK ÇIKTI:
            "Ah, maceracı! Sonunda geldin. Ejderhalar dağlarda 
            uyanmış, köylüler korkudan titriyor. Efsanevi kılıcı 
            bulmak için yola çıkmalıyız, ama önce hazırlık yapmalıyız!"
            """
        }
    }
    
    # ========================================================================
    # KONFİGÜRASYONLARI GÖSTER
    # ========================================================================
    
    for use_case, config in configs.items():
        print(f"\n{use_case}")
        print("=" * 80)
        
        # Parametreleri göster
        print("\n📊 PARAMETRELER:")
        print("-" * 80)
        
        param_descriptions = {
            "temperature": "🌡️ Randomness Seviyesi",
            "top_p": "🎯 Nucleus Sampling",
            "top_k": "🔢 Token Çeşitliliği",
            "repetition_penalty": "🔁 Tekrar Cezası",
            "max_new_tokens": "📏 Maksimum Token",
            "do_sample": "🎲 Sampling Durumu"
        }
        
        for param in ["temperature", "top_p", "top_k", "repetition_penalty", 
                      "max_new_tokens", "do_sample"]:
            if param in config:
                value = config[param]
                desc = param_descriptions.get(param, param)
                print(f"  {desc:<25} : {value}")
        
        # Açıklamayı göster
        print("\n📖 AÇIKLAMA:")
        print("-" * 80)
        print(config["description"].strip())
    
    # ========================================================================
    # PARAMETRE ETKİLEŞİMLERİ
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("🔄 PARAMETRE ETKİLEŞİMLERİ")
    print("=" * 80)
    print("""
    PARAMETRE KOMBİNASYONLARI:
    -------------------------
    
    1️⃣ Deterministik Mod (Her zaman aynı çıktı):
       temperature = 0.0
       do_sample = False
       → Soru-cevap, dokümantasyon için
    
    2️⃣ Kontrollü Yaratıcılık:
       temperature = 0.7
       top_p = 0.9
       top_k = 50
       repetition_penalty = 1.1
       → Chatbot, blog yazıları
    
    3️⃣ Maksimum Çeşitlilik:
       temperature = 1.5
       top_p = 0.95
       top_k = 100
       repetition_penalty = 1.2
       → Brainstorming, yaratıcı yazarlık
    
    4️⃣ Teknik İçerik:
       temperature = 0.2
       top_p = 0.6
       repetition_penalty = 1.0
       → Kod, teknik açıklamalar
    
    ⚠️ DİKKAT:
    ----------
    • temperature=0 iken top_p ve top_k etkisizdir
    • Çok yüksek repetition_penalty tutarsız çıktılar verebilir
    • Uzun metinler için max_tokens'ı artırın
    • Her model farklı tepki verebilir, test edin!
    """)
    
    # ========================================================================
    # PRATIK KOD ÖRNEĞİ
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("💻 PRATIK KOD ÖRNEĞİ")
    print("=" * 80)
    
    code_example = '''
# Generation config oluşturma
from transformers import GenerationConfig

# Yaratıcı yazarlık için config
creative_config = GenerationConfig(
    max_new_tokens=200,
    temperature=0.8,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.1,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

# Faktüel QA için config
factual_config = GenerationConfig(
    max_new_tokens=100,
    temperature=0.1,
    top_p=0.5,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id
)

# Kullanım
outputs = model.generate(
    input_ids,
    generation_config=creative_config  # veya factual_config
)

# Alternatif: Doğrudan parametre
outputs = model.generate(
    input_ids,
    max_new_tokens=150,
    temperature=0.7,
    top_p=0.9,
    top_k=40,
    repetition_penalty=1.1,
    do_sample=True
)
'''
    
    print(code_example)
    
    print("=" * 80)
    print("✅ Generation configuration demonstrasyonu tamamlandı!\n")


# ============================================================================
# INFERENCE HIZ BENCHMARK
# ============================================================================

def benchmark_inference_speed():
    """
    Inference Hız Benchmark ve Performans Analizi
    ==============================================
    
    Bu fonksiyon, farklı optimizasyon tekniklerinin inference hızına
    etkisini gösterir ve karşılaştırır.
    
    NEDEN BENCHMARK GEREKLİ?
    -----------------------
    Production ortamında:
    • Kullanıcı deneyimi: <200ms ideal
    • Throughput: Saniyede işlenebilecek istek sayısı
    • Maliyet: Daha hızlı = daha az sunucu
    • SLA (Service Level Agreement) gereklilikleri
    
    BENCHMARK METRİKLERİ:
    --------------------
    • Latency: Tek bir inference süresi (ms)
    • Throughput: Saniyede işlenebilen istek sayısı
    • GPU/CPU Kullanımı: Kaynak verimliliği (%)
    • Memory Footprint: Bellek kullanımı (GB)
    • Cost per 1M tokens: Maliyet analizi
    
    HIZ OPTİMİZASYONUNU ETKİLEYEN FAKTÖRLER:
    ---------------------------------------
    1. Model Boyutu (7B, 13B, 70B parametreler)
    2. Quantization Seviyesi (FP32, FP16, INT8, INT4)
    3. Hardware (CPU, GPU, TPU)
    4. Batch Size (1, 8, 32, 64)
    5. Sequence Length (input + output token sayısı)
    """
    
    print("\n" + "=" * 80)
    print("⚡ INFERENCE HIZ BENCHMARK")
    print("=" * 80)
    
    # ========================================================================
    # SENARYO 1: QUANTIZATION ETKİSİ (7B Model, GPU, Batch=1, 100 tokens)
    # ========================================================================
    
    print("\n📊 SENARYO 1: QUANTIZATION ETKİSİNİN HIZ ÜZERİNDEKİ ETKİSİ")
    print("-" * 80)
    print("Test Koşulları: 7B Model, NVIDIA A100 GPU, Batch Size=1, 100 token")
    print("-" * 80)
    
    scenarios_quantization = {
        "FP32 (Tam Hassasiyet)": {
            "latency_ms": 1500,
            "throughput": 0.67,      # requests/second
            "memory_gb": 28,
            "gpu_util": 85,
            "quality": 100
        },
        "FP16 (Yarı Hassasiyet)": {
            "latency_ms": 850,
            "throughput": 1.18,
            "memory_gb": 14,
            "gpu_util": 75,
            "quality": 99.9
        },
        "INT8 (8-bit Quantization)": {
            "latency_ms": 580,
            "throughput": 1.72,
            "memory_gb": 7,
            "gpu_util": 65,
            "quality": 98.5
        },
        "INT4 (4-bit Quantization)": {
            "latency_ms": 420,
            "throughput": 2.38,
            "memory_gb": 3.5,
            "gpu_util": 55,
            "quality": 96.0
        }
    }
    
    print(f"\n{'Yöntem':<30} {'Latency':>12} {'Throughput':>12} {'Bellek':>10} {'GPU':>8} {'Kalite':>8}")
    print(f"{'':<30} {'(ms)':>12} {'(req/s)':>12} {'(GB)':>10} {'(%)':>8} {'(%)':>8}")
    print("─" * 90)
    
    baseline_latency = scenarios_quantization["FP32 (Tam Hassasiyet)"]["latency_ms"]
    
    for scenario, metrics in scenarios_quantization.items():
        latency = metrics["latency_ms"]
        throughput = metrics["throughput"]
        memory = metrics["memory_gb"]
        gpu_util = metrics["gpu_util"]
        quality = metrics["quality"]
        
        speedup = baseline_latency / latency
        
        print(f"{scenario:<30} {latency:>10.0f}ms {throughput:>10.2f}  {memory:>8.1f}GB {gpu_util:>6}% {quality:>6.1f}%")
        print(f"{'':>30} {'↑ ' + str(round(speedup, 1)) + 'x hızlı':>12} {'':>12} {'↓ ' + str(round((1-memory/28)*100, 1)) + '% azalma':>10}")
        print()
    
    # ========================================================================
    # SENARYO 2: CPU vs GPU KARŞILAŞTIRMASI
    # ========================================================================
    
    print("\n📊 SENARYO 2: CPU vs GPU PERFORMANS KARŞILAŞTIRMASI")
    print("-" * 80)
    print("Test Koşulları: 7B Model, INT8 Quantization, Batch Size=1, 100 token")
    print("-" * 80)
    
    scenarios_hardware = {
        "CPU - Intel Xeon": {
            "latency_ms": 3200,
            "cost_per_hour": 0.50,
            "throughput": 0.31
        },
        "CPU - Optimized (ONNX)": {
            "latency_ms": 1800,
            "cost_per_hour": 0.50,
            "throughput": 0.56
        },
        "GPU - NVIDIA T4": {
            "latency_ms": 850,
            "cost_per_hour": 0.95,
            "throughput": 1.18
        },
        "GPU - NVIDIA A100": {
            "latency_ms": 420,
            "cost_per_hour": 3.50,
            "throughput": 2.38
        },
        "GPU - NVIDIA H100": {
            "latency_ms": 210,
            "cost_per_hour": 8.00,
            "throughput": 4.76
        }
    }
    
    print(f"\n{'Hardware':<25} {'Latency':>12} {'Throughput':>14} {'Maliyet':>12} {'Verimlilik':>15}")
    print(f"{'':<25} {'(ms)':>12} {'(req/s)':>14} {'($/saat)':>12} {'(req/$/saat)':>15}")
    print("─" * 80)
    
    for hw, metrics in scenarios_hardware.items():
        latency = metrics["latency_ms"]
        cost = metrics["cost_per_hour"]
        throughput = metrics["throughput"]
        efficiency = throughput * 3600 / cost  # requests per dollar per hour
        
        print(f"{hw:<25} {latency:>10.0f}ms {throughput:>12.2f}  ${cost:>10.2f}  {efficiency:>12.0f}")
    
    # ========================================================================
    # SENARYO 3: BATCH SIZE ETKİSİ
    # ========================================================================
    
    print("\n\n📊 SENARYO 3: BATCH SIZE'IN THROUGHPUT ÜZERİNDEKİ ETKİSİ")
    print("-" * 80)
    print("Test Koşulları: 7B Model, NVIDIA A100 GPU, INT8, 100 token")
    print("-" * 80)
    
    scenarios_batch = {
        "Batch Size = 1": {
            "latency_ms": 420,
            "throughput": 2.38,
            "gpu_util": 45,
            "memory_gb": 7.2
        },
        "Batch Size = 4": {
            "latency_ms": 580,
            "throughput": 6.90,      # 4 items in 580ms
            "gpu_util": 72,
            "memory_gb": 9.5
        },
        "Batch Size = 8": {
            "latency_ms": 920,
            "throughput": 8.70,      # 8 items in 920ms
            "gpu_util": 88,
            "memory_gb": 12.8
        },
        "Batch Size = 16": {
            "latency_ms": 1650,
            "throughput": 9.70,      # 16 items in 1650ms
            "gpu_util": 95,
            "memory_gb": 18.5
        },
        "Batch Size = 32": {
            "latency_ms": 3100,
            "throughput": 10.32,     # 32 items in 3100ms
            "gpu_util": 98,
            "memory_gb": 28.0
        }
    }
    
    print(f"\n{'Batch Size':<20} {'Latency':>12} {'Throughput':>14} {'GPU Kullanımı':>16} {'Bellek':>12}")
    print(f"{'':<20} {'(ms)':>12} {'(req/s)':>14} {'(%)':>16} {'(GB)':>12}")
    print("─" * 80)
    
    for batch, metrics in scenarios_batch.items():
        latency = metrics["latency_ms"]
        throughput = metrics["throughput"]
        gpu_util = metrics["gpu_util"]
        memory = metrics["memory_gb"]
        
        print(f"{batch:<20} {latency:>10.0f}ms {throughput:>12.2f}  {gpu_util:>14}%  {memory:>10.1f}GB")
    
    # ========================================================================
    # SENARYO 4: MODEL BOYUTU ETKİSİ
    # ========================================================================
    
    print("\n\n📊 SENARYO 4: MODEL BOYUTUNUN PERFORMANSA ETKİSİ")
    print("-" * 80)
    print("Test Koşulları: NVIDIA A100 GPU, INT8, Batch Size=1, 100 token")
    print("-" * 80)
    
    scenarios_model_size = {
        "1.3B Parametreli Model": {
            "latency_ms": 85,
            "memory_gb": 1.3,
            "quality_score": 75
        },
        "7B Parametreli Model": {
            "latency_ms": 420,
            "memory_gb": 7.0,
            "quality_score": 85
        },
        "13B Parametreli Model": {
            "latency_ms": 780,
            "memory_gb": 13.0,
            "quality_score": 90
        },
        "30B Parametreli Model": {
            "latency_ms": 1850,
            "memory_gb": 30.0,
            "quality_score": 93
        },
        "70B Parametreli Model": {
            "latency_ms": 4200,
            "memory_gb": 70.0,
            "quality_score": 96
        }
    }
    
    print(f"\n{'Model':<28} {'Latency':>12} {'Bellek':>12} {'Kalite':>12}")
    print(f"{'':<28} {'(ms)':>12} {'(GB)':>12} {'(Skor)':>12}")
    print("─" * 70)
    
    for model, metrics in scenarios_model_size.items():
        latency = metrics["latency_ms"]
        memory = metrics["memory_gb"]
        quality = metrics["quality_score"]
        
        print(f"{model:<28} {latency:>10.0f}ms {memory:>10.1f}GB {quality:>10}/100")
    
    # ========================================================================
    # ÖNERİLER ve EN İYİ UYGULAMALAR
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("💡 PERFORMANS OPTİMİZASYON ÖNERİLERİ")
    print("=" * 80)
    print("""
    HIZLI BAŞLANGIÇ (Quick Wins):
    -----------------------------
    1. INT8 Quantization kullan
       → 2-3x hız artışı, minimal kalite kaybı
    
    2. Batch Processing uygula
       → Throughput'u 3-5x artırabilir
    
    3. KV-Cache'i aktif tut
       → Model.generate() default olarak açık
    
    4. GPU kullan (mümkünse)
       → CPU'ya göre 5-10x hızlı
    
    SENARYOYA GÖRE SEÇİM:
    --------------------
    
    💰 Düşük Maliyet:
       • CPU + INT8/INT4
       • Büyük batch size
       • Küçük model (7B)
    
    ⚡ Düşük Latency:
       • GPU (A100/H100)
       • INT8/FP16
       • Batch size = 1-4
    
    📈 Yüksek Throughput:
       • GPU + Batch size 16-32
       • INT8 quantization
       • Dynamic batching
    
    🎯 Yüksek Kalite:
       • Büyük model (30B-70B)
       • FP16/FP32
       • GPU gerekli
    
    BENCHMARK NASIL YAPILIR:
    -----------------------
    1. Üretim senaryonuza benzer test verisi hazırlayın
    2. Farklı konfigürasyonları test edin
    3. Latency, throughput, maliyet ölçün
    4. Trade-off analizi yapın
    5. Production'a en uygun olanı seçin
    
    ÖRNEK KOD:
    ---------
    import time
    
    # Benchmark fonksiyonu
    def benchmark_model(model, inputs, num_runs=100):
        times = []
        for _ in range(num_runs):
            start = time.time()
            model.generate(**inputs)
            times.append(time.time() - start)
        
        avg_latency = sum(times) / len(times)
        throughput = 1 / avg_latency
        
        print(f"Avg Latency: {avg_latency*1000:.2f}ms")
        print(f"Throughput: {throughput:.2f} req/s")
    """)
    
    print("=" * 80)
    print("✅ Inference hız benchmark demonstrasyonu tamamlandı!\n")


# ============================================================================
# INFERENCE PIPELINE OLUŞTURMA
# ============================================================================

def create_inference_pipeline():
    """
    Production-Ready Inference Pipeline
    ====================================
    
    Bu fonksiyon, model inference'ı için eksiksiz bir pipeline (boru hattı)
    oluşturmayı ve farklı pipeline yaklaşımlarını gösterir.
    
    PIPELINE NEDİR?
    ---------------
    Inference pipeline, input'dan output'a kadar tüm adımları
    otomatize eden bir sistemdir:
    
    Input → Preprocessing → Tokenization → Model Inference → 
    Postprocessing → Output
    
    PIPELINE AVANTAJLARI:
    --------------------
    ✓ Tutarlılık: Her zaman aynı adımlar
    ✓ Hata Yönetimi: Merkezi error handling
    ✓ Logging: Her adım loglanabilir
    ✓ Monitoring: Performans takibi
    ✓ Testing: Birim ve entegrasyon testleri
    ✓ Ölçeklenebilirlik: Kolayca scale edilebilir
    
    PIPELINE ADIMLARI:
    -----------------
    1. INPUT PREPROCESSING
       • Input validasyonu
       • Text cleaning
       • Format dönüşümü
    
    2. TOKENIZATION
       • Text → Token ID dönüşümü
       • Padding/truncation
       • Attention mask oluşturma
    
    3. MODEL INFERENCE
       • GPU'ya taşıma
       • Model forward pass
       • Generation veya classification
    
    4. OUTPUT POSTPROCESSING
       • Decode işlemi
       • Format dönüşümü
       • Filtering
    
    5. RESPONSE FORMATTING
       • JSON yapısı
       • Metadata ekleme
       • Error handling
    """
    
    print("\n" + "=" * 80)
    print("🔄 INFERENCE PIPELINE OLUŞTURMA")
    print("=" * 80)
    
    # ========================================================================
    # PIPELINE WORKFLOW
    # ========================================================================
    
    print("\n1️⃣ INFERENCE PIPELINE ADIMLARI")
    print("-" * 80)
    
    steps = [
        {
            "step": "1. Input Preprocessing",
            "description": "Gelen veriyi temizleme ve hazırlama",
            "operations": [
                "• Text cleaning (trim, normalize)",
                "• Input validation (length, format)",
                "• Security checks (XSS, injection prevention)",
                "• Language detection (opsiyonel)"
            ]
        },
        {
            "step": "2. Tokenization",
            "description": "Metni sayısal temsile dönüştürme",
            "operations": [
                "• Text → Token ID mapping",
                "• Padding ve truncation",
                "• Attention mask oluşturma",
                "• Special tokens ekleme"
            ]
        },
        {
            "step": "3. Model Inference",
            "description": "Model ile tahmin yapma",
            "operations": [
                "• GPU/CPU'ya tensor taşıma",
                "• Model forward pass",
                "• Generation/classification",
                "• Result collection"
            ]
        },
        {
            "step": "4. Output Postprocessing",
            "description": "Model çıktısını işleme",
            "operations": [
                "• Token ID → Text decode",
                "• Special token removal",
                "• Text cleaning ve formatting",
                "• Confidence calculation"
            ]
        },
        {
            "step": "5. Response Formatting",
            "description": "Son yanıtı yapılandırma",
            "operations": [
                "• JSON response oluşturma",
                "• Metadata ekleme (timestamp, model_version)",
                "• Error handling ve logging",
                "• Performance metrics"
            ]
        }
    ]
    
    for item in steps:
        print(f"\n{item['step']}")
        print(f"   {item['description']}")
        print()
        for op in item['operations']:
            print(f"   {op}")
    
    # ========================================================================
    # HUGGING FACE PIPELINE KULLANIMI
    # ========================================================================
    
    print("\n\n2️⃣ HUGGING FACE PIPELINE API")
    print("-" * 80)
    print("""
    Hugging Face, high-level bir Pipeline API sunar.
    Tüm preprocessing, inference ve postprocessing otomatik yapılır.
    
    AVANTAJLARI:
    -----------
    ✓ Kolay kullanım (tek satır kod)
    ✓ Otomatik preprocessing/postprocessing
    ✓ Birçok task için hazır (text-generation, classification, vb.)
    ✓ Batch processing desteği
    
    DEZAVANTAJLARI:
    --------------
    ✗ Daha az kontrol
    ✗ Custom preprocessing zor
    ✗ Memory overhead biraz daha fazla
    """)
    
    pipeline_example = '''
# ============================================================================
# ÖRNEK 1: TEKSTüretimi PIPELINE
# ============================================================================

from transformers import pipeline, AutoTokenizer
import torch

# Tokenizer yükle ve pad token ayarla
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Text generation pipeline oluştur
generator = pipeline(
    "text-generation",                          # Task tipi
    model="./fine_tuned_model",                # Model path
    tokenizer=tokenizer,                        # Tokenizer
    device=0 if torch.cuda.is_available() else -1,  # GPU (0) veya CPU (-1)
    torch_dtype="auto",                         # Otomatik dtype
    max_new_tokens=100,                         # Default max tokens
)

# KULLANIM 1: Deterministik (Faktüel içerik)
result_factual = generator(
    "Yapay zeka nedir?",
    max_new_tokens=50,
    do_sample=False,                    # Sampling kapalı
    temperature=0.0,                    # Deterministik
    pad_token_id=tokenizer.eos_token_id,
    num_return_sequences=1              # Tek sonuç
)

print("Faktüel Yanıt:")
print(result_factual[0]['generated_text'])
print()

# KULLANIM 2: Yaratıcı (Hikaye/Blog)
result_creative = generator(
    "Bir zamanlar uzak bir galakside...",
    max_new_tokens=150,
    do_sample=True,                     # Sampling açık
    temperature=0.8,                    # Yüksek yaratıcılık
    top_p=0.9,                          # Nucleus sampling
    top_k=50,                           # Top-k filtering
    repetition_penalty=1.1,             # Tekrar önleme
    pad_token_id=tokenizer.eos_token_id,
    num_return_sequences=1
)

print("Yaratıcı Yanıt:")
print(result_creative[0]['generated_text'])
print()

# KULLANIM 3: Batch Processing
prompts = [
    "Python nedir?",
    "Machine learning nasıl çalışır?",
    "Deep learning ile machine learning farkı nedir?"
]

results = generator(
    prompts,
    max_new_tokens=80,
    do_sample=True,
    temperature=0.7,
    batch_size=3,                       # Batch processing
    pad_token_id=tokenizer.eos_token_id
)

for i, result in enumerate(results):
    print(f"Soru {i+1}: {prompts[i]}")
    print(f"Yanıt: {result[0]['generated_text']}")
    print()

# ============================================================================
# ÖRNEK 2: CLASSIFICATION PIPELINE
# ============================================================================

# Text classification pipeline
classifier = pipeline(
    "text-classification",
    model="./sentiment_model",
    device=0 if torch.cuda.is_available() else -1
)

# Tekli sınıflandırma
text = "Bu film gerçekten harikaydı, kesinlikle tavsiye ederim!"
result = classifier(text)

print(f"Metin: {text}")
print(f"Tahmin: {result[0]['label']}")
print(f"Güven: {result[0]['score']:.2%}")
print()

# Batch sınıflandırma
texts = [
    "Harika bir ürün, çok memnunum!",
    "Berbat bir deneyimdi, asla tekrar almam.",
    "İdare eder, fena değil."
]

results = classifier(texts, batch_size=3)

for text, result in zip(texts, results):
    print(f"Metin: {text}")
    print(f"Duygu: {result['label']} ({result['score']:.2%})")
    print()
'''
    
    print(pipeline_example)
    
    # ========================================================================
    # CUSTOM PIPELINE SINIFI
    # ========================================================================
    
    print("\n3️⃣ CUSTOM INFERENCE PIPELINE SINIFI")
    print("-" * 80)
    print("Daha fazla kontrol için özel pipeline sınıfı:")
    print()
    
    custom_pipeline = '''
# ============================================================================
# CUSTOM INFERENCE PIPELINE
# ============================================================================

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import logging
from typing import Dict, List, Union

class ProductionInferencePipeline:
    """
    Production-Ready Inference Pipeline
    
    Özellikleri:
    • Input preprocessing ve validation
    • Batch processing desteği
    • Error handling ve logging
    • Performance monitoring
    • Flexible generation configs
    """
    
    def __init__(self, model_path: str, device: str = "auto"):
        """Pipeline'ı başlat"""
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        
        # Model ve tokenizer yükle
        self.logger.info(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype="auto"
        )
        
        # Pad token ayarla
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        self.logger.info("Model loaded successfully")
    
    def preprocess(self, text: Union[str, List[str]]) -> Dict:
        """
        Input preprocessing
        
        Args:
            text: Tek string veya string listesi
        
        Returns:
            Preprocessed ve tokenized inputs
        """
        # Liste değilse listeye çevir
        if isinstance(text, str):
            text = [text]
        
        # Input validation
        for t in text:
            if not t or not isinstance(t, str):
                raise ValueError(f"Invalid input: {t}")
            if len(t) > 5000:  # Max karakter limiti
                raise ValueError("Input too long (>5000 chars)")
        
        # Tokenization
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # GPU'ya taşı
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        return inputs
    
    def postprocess(self, outputs, inputs):
        """
        Output postprocessing
        
        Args:
            outputs: Model outputs
            inputs: Original inputs
        
        Returns:
            Decoded text (sadece yeni tokenlar)
        """
        # Sadece yeni üretilen tokenları decode et
        input_length = inputs['input_ids'].shape[-1]
        new_tokens = outputs[:, input_length:]
        
        # Decode
        texts = self.tokenizer.batch_decode(
            new_tokens,
            skip_special_tokens=True
        )
        
        return [t.strip() for t in texts]
    
    def generate(
        self,
        prompts: Union[str, List[str]],
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        deterministic: bool = False
    ) -> List[Dict]:
        """
        Text generation
        
        Returns:
            List of dicts with 'text', 'latency_ms', etc.
        """
        start_time = time.time()
        
        try:
            # Preprocessing
            inputs = self.preprocess(prompts)
            
            # Generation config
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            if deterministic:
                gen_kwargs.update({
                    "do_sample": False,
                    "temperature": 0.0,
                })
            else:
                gen_kwargs.update({
                    "do_sample": True,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": 50,
                })
            
            # Inference
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)
            
            # Postprocessing
            texts = self.postprocess(outputs, inputs)
            
            # Performance metrics
            latency_ms = (time.time() - start_time) * 1000
            
            # Format results
            results = []
            for text in texts:
                results.append({
                    "generated_text": text,
                    "latency_ms": latency_ms / len(texts),
                    "model": self.model_path,
                    "timestamp": time.time()
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Inference error: {e}")
            raise

# ============================================================================
# KULLANIM ÖRNEĞİ
# ============================================================================

# Pipeline oluştur
pipeline = ProductionInferencePipeline(
    model_path="./fine_tuned_model",
    device="auto"
)

# Tekli inference
result = pipeline.generate(
    "Python nedir?",
    max_new_tokens=50,
    deterministic=True
)

print(f"Sonuç: {result[0]['generated_text']}")
print(f"Latency: {result[0]['latency_ms']:.2f}ms")

# Batch inference
results = pipeline.generate(
    ["Python nedir?", "AI nasıl çalışır?", "ML nedir?"],
    max_new_tokens=80,
    temperature=0.7
)

for i, res in enumerate(results):
    print(f"\\nSonuç {i+1}: {res['generated_text']}")
    print(f"Latency: {res['latency_ms']:.2f}ms")
'''
    
    print(custom_pipeline)
    
    # ========================================================================
    # BEST PRACTICES
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("💡 PIPELINE BEST PRACTICES")
    print("=" * 80)
    print("""
    1. ERROR HANDLING:
       ✓ Try-except blokları kullan
       ✓ Input validation yap
       ✓ Timeout mekanizması ekle
       ✓ Graceful degradation
    
    2. LOGGING:
       ✓ Her adımı logla
       ✓ Performance metrics kaydet
       ✓ Error stack trace'leri sakla
       ✓ Request/response örnekleri kaydet
    
    3. MONITORING:
       ✓ Latency tracking
       ✓ Throughput measurement
       ✓ Error rate monitoring
       ✓ Resource utilization (GPU/CPU/Memory)
    
    4. TESTING:
       ✓ Unit tests (her adım için)
       ✓ Integration tests
       ✓ Load tests
       ✓ Edge case tests
    
    5. OPTIMIZATION:
       ✓ Batch processing kullan
       ✓ Cache sık kullanılan sonuçları
       ✓ Connection pooling
       ✓ Async processing (gerekirse)
    
    6. SECURITY:
       ✓ Input sanitization
       ✓ Rate limiting
       ✓ Authentication/Authorization
       ✓ Data encryption
    """)
    
    print("=" * 80)
    print("✅ Inference pipeline demonstrasyonu tamamlandı!\n")


# ============================================================================
# MODEL DEPLOYMENT STRATEJİLERİ
# ============================================================================

def demonstrate_model_deployment():
    """
    Model Deployment (Dağıtım) Stratejileri ve Best Practices
    ==========================================================
    
    Bu fonksiyon, fine-tune edilmiş modellerin production ortamına
    nasıl deploy edileceğini, farklı deployment stratejilerini ve
    her birinin artı/eksilerini gösterir.
    
    DEPLOYMENT NEDİR?
    -----------------
    Model deployment, eğitilmiş bir modeli gerçek kullanıcılara
    hizmet verebilecek şekilde production ortamına alma sürecidir.
    
    DEPLOYMENT AŞAMALARI:
    --------------------
    1. Model Hazırlama (Optimization, Quantization)
    2. Containerization (Docker)
    3. API Servisi Oluşturma (REST, gRPC)
    4. Orchestration (Kubernetes)
    5. Monitoring & Logging
    6. CI/CD Pipeline
    7. Scaling Strategy
    
    DEPLOYMENT FAKTÖRLERI:
    ---------------------
    • Latency gereksinimleri
    • Throughput ihtiyacı
    • Maliyet kısıtları
    • Güvenlik gereksinimleri
    • Ölçeklenebilirlik
    • Bakım kolaylığı
    """
    
    print("\n" + "=" * 80)
    print("🚀 MODEL DEPLOYMENT STRATEJİLERİ")
    print("=" * 80)
    
    # ========================================================================
    # DEPLOYMENT STRATEJİLERİ
    # ========================================================================
    
    print("\n📋 DEPLOYMENT STRATEJİLERİ KARŞILAŞTIRMASI")
    print("-" * 80)
    
    strategies = {
        "🌐 REST API (Flask/FastAPI)": {
            "description": "HTTP tabanlı web servisi",
            "best_for": "Web uygulamaları, mobil apps, genel kullanım",
            "pros": [
                "✓ Kolay implement edilir",
                "✓ Platform bağımsız",
                "✓ Yaygın protokol (HTTP)",
                "✓ Load balancing kolay",
                "✓ Debugging/testing basit"
            ],
            "cons": [
                "✗ gRPC'ye göre daha yavaş",
                "✗ HTTP overhead",
                "✗ Binary data için verimsiz"
            ],
            "latency": "10-50ms (overhead)",
            "complexity": "Düşük",
            "use_case": "Chatbot API, content generation service"
        },
        
        "⚡ gRPC": {
            "description": "Yüksek performanslı RPC framework",
            "best_for": "Microservices, low-latency uygulamalar",
            "pros": [
                "✓ REST'e göre 2-5x hızlı",
                "✓ Binary protocol (Protobuf)",
                "✓ Bi-directional streaming",
                "✓ Type-safe",
                "✓ Code generation"
            ],
            "cons": [
                "✗ Daha karmaşık setup",
                "✗ Browser desteği sınırlı",
                "✗ Debugging zor"
            ],
            "latency": "2-10ms (overhead)",
            "complexity": "Orta",
            "use_case": "Real-time inference, internal services"
        },
        
        "🐳 Docker Container": {
            "description": "Containerized deployment",
            "best_for": "Portable, reproducible deployments",
            "pros": [
                "✓ Environment consistency",
                "✓ Kolay versiyonlama",
                "✓ İzolasyon (security)",
                "✓ Ölçeklenebilir",
                "✓ CI/CD uyumlu"
            ],
            "cons": [
                "✗ Overhead (düşük)",
                "✗ Storage gereksinimi",
                "✗ GPU pass-through kompleksliği"
            ],
            "latency": "Minimal (konteyner overhead: <5ms)",
            "complexity": "Orta",
            "use_case": "Tüm production deploymentlar"
        },
        
        "☸️ Kubernetes": {
            "description": "Container orchestration platform",
            "best_for": "Large-scale, auto-scaling deployments",
            "pros": [
                "✓ Auto-scaling (HPA)",
                "✓ Self-healing",
                "✓ Load balancing",
                "✓ Rolling updates",
                "✓ Multi-cloud support"
            ],
            "cons": [
                "✗ Steep learning curve",
                "✗ Overhead (cluster management)",
                "✗ Karmaşık konfigürasyon"
            ],
            "latency": "Minimal",
            "complexity": "Yüksek",
            "use_case": "Enterprise applications, high-traffic services"
        },
        
        "📱 Edge Deployment": {
            "description": "Mobile/IoT cihazlarda çalıştırma",
            "best_for": "Offline inference, privacy-critical apps",
            "pros": [
                "✓ Sıfır latency (network yok)",
                "✓ Privacy (data local)",
                "✓ Offline çalışma",
                "✓ Düşük bandwidth"
            ],
            "cons": [
                "✗ Sınırlı compute",
                "✗ Model boyutu kısıtları",
                "✗ Update zorluğu",
                "✗ Quantization gerekli"
            ],
            "latency": "~10-100ms (cihaza bağlı)",
            "complexity": "Yüksek",
            "use_case": "Mobile apps, IoT devices, offline tools"
        },
        
        "☁️ Serverless (AWS Lambda, Cloud Functions)": {
            "description": "Event-driven, auto-scaling functions",
            "best_for": "Sporadic usage, cost optimization",
            "pros": [
                "✓ Sıfır bakım (infrastructure)",
                "✓ Auto-scaling",
                "✓ Pay-per-use pricing",
                "✓ Kolay deploy"
            ],
            "cons": [
                "✗ Cold start (3-10 saniye)",
                "✗ Execution time limit",
                "✗ Memory limits",
                "✗ Büyük modeller zor"
            ],
            "latency": "50ms-10s (cold start dahil)",
            "complexity": "Düşük-Orta",
            "use_case": "Düşük frekanslı inference, batch processing"
        },
        
        "🖥️ Dedicated Server": {
            "description": "Fiziksel veya sanal sunucu",
            "best_for": "Predictable traffic, high GPU requirements",
            "pros": [
                "✓ Maksimum performans",
                "✓ Tam kontrol",
                "✓ GPU erişimi kolay",
                "✓ No cold start"
            ],
            "cons": [
                "✗ Yüksek maliyet",
                "✗ Manual scaling",
                "✗ Bakım gereksinimi",
                "✗ Under-utilization riski"
            ],
            "latency": "Minimal",
            "complexity": "Orta",
            "use_case": "High-throughput services, GPU-intensive tasks"
        },
        
        "🔌 Model-as-a-Service (Hugging Face Inference API)": {
            "description": "Managed inference service",
            "best_for": "Hızlı prototipleme, küçük ölçek",
            "pros": [
                "✓ Sıfır infrastructure",
                "✓ Anında başlatma",
                "✓ Auto-scaling",
                "✓ Free tier mevcut"
            ],
            "cons": [
                "✗ Yüksek maliyet (scale'de)",
                "✗ Sınırlı kontrol",
                "✗ Vendor lock-in",
                "✗ Privacy concerns"
            ],
            "latency": "50-200ms",
            "complexity": "Çok Düşük",
            "use_case": "MVP, demo, low-traffic apps"
        }
    }
    
    for strategy_name, details in strategies.items():
        print(f"\n{strategy_name}")
        print("=" * 80)
        print(f"📝 Açıklama: {details['description']}")
        print(f"🎯 En İyi Kullanım: {details['best_for']}")
        
        print("\n✅ Avantajlar:")
        for pro in details['pros']:
            print(f"   {pro}")
        
        print("\n❌ Dezavantajlar:")
        for con in details['cons']:
            print(f"   {con}")
        
        print(f"\n⏱️ Latency: {details['latency']}")
        print(f"🔧 Komplekslik: {details['complexity']}")
        print(f"💼 Kullanım Örneği: {details['use_case']}")
        print()
    
    # ========================================================================
    # DEPLOYMENT CHECKLIST
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("✅ DEPLOYMENT CHECKLIST")
    print("=" * 80)
    
    checklist = {
        "🎯 Model Hazırlık": [
            "☐ Model quantization (INT8/FP16)",
            "☐ Model serialization (ONNX, TorchScript)",
            "☐ Benchmark performans (latency, throughput)",
            "☐ Memory profiling",
            "☐ Model versiyonlama"
        ],
        
        "🔒 Güvenlik": [
            "☐ API authentication (JWT, OAuth)",
            "☐ Rate limiting",
            "☐ Input validation ve sanitization",
            "☐ HTTPS/TLS encryption",
            "☐ Secrets management (API keys)",
            "☐ CORS configuration"
        ],
        
        "📊 Monitoring & Logging": [
            "☐ Request/response logging",
            "☐ Performance metrics (Prometheus/Grafana)",
            "☐ Error tracking (Sentry)",
            "☐ Latency monitoring",
            "☐ Resource utilization (GPU/CPU/Memory)",
            "☐ Alerting system"
        ],
        
        "⚡ Performance": [
            "☐ Batch processing implementation",
            "☐ Caching strategy",
            "☐ Load balancing",
            "☐ Connection pooling",
            "☐ Async processing (gerekirse)",
            "☐ CDN (static content için)"
        ],
        
        "🧪 Testing": [
            "☐ Unit tests",
            "☐ Integration tests",
            "☐ Load tests (Apache Bench, Locust)",
            "☐ Stress tests",
            "☐ Canary deployment tests",
            "☐ Rollback plan"
        ],
        
        "📈 Scalability": [
            "☐ Horizontal scaling stratejisi",
            "☐ Auto-scaling rules",
            "☐ Database optimization",
            "☐ Queue system (Celery, RabbitMQ)",
            "☐ Cache layer (Redis)",
            "☐ Multi-region deployment (opsiyonel)"
        ],
        
        "🔄 CI/CD": [
            "☐ Automated testing pipeline",
            "☐ Automated deployment",
            "☐ Blue-green deployment",
            "☐ Rollback mechanism",
            "☐ Version tagging",
            "☐ Environment configs (dev, staging, prod)"
        ],
        
        "📚 Dokümantasyon": [
            "☐ API documentation (Swagger/OpenAPI)",
            "☐ Usage examples",
            "☐ Error codes documentation",
            "☐ SLA definition",
            "☐ Runbook (troubleshooting)",
            "☐ Architecture diagram"
        ]
    }
    
    for category, items in checklist.items():
        print(f"\n{category}")
        print("-" * 80)
        for item in items:
            print(f"  {item}")
    
    # ========================================================================
    # DEPLOYMENT WORKFLOW ÖRNEĞİ
    # ========================================================================
    
    print("\n\n" + "=" * 80)
    print("🔄 ÖRNEK DEPLOYMENT WORKFLOW")
    print("=" * 80)
    
    workflow = """
    1. GELIŞTIRME AŞAMASI (Development)
       ├─ Model training ve fine-tuning
       ├─ Local testing
       ├─ Performance benchmarking
       └─ Code review
    
    2. HAZIRLIK AŞAMASI (Preparation)
       ├─ Model optimization (quantization)
       ├─ Containerization (Dockerfile oluştur)
       ├─ Unit/integration tests
       └─ Documentation update
    
    3. STAGING ORTAMI (Staging)
       ├─ Deploy to staging environment
       ├─ Integration testing
       ├─ Load testing
       ├─ Security scanning
       └─ Stakeholder approval
    
    4. PRODUCTION DEPLOYMENT (Production)
       ├─ Canary deployment (5% traffic)
       ├─ Monitor metrics closely
       ├─ Gradual rollout (10% → 50% → 100%)
       └─ Full deployment
    
    5. MONİTORING (Post-Deployment)
       ├─ Real-time performance monitoring
       ├─ Error rate tracking
       ├─ User feedback collection
       └─ Continuous optimization
    
    6. BAKIMIMLAMA (Maintenance)
       ├─ Regular model updates
       ├─ Bug fixes
       ├─ Performance tuning
       └─ Infrastructure updates
    """
    
    print(workflow)
    
    # ========================================================================
    # ÖRNEK DEPLOYMENT KODU
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("💻 ÖRNEK DEPLOYMENT KODU (FastAPI + Docker)")
    print("=" * 80)
    
    deployment_code = '''
# ============================================================================
# FILE: main.py (FastAPI Application)
# ============================================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
from typing import Optional

app = FastAPI(
    title="LLM Inference API",
    description="Production-ready LLM inference service",
    version="1.0.0"
)

# Model yükleme (startup sırasında)
@app.on_event("startup")
async def load_model():
    global model, tokenizer
    
    model_path = "./fine_tuned_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    print("✅ Model loaded successfully")

# Request/Response models
class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    deterministic: bool = False

class GenerateResponse(BaseModel):
    generated_text: str
    latency_ms: float
    model_version: str = "1.0.0"

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "gpu_available": torch.cuda.is_available()
    }

# Generation endpoint
@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    start_time = time.time()
    
    try:
        # Tokenization
        inputs = tokenizer(
            request.prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generation config
        gen_kwargs = {
            "max_new_tokens": request.max_new_tokens,
            "pad_token_id": tokenizer.eos_token_id,
        }
        
        if request.deterministic:
            gen_kwargs.update({"do_sample": False, "temperature": 0.0})
        else:
            gen_kwargs.update({
                "do_sample": True,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "top_k": 50
            })
        
        # Inference
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        
        # Decode
        new_tokens = outputs[0][inputs['input_ids'].shape[-1]:]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        latency_ms = (time.time() - start_time) * 1000
        
        return GenerateResponse(
            generated_text=generated_text.strip(),
            latency_ms=latency_ms
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run: uvicorn main:app --host 0.0.0.0 --port 8000

# ============================================================================
# FILE: Dockerfile
# ============================================================================

FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Python kurulumu
RUN apt-get update && apt-get install -y python3-pip

WORKDIR /app

# Dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Model ve kod kopyala
COPY ./fine_tuned_model /app/fine_tuned_model
COPY main.py .

# Expose port
EXPOSE 8000

# Start server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# Build: docker build -t llm-inference:v1 .
# Run: docker run -p 8000:8000 --gpus all llm-inference:v1

# ============================================================================
# FILE: docker-compose.yml
# ============================================================================

version: '3.8'

services:
  llm-inference:
    build: .
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - MODEL_PATH=/app/fine_tuned_model
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

# Run: docker-compose up
'''
    
    print(deployment_code)
    
    print("\n" + "=" * 80)
    print("✅ Model deployment demonstrasyonu tamamlandı!")
    print("=" * 80)


# ============================================================================
# ANA PROGRAM (MAIN)
# ============================================================================

if __name__ == "__main__":
    """
    Ana Demonstrasyon Programı
    ===========================
    
    Bu program, inference ve kişiselleştirme konusundaki tüm
    teknikleri demonstre eder.
    
    PROGRAM AKIŞI:
    --------------
    1. Inference optimizasyon teknikleri
    2. Kişiselleştirilmiş chatbot sistemi
    3. Generation configuration parametreleri
    4. Performance benchmarking
    5. Inference pipeline oluşturma
    6. Model deployment stratejileri
    7. Pratik kullanım örnekleri
    """
    
    # ========================================================================
    # PROGRAM BAŞLANGIÇ
    # ========================================================================
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 15 + "INFERENCE VE KİŞİSELLEŞTİRME BOOTCAMP" + " " * 25 + "█")
    print("█" + " " * 78 + "█")
    print("█" + " " * 20 + "Production-Ready LLM Deployment" + " " * 27 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    
    print("\n📚 KONU BAŞLIKLARI:")
    print("-" * 80)
    topics = [
        "1️⃣ Inference Optimizasyon Teknikleri",
        "2️⃣ Kişiselleştirilmiş Chatbot Sistemleri",
        "3️⃣ Generation Configuration Parametreleri",
        "4️⃣ Performance Benchmarking ve Analiz",
        "5️⃣ Production Inference Pipeline",
        "6️⃣ Model Deployment Stratejileri",
        "7️⃣ Pratik Kullanım Örnekleri"
    ]
    
    for topic in topics:
        print(f"   {topic}")
    
    print("\n" + "=" * 80)
    print("🚀 DEMONSTRASYONLARIgeliştirilmiş BAŞLIYOR...")
    print("=" * 80)
    
    # ========================================================================
    # DEMONSTRASYONLARI ÇALIŞTIR
    # ========================================================================
    
    try:
        # 1. Inference Optimizasyon Teknikleri
        demonstrate_inference_optimization()
        
        # 2. Kişiselleştirilmiş Chatbot
        create_personalized_chatbot()
        
        # 3. Generation Configuration
        demonstrate_generation_config()
        
        # 4. Performance Benchmarking
        benchmark_inference_speed()
        
        # 5. Inference Pipeline
        create_inference_pipeline()
        
        # 6. Model Deployment
        demonstrate_model_deployment()
        
    except Exception as e:
        print(f"\n❌ HATA: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # PRATİK KULLANIM ÖRNEKLERİ
    # ========================================================================
    
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 25 + "PRATİK KULLANIM ÖRNEKLERİ" + " " * 29 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    
    print("\n" + "=" * 80)
    print("💻 KOD ÖRNEKLERİ")
    print("=" * 80)
    
    example_usage = '''
# ============================================================================
# ÖRNEK 1: DETERMİNİSTİK METİN ÜRETİMİ (Faktüel/Teknik İçerik)
# ============================================================================

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Model ve tokenizer yükleme
model_path = "./fine_tuned_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16  # FP16 for efficiency
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.eval()

# Deterministik üretim (her zaman aynı sonuç)
prompt = "Python programlama dilinin temel özellikleri nelerdir?"

inputs = tokenizer(prompt, return_tensors="pt", padding=True)
if torch.cuda.is_available():
    inputs = {k: v.cuda() for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,        # Sampling KAPALI
        temperature=0.0,        # Deterministik
        pad_token_id=tokenizer.eos_token_id
    )

# Sadece yeni tokenları decode et
new_tokens = outputs[0][inputs['input_ids'].shape[-1]:]
result = tokenizer.decode(new_tokens, skip_special_tokens=True)

print("=" * 70)
print("SORU:", prompt)
print("-" * 70)
print("YANIT:", result.strip())
print("=" * 70)

# ============================================================================
# ÖRNEK 2: YARATICI METİN ÜRETİMİ (Hikaye/Blog)
# ============================================================================

# Yaratıcı üretim (her çalıştırmada farklı sonuç)
creative_prompt = "Bir zamanlar, teknolojinin insanlığı değiştirdiği bir dünyada..."

inputs = tokenizer(creative_prompt, return_tensors="pt", padding=True)
if torch.cuda.is_available():
    inputs = {k: v.cuda() for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,          # Sampling AÇIK
        temperature=0.8,         # Yüksek yaratıcılık
        top_p=0.9,              # Nucleus sampling
        top_k=50,               # Top-k filtering
        repetition_penalty=1.1,  # Tekrar önleme
        pad_token_id=tokenizer.eos_token_id
    )

new_tokens = outputs[0][inputs['input_ids'].shape[-1]:]
creative_result = tokenizer.decode(new_tokens, skip_special_tokens=True)

print("\\n" + "=" * 70)
print("PROMPT:", creative_prompt)
print("-" * 70)
print("HİKAYE:", creative_result.strip())
print("=" * 70)

# ============================================================================
# ÖRNEK 3: SINIFLANDIRMA (Sentiment Analysis)
# ============================================================================

from transformers import AutoModelForSequenceClassification

# Sınıflandırma modeli yükleme
classifier_model = AutoModelForSequenceClassification.from_pretrained(
    "./sentiment_model",
    device_map="auto"
)

classifier_model.eval()

# Sınıflandırma
text = "Bu ürün gerçekten harika, kesinlikle tavsiye ederim!"
label_names = ["Negatif", "Nötr", "Pozitif"]

inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
if torch.cuda.is_available():
    inputs = {k: v.cuda() for k, v in inputs.items()}

with torch.no_grad():
    outputs = classifier_model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    probabilities = predictions.cpu().numpy()[0]
    predicted_class = probabilities.argmax()

print("\\n" + "=" * 70)
print("METİN:", text)
print("-" * 70)
print(f"TAHMİN: {label_names[predicted_class]}")
print(f"GÜVEN SKORU: {probabilities[predicted_class]:.2%}")
print("\\nTÜM OLASLIKLAR:")
for i, (label, prob) in enumerate(zip(label_names, probabilities)):
    bar = "█" * int(prob * 50)
    print(f"  {label:<10} [{prob:>6.2%}] {bar}")
print("=" * 70)

# ============================================================================
# ÖRNEK 4: BATCH PROCESSING (Toplu İşleme)
# ============================================================================

# Birden fazla prompt'u aynı anda işleme
batch_prompts = [
    "Yapay zeka nedir?",
    "Machine learning nasıl çalışır?",
    "Python neden popülerdir?"
]

# Batch tokenization
batch_inputs = tokenizer(
    batch_prompts,
    return_tensors="pt",
    padding=True,
    truncation=True
)

if torch.cuda.is_available():
    batch_inputs = {k: v.cuda() for k, v in batch_inputs.items()}

# Batch generation
with torch.no_grad():
    batch_outputs = model.generate(
        **batch_inputs,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

# Her sonucu ayrı decode et
print("\\n" + "=" * 70)
print("BATCH PROCESSING SONUÇLARI")
print("=" * 70)

for i, (prompt, output) in enumerate(zip(batch_prompts, batch_outputs)):
    input_length = len(tokenizer.encode(prompt))
    new_tokens = output[input_length:]
    result = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    print(f"\\n{i+1}. SORU: {prompt}")
    print(f"   YANIT: {result.strip()}")

print("=" * 70)

# ============================================================================
# ÖRNEK 5: LORA MODEL KULLANIMI
# ============================================================================

from peft import PeftModel

# Base model ve LoRA adapter yükleme
base_model_path = "meta-llama/Llama-2-7b-hf"
lora_adapter_path = "./lora_adapters"

print("\\n" + "=" * 70)
print("LORA MODEL YÜKLEME")
print("=" * 70)

base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",
    torch_dtype=torch.float16
)

# LoRA adapter ekle
lora_model = PeftModel.from_pretrained(base_model, lora_adapter_path)
lora_model.eval()

print("✅ LoRA model yüklendi")

# LoRA model ile inference
lora_prompt = "Fine-tuning yapılmış konuda soru..."
lora_inputs = base_tokenizer(lora_prompt, return_tensors="pt")

if torch.cuda.is_available():
    lora_inputs = {k: v.cuda() for k, v in lora_inputs.items()}

with torch.no_grad():
    lora_outputs = lora_model.generate(
        **lora_inputs,
        max_new_tokens=100,
        temperature=0.7
    )

result = base_tokenizer.decode(lora_outputs[0], skip_special_tokens=True)
print(f"\\nLoRA Model Yanıtı: {result}")
print("=" * 70)

# ============================================================================
# ÖRNEK 6: QUANTIZED MODEL KULLANIMI
# ============================================================================

print("\\n" + "=" * 70)
print("QUANTIZED MODEL (INT8) KULLANIMI")
print("=" * 70)

# INT8 quantized model yükleme
quantized_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    load_in_8bit=True,
    device_map="auto"
)

quantized_model.eval()

print("✅ INT8 quantized model yüklendi")
print("💾 Bellek tasarrufu: ~75% (FP32'ye göre)")

# Quantized model ile inference
# (Kullanımı normal model ile aynı)

print("=" * 70)
'''
    
    print(example_usage)
    
    # ========================================================================
    # GERÇEKmodel INFERENCE ÖRNEĞİ (Opsiyonel)
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("🎯 GERÇEKgerçek INFERENCE ÖRNEĞİ")
    print("=" * 80)
    print("""
    Gerçek bir model ile inference yapmak için aşağıdaki kodu uncomment edin:
    
    NOT: Bu kod çalıştırılabilmesi için:
    • Fine-tune edilmiş bir model gereklidir
    • Model path'i doğru ayarlanmalıdır
    • Gerekli kütüphaneler yüklenmiş olmalıdır
    """)
    
    real_inference_code = '''
    # Gerçek model inference (uncomment to run)
    try:
        model_path = "./fine_tuned_model"  # Model path'inizi buraya yazın
        
        print(f"\\n📦 Model yükleniyor: {model_path}")
        inference_engine = PersonalizedInference(model_path, model_type="causal_lm")
        
        # Test 1: Deterministik üretim
        print("\\n🔬 Test 1: Deterministik Üretim")
        print("-" * 70)
        factual_result = inference_engine.generate_text(
            "Python programlama dilinde liste (list) nedir?",
            max_new_tokens=50,
            deterministic=True
        )
        print(f"Sonuç: {factual_result}")
        
        # Test 2: Yaratıcı üretim
        print("\\n🎨 Test 2: Yaratıcı Üretim")
        print("-" * 70)
        creative_result = inference_engine.generate_text(
            "Bir zamanlar uzak bir galakside...",
            max_new_tokens=150,
            temperature=0.9,
            deterministic=False
        )
        print(f"Hikaye: {creative_result}")
        
        print("\\n✅ Gerçek inference testleri tamamlandı!")
        
    except FileNotFoundError:
        print("\\n⚠️ Model bulunamadı. Önce bir model fine-tune etmelisiniz.")
    except Exception as e:
        print(f"\\n❌ Hata: {e}")
    '''
    
    print(real_inference_code)
    
    # ========================================================================
    # PROGRAM SONU
    # ========================================================================
    
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 25 + "✅ DEMONSTRASYON TAMAMLANDI!" + " " * 24 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    
    print("\n📚 ÖZET:")
    print("-" * 80)
    
    summary = """
    Bu demonstrasyonda öğrendikleriniz:
    
    ✓ Inference optimizasyon teknikleri (Quantization, Batching, KV-Cache)
    ✓ Kişiselleştirilmiş chatbot sistemi tasarımı
    ✓ Generation parametreleri ve kullanım senaryoları
    ✓ Performance benchmarking ve analiz
    ✓ Production-ready inference pipeline oluşturma
    ✓ Model deployment stratejileri ve best practices
    ✓ Pratik kod örnekleri ve kullanım senaryoları
    
    🎯 BİR SONRAKİ ADIMLAR:
    ----------------------
    1. Kendi modelinizi fine-tune edin (1_peft_lora.py)
    2. Bu dosyadaki teknikleri kendi modelinizle deneyin
    3. Production deployment için bir strateji seçin
    4. Performance testleri yapın ve optimize edin
    5. Monitoring ve logging sistemleri kurun
    
    📖 İLAVE KAYNAKLAR:
    ------------------
    • Hugging Face Documentation: https://huggingface.co/docs
    • PEFT Library: https://github.com/huggingface/peft
    • FastAPI: https://fastapi.tiangolo.com
    • Docker: https://docs.docker.com
    • Kubernetes: https://kubernetes.io/docs
    
    💬 SORULARINIZ İÇİN:
    -------------------
    Kairu AI - Build with LLMs Bootcamp
    """
    
    print(summary)
    
    print("\n" + "=" * 80)
    print("🎓 İyi Çalışmalar!")
    print("=" * 80)
    print()
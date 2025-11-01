"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                 PEFT (Parameter Efficient Fine-Tuning) ve LoRA              ║
║                          Detaylı Eğitim Scripti                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

GENEL BAKIŞ:
-----------
Bu script, modern dil modellerini verimli bir şekilde fine-tune etmek için 
PEFT (Parameter Efficient Fine-Tuning) yaklaşımını ve özellikle LoRA 
(Low-Rank Adaptation) tekniğini kullanmayı gösterir.

TEMEL KAVRAMLAR:
---------------
1. PEFT (Parameter Efficient Fine-Tuning):
   - Büyük modellerin tüm parametrelerini eğitmek yerine, sadece küçük bir 
     kısmını veya ek adapter katmanlarını eğitme yaklaşımı
   - Avantajlar: Daha az bellek, daha hızlı eğitim, daha az disk kullanımı

2. LoRA (Low-Rank Adaptation):
   - PEFT'in en popüler tekniklerinden biri
   - Orijinal model ağırlıklarını dondurup, düşük rankli (low-rank) 
     matrisler ekleyerek adaptasyon sağlar
   - Matematiksel formül: W = W₀ + ΔW = W₀ + BA
     * W₀: Orijinal ağırlık matrisi (dondurulmuş)
     * B, A: Düşük rankli matrisler (eğitilebilir)
     * rank(BA) << rank(W₀)

KULLANIM ALANLARI:
-----------------
- Chatbot özelleştirme
- Domain-specific dil modelleri (tıbbi, hukuki, teknik vb.)
- Çok görevli öğrenme (multi-task learning)
- Kişiselleştirilmiş asistanlar
- Düşük kaynaklı ortamlarda model eğitimi

DOSYA YAPISI:
------------
1. Kütüphane İmportları ve Yapılandırma
2. Model Kurulum Fonksiyonları
3. Veri Hazırlama Fonksiyonları
4. Eğitim Fonksiyonları
5. Analiz ve Karşılaştırma Fonksiyonları
6. Ana Çalıştırma Bloğu

Yazar: Kairu LLM Bootcamp - Hafta 6
Tarih: 2025
Versiyon: 2.0 (Geliştirilmiş ve Türkçeleştirilmiş)
"""

# ============================================================================
# BÖLÜM 1: KÜTÜPHANE İMPORTLARI
# ============================================================================

import torch  # PyTorch - Derin öğrenme framework'ü
from transformers import (
    AutoTokenizer,           # Otomatik tokenizer yükleyici
    AutoModelForCausalLM,    # Causal Language Model (GPT-tarzı) otomatik yükleyici
    TrainingArguments,        # Eğitim parametrelerini yapılandırma sınıfı
    Trainer,                  # Hugging Face'in yüksek seviye eğitim API'si
    DataCollatorForLanguageModeling  # Language modeling için data collator
)
from peft import (
    LoraConfig,      # LoRA yapılandırma sınıfı
    get_peft_model,  # Standart modeli PEFT modeline dönüştüren fonksiyon
    TaskType         # Görev tiplerini tanımlayan enum
)
from datasets import Dataset  # Hugging Face Datasets kütüphanesi


# ============================================================================
# BÖLÜM 2: GLOBAL YAPILANDIRMA VE SABITLER
# ============================================================================

# Model Yapılandırması
DEFAULT_MODEL_NAME = "microsoft/DialoGPT-medium"  # Varsayılan kullanılacak model
MAX_SEQUENCE_LENGTH = 512                         # Maksimum token uzunluğu

# LoRA Hiperparametreleri
LORA_RANK = 16              # Low-rank matrisin rank'ı (r parametresi)
                            # Daha yüksek r = daha fazla kapasite ama daha fazla parametre
                            # Tipik değerler: 4, 8, 16, 32, 64
                            
LORA_ALPHA = 32             # LoRA scaling faktörü (α parametresi)
                            # Genellikle r'nin 1-2 katı olarak ayarlanır
                            # Formül: scaling = α / r
                            
LORA_DROPOUT = 0.1          # LoRA katmanlarında dropout oranı
                            # Overfitting'i önlemek için kullanılır
                            
# DialoGPT için hedef modüller
# Bu modüller, transformer'ın attention ve projection katmanlarıdır
LORA_TARGET_MODULES = ["c_attn", "c_proj"]

# Eğitim Yapılandırması
OUTPUT_DIR = "./lora_results"           # Checkpoint'lerin kaydedileceği klasör
FINAL_MODEL_DIR = "./lora_model"        # Final modelin kaydedileceği klasör
NUM_TRAIN_EPOCHS = 3                    # Toplam eğitim epoch sayısı
BATCH_SIZE = 2                          # Her cihaz için batch size
GRADIENT_ACCUMULATION_STEPS = 4         # Gradient accumulation adımları
WARMUP_STEPS = 100                      # Learning rate warmup adımları
LOGGING_STEPS = 10                      # Her kaç adımda log yazılacağı


# ============================================================================
# BÖLÜM 3: MODEL KURULUM FONKSİYONLARI
# ============================================================================

def setup_lora_model(model_name: str = DEFAULT_MODEL_NAME) -> tuple:
    """
    LoRA konfigürasyonu ile dil modelini hazırlar ve yükler.
    
    Bu fonksiyon şu adımları gerçekleştirir:
    1. Pre-trained model ve tokenizer'ı indirir/yükler
    2. Tokenizer yapılandırmasını optimize eder
    3. LoRA konfigürasyonunu oluşturur
    4. Standart modeli PEFT modeline dönüştürür
    5. Eğitilebilir parametreleri analiz eder ve raporlar
    
    LoRA NEDİR?
    -----------
    LoRA (Low-Rank Adaptation), büyük dil modellerini eğitmek için 
    memory-efficient bir yöntemdir. Temel prensibi:
    
    - Orijinal model ağırlıkları W₀ dondurulur (freeze edilir)
    - Bunun yerine, düşük rankli iki matris (A ve B) eğitilir
    - Final çıktı: W = W₀ + BA şeklinde hesaplanır
    - rank(BA) << rank(W₀) olduğu için çok daha az parametre eğitilir
    
    Örnek: 
    - Orijinal ağırlık matrisi: 1024x1024 = 1,048,576 parametre
    - LoRA ile (r=16): (1024x16) + (16x1024) = 32,768 parametre
    - %96.9 parametre tasarrufu!
    
    Parametreler:
    ------------
    model_name : str
        Hugging Face model hub'dan yüklenecek model adı.
        Varsayılan: "microsoft/DialoGPT-medium"
        Diğer örnekler: "gpt2", "gpt2-large", "EleutherAI/gpt-neo-1.3B"
    
    Döndürür:
    --------
    tuple : (model, tokenizer)
        model : PeftModel
            LoRA adapter'ları eklenmiş PEFT modeli
        tokenizer : PreTrainedTokenizer
            Model ile uyumlu tokenizer
    
    Örnek Kullanım:
    --------------
    >>> model, tokenizer = setup_lora_model("gpt2")
    >>> model.print_trainable_parameters()
    trainable params: 294,912 || all params: 124,734,720 || trainable%: 0.2364
    """
    
    print(f"\n{'='*80}")
    print(f"🔧 MODEL KURULUMU BAŞLIYOR")
    print(f"{'='*80}")
    print(f"📦 Model: {model_name}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Adım 1: Tokenizer'ı yükle
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[1/5] Tokenizer yükleniyor...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Pad token kontrolü ve ayarlaması
    # Bazı modellerde pad_token tanımlı değildir, bu durumda eos_token kullanılır
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"   ℹ️  Pad token ayarlandı: '{tokenizer.eos_token}'")
    
    print(f"   ✅ Tokenizer başarıyla yüklendi")
    print(f"   📊 Vocabulary boyutu: {len(tokenizer):,} token")
    
    # ─────────────────────────────────────────────────────────────────
    # Adım 2: Pre-trained modeli yükle
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[2/5] Pre-trained model yükleniyor...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Model bilgilerini göster
    total_params_original = sum(p.numel() for p in model.parameters())
    print(f"   ✅ Model başarıyla yüklendi")
    print(f"   📊 Toplam parametre sayısı: {total_params_original:,}")
    print(f"   💾 Tahmini boyut: {total_params_original * 4 / (1024**3):.2f} GB (float32)")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Adım 3: LoRA konfigürasyonunu oluştur
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[3/5] LoRA konfigürasyonu oluşturuluyor...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,        # Görev tipi: Causal Language Modeling
        r=LORA_RANK,                          # Low-rank decomposition rank değeri
        lora_alpha=LORA_ALPHA,                # LoRA scaling faktörü
        lora_dropout=LORA_DROPOUT,            # Dropout oranı (regularization)
        target_modules=LORA_TARGET_MODULES,   # LoRA'nın uygulanacağı modüller
        bias="none",                          # Bias parametrelerini eğitme ("none", "all", "lora_only")
        inference_mode=False,                 # Eğitim modu (False = eğitim, True = inference)
    )
    
    print(f"   ✅ LoRA konfigürasyonu oluşturuldu")
    print(f"   📐 Rank (r): {LORA_RANK}")
    print(f"   🎚️  Alpha (α): {LORA_ALPHA}")
    print(f"   📉 Scaling faktörü (α/r): {LORA_ALPHA/LORA_RANK}")
    print(f"   💧 Dropout: {LORA_DROPOUT}")
    print(f"   🎯 Hedef modüller: {', '.join(LORA_TARGET_MODULES)}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Adım 4: PEFT modelini oluştur
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[4/5] PEFT modeli oluşturuluyor...")
    model = get_peft_model(model, lora_config)
    
    print(f"   ✅ Model PEFT modeline dönüştürüldü")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Adım 5: Eğitilebilir parametreleri analiz et
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[5/5] Parametre analizi yapılıyor...")
    print(f"\n{'─'*80}")
    
    # Hugging Face PEFT'in built-in fonksiyonu ile özet bilgi
    model.print_trainable_parameters()
    
    # Detaylı analiz
    print(f"\n📋 DETAYLI PARAMETRE ANALİZİ:")
    print(f"{'─'*80}")
    
    # Eğitilebilir parametreler
    print(f"\n✅ EĞİTİLEBİLİR PARAMETRELER (LoRA Adapter'ları):")
    print(f"{'─'*80}")
    trainable_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"   • {name:50s} → {str(param.shape):20s} ({param.numel():,} params)")
            trainable_count += 1
    
    print(f"\n   📊 Toplam eğitilebilir katman sayısı: {trainable_count}")
    
    # Dondurulmuş parametreler (örnekleme)
    print(f"\n❄️  DONDURULMUŞ PARAMETRELER (Orijinal Model):")
    print(f"{'─'*80}")
    frozen_count = 0
    total_frozen = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            total_frozen += 1
            if frozen_count < 5:  # İlk 5 örneği göster
                print(f"   • {name:50s} → {str(param.shape):20s} ({param.numel():,} params)")
                frozen_count += 1
    
    if total_frozen > 5:
        print(f"   ... ve {total_frozen - 5:,} adet daha dondurulmuş parametre")
    
    print(f"\n   📊 Toplam dondurulmuş katman sayısı: {total_frozen}")
    
    # Özet istatistikler
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*80}")
    print(f"📊 ÖZET İSTATİSTİKLER:")
    print(f"{'='*80}")
    print(f"   Toplam parametreler      : {total_params:,}")
    print(f"   Eğitilebilir parametreler: {trainable_params:,}")
    print(f"   Dondurulmuş parametreler : {total_params - trainable_params:,}")
    print(f"   Eğitilebilir oran        : {100 * trainable_params / total_params:.4f}%")
    print(f"   Bellek tasarrufu         : ~{100 * (1 - trainable_params / total_params):.2f}%")
    print(f"{'='*80}\n")
    
    return model, tokenizer


# ============================================================================
# BÖLÜM 4: VERİ HAZIRLAMA FONKSİYONLARI
# ============================================================================

def prepare_dataset(tokenizer, max_length: int = MAX_SEQUENCE_LENGTH) -> Dataset:
    """
    Model eğitimi için basit bir konuşma dataseti oluşturur ve tokenize eder.
    
    Bu fonksiyon şu işlemleri gerçekleştirir:
    1. Örnek konuşma metinlerini oluşturur
    2. Metinleri tokenize eder (sayısal forma çevirir)
    3. Padding ve truncation uygular
    4. Hugging Face Dataset formatına dönüştürür
    
    TOKENIZATION NEDİR?
    ------------------
    Tokenization, metinleri model tarafından işlenebilir sayısal forma
    dönüştürme işlemidir. Örnek:
    
    Metin: "Merhaba dünya!"
    Tokenlar: ["Mer", "haba", " dün", "ya", "!"]
    Token IDs: [5673, 8901, 2134, 6789, 0]
    
    PADDING VE TRUNCATION:
    ---------------------
    - Padding: Kısa cümleleri sabit uzunluğa getirmek için token ekleme
    - Truncation: Uzun cümleleri max_length'e göre kesme
    
    Parametreler:
    ------------
    tokenizer : PreTrainedTokenizer
        Metinleri tokenize etmek için kullanılacak tokenizer
    max_length : int
        Maksimum sequence uzunluğu (varsayılan: 512)
        Daha uzun sequenceler kesilir, daha kısa olanlar padding ile doldurulur
    
    Döndürür:
    --------
    Dataset
        Tokenize edilmiş Hugging Face Dataset objesi
        Her örnek şu sütunları içerir:
        - input_ids: Token ID'leri
        - attention_mask: Hangi tokenlerin gerçek, hangilerinin padding olduğunu gösterir
        - labels: Eğitim için hedef token'lar (input_ids ile aynı)
    
    Örnek Kullanım:
    --------------
    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
    >>> dataset = prepare_dataset(tokenizer, max_length=128)
    >>> print(dataset)
    Dataset({
        features: ['input_ids', 'attention_mask', 'labels'],
        num_rows: 10
    })
    """
    
    print(f"\n{'='*80}")
    print(f"📚 VERİ SETİ HAZIRLANIYOR")
    print(f"{'='*80}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Adım 1: Örnek konuşma verilerini tanımla
    # ─────────────────────────────────────────────────────────────────────────
    # NOT: Gerçek bir projede bu veriler bir dosyadan veya veritabanından
    # yüklenirdi. Bu örnekte gösterim amaçlı basit veriler kullanıyoruz.
    
    conversations = [
        # Temel selamlaşma
        "Kullanıcı: Merhaba! Asistan: Merhaba! Size nasıl yardımcı olabilirim?",
        
        # Programlama soruları
        "Kullanıcı: Python öğrenmek istiyorum. Asistan: Harika! Python'a nereden başlamak istiyorsunuz? Temel sözdizimi mi yoksa belirli bir proje mi?",
        "Kullanıcı: Makine öğrenmesi nedir? Asistan: Makine öğrenmesi, bilgisayarların verilerden öğrenmesini ve deneyimle performanslarını geliştirmelerini sağlayan bir tekniktir.",
        
        # Teknik sorular
        "Kullanıcı: Derin öğrenme ile makine öğrenmesi arasındaki fark nedir? Asistan: Derin öğrenme, makine öğrenmesinin bir alt dalıdır ve yapay sinir ağları kullanır.",
        "Kullanıcı: Neural network nasıl çalışır? Asistan: Neural network, birbirine bağlı nöronlardan oluşan katmanlar halinde çalışır ve veriyi işleyerek öğrenir.",
        
        # Pratik örnekler
        "Kullanıcı: Bir chatbot nasıl yapabilirim? Asistan: Chatbot yapmak için önce bir dil modeli seçmeli, sonra kendi verinizle fine-tune etmelisiniz.",
        "Kullanıcı: Fine-tuning nedir? Asistan: Fine-tuning, önceden eğitilmiş bir modeli kendi özel göreviniz için uyarlama sürecidir.",
        
        # İleri seviye
        "Kullanıcı: LoRA ne işe yarar? Asistan: LoRA, büyük modelleri verimli şekilde fine-tune etmek için kullanılan bir tekniktir ve çok az parametre güncelleyerek iyi sonuçlar verir.",
        "Kullanıcı: Transformer mimarisi nedir? Asistan: Transformer, attention mekanizması kullanan ve modern NLP'nin temelini oluşturan bir sinir ağı mimarisidir.",
        "Kullanıcı: GPT modelleri nasıl çalışır? Asistan: GPT modelleri, transformer decoder kullanarak önceki tokenlere bakarak sonraki tokenı tahmin eder.",
    ]
    
    print(f"   📝 Toplam konuşma örneği: {len(conversations)}")
    print(f"   📏 Maksimum sequence uzunluğu: {max_length}")
    
    # Örnek bir konuşmayı göster
    print(f"\n   💬 Örnek konuşma:")
    print(f"   {'-'*76}")
    print(f"   {conversations[0]}")
    print(f"   {'-'*76}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Adım 2: Tokenization fonksiyonunu tanımla
    # ─────────────────────────────────────────────────────────────────────────
    def tokenize_function(examples):
        """
        Batch içindeki metinleri tokenize eder.
        
        Bu iç fonksiyon, dataset.map() tarafından çağrılır ve her batch
        için tokenization işlemini gerçekleştirir.
        """
        result = tokenizer(
            examples["text"],                  # Tokenize edilecek metinler
            truncation=True,                   # Uzun metinleri max_length'e göre kes
            padding="max_length",              # Tüm sequenceleri aynı uzunluğa getir
            max_length=max_length,             # Maksimum sequence uzunluğu
            return_tensors=None,               # Dataset map için None kullan (batching için)
        )
        # Causal LM için labels ekliyoruz (input_ids'in kopyası)
        result["labels"] = result["input_ids"].copy()
        return result
    
    # ─────────────────────────────────────────────────────────────────────────
    # Adım 3: Dataset oluştur ve tokenize et
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n   🔄 Tokenization işlemi başlıyor...")
    
    # Ham dataseti oluştur
    dataset = Dataset.from_dict({"text": conversations})
    
    # Tokenization uygula (batched=True daha hızlı işlem sağlar)
    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True,
        remove_columns=["text"],  # Orijinal text kolonunu kaldır (artık gerekli değil)
        desc="Tokenizing"         # Progress bar için açıklama
    )
    
    print(f"   ✅ Tokenization tamamlandı")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Adım 4: Dataset istatistiklerini göster
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n   📊 Dataset İstatistikleri:")
    print(f"   {'-'*76}")
    print(f"   • Toplam örnek sayısı: {len(tokenized_dataset)}")
    print(f"   • Özellikler: {list(tokenized_dataset.features.keys())}")
    print(f"   • Her sequence uzunluğu: {max_length} token")
    
    # İlk örneğin detaylarını göster
    if len(tokenized_dataset) > 0:
        first_example = tokenized_dataset[0]
        print(f"\n   🔍 İlk Örnek Detayı:")
        print(f"   {'-'*76}")
        print(f"   • Input IDs sayısı: {len(first_example['input_ids'])}")
        print(f"   • Attention mask sayısı: {len(first_example['attention_mask'])}")
        print(f"   • Labels sayısı: {len(first_example['labels'])}")
        print(f"   • Gerçek token sayısı: {sum(first_example['attention_mask'])}")
        print(f"   • Padding token sayısı: {len(first_example['attention_mask']) - sum(first_example['attention_mask'])}")
    
    print(f"{'='*80}\n")
    
    return tokenized_dataset


# ============================================================================
# BÖLÜM 5: EĞİTİM FONKSİYONLARI
# ============================================================================

def train_lora_model() -> tuple:
    """
    LoRA kullanarak dil modelini fine-tune eder.
    
    Bu fonksiyon, complete bir eğitim pipeline'ını orchestrate eder:
    1. Model ve tokenizer kurulumu
    2. Dataset hazırlığı ve preprocessing
    3. Training arguments yapılandırması
    4. Hugging Face Trainer ile eğitim
    5. Model ve tokenizer kaydetme
    
    EĞİTİM SÜRECİ:
    --------------
    Eğitim şu aşamalardan oluşur:
    
    1. Forward Pass: Input verisi modelden geçirilir, çıktı alınır
    2. Loss Hesaplama: Tahmin ile gerçek değer arasındaki fark hesaplanır
    3. Backward Pass: Gradient'lar hesaplanır (backpropagation)
    4. Optimizer Step: Parametreler güncellenir
    5. Tekrar (epoch ve batch sayısına göre)
    
    GRADIENT ACCUMULATION:
    ---------------------
    Büyük batch size kullanamıyorsak, gradient accumulation ile
    birden fazla küçük batch'in gradient'larını biriktirip tek seferde
    update yapabiliriz.
    
    Örnek:
    - Gerçek batch size = batch_size × gradient_accumulation_steps
    - batch_size=2, accumulation=4 → Efektif batch size = 8
    
    WARMUP:
    -------
    Eğitimin başında learning rate'i düşük tutup yavaşça artırma stratejisi.
    Ani büyük güncellemelerin modeli bozmasını önler.
    
    Döndürür:
    --------
    tuple : (model, tokenizer)
        model : PeftModel
            Eğitilmiş LoRA modeli
        tokenizer : PreTrainedTokenizer
            Model ile uyumlu tokenizer
    
    Örnek Kullanım:
    --------------
    >>> model, tokenizer = train_lora_model()
    🔧 MODEL KURULUMU BAŞLIYOR...
    📚 VERİ SETİ HAZIRLANIYOR...
    🚀 EĞİTİM BAŞLIYOR...
    ✅ Eğitim tamamlandı!
    """
    
    print(f"\n{'#'*80}")
    print(f"{'#'*80}")
    print(f"##{' '*76}##")
    print(f"##{'LoRA MODEL EĞİTİM PİPELINE':^76}##")
    print(f"##{' '*76}##")
    print(f"{'#'*80}")
    print(f"{'#'*80}\n")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Adım 1: Model ve Tokenizer Kurulumu
    # ─────────────────────────────────────────────────────────────────────────
    print(f"[ADIM 1/4] Model Kurulumu")
    model, tokenizer = setup_lora_model()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Adım 2: Dataset Hazırlığı
    # ─────────────────────────────────────────────────────────────────────────
    print(f"[ADIM 2/4] Dataset Hazırlığı")
    train_dataset = prepare_dataset(tokenizer)
    
    # Data Collator oluştur
    # Bu, batch'leri dinamik olarak padding yapar ve labels'ı düzenler
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM için False (Masked LM için True olur)
    )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Adım 3: Training Arguments Yapılandırması
    # ─────────────────────────────────────────────────────────────────────────
    print(f"[ADIM 3/4] Training Arguments Yapılandırması")
    print(f"{'='*80}")
    
    training_args = TrainingArguments(
        # ─── Çıktı ve Loglama ───
        output_dir=OUTPUT_DIR,                      # Checkpoint'lerin kaydedileceği klasör
        logging_dir=f"{OUTPUT_DIR}/logs",           # TensorBoard logları
        logging_steps=LOGGING_STEPS,                # Her X step'te log kaydı
        
        # ─── Eğitim Hiperparametreleri ───
        num_train_epochs=NUM_TRAIN_EPOCHS,          # Kaç epoch eğitim yapılacak
        per_device_train_batch_size=BATCH_SIZE,     # Her GPU/CPU için batch size
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,  # Gradient biriktirme adımları
        
        # ─── Optimizer ve Learning Rate ───
        learning_rate=5e-5,                         # Başlangıç learning rate
        warmup_steps=WARMUP_STEPS,                  # LR warmup adım sayısı
        lr_scheduler_type="linear",                 # LR scheduler tipi (linear, cosine, etc.)
        weight_decay=0.01,                          # L2 regularization (overfitting önler)
        
        # ─── Checkpoint ve Kaydetme ───
        save_strategy="epoch",                      # Her epoch'ta kaydet
        save_steps=500,                             # Alternatif: Her X step'te kaydet
        save_total_limit=2,                         # En fazla kaç checkpoint sakla
        load_best_model_at_end=False,               # Eğitim sonunda en iyi modeli yükle
        
        # ─── Performans Optimizasyonları ───
        fp16=False,                                 # 16-bit floating point (GPU'da hızlandırma)
        # bf16=False,                               # Brain float 16 (yeni GPU'larda)
        gradient_checkpointing=False,               # Bellek tasarrufu (daha yavaş ama daha az bellek)
        
        # ─── Diğer Ayarlar ───
        report_to="none",                           # Loglama servisi (wandb, tensorboard, etc.)
        disable_tqdm=False,                         # Progress bar'ı kapat (False = açık)
        remove_unused_columns=True,                 # Kullanılmayan kolonları kaldır
        dataloader_num_workers=0,                   # Data loading için worker sayısı
        
        # ─── Seed (Tekrarlanabilirlik için) ───
        seed=42,                                    # Random seed
    )
    
    print(f"\n   ⚙️  EĞİTİM AYARLARI:")
    print(f"   {'-'*76}")
    print(f"   📁 Output directory        : {OUTPUT_DIR}")
    print(f"   🔄 Epoch sayısı            : {NUM_TRAIN_EPOCHS}")
    print(f"   📦 Batch size              : {BATCH_SIZE}")
    print(f"   🔢 Gradient accumulation   : {GRADIENT_ACCUMULATION_STEPS}")
    print(f"   📊 Efektif batch size      : {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"   📈 Learning rate           : {training_args.learning_rate}")
    print(f"   🌡️  Warmup steps           : {WARMUP_STEPS}")
    print(f"   💾 Checkpoint stratejisi   : {training_args.save_strategy}")
    print(f"   🎲 Random seed             : {training_args.seed}")
    print(f"{'='*80}\n")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Adım 4: Trainer Oluşturma ve Eğitimi Başlatma
    # ─────────────────────────────────────────────────────────────────────────
    print(f"[ADIM 4/4] Trainer Oluşturma ve Eğitim")
    print(f"{'='*80}")
    
    # Trainer objesi oluştur
    # Trainer, Hugging Face'in high-level eğitim API'sidir
    # Training loop, logging, checkpointing gibi tüm detayları halleder
    trainer = Trainer(
        model=model,                    # Eğitilecek model (LoRA eklenmiş)
        args=training_args,             # Training ayarları
        train_dataset=train_dataset,    # Eğitim dataseti
        tokenizer=tokenizer,            # Tokenizer (kaydetme için gerekli)
        data_collator=data_collator,    # Data collator (batch hazırlama)
        # compute_metrics=...,          # Metrik hesaplama fonksiyonu (opsiyonel)
    )
    
    print(f"\n   ✅ Trainer başarıyla oluşturuldu")
    print(f"\n{'='*80}")
    print(f"🚀 EĞİTİM BAŞLIYOR...")
    print(f"{'='*80}\n")
    
    # EĞİTİMİ BAŞLAT!
    # Bu satır tüm eğitim sürecini çalıştırır
    train_result = trainer.train()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Eğitim Sonrası İşlemler
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"✅ EĞİTİM TAMAMLANDI!")
    print(f"{'='*80}")
    
    # Eğitim metriklerini göster
    print(f"\n📊 EĞİTİM METRİKLERİ:")
    print(f"{'-'*80}")
    print(f"   • Training Loss     : {train_result.training_loss:.4f}")
    print(f"   • Toplam Adım       : {train_result.global_step}")
    print(f"   • Epoch Sayısı      : {NUM_TRAIN_EPOCHS}")
    
    # Model ve tokenizer'ı kaydet
    print(f"\n💾 MODEL KAYDEDİLİYOR...")
    print(f"{'-'*80}")
    
    # LoRA adapter'larını kaydet
    trainer.save_model(FINAL_MODEL_DIR)
    print(f"   ✅ LoRA adapter'ları kaydedildi: {FINAL_MODEL_DIR}")
    
    # Tokenizer'ı kaydet
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    print(f"   ✅ Tokenizer kaydedildi: {FINAL_MODEL_DIR}")
    
    # Training arguments'ı kaydet (referans için)
    training_args_path = f"{FINAL_MODEL_DIR}/training_args.txt"
    with open(training_args_path, "w", encoding="utf-8") as f:
        f.write(training_args.to_json_string())
    print(f"   ✅ Training arguments kaydedildi: {training_args_path}")
    
    print(f"\n{'='*80}")
    print(f"🎉 TÜM İŞLEMLER BAŞARIYLA TAMAMLANDI!")
    print(f"{'='*80}")
    print(f"\n📂 Kaydedilen dosyalar:")
    print(f"   • LoRA Adapter     : {FINAL_MODEL_DIR}/adapter_model.bin")
    print(f"   • Adapter Config   : {FINAL_MODEL_DIR}/adapter_config.json")
    print(f"   • Tokenizer        : {FINAL_MODEL_DIR}/tokenizer.json")
    print(f"   • Training Args    : {FINAL_MODEL_DIR}/training_args.txt")
    print(f"\n💡 Modeli yüklemek için:")
    print(f'   >>> from peft import PeftModel')
    print(f'   >>> base_model = AutoModelForCausalLM.from_pretrained("{DEFAULT_MODEL_NAME}")')
    print(f'   >>> model = PeftModel.from_pretrained(base_model, "{FINAL_MODEL_DIR}")')
    print(f"{'='*80}\n")
    
    return model, tokenizer


# ============================================================================
# BÖLÜM 6: ANALİZ VE KARŞILAŞTIRMA FONKSİYONLARI
# ============================================================================

def demonstrate_lora_benefits():
    """
    LoRA'nın sağladığı faydaları detaylı bir şekilde gösterir ve analiz eder.
    
    Bu fonksiyon şunları yapar:
    1. LoRA'nın avantajlarını açıklar
    2. Gerçek bir model üzerinde parametre analizi yapar
    3. Bellek ve hesaplama tasarrufunu hesaplar
    4. Traditional fine-tuning ile karşılaştırma yapar
    
    LoRA AVANTAJLARI:
    ----------------
    
    1. BELLEK VERİMLİLİĞİ:
       - Sadece adapter katmanları GPU belleğinde tutulur
       - Base model ağırlıkları donuk olduğu için gradientleri saklanmaz
       - Örnek: GPT-3 175B → Sadece ~100MB adapter
    
    2. HIZLI EĞİTİM:
       - Daha az parametre = daha az backpropagation
       - Daha az optimizer state (Adam için önemli)
       - 2-3x daha hızlı eğitim süresi
    
    3. MODÜLER YAPILAR:
       - Farklı görevler için farklı adapter'lar
       - Base model tek, görev başına adapter
       - Kolay geçiş: Adapter'ı değiştir, görev değişti!
    
    4. DÜŞÜK DİSK KULLANIMI:
       - Base model: 500MB-5GB
       - Her adapter: ~1-10MB
       - 100 görev için: 1 base + 100 adapter ≈ 1GB total
    
    5. DEPLOYMENT KOLAYLIĞI:
       - Base model bir kere deploy edilir
       - Her kullanıcı/görev için sadece adapter yüklenir
       - Multi-tenant senaryolar için ideal
    
    Döndürür:
    --------
    None
        Fonksiyon sonuç döndürmez, sadece detaylı analiz çıktıları verir
    
    Örnek Kullanım:
    --------------
    >>> demonstrate_lora_benefits()
    ╔══════════════════════════════════════════════════════════════════╗
    ║                    LoRA AVANTAJLARI ANALİZİ                      ║
    ╚══════════════════════════════════════════════════════════════════╝
    ...
    """
    
    print(f"\n{'╔' + '═'*78 + '╗'}")
    print(f"║{'LoRA AVANTAJLARI - DETAYLI ANALİZ':^78}║")
    print(f"{'╚' + '═'*78 + '╝'}\n")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Bölüm 1: Teorik Avantajlar
    # ─────────────────────────────────────────────────────────────────────────
    print(f"{'='*80}")
    print(f"📚 1. TEORİK AVANTAJLAR")
    print(f"{'='*80}\n")
    
    advantages = [
        {
            "icon": "💾",
            "title": "Bellek Verimliliği",
            "description": "Sadece adapter katmanları eğitilir, base model dondurulur",
            "benefit": "GPU belleği %70-90 oranında azalır"
        },
        {
            "icon": "⚡",
            "title": "Hızlı Eğitim",
            "description": "Daha az parametre güncellenir, daha az gradient hesaplanır",
            "benefit": "Eğitim süresi 2-3x daha kısa"
        },
        {
            "icon": "🔧",
            "title": "Modülerlik",
            "description": "Her görev için ayrı adapter, tek base model",
            "benefit": "Multi-task learning kolaylaşır"
        },
        {
            "icon": "💿",
            "title": "Düşük Disk Kullanımı",
            "description": "Sadece adapter ağırlıkları kaydedilir (~1-10 MB)",
            "benefit": "100 görev için bile minimal disk kullanımı"
        },
        {
            "icon": "🚀",
            "title": "Kolay Deployment",
            "description": "Base model bir kere, adapter'lar ihtiyaç anında yüklenir",
            "benefit": "Multi-tenant sistemler için ideal"
        },
        {
            "icon": "🎯",
            "title": "Düşük Catastrophic Forgetting",
            "description": "Base model değişmediği için orijinal yetenekler korunur",
            "benefit": "Genel yetenekler + özel görev becerileri"
        }
    ]
    
    for i, adv in enumerate(advantages, 1):
        print(f"{adv['icon']} {adv['title']}")
        print(f"   └─ Açıklama : {adv['description']}")
        print(f"   └─ Fayda    : {adv['benefit']}")
        print()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Bölüm 2: Pratik Demonstrasyon
    # ─────────────────────────────────────────────────────────────────────────
    print(f"{'='*80}")
    print(f"🔬 2. PRATİK DEMONSTRASYON - GERÇEK MODEL ANALİZİ")
    print(f"{'='*80}\n")
    
    print(f"Model yükleniyor (lütfen bekleyin)...")
    model, tokenizer = setup_lora_model()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Bölüm 3: Detaylı Parametre Analizi
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"📊 3. DETAYLI PARAMETRE ANALİZİ")
    print(f"{'='*80}\n")
    
    # Parametre istatistiklerini hesapla
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    trainable_percentage = 100 * trainable_params / total_params
    
    # Bellek hesaplamaları (float32 için 4 byte/param)
    bytes_per_param = 4  # float32
    total_memory_mb = (total_params * bytes_per_param) / (1024**2)
    trainable_memory_mb = (trainable_params * bytes_per_param) / (1024**2)
    frozen_memory_mb = (frozen_params * bytes_per_param) / (1024**2)
    
    # Gradient belleği (sadece trainable parametreler için)
    # Adam optimizer: params + momentum + variance = 3x
    gradient_memory_mb = trainable_memory_mb * 3
    total_training_memory_mb = frozen_memory_mb + trainable_memory_mb + gradient_memory_mb
    
    # Traditional fine-tuning ile karşılaştırma
    traditional_gradient_memory_mb = total_memory_mb * 3
    traditional_total_memory_mb = total_memory_mb + traditional_gradient_memory_mb
    memory_saving_percentage = 100 * (1 - total_training_memory_mb / traditional_total_memory_mb)
    
    # Sonuçları göster
    print(f"┌{'─'*78}┐")
    print(f"│ {'PARAMETRE İSTATİSTİKLERİ':^76} │")
    print(f"├{'─'*78}┤")
    print(f"│ Toplam Parametreler          │ {total_params:>15,} params │ {total_memory_mb:>8.2f} MB │")
    print(f"│ Eğitilebilir Parametreler    │ {trainable_params:>15,} params │ {trainable_memory_mb:>8.2f} MB │")
    print(f"│ Dondurulmuş Parametreler     │ {frozen_params:>15,} params │ {frozen_memory_mb:>8.2f} MB │")
    print(f"├{'─'*78}┤")
    print(f"│ Eğitilebilir Oran            │ {trainable_percentage:>15.4f} %      │            │")
    print(f"└{'─'*78}┘")
    
    print(f"\n┌{'─'*78}┐")
    print(f"│ {'BELLEK KULLANIMI ANALİZİ (Adam Optimizer ile)':^76} │")
    print(f"├{'─'*78}┤")
    print(f"│ Model Ağırlıkları (Frozen)   │                         │ {frozen_memory_mb:>8.2f} MB │")
    print(f"│ Model Ağırlıkları (Trainable)│                         │ {trainable_memory_mb:>8.2f} MB │")
    print(f"│ Gradient + Optimizer States  │                         │ {gradient_memory_mb:>8.2f} MB │")
    print(f"├{'─'*78}┤")
    print(f"│ LoRA Toplam Eğitim Belleği   │                         │ {total_training_memory_mb:>8.2f} MB │")
    print(f"└{'─'*78}┘")
    
    print(f"\n┌{'─'*78}┐")
    print(f"│ {'TRADITIONAL FINE-TUNING KARŞILAŞTIRMASI':^76} │")
    print(f"├{'─'*78}┤")
    print(f"│ Traditional Model Belleği    │                         │ {total_memory_mb:>8.2f} MB │")
    print(f"│ Traditional Gradient Belleği │                         │ {traditional_gradient_memory_mb:>8.2f} MB │")
    print(f"│ Traditional Toplam           │                         │ {traditional_total_memory_mb:>8.2f} MB │")
    print(f"├{'─'*78}┤")
    print(f"│ LoRA ile Bellek Tasarrufu    │                         │ {memory_saving_percentage:>7.2f} %  │")
    print(f"└{'─'*78}┘")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Bölüm 4: Disk Kullanımı Projeksiyonu
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"💿 4. DİSK KULLANIMI PROJEKSİYONU")
    print(f"{'='*80}\n")
    
    # Adapter boyutu (sadece trainable parametreler)
    adapter_size_mb = trainable_memory_mb
    
    # Farklı senaryo örnekleri
    scenarios = [
        ("1 görev", 1),
        ("10 görev", 10),
        ("50 görev", 50),
        ("100 görev", 100),
    ]
    
    print(f"┌{'─'*78}┐")
    print(f"│ {'Senaryo':^20} │ {'Traditional':^15} │ {'LoRA':^15} │ {'Tasarruf':^15} │")
    print(f"├{'─'*78}┤")
    
    for scenario_name, num_tasks in scenarios:
        traditional_total = total_memory_mb * num_tasks
        lora_total = total_memory_mb + (adapter_size_mb * num_tasks)
        saving = traditional_total - lora_total
        saving_pct = 100 * saving / traditional_total
        
        print(f"│ {scenario_name:^20} │ {traditional_total:>12.1f} MB │ {lora_total:>12.1f} MB │ {saving_pct:>12.1f} % │")
    
    print(f"└{'─'*78}┘")
    
    print(f"\n💡 Açıklama:")
    print(f"   • Traditional: Her görev için tam model kopyası")
    print(f"   • LoRA: 1 base model + her görev için adapter")
    print(f"   • Adapter boyutu: ~{adapter_size_mb:.2f} MB")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Bölüm 5: Eğitim Hızı Tahmini
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"⚡ 5. EĞİTİM HIZI TAHMİNİ")
    print(f"{'='*80}\n")
    
    # Forward pass: Her iki yöntemde de aynı
    # Backward pass: Sadece trainable parametrelere göre
    backward_speedup = total_params / trainable_params
    
    print(f"┌{'─'*78}┐")
    print(f"│ {'Metrik':^35} │ {'Traditional':^15} │ {'LoRA':^15} │")
    print(f"├{'─'*78}┤")
    print(f"│ {'Forward Pass (göreceli)':^35} │ {'1.0x':>15} │ {'1.0x':>15} │")
    print(f"│ {'Backward Pass (göreceli)':^35} │ {'1.0x':>15} │ {f'{1/backward_speedup:.3f}x':>15} │")
    print(f"│ {'Optimizer Step (göreceli)':^35} │ {'1.0x':>15} │ {f'{1/backward_speedup:.3f}x':>15} │")
    print(f"├{'─'*78}┤")
    print(f"│ {'Tahmini Hızlanma':^35} │ {'1.0x':>15} │ {f'{backward_speedup/3:.2f}x':>15} │")
    print(f"└{'─'*78}┘")
    
    print(f"\n💡 Not:")
    print(f"   • Gerçek hızlanma, model boyutu ve donanıma göre değişir")
    print(f"   • Genellikle 2-3x hızlanma gözlemlenir")
    print(f"   • Daha büyük modellerde fark daha belirgin olur")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Özet ve Öneriler
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"✨ ÖZET VE ÖNERİLER")
    print(f"{'='*80}\n")
    
    print(f"🎯 LoRA kullanmanız önerilen durumlar:")
    print(f"   ✓ Sınırlı GPU belleğiniz varsa")
    print(f"   ✓ Hızlı iterasyon yapmanız gerekiyorsa")
    print(f"   ✓ Çoklu görev/domain için model özelleştiriyorsanız")
    print(f"   ✓ Production'da çoklu kullanıcıya özel modeller sunacaksanız")
    print(f"   ✓ Maliyeti düşük tutmak istiyorsanız")
    
    print(f"\n⚠️  Traditional fine-tuning tercih edilebilecek durumlar:")
    print(f"   • Çok spesifik/dar bir domain'de maksimum performans gerekiyorsa")
    print(f"   • Modelin tüm katmanlarını ciddi şekilde değiştirmek gerekiyorsa")
    print(f"   • Bellek ve hesaplama kaynağı bol ise")
    
    print(f"\n{'='*80}")
    print(f"🎓 LoRA Hiperparametre Önerileri:")
    print(f"{'='*80}\n")
    
    print(f"   r (rank) seçimi:")
    print(f"   • Küçük görevler: r=4 veya r=8")
    print(f"   • Orta görevler: r=16 veya r=32")
    print(f"   • Büyük/kompleks görevler: r=64 veya r=128")
    print(f"   • Şu anki değer: r={LORA_RANK}")
    
    print(f"\n   α (alpha) seçimi:")
    print(f"   • Genellikle α = r veya α = 2×r")
    print(f"   • Scaling = α/r → Tipik değer: 1 veya 2")
    print(f"   • Şu anki değer: α={LORA_ALPHA} (scaling={LORA_ALPHA/LORA_RANK})")
    
    print(f"\n   Hedef modüller:")
    print(f"   • Attention: Query, Key, Value, Output projection")
    print(f"   • Feed-forward: Intermediate ve output layers")
    print(f"   • Şu anki hedefler: {', '.join(LORA_TARGET_MODULES)}")
    
    print(f"\n{'='*80}\n")


# ============================================================================
# BÖLÜM 7: YARDIMCI FONKSİYONLAR
# ============================================================================

def print_welcome_message():
    """
    Program başlangıcında hoş geldiniz mesajı ve bilgilendirme yapar.
    """
    print(f"\n")
    print(f"{'╔' + '═'*78 + '╗'}")
    print(f"║{' '*78}║")
    print(f"║{'PEFT (Parameter Efficient Fine-Tuning) ve LoRA':^78}║")
    print(f"║{'Detaylı Eğitim ve Demonstrasyon Scripti':^78}║")
    print(f"║{' '*78}║")
    print(f"{'╚' + '═'*78 + '╝'}")
    
    print(f"\n📖 Bu script şunları içerir:")
    print(f"   1. LoRA model kurulumu ve yapılandırması")
    print(f"   2. Veri hazırlama ve tokenization")
    print(f"   3. Model eğitimi (training pipeline)")
    print(f"   4. Detaylı avantaj analizi ve karşılaştırmalar")
    
    print(f"\n⚙️  Yapılandırma:")
    print(f"   • Model: {DEFAULT_MODEL_NAME}")
    print(f"   • LoRA Rank: {LORA_RANK}")
    print(f"   • LoRA Alpha: {LORA_ALPHA}")
    print(f"   • Epoch: {NUM_TRAIN_EPOCHS}")
    print(f"   • Batch Size: {BATCH_SIZE}")
    
    print(f"\n{'='*80}\n")


# ============================================================================
# BÖLÜM 8: ANA ÇALIŞTIRMA BLOĞU
# ============================================================================

if __name__ == "__main__":
    """
    Ana çalıştırma bloğu.
    
    Bu bölüm, scriptin doğrudan çalıştırılması durumunda (import edilmediğinde)
    çalışır ve tüm demonstrasyonu orchestrate eder.
    
    Kullanım:
    --------
    1. Sadece demonstrasyon (varsayılan):
       python 1_peft_lora.py
    
    2. Tam eğitim için, aşağıdaki satırın yorumunu kaldırın:
       model, tokenizer = train_lora_model()
    """
    
    # Hoş geldiniz mesajı
    print_welcome_message()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Demonstrasyon Modu (Varsayılan)
    # ─────────────────────────────────────────────────────────────────────────
    # LoRA'nın avantajlarını ve parametre analizini göster
    demonstrate_lora_benefits()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Eğitim Modu (Manuel Aktivasyon Gerekli)
    # ─────────────────────────────────────────────────────────────────────────
    # Eğitim yapmak için aşağıdaki satırların yorumunu kaldırın:
    
    print(f"\n{'='*80}")
    print(f"⚠️  DİKKAT: Eğitim modu aktif!")
    print(f"{'='*80}\n")
    
    model, tokenizer = train_lora_model()
    
    # Eğitim sonrası basit bir inference örneği
    print(f"\n{'='*80}")
    print(f"🤖 BASİT INFERENCE ÖRNEĞİ")
    print(f"{'='*80}\n")
    
    prompt = "Kullanıcı: Merhaba! Asistan:"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50, num_return_sequences=1)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"Prompt: {prompt}")
    print(f"Çıktı: {generated_text}")
    print(f"\n{'='*80}\n")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Bilgilendirme Mesajı
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n💡 İPUCU:")
    print(f"   • Eğitim modu çalıştırıldı!")
    print(f"   • Model başarıyla eğitildi ve kaydedildi")
    print(f"   • Kaydedilen konum: {FINAL_MODEL_DIR}")
    
    print(f"\n📚 Öğrenim Kaynakları:")
    print(f"   • LoRA Paper: https://arxiv.org/abs/2106.09685")
    print(f"   • Hugging Face PEFT: https://huggingface.co/docs/peft")
    print(f"   • Transformers Docs: https://huggingface.co/docs/transformers")
    
    print(f"\n{'='*80}")
    print(f"✅ PROGRAM BAŞARIYLA TAMAMLANDI")
    print(f"{'='*80}\n")
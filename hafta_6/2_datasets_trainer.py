"""
================================================================================
HUGGING FACE DATASETS VE TRAINER API KULLANIMI
================================================================================

Bu script, modern NLP model eğitiminin temel taşlarından olan Hugging Face 
Datasets ve Trainer API'sini kapsamlı bir şekilde gösterir.

KONU BAŞLIKLARI:
1. Datasets Kütüphanesi: Veri yönetimi ve dönüşümleri
2. Trainer API: Otomatik eğitim ve değerlendirme
3. Training Arguments: Eğitim parametreleri
4. Metrics ve Evaluation: Model performans ölçümü
5. Best Practices: Gelişmiş optimizasyonlar

ÖĞRENME HEDEFLERİ:
- Datasets kütüphanesi ile veri yükleme ve hazırlama
- Trainer API ile otomatik eğitim pipeline'ı oluşturma
- Özel metrik hesaplama ve model değerlendirme
- Production-ready model eğitim süreçleri tasarlama

Yazar: Kairu AI Bootcamp - Hafta 6
Tarih: Kasım 2025
================================================================================
"""

# =============================================================================
# KÜTÜPHANE İÇE AKTARMALARI
# =============================================================================

import torch
from transformers import (
    AutoTokenizer,                          # Otomatik tokenizer yükleyici
    AutoModelForSequenceClassification,     # Sınıflandırma modelleri
    TrainingArguments,                      # Eğitim parametreleri
    Trainer,                                # Ana eğitim sınıfı
    DataCollatorWithPadding                 # Dinamik padding için
)
from datasets import Dataset, load_dataset  # Veri yönetimi
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# =============================================================================
# VERİ YÜKLEME VE HAZIRLAMA FONKSİYONLARI
# =============================================================================

def load_and_prepare_dataset():
    """
    Dataset Yükleme ve Hazırlama
    
    Bu fonksiyon, model eğitimi için gerekli veri setini yükler ve hazırlar.
    
    DATASETS KÜTÜPHANESİ HAKKINDA:
    - Hugging Face Datasets: Büyük veri setlerini verimli yönetir
    - Memory mapping: RAM'i optimize eder, büyük veri setleri için idealdir
    - Arrow format: Hızlı okuma/yazma için Apache Arrow kullanır
    - Lazy loading: Sadece gerekli veriyi yükler
    
    GERÇEK DÜNYA KULLANIMI:
    - load_dataset("imdb"): Film yorumları sentiment analizi
    - load_dataset("squad"): Question answering veri seti
    - load_dataset("glue", "sst2"): Benchmark veri setleri
    
    Returns:
        tuple: (train_dataset, eval_dataset) - Eğitim ve değerlendirme setleri
    """
    print("📚 Dataset yükleniyor...")
    
    # Örnek sentiment analysis dataset'i
    # Production'da: load_dataset("imdb") veya kendi veri setiniz
    sample_data = {
        "text": [
            "Bu film gerçekten harikaydı! Çok beğendim.",
            "Berbat bir film, hiç beğenmedim.",
            "Ortalama bir yapım, fena değil.",
            "Muhteşem oyunculuk ve senaryo!",
            "Sıkıcı ve anlamsız bir film.",
            "Güzel bir aile filmi, tavsiye ederim.",
            "Vakit kaybı, izlemeyin.",
            "Etkileyici görsel efektler ve müzik.",
            "Hayal kırıklığı yaratan bir devam filmi.",
            "Mükemmel yönetim ve karakterler."
        ],
        "label": [1, 0, 1, 1, 0, 1, 0, 1, 0, 1]  # 1: pozitif, 0: negatif
    }
    
    # Dataset.from_dict(): Dictionary'den Dataset objesi oluşturur
    # Alternatifler:
    # - Dataset.from_pandas(): Pandas DataFrame'den
    # - Dataset.from_csv(): CSV dosyasından
    # - Dataset.from_json(): JSON dosyasından
    dataset = Dataset.from_dict(sample_data)
    
    # VERİ BÖLÜMLEME (Train/Validation Split)
    # Best Practice: %80 train, %20 validation oranı
    # Burada küçük veri seti için 8-2 split kullanıyoruz
    train_dataset = dataset.select(range(8))
    eval_dataset = dataset.select(range(8, 10))
    
    print(f"✓ Train örnekleri: {len(train_dataset)}")
    print(f"✓ Validation örnekleri: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset


# =============================================================================
# MODEL VE TOKENIZER KURULUM FONKSİYONLARI
# =============================================================================

def setup_tokenizer_and_model(model_name="distilbert-base-uncased"):
    """
    Model ve Tokenizer Kurulumu
    
    Bu fonksiyon, pre-trained bir model ve ilgili tokenizer'ı yükler.
    
    AUTOTOKENIZER VE AUTOMODEL HAKKINDA:
    - Auto* sınıfları: Model ismine göre otomatik olarak doğru sınıfı seçer
    - Pre-trained: Model, büyük veri setleri üzerinde önceden eğitilmiştir
    - Fine-tuning: Mevcut ağırlıkları kullanarak spesifik görev için uyarlama
    
    MODEL SEÇİMLERİ:
    - distilbert-base-uncased: Hızlı ve hafif (66M parametre)
    - bert-base-uncased: Standart BERT (110M parametre)
    - roberta-base: Geliştirilmiş BERT varyantı (125M parametre)
    - xlm-roberta-base: Çok dilli destek (270M parametre)
    
    TOKENİZER GÖREVİ:
    1. Metni küçük parçalara (token) böler
    2. Her tokeni sayısal ID'ye çevirir
    3. Özel tokenler ekler ([CLS], [SEP], [PAD])
    4. Attention mask oluşturur
    
    Args:
        model_name (str): Hugging Face Hub'dan yüklenecek model adı
        
    Returns:
        tuple: (tokenizer, model) - Tokenizer ve model nesneleri
    """
    print(f"🤖 Model yükleniyor: {model_name}")
    
    # TOKENIZER YÜKLEME
    # from_pretrained(): Hub'dan veya local'den model yükler
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # MODEL YÜKLEME
    # AutoModelForSequenceClassification: Metin sınıflandırma için
    # num_labels: Çıkış sınıf sayısı (binary classification için 2)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2,  # Binary classification: pozitif/negatif
        # İsteğe bağlı parametreler:
        # hidden_dropout_prob=0.1,    # Dropout oranı
        # attention_probs_dropout_prob=0.1,  # Attention dropout
        # classifier_dropout=0.1      # Sınıflandırıcı dropout
    )
    
    print(f"✓ Model yüklendi: {model.config.model_type}")
    print(f"✓ Parametre sayısı: {model.num_parameters():,}")
    
    return tokenizer, model


# =============================================================================
# TOKENİZASYON FONKSİYONLARI
# =============================================================================

def tokenize_dataset(dataset, tokenizer, max_length=128):
    """
    Dataset Tokenization (Metni Sayısal Veriye Dönüştürme)
    
    Bu fonksiyon, tüm veri setini tokenize eder ve model için hazır hale getirir.
    
    TOKENİZASYON SÜRECİ:
    1. Metin → Token'lara bölme (kelime/alt-kelime parçaları)
    2. Token → ID dönüşümü (vocabulary kullanarak)
    3. Padding ve Truncation uygulama
    4. Attention mask oluşturma
    
    ÖRNEK TOKENİZASYON:
    Metin: "Bu harika bir film!"
    Token'lar: ["bu", "harika", "bir", "film", "!"]
    ID'ler: [101, 2023, 5919, 2019, 2143, 999, 102]
    
    PADDING STRATEJİLERİ:
    - padding=False: Batch sırasında dinamik padding (önerilen)
    - padding=True: Her örneği hemen max_length'e getir
    - padding="max_length": Sabit uzunluk padding
    
    TRUNCATION:
    - truncation=True: max_length'i aşan metinleri kes
    - truncation=False: Uzun metinler hata verir
    
    Args:
        dataset: Tokenize edilecek Dataset objesi
        tokenizer: Kullanılacak tokenizer
        max_length (int): Maksimum token uzunluğu
        
    Returns:
        Dataset: Tokenize edilmiş dataset
    """
    
    def tokenize_function(examples):
        """
        Batch tokenization için yardımcı fonksiyon
        
        Bu fonksiyon, dataset.map() tarafından her batch için çağrılır.
        """
        # Tokenization
        tokenized = tokenizer(
            examples["text"],           # Tokenize edilecek metinler
            truncation=True,            # Uzun metinleri kes
            padding=False,              # Dynamic padding kullanacağız (DataCollator ile)
            max_length=max_length,      # Maksimum uzunluk
            # İsteğe bağlı parametreler:
            # add_special_tokens=True,  # [CLS], [SEP] tokenlerini ekle
            # return_attention_mask=True,  # Attention mask döndür
            # return_token_type_ids=True,  # Segment ID'leri (BERT için)
        )
        # Label'ı labels olarak adlandır (Trainer'ın beklediği format)
        tokenized["labels"] = examples["label"]
        return tokenized
    
    # BATCH TOKENIZATION
    # map(): Her örneğe veya batch'e fonksiyon uygular
    # batched=True: Verimliliği artırır, tüm örnekleri tek tek işlemekten daha hızlı
    # remove_columns: Tokenization sonrası orijinal sütunları kaldır
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,               # Batch processing (daha hızlı)
        remove_columns=["text", "label"],  # Orijinal text ve label sütunlarını kaldır
        desc="Tokenizing dataset"   # Progress bar açıklaması
    )
    
    print(f"✓ Dataset tokenize edildi: {len(tokenized_dataset)} örnek")
    
    return tokenized_dataset


# =============================================================================
# METRİK HESAPLAMA FONKSİYONLARI
# =============================================================================

def compute_metrics(eval_pred):
    """
    Evaluation Metrics Hesaplama
    
    Bu fonksiyon, model performansını değerlendirmek için çeşitli metrikler hesaplar.
    Trainer tarafından her evaluation adımında otomatik olarak çağrılır.
    
    METRİKLER HAKKINDA:
    
    1. ACCURACY (Doğruluk):
       - Doğru tahmin edilen örneklerin oranı
       - Formül: (TP + TN) / Toplam
       - İyi: Dengeli veri setleri için
       - Kötü: Dengesiz veri setleri için yanıltıcı olabilir
    
    2. PRECISION (Kesinlik):
       - Pozitif dediğimizin gerçekten pozitif olma oranı
       - Formül: TP / (TP + FP)
       - Örnek: 100 spam emailden 90'ı gerçekten spam
    
    3. RECALL (Duyarlılık / Hatırlama):
       - Tüm pozitiflerin ne kadarını yakaladık
       - Formül: TP / (TP + FN)
       - Örnek: Tüm spam emaillerden %95'ini yakaladık
    
    4. F1-SCORE:
       - Precision ve Recall'un harmonik ortalaması
       - Formül: 2 * (Precision * Recall) / (Precision + Recall)
       - Dengeli bir metrik
    
    WEIGHTED AVERAGE:
    - Her sınıfın metriği, o sınıftaki örnek sayısına göre ağırlıklandırılır
    - Dengesiz veri setleri için daha adil bir değerlendirme
    
    Args:
        eval_pred: (predictions, labels) tuple'ı
        
    Returns:
        dict: Hesaplanan metrikler
    """
    predictions, labels = eval_pred
    
    # SOFTMAX ÇIKIŞINDAN TAHMİN ALMA
    # Model çıkışı: [batch_size, num_classes] şeklinde logit değerleri
    # argmax ile en yüksek olasılıklı sınıfı seçiyoruz
    predictions = np.argmax(predictions, axis=1)
    
    # PRECISION, RECALL, F1 HESAPLAMA
    # average='weighted': Sınıf dengesizliğini dikkate al
    # Alternatifler:
    # - 'micro': Tüm örnekleri eşit ağırlıkta say
    # - 'macro': Her sınıfa eşit ağırlık ver
    # - None: Her sınıf için ayrı metrik döndür
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, 
        predictions, 
        average='weighted',
        zero_division=0  # Bölme hatalarını önle
    )
    
    # ACCURACY HESAPLAMA
    accuracy = accuracy_score(labels, predictions)
    
    # METRİKLERİ DICTIONARY OLARAK DÖNDÜR
    # Bu metrikler Trainer tarafından loglanır ve görselleştirilir
    return {
        'accuracy': accuracy,      # Genel doğruluk
        'f1': f1,                  # F1 skoru
        'precision': precision,    # Kesinlik
        'recall': recall          # Duyarlılık
    }


# =============================================================================
# EĞİTİM PARAMETRELERI KONFIGÜRASYONU
# =============================================================================

def setup_training_arguments():
    """
    Training Arguments Konfigürasyonu
    
    TrainingArguments, Trainer'ın tüm eğitim davranışını kontrol eder.
    Bu parametreler model performansını ve eğitim süresini doğrudan etkiler.
    
    ÖNEMLİ PARAMETRELER VE AÇIKLAMALARI:
    
    1. ÇIKTI VE KAYIT:
       - output_dir: Model checkpoint'lerinin kaydedileceği klasör
       - logging_dir: TensorBoard/WandB loglarının kaydedileceği yer
       - logging_steps: Kaç adımda bir log kaydedilecek
    
    2. EĞİTİM DÖNGÜSÜ:
       - num_train_epochs: Kaç epoch eğitim yapılacak
       - per_device_train_batch_size: Her GPU/CPU'da batch boyutu
       - gradient_accumulation_steps: Efektif batch boyutunu artırır
    
    3. LEARNING RATE:
       - learning_rate: Varsayılan 5e-5 (çoğu BERT modeli için uygun)
       - warmup_steps: Başlangıçta LR'yi kademeli artır (stabilite için)
       - lr_scheduler_type: LR azaltma stratejisi (linear, cosine, etc.)
    
    4. REGÜLARİZASYON:
       - weight_decay: L2 regularization (overfitting önler)
       - max_grad_norm: Gradient clipping (training stability)
    
    5. EVALUATION STRATEJİSİ:
       - eval_strategy: Ne zaman evaluate edileceği
         * "no": Hiç evaluate etme
         * "steps": Her N adımda bir
         * "epoch": Her epoch sonunda
       - eval_steps: steps stratejisinde kaç adımda bir
    
    6. KAYIT STRATEJİSİ:
       - save_strategy: Model kaydedme stratejisi
       - save_steps: Kaç adımda bir kaydet
       - save_total_limit: Maksimum checkpoint sayısı
       - load_best_model_at_end: En iyi modeli eğitim sonunda yükle
    
    7. OPTİMİZASYON:
       - fp16: 16-bit floating point (2x hızlanma, %50 RAM tasarrufu)
       - dataloader_num_workers: Paralel veri yükleme
       - gradient_checkpointing: RAM tasarrufu (ama yavaşlatır)
    
    Returns:
        TrainingArguments: Eğitim parametreleri objesi
    """
    training_args = TrainingArguments(
        # ÇIKTI AYARLARI
        output_dir='./results',              # Checkpoint kayıt klasörü
        logging_dir='./logs',                # Log dosyaları klasörü
        logging_steps=10,                    # Her 10 adımda log kaydet
        
        # EĞİTİM HİPERPARAMETRELERİ
        num_train_epochs=3,                  # 3 epoch eğitim
        per_device_train_batch_size=8,       # Her cihazda 8 örnek/batch
        per_device_eval_batch_size=8,        # Evaluation için 8 örnek/batch
        
        # LEARNING RATE VE OPTİMİZASYON
        learning_rate=5e-5,                  # Varsayılan BERT learning rate
        warmup_steps=50,                     # İlk 50 adımda warmup
        weight_decay=0.01,                   # L2 regularization katsayısı
        
        # EVALUATION VE KAYIT STRATEJİSİ
        eval_strategy="epoch",               # Her epoch sonunda evaluate et
        save_strategy="epoch",               # Her epoch sonunda kaydet
        load_best_model_at_end=True,         # En iyi modeli eğitim sonunda yükle
        metric_for_best_model="accuracy",    # Hangi metrik için "en iyi"
        greater_is_better=True,              # Yüksek accuracy daha iyi
        
        # DİĞER AYARLAR
        report_to=None,                      # WandB/TensorBoard kapalı
        seed=42,                             # Reproducibility için sabit seed
        
        # GELİŞMİŞ OPTİMİZASYON (isteğe bağlı)
        # fp16=True,                         # Mixed precision training (GPU varsa)
        # gradient_accumulation_steps=2,     # Efektif batch size = 8 * 2 = 16
        # max_grad_norm=1.0,                 # Gradient clipping
        # dataloader_num_workers=4,          # Paralel veri yükleme
    )
    
    print("✓ Training arguments hazırlandı")
    
    return training_args


# =============================================================================
# MODEL EĞİTİM FONKSİYONU - ANA PİPELINE
# =============================================================================

def train_model_with_trainer():
    """
    Trainer API Kullanarak Model Eğitimi
    
    Bu fonksiyon, modern NLP model eğitiminin tüm adımlarını içerir.
    Trainer API, karmaşık eğitim döngülerini otomatikleştirir.
    
    TRAINER API'NİN AVANTAJLARI:
    - Otomatik gradient hesaplama ve backpropagation
    - GPU/TPU/CPU otomatik yönetimi
    - Checkpoint kaydetme ve yükleme
    - Early stopping desteği
    - Distributed training (çoklu GPU) desteği
    - Mixed precision training
    - Gradient accumulation
    - Logging ve monitoring entegrasyonu
    
    EĞİTİM PİPELINE ADIMLARI:
    1. Dataset yükleme ve hazırlama
    2. Model ve tokenizer kurulumu
    3. Dataset tokenization
    4. Data collator oluşturma (dynamic padding için)
    5. Training arguments tanımlama
    6. Trainer objesi oluşturma
    7. Model eğitimi
    8. Model değerlendirme
    9. Model kaydetme
    
    Returns:
        tuple: (trainer, model, tokenizer)
    """
    print("\n" + "="*80)
    print("🚀 MODEL EĞİTİMİ BAŞLIYOR")
    print("="*80 + "\n")
    
    # -------------------------------------------------------------------------
    # ADIM 1: DATASET HAZIRLAMA
    # -------------------------------------------------------------------------
    print("📊 ADIM 1: Dataset Hazırlama")
    train_dataset, eval_dataset = load_and_prepare_dataset()
    print(f"   └─ Train dataset: {len(train_dataset)} örnek")
    print(f"   └─ Eval dataset: {len(eval_dataset)} örnek\n")
    
    # -------------------------------------------------------------------------
    # ADIM 2: MODEL VE TOKENIZER YÜKLEME
    # -------------------------------------------------------------------------
    print("🤖 ADIM 2: Model ve Tokenizer Yükleme")
    tokenizer, model = setup_tokenizer_and_model()
    print()
    
    # -------------------------------------------------------------------------
    # ADIM 3: DATASET TOKENİZASYONU
    # -------------------------------------------------------------------------
    print("🔤 ADIM 3: Dataset Tokenization")
    train_tokenized = tokenize_dataset(train_dataset, tokenizer)
    eval_tokenized = tokenize_dataset(eval_dataset, tokenizer)
    print()
    
    # -------------------------------------------------------------------------
    # ADIM 4: DATA COLLATOR OLUŞTURMA
    # -------------------------------------------------------------------------
    print("📦 ADIM 4: Data Collator Hazırlama")
    # DataCollatorWithPadding: Batch oluştururken dinamik padding uygular
    # Avantaj: Her batch'teki en uzun örneğe göre padding yapar (RAM tasarrufu)
    # Alternatif: DataCollatorForLanguageModeling (LM için)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    print("   └─ Dynamic padding aktif\n")
    
    # -------------------------------------------------------------------------
    # ADIM 5: TRAINING ARGUMENTS
    # -------------------------------------------------------------------------
    print("⚙️  ADIM 5: Training Arguments Konfigürasyonu")
    training_args = setup_training_arguments()
    print()
    
    # -------------------------------------------------------------------------
    # ADIM 6: TRAINER OLUŞTURMA
    # -------------------------------------------------------------------------
    print("🎯 ADIM 6: Trainer Oluşturma")
    trainer = Trainer(
        model=model,                          # Eğitilecek model
        args=training_args,                   # Eğitim parametreleri
        train_dataset=train_tokenized,        # Eğitim veri seti
        eval_dataset=eval_tokenized,          # Değerlendirme veri seti
        tokenizer=tokenizer,                  # Tokenizer (kaydetme için gerekli)
        data_collator=data_collator,          # Batch oluşturucu
        compute_metrics=compute_metrics,      # Metrik hesaplama fonksiyonu
        
        # İsteğe bağlı callback'ler:
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        # optimizers=(optimizer, scheduler),  # Custom optimizer/scheduler
    )
    print("   └─ Trainer hazır\n")
    
    # -------------------------------------------------------------------------
    # ADIM 7: MODEL EĞİTİMİ
    # -------------------------------------------------------------------------
    print("🏋️  ADIM 7: Model Eğitimi Başlıyor...")
    print("-" * 80)
    
    # train() methodu tüm eğitim döngüsünü yönetir:
    # - Forward pass
    # - Loss hesaplama
    # - Backward pass (gradient hesaplama)
    # - Optimizer step (ağırlık güncelleme)
    # - Logging
    # - Evaluation (belirtilen aralıklarla)
    # - Checkpoint kaydetme
    train_result = trainer.train()
    
    print("-" * 80)
    print("✓ Eğitim tamamlandı!\n")
    
    # -------------------------------------------------------------------------
    # ADIM 8: FINAL EVALUATION
    # -------------------------------------------------------------------------
    print("📈 ADIM 8: Final Evaluation")
    eval_results = trainer.evaluate()
    
    print("\n📊 Evaluation Sonuçları:")
    print("-" * 40)
    for metric, value in eval_results.items():
        print(f"   {metric:20s}: {value:.4f}")
    print("-" * 40 + "\n")
    
    # -------------------------------------------------------------------------
    # ADIM 9: MODEL KAYDETME
    # -------------------------------------------------------------------------
    print("💾 ADIM 9: Model Kaydetme")
    save_path = "./fine_tuned_model"
    
    # Model ve tokenizer'ı kaydet
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    
    print(f"   └─ Model kaydedildi: {save_path}")
    print(f"   └─ Dosyalar: config.json, pytorch_model.bin, tokenizer.json\n")
    
    print("="*80)
    print("✅ TÜM İŞLEMLER BAŞARIYLA TAMAMLANDI!")
    print("="*80 + "\n")
    
    return trainer, model, tokenizer


# =============================================================================
# DATASET OPERASYONLARI DEMO FONKSİYONU
# =============================================================================

def demonstrate_dataset_operations():
    """
    Datasets Kütüphanesi - Gelişmiş Operasyonlar
    
    Bu fonksiyon, Hugging Face Datasets kütüphanesinin güçlü özelliklerini gösterir.
    
    DATASETS KÜTÜPHANESİNİN ÖZELLİKLERİ:
    
    1. VERİMLİ VERİ YÖNETİMİ:
       - Apache Arrow formatı: Hızlı okuma/yazma
       - Memory mapping: RAM'e sığmayan büyük veri setleri
       - Lazy loading: İhtiyaç duyulan veri yüklenir
    
    2. DÖNÜŞÜM OPERASYONLARİ:
       - map(): Her örneğe veya batch'e fonksiyon uygula
       - filter(): Belirli koşullara göre filtrele
       - select(): Index ile seçim yap
       - sort(): Sıralama
       - shuffle(): Karıştırma
    
    3. VERİ BİRLEŞTİRME:
       - concatenate(): Veri setlerini birleştir
       - interleave(): Veri setlerini iç içe geçir
    
    4. BATCH PROCESSING:
       - batched=True: Verimliliği artırır
       - num_proc: Paralel işleme
    
    5. CACHING:
       - Otomatik caching: İşlenen veri cache'lenir
       - load_from_cache_file: Cache'den yükle
    """
    print("\n" + "="*80)
    print("📚 DATASETS KÜTÜPHANESİ - GELİŞMİŞ OPERASYONLAR")
    print("="*80 + "\n")
    
    # Dataset yükleme
    train_dataset, eval_dataset = load_and_prepare_dataset()
    
    # -------------------------------------------------------------------------
    # DATASET ÖZELLİKLERİ
    # -------------------------------------------------------------------------
    print("🔍 Dataset Özellikleri:")
    print(f"   └─ Columns: {train_dataset.column_names}")
    print(f"   └─ Features: {train_dataset.features}")
    print(f"   └─ Toplam örnek sayısı: {len(train_dataset)}")
    print(f"   └─ İlk örnek: {train_dataset[0]}\n")
    
    # -------------------------------------------------------------------------
    # FILTER OPERASYONU
    # -------------------------------------------------------------------------
    print("🔎 Filter Operasyonu - Pozitif Örnekleri Seç:")
    # filter(): Lambda fonksiyonu ile veri filtreleme
    # Her örnek için True/False döndürür, True olanları tutar
    positive_samples = train_dataset.filter(lambda x: x["label"] == 1)
    negative_samples = train_dataset.filter(lambda x: x["label"] == 0)
    
    print(f"   └─ Pozitif örnekler: {len(positive_samples)}")
    print(f"   └─ Negatif örnekler: {len(negative_samples)}\n")
    
    # -------------------------------------------------------------------------
    # MAP OPERASYONU - TEK ÖRNEK
    # -------------------------------------------------------------------------
    print("🔧 Map Operasyonu - Yeni Sütun Ekleme:")
    
    def add_length(example):
        """Her örneğe metin uzunluğu ekle"""
        example["text_length"] = len(example["text"])
        return example
    
    # map() ile her örneğe fonksiyon uygula
    dataset_with_length = train_dataset.map(add_length)
    print(f"   └─ Yeni sütun eklendi: 'text_length'")
    print(f"   └─ Örnek: {dataset_with_length[0]}\n")
    
    # -------------------------------------------------------------------------
    # MAP OPERASYONU - BATCH PROCESSİNG
    # -------------------------------------------------------------------------
    print("⚡ Batch Processing - Toplu İşleme:")
    
    def uppercase_text(batch):
        """Tüm metinleri büyük harfe çevir (batch modunda)"""
        # batch: dictionary, her key bir liste içerir
        batch["text"] = [text.upper() for text in batch["text"]]
        return batch
    
    # batched=True: Tek tek değil, batch halinde işler (çok daha hızlı!)
    uppercase_dataset = train_dataset.map(uppercase_text, batched=True)
    print(f"   └─ Metinler büyük harfe çevrildi")
    print(f"   └─ Örnek: {uppercase_dataset[0]['text']}\n")
    
    # -------------------------------------------------------------------------
    # SELECT ve SHUffle OPERASYONLARI
    # -------------------------------------------------------------------------
    print("🎲 Select ve Shuffle Operasyonları:")
    
    # shuffle(): Veriyi karıştır
    shuffled_dataset = train_dataset.shuffle(seed=42)
    print(f"   └─ Dataset karıştırıldı (reproducibility için seed=42)")
    
    # select(): Index ile seçim
    subset = train_dataset.select([0, 2, 4, 6])
    print(f"   └─ Alt küme seçildi: {len(subset)} örnek\n")
    
    # -------------------------------------------------------------------------
    # GELIŞMIŞ ÖZELLIKLER
    # -------------------------------------------------------------------------
    print("🚀 Gelişmiş Özellikler:")
    print("   ├─ num_proc: Paralel işleme (çoklu CPU core)")
    print("   ├─ remove_columns: İstenmeyen sütunları kaldır")
    print("   ├─ load_from_cache_file: Cache kullanımı (hız için)")
    print("   └─ keep_in_memory: Tüm veriyi RAM'de tut\n")
    
    # Paralel işleme örneği (uncomment to use)
    # dataset_parallel = train_dataset.map(
    #     add_length, 
    #     num_proc=4,  # 4 core kullan
    #     desc="Processing with 4 cores"
    # )
    
    print("="*80 + "\n")


# =============================================================================
# EĞİTİM STRATEJİLERİ DEMO FONKSİYONU
# =============================================================================

def demonstrate_training_strategies():
    """
    Training Strategies - Eğitim Stratejileri
    
    Bu fonksiyon, farklı eğitim stratejilerini ve TrainingArguments'in
    önemli parametrelerini açıklar.
    
    EĞİTİM STRATEJİLERİ:
    
    1. EVALUATION STRATEGY:
       - Modelin ne sıklıkla değerlendirileceğini belirler
       - Overfitting'i erken tespit etmek için önemli
    
    2. SAVE STRATEGY:
       - Checkpoint kaydetme sıklığı
       - Disk alanı ve eğitim süresi dengesi
    
    3. EARLY STOPPING:
       - Performans artmayı bırakınca eğitimi durdur
       - Zaman ve kaynak tasarrufu
    
    4. LEARNING RATE SCHEDULING:
       - Eğitim boyunca LR'yi ayarla
       - Daha iyi konverjans için
    """
    print("\n" + "="*80)
    print("📊 EĞİTİM STRATEJİLERİ VE BEST PRACTICES")
    print("="*80 + "\n")
    
    # -------------------------------------------------------------------------
    # EVALUATION STRATEJİLERİ
    # -------------------------------------------------------------------------
    print("🎯 Evaluation Stratejileri:")
    print("-" * 40)
    
    strategies = {
        "no": "Değerlendirme yapılmaz (sadece eğitim)",
        "epoch": "Her epoch sonunda değerlendir (önerilen)",
        "steps": "Belirli step sayısında değerlendir (büyük veri setleri için)"
    }
    
    for strategy, description in strategies.items():
        print(f"   └─ '{strategy}': {description}")
    
    print("\n   💡 Öneri: Küçük veri setleri için 'epoch', büyük veri için 'steps'")
    print()
    
    # -------------------------------------------------------------------------
    # SAVE STRATEJİLERİ
    # -------------------------------------------------------------------------
    print("💾 Save Stratejileri:")
    print("-" * 40)
    print("   ├─ save_strategy='no': Checkpoint kaydetme")
    print("   ├─ save_strategy='epoch': Her epoch'ta kaydet")
    print("   ├─ save_strategy='steps': Her N adımda kaydet")
    print("   └─ save_total_limit: Maksimum checkpoint sayısı (disk tasarrufu)")
    print()
    
    # -------------------------------------------------------------------------
    # EARLY STOPPING
    # -------------------------------------------------------------------------
    print("⏹️  Early Stopping:")
    print("-" * 40)
    print("   Kullanımı:")
    print("   ```python")
    print("   from transformers import EarlyStoppingCallback")
    print("   ")
    print("   trainer = Trainer(")
    print("       ...,")
    print("       callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]")
    print("   )")
    print("   ```")
    print("   └─ patience=3: 3 epoch boyunca gelişme yoksa dur")
    print()
    
    # -------------------------------------------------------------------------
    # TRAININGARGUMENTS ÖNEMLİ PARAMETRELER
    # -------------------------------------------------------------------------
    print("⚙️  TrainingArguments - Önemli Parametreler:")
    print("="*80)
    
    params = [
        ("learning_rate", "5e-5", "Öğrenme oranı (BERT için önerilen)"),
        ("weight_decay", "0.01", "L2 regularization (overfitting önler)"),
        ("warmup_steps", "500", "LR warmup (başlangıç stabilitesi)"),
        ("warmup_ratio", "0.1", "Warmup oranı (steps yerine kullanılabilir)"),
        ("gradient_accumulation_steps", "2", "Efektif batch size artırma"),
        ("fp16", "True", "16-bit precision (2x hızlanma, GPU gerekir)"),
        ("max_grad_norm", "1.0", "Gradient clipping (training stability)"),
        ("lr_scheduler_type", "'linear'", "LR azaltma stratejisi"),
        ("dataloader_num_workers", "4", "Paralel veri yükleme"),
        ("logging_first_step", "True", "İlk adımı da logla"),
        ("load_best_model_at_end", "True", "En iyi modeli eğitim sonunda yükle"),
        ("metric_for_best_model", "'eval_loss'", "En iyi model için metrik"),
    ]
    
    for param, default, desc in params:
        print(f"   {param:30s} = {default:15s}  # {desc}")
    
    print("="*80 + "\n")
    
    # -------------------------------------------------------------------------
    # LR SCHEDULER TİPLERİ
    # -------------------------------------------------------------------------
    print("📈 Learning Rate Scheduler Tipleri:")
    print("-" * 40)
    
    schedulers = {
        "linear": "Lineer azalma (varsayılan, çoğu durumda iyi)",
        "cosine": "Cosine azalma (yumuşak azalma)",
        "cosine_with_restarts": "Periyodik cosine (uzun eğitimler için)",
        "polynomial": "Polinom azalma",
        "constant": "Sabit LR (warmup sonrası)",
        "constant_with_warmup": "Warmup sonrası sabit"
    }
    
    for scheduler, description in schedulers.items():
        print(f"   └─ {scheduler:25s}: {description}")
    
    print()
    
    # -------------------------------------------------------------------------
    # BATCH SIZE OPTİMİZASYONU
    # -------------------------------------------------------------------------
    print("📦 Batch Size Optimizasyonu:")
    print("-" * 40)
    print("   Gradient Accumulation ile efektif batch size artırma:")
    print()
    print("   per_device_train_batch_size = 8")
    print("   gradient_accumulation_steps = 4")
    print("   num_gpus = 2")
    print("   → Efektif batch size = 8 × 4 × 2 = 64")
    print()
    print("   💡 Avantaj: GPU memory'ye sığmayan büyük batch'ler kullanabilirsiniz")
    print()
    
    # -------------------------------------------------------------------------
    # MIXED PRECISION TRAINING
    # -------------------------------------------------------------------------
    print("⚡ Mixed Precision Training (FP16/BF16):")
    print("-" * 40)
    print("   FP16 (Float16):")
    print("   ├─ 2x hızlanma")
    print("   ├─ %50 RAM tasarrufu")
    print("   └─ Modern GPU gerekir (V100, A100, RTX series)")
    print()
    print("   BF16 (BFloat16):")
    print("   ├─ FP16'ya benzer performans")
    print("   ├─ Daha geniş dinamik aralık")
    print("   └─ A100, TPU v3/v4 için önerilen")
    print()
    
    # -------------------------------------------------------------------------
    # BEST PRACTICES
    # -------------------------------------------------------------------------
    print("✨ Best Practices - En İyi Uygulamalar:")
    print("="*80)
    print("   1. Küçük learning rate ile başla (5e-5 veya 3e-5)")
    print("   2. Warmup kullan (total_steps'in %10'u)")
    print("   3. Weight decay uygula (0.01)")
    print("   4. Gradient clipping kullan (max_grad_norm=1.0)")
    print("   5. Evaluation strategy ayarla (overfitting kontrolü)")
    print("   6. Best model kaydetme aktif et (load_best_model_at_end=True)")
    print("   7. Seed sabitle (reproducibility için)")
    print("   8. FP16/BF16 kullan (GPU varsa)")
    print("   9. Logging ve monitoring ekle (WandB, TensorBoard)")
    print("   10. Early stopping kullan (gereksiz eğitimi önle)")
    print("="*80 + "\n")


# =============================================================================
# ANA PROGRAM - MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    """
    Ana Program Akışı
    
    Bu bölüm, script'in tüm özelliklerini sırasıyla gösterir.
    Eğitimi çalıştırmak için ilgili satırların comment'ini kaldırın.
    
    KULLANIM ÖRNEKLERİ:
    
    1. Sadece demo'ları çalıştır:
       python 2_datasets_trainer.py
    
    2. Model eğitimi için (uncomment gerekli):
       - train_model_with_trainer() satırının comment'ini kaldır
       - GPU varsa fp16=True ekle (hızlanma için)
    
    3. Checkpoint'ten devam etme:
       trainer.train(resume_from_checkpoint="./results/checkpoint-xxx")
    """
    
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "  HUGGING FACE DATASETS & TRAINER API - KAPSAMLI EĞİTİM  ".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    
    # -------------------------------------------------------------------------
    # BÖLÜM 1: DATASET OPERASYONLARI
    # -------------------------------------------------------------------------
    demonstrate_dataset_operations()
    
    # -------------------------------------------------------------------------
    # BÖLÜM 2: EĞİTİM STRATEJİLERİ
    # -------------------------------------------------------------------------
    demonstrate_training_strategies()
    
    # -------------------------------------------------------------------------
    # BÖLÜM 3: MODEL EĞİTİMİ (İSTEĞE BAĞLI)
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("🎓 MODEL EĞİTİMİ")
    print("="*80)
    print()
    print("   Model eğitimini çalıştırmak için aşağıdaki satırın comment'ini kaldırın:")
    print("   → trainer, model, tokenizer = train_model_with_trainer()")
    print()
    print("   ⚠️  Not: Eğitim birkaç dakika sürebilir ve GPU kullanımını önerilir.")
    print()
    
    # Model eğitimi (uncomment to run / Çalıştırmak için comment'i kaldır)
    trainer, model, tokenizer = train_model_with_trainer()
    
    # Eğitimden sonra inference örneği
    print("\n=== Inference Örneği ===")
    test_text = "Bu muhteşem bir film, çok beğendim!"
    inputs = tokenizer(test_text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=-1)
    sentiment = "POZİTİF" if prediction.item() == 1 else "NEGATİF"
    print(f"Metin: {test_text}")
    print(f"Tahmin: {sentiment}")
    
    # -------------------------------------------------------------------------
    # SONUÇ
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("✅ DEMO TAMAMLANDI!")
    print("="*80)
    print()
    print("📚 ÖĞRENİLEN KONULAR:")
    print("   ✓ Hugging Face Datasets kütüphanesi kullanımı")
    print("   ✓ Dataset yükleme, hazırlama ve dönüşümleri")
    print("   ✓ Tokenization ve data collation")
    print("   ✓ Trainer API ile otomatik eğitim")
    print("   ✓ Training arguments ve hiperparametre ayarlama")
    print("   ✓ Metric hesaplama ve model değerlendirme")
    print("   ✓ Model kaydetme ve yükleme")
    print()
    print("🚀 SONRAKI ADIMLAR:")
    print("   1. Kendi veri setiniz ile deneyin")
    print("   2. Farklı modeller test edin (BERT, RoBERTa, etc.)")
    print("   3. Hiperparametre optimizasyonu yapın")
    print("   4. WandB/TensorBoard ile monitoring ekleyin")
    print("   5. Production deployment için model optimize edin")
    print()
    print("="*80 + "\n")


# =============================================================================
# EK NOTLAR VE KAYNAKLAR
# =============================================================================

"""
FAYDALΙ KAYNAKLAR:

1. Hugging Face Datasets Dokümantasyonu:
   https://huggingface.co/docs/datasets/

2. Transformers Trainer Guide:
   https://huggingface.co/docs/transformers/main_classes/trainer

3. Training Arguments Full List:
   https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments

4. Model Hub:
   https://huggingface.co/models

5. Dataset Hub:
   https://huggingface.co/datasets


SΙKÇA SORULAN SORULAR (FAQ):

Q: GPU olmadan eğitim yapabilir miyim?
A: Evet, ancak CPU'da çok daha yavaş olacaktır. Küçük modeller ve veri setleri 
   ile başlayın.

Q: Out of memory hatası alıyorum, ne yapmalıyım?
A: 1) Batch size'ı küçült
   2) Gradient accumulation kullan
   3) FP16 kullan
   4) Daha küçük model seç

Q: En iyi hiperparametreleri nasıl bulurum?
A: Hyperparameter search kullanın (Optuna, Ray Tune ile entegre)
   https://huggingface.co/docs/transformers/hpo_train

Q: Çoklu GPU'da eğitim nasıl yapılır?
A: `python -m torch.distributed.launch --nproc_per_node=NUM_GPUS script.py`
   veya `accelerate launch script.py`

Q: Custom loss function nasıl kullanırım?
A: Trainer'ı subclass yapın ve compute_loss() methodunu override edin.


PERFORMANS OPTİMİZASYON İPUÇLARI:

1. Veri Yükleme:
   - dataloader_num_workers > 0 (paralel yükleme)
   - pin_memory=True (GPU için)
   - prefetch_factor > 1 (önceden yükleme)

2. Model Optimizasyonu:
   - gradient_checkpointing: RAM tasarrufu
   - fp16/bf16: Hızlanma
   - torch.compile(): PyTorch 2.0+ hızlanma

3. Batch Size:
   - Mümkün olan en büyük batch size kullanın
   - Gradient accumulation ile efektif size artırın

4. Learning Rate:
   - Batch size ile birlikte scale edin
   - Linear warmup kullanın
   - Learning rate finder kullanın

5. Distributed Training:
   - DDP (DistributedDataParallel)
   - FSDP (Fully Sharded Data Parallel) büyük modeller için
   - DeepSpeed entegrasyonu


PRODUCTION DEPLOYMENT:

1. Model Kaydetme:
   - trainer.save_model() veya model.save_pretrained()
   - Tokenizer'ı da kaydetmeyi unutmayın

2. Model Boyutu Küçültme:
   - Quantization (int8, int4)
   - Pruning (ağırlık budama)
   - Distillation (öğretmen-öğrenci)

3. Inference Optimizasyonu:
   - ONNX export
   - TensorRT
   - TorchScript

4. API Servisi:
   - FastAPI ile REST API
   - Triton Inference Server
   - HuggingFace Inference API


GÜVENLİK VE ETİK HUSUSLAR:

- Veri gizliliği: Hassas veriyi model eğitiminde kullanmayın
- Bias kontrolü: Model çıktılarını adalet açısından değerlendirin
- Model kötüye kullanımı: Responsible AI prensiplerini takip edin
- Lisanslama: Model ve veri lisanslarına dikkat edin


Son Güncelleme: Kasım 2025
Kairu AI Bootcamp - Hafta 6
"""
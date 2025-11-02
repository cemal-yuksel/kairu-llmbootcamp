"""
================================================================================
3. LORA SUMMARIZER TRAINING - Intelligent Review Summarizer
================================================================================

LoRA ile BART/T5 modelini summarization görevi için fine-tune eder.

Bu script:
1. Summarization data yükler
2. BART/T5 base model yükler
3. LoRA configuration uygular
4. Model'i eğitir
5. Fine-tuned modeli kaydeder
6. Evaluation metrikleri hesaplar

KULLANIM:
---------
python 3_lora_summarizer_training.py

veya

python 3_lora_summarizer_training.py --epochs 5 --batch-size 8

Yazar: Kairu AI - Build with LLMs Bootcamp
Tarih: 2 Kasım 2025
================================================================================
"""

import sys
import os
from pathlib import Path
import argparse
import json
from loguru import logger
import torch
from tqdm import tqdm

# Wandb'yi devre dışı bırak
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

# Proje root'una path ekle
sys.path.append(str(Path(__file__).parent))

from config import config
from utils.metrics import MetricsCalculator


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging():
    """Logging yapılandırması"""
    logger.remove()
    logger.add(
        sys.stderr,
        format=config.LOG_FORMAT,
        level=config.LOG_LEVEL,
        colorize=True
    )
    logger.add(
        config.LOG_FILE,
        format=config.LOG_FORMAT,
        level=config.LOG_LEVEL,
        rotation="10 MB"
    )


# ============================================================================
# DATA LOADING
# ============================================================================

def load_summarization_data():
    """
    Summarization datasını yükle
    
    Returns:
        (train_data, test_data) tuple
    """
    logger.info("=" * 80)
    logger.info("📥 SUMMARIZATION DATA YÜKLEME")
    logger.info("=" * 80)
    
    train_path = config.PROCESSED_DATA_DIR / "train.json"
    test_path = config.PROCESSED_DATA_DIR / "test.json"
    
    if not train_path.exists() or not test_path.exists():
        logger.error("❌ Data dosyaları bulunamadı!")
        logger.error("   Önce 1_data_preparation.py çalıştırın!")
        raise FileNotFoundError("train.json veya test.json bulunamadı")
    
    logger.info(f"📂 Train: {train_path}")
    logger.info(f"📂 Test: {test_path}")
    
    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    logger.info(f"✅ Veri yüklendi:")
    logger.info(f"  • Train: {len(train_data):,} samples")
    logger.info(f"  • Test: {len(test_data):,} samples")
    
    return train_data, test_data


def create_dataset_from_data(data, tokenizer, max_samples=None):
    """
    Dict data'dan Hugging Face Dataset oluştur
    
    Args:
        data: JSON data listesi
        tokenizer: Tokenizer
        max_samples: Maksimum sample sayısı
        
    Returns:
        Dataset
    """
    from datasets import Dataset
    
    if max_samples:
        data = data[:max_samples]
    
    # Dataset formatına çevir
    dataset_dict = {
        "text": [item["text"] for item in data],
        "summary": [item["summary"] for item in data],
        "label": [item["label"] for item in data]
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    
    return dataset


# ============================================================================
# MODEL SETUP
# ============================================================================

def load_model_and_tokenizer(model_name: str = None):
    """
    Base summarization modelini yükle
    
    Args:
        model_name: Model adı (None = config'den)
        
    Returns:
        (model, tokenizer) tuple
    """
    logger.info("\n" + "=" * 80)
    logger.info("🤖 MODEL VE TOKENIZER YÜKLEME")
    logger.info("=" * 80)
    
    if model_name is None:
        model_name = config.BASE_MODEL_NAME
    
    logger.info(f"📦 Model: {model_name}")
    
    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM
    )
    
    # Tokenizer
    logger.info("📝 Tokenizer yükleniyor...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info(f"  ✅ Vocab size: {len(tokenizer):,}")
    
    # Model
    logger.info("🧠 Model yükleniyor...")
    
    # CPU'da float16 sorunlu olabilir, float32 kullan
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = config.FP16 and device == "cuda"
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name
    )
    
    # Tüm parametreleri freeze et (LoRA sadece adapter'ları train edecek)
    for param in model.parameters():
        param.requires_grad = False
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  ✅ Model yüklendi!")
    logger.info(f"  📊 Total parameters: {total_params:,}")
    logger.info(f"  💾 Model size: ~{total_params * 2 / 1024**3:.2f} GB")
    logger.info(f"  🔧 dtype: {next(model.parameters()).dtype}")
    logger.info(f"  🖥️  device: {device}")
    
    return model, tokenizer


def apply_lora(model):
    """
    Modele LoRA uygula
    
    Args:
        model: Base model
        
    Returns:
        PEFT model
    """
    logger.info("\n" + "=" * 80)
    logger.info("🔧 LORA UYGULAMA")
    logger.info("=" * 80)
    
    from peft import LoraConfig, get_peft_model, TaskType
    
    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,  # Sequence-to-sequence task
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,  # None = auto-detect
        bias=config.LORA_BIAS,
        inference_mode=False
    )
    
    logger.info("📐 LoRA Configuration:")
    logger.info(f"  • Rank (r): {config.LORA_R}")
    logger.info(f"  • Alpha (α): {config.LORA_ALPHA}")
    logger.info(f"  • Dropout: {config.LORA_DROPOUT}")
    logger.info(f"  • Scaling: {config.LORA_ALPHA / config.LORA_R}")
    
    # LoRA uygula
    logger.info("\n🔄 PEFT modeli oluşturuluyor...")
    model = get_peft_model(model, lora_config)
    
    # Trainable parameters
    logger.info("\n📊 PARAMETRE ANALİZİ:")
    model.print_trainable_parameters()
    
    # Gradient kontrolü
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  • Gradient-enabled params: {trainable_params:,}")
    logger.info(f"  • Total params: {total_params:,}")
    logger.info(f"  • Trainable ratio: {100 * trainable_params / total_params:.2f}%")
    
    return model


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def preprocess_function(examples, tokenizer):
    """
    Summarization için data preprocessing
    
    Args:
        examples: Batch examples
        tokenizer: Tokenizer
        
    Returns:
        Tokenized batch
    """
    # Input (review text)
    inputs = examples["text"]
    
    # Target (summary)
    targets = examples["summary"]
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=config.CHUNK_SIZE,
        truncation=True,
        padding="max_length",
        return_tensors=None  # Dataset için None
    )
    
    # Tokenize targets
    labels = tokenizer(
        targets,
        max_length=config.MAX_SUMMARY_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors=None
    )
    
    # Labels'ı model_inputs'a ekle
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs


def prepare_datasets(train_data, test_data, tokenizer, max_train=None, max_test=None):
    """
    Dataset'leri hazırla ve tokenize et
    
    Args:
        train_data: Train data
        test_data: Test data
        tokenizer: Tokenizer
        max_train: Max train samples
        max_test: Max test samples
        
    Returns:
        (train_dataset, test_dataset) tokenized
    """
    logger.info("\n" + "=" * 80)
    logger.info("🔄 DATASET HAZIRLAMA VE TOKENİZASYON")
    logger.info("=" * 80)
    
    # Datasets oluştur
    train_dataset = create_dataset_from_data(train_data, tokenizer, max_train)
    test_dataset = create_dataset_from_data(test_data, tokenizer, max_test)
    
    logger.info(f"📊 Dataset boyutları:")
    logger.info(f"  • Train: {len(train_dataset):,}")
    logger.info(f"  • Test: {len(test_dataset):,}")
    
    # Tokenize
    logger.info("\n📝 Tokenization...")
    
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=["text", "summary", "label"],
        desc="Tokenizing train"
    )
    
    test_dataset = test_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=["text", "summary", "label"],
        desc="Tokenizing test"
    )
    
    logger.info("✅ Tokenization tamamlandı!")
    
    return train_dataset, test_dataset


# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, tokenizer, train_dataset, test_dataset, args):
    """
    Model'i eğit
    
    Args:
        model: PEFT model
        tokenizer: Tokenizer
        train_dataset: Train dataset
        test_dataset: Test dataset
        args: CLI arguments
        
    Returns:
        Trained model
    """
    logger.info("\n" + "=" * 80)
    logger.info("🎓 MODEL EĞİTİMİ")
    logger.info("=" * 80)
    
    from transformers import (
        Trainer,
        TrainingArguments,
        DataCollatorForSeq2Seq
    )
    
    # Training arguments
    training_args = TrainingArguments(
        # Çıktı
        output_dir=str(config.LORA_MODEL_DIR),
        logging_dir=str(config.LOGS_DIR / "training"),
        
        # Eğitim
        num_train_epochs=args.epochs if args.epochs else config.NUM_EPOCHS,
        per_device_train_batch_size=args.batch_size if args.batch_size else config.BATCH_SIZE_TRAIN,
        per_device_eval_batch_size=config.BATCH_SIZE_EVAL,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        
        # Optimization
        learning_rate=config.LEARNING_RATE,
        warmup_steps=config.WARMUP_STEPS,
        weight_decay=config.WEIGHT_DECAY,
        max_grad_norm=config.MAX_GRAD_NORM,
        
        # Mixed precision
        fp16=config.FP16 and torch.cuda.is_available(),
        
        # Logging ve saving
        logging_steps=config.LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=config.EVAL_STEPS,
        save_strategy="steps",
        save_steps=config.SAVE_STEPS,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Diğer
        report_to=None,  # WandB kapalı
        seed=42,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        
        # Gradient checkpointing kapalı (LoRA ile uyumsuz olabilir)
        gradient_checkpointing=False,
    )
    
    # Data collator (dynamic padding)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Trainer
    logger.info("🔧 Trainer oluşturuluyor...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Eğitim bilgisi
    logger.info("\n📋 EĞİTİM PARAMETRELERİ:")
    logger.info(f"  • Epochs: {training_args.num_train_epochs}")
    logger.info(f"  • Batch size (per device): {training_args.per_device_train_batch_size}")
    logger.info(f"  • Gradient accumulation: {training_args.gradient_accumulation_steps}")
    logger.info(f"  • Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size}")
    logger.info(f"  • Learning rate: {training_args.learning_rate:.0e}")
    logger.info(f"  • Warmup steps: {training_args.warmup_steps}")
    logger.info(f"  • FP16: {training_args.fp16}")
    logger.info(f"  • Device: {training_args.device}")
    
    # Eğitim
    logger.info("\n🚀 Eğitim başlıyor...\n")
    
    try:
        trainer.train()
        logger.info("\n✅ Eğitim tamamlandı!")
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Eğitim kullanıcı tarafından durduruldu!")
        logger.info("💾 Mevcut checkpoint kaydediliyor...")
        trainer.save_model(str(config.LORA_MODEL_DIR / "interrupted"))
        raise
    
    return trainer.model, trainer


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, tokenizer, test_data, num_samples=100):
    """
    Model'i değerlendir ve örnekler oluştur
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        test_data: Test data
        num_samples: Kaç sample değerlendirilecek
        
    Returns:
        Evaluation results dict
    """
    logger.info("\n" + "=" * 80)
    logger.info("📊 MODEL DEĞERLENDİRME")
    logger.info("=" * 80)
    
    device = config.DEVICE
    model = model.to(device)
    model.eval()
    
    # Sample'ları al
    test_samples = test_data[:num_samples]
    
    logger.info(f"🔍 {len(test_samples)} sample değerlendiriliyor...")
    
    predictions = []
    references = []
    
    with torch.no_grad():
        for item in tqdm(test_samples, desc="Generating summaries"):
            text = item["text"]
            reference = item["summary"]
            
            # Tokenize input
            inputs = tokenizer(
                text,
                max_length=config.CHUNK_SIZE,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            ).to(device)
            
            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.MAX_SUMMARY_LENGTH,
                min_length=config.MIN_SUMMARY_LENGTH,
                num_beams=config.NUM_BEAMS,
                length_penalty=config.LENGTH_PENALTY,
                no_repeat_ngram_size=config.NO_REPEAT_NGRAM_SIZE,
                early_stopping=config.EARLY_STOPPING,
            )
            
            # Decode
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            predictions.append(prediction)
            references.append(reference)
    
    # Metrikleri hesapla
    logger.info("\n📈 Metrikler hesaplanıyor...")
    metrics_calc = MetricsCalculator(config)
    
    results = metrics_calc.evaluate_summarization(
        predictions=predictions,
        references=references,
        include_bertscore=False  # Hızlı değerlendirme için kapalı
    )
    
    # Sonuçları göster
    logger.info("\n" + "=" * 80)
    logger.info("📊 DEĞERLENDIRME SONUÇLARI")
    logger.info("=" * 80)
    
    for metric, value in results.items():
        if isinstance(value, float):
            logger.info(f"  • {metric}: {value:.4f}")
        else:
            logger.info(f"  • {metric}: {value}")
    
    # Örnek çıktılar göster
    logger.info("\n" + "=" * 80)
    logger.info("📝 ÖRNEK ÇIKTILAR")
    logger.info("=" * 80)
    
    for i in range(min(3, len(predictions))):
        logger.info(f"\n[Example {i+1}]")
        logger.info(f"Input: {test_samples[i]['text'][:200]}...")
        logger.info(f"\nReference: {references[i]}")
        logger.info(f"\nPrediction: {predictions[i]}")
        logger.info("-" * 80)
    
    return results


# ============================================================================
# SAVE MODEL
# ============================================================================

def save_final_model(model, tokenizer):
    """
    Final modeli kaydet
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
    """
    logger.info("\n" + "=" * 80)
    logger.info("💾 FİNAL MODEL KAYDETME")
    logger.info("=" * 80)
    
    save_dir = config.LORA_MODEL_DIR / "final"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📁 Save directory: {save_dir}")
    
    # Model kaydet
    logger.info("🧠 Model kaydediliyor...")
    model.save_pretrained(str(save_dir))
    
    # Tokenizer kaydet
    logger.info("📝 Tokenizer kaydediliyor...")
    tokenizer.save_pretrained(str(save_dir))
    
    # Config kaydet
    logger.info("⚙️  Config kaydediliyor...")
    config_info = {
        "base_model": config.BASE_MODEL_NAME,
        "lora_config": {
            "r": config.LORA_R,
            "alpha": config.LORA_ALPHA,
            "dropout": config.LORA_DROPOUT
        },
        "training": {
            "epochs": config.NUM_EPOCHS,
            "batch_size": config.BATCH_SIZE_TRAIN,
            "learning_rate": config.LEARNING_RATE
        }
    }
    
    with open(save_dir / "training_config.json", "w") as f:
        json.dump(config_info, f, indent=2)
    
    logger.info(f"✅ Model kaydedildi: {save_dir}")
    
    # Boyut bilgisi
    total_size = sum(
        f.stat().st_size for f in save_dir.rglob("*") if f.is_file()
    )
    logger.info(f"💾 Total size: {total_size / 1024 / 1024:.1f} MB")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main(args):
    """Ana eğitim pipeline'ı"""
    
    logger.info("\n")
    logger.info("╔" + "═" * 78 + "╗")
    logger.info("║" + " " * 18 + "LORA SUMMARIZER TRAINING PIPELINE" + " " * 27 + "║")
    logger.info("╚" + "═" * 78 + "╝")
    logger.info("\n")
    
    # Configuration
    config.print_config()
    
    # 1. Data yükle
    train_data, test_data = load_summarization_data()
    
    # 2. Model ve tokenizer yükle
    model, tokenizer = load_model_and_tokenizer(args.model)
    
    # 3. LoRA uygula
    model = apply_lora(model)
    
    # 4. Datasets hazırla
    train_dataset, test_dataset = prepare_datasets(
        train_data, test_data, tokenizer,
        max_train=args.max_train,
        max_test=args.max_test
    )
    
    # 5. Eğitim
    model, trainer = train_model(model, tokenizer, train_dataset, test_dataset, args)
    
    # 6. Değerlendirme
    if args.evaluate:
        results = evaluate_model(model, tokenizer, test_data, num_samples=args.eval_samples)
        
        # NumPy tiplerini Python tiplerine dönüştür
        def convert_to_python_types(obj):
            """NumPy ve diğer özel tipleri Python tiplerine dönüştür"""
            import numpy as np
            if isinstance(obj, dict):
                return {k: convert_to_python_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_python_types(v) for v in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        results = convert_to_python_types(results)
        
        # Sonuçları kaydet
        results_path = config.EVAL_RESULTS_DIR / "summarization_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"💾 Results saved: {results_path}")
    
    # 7. Kaydet
    save_final_model(model, tokenizer)
    
    # Final
    logger.info("\n")
    logger.info("╔" + "═" * 78 + "╗")
    logger.info("║" + " " * 28 + "✅ BAŞARILI! ✅" + " " * 35 + "║")
    logger.info("╚" + "═" * 78 + "╝")
    logger.info("\n")
    logger.info("📁 Model: " + str(config.LORA_MODEL_DIR / "final"))
    logger.info("\n🚀 Sonraki adım: python 4_rag_qa_system.py")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    setup_logging()
    
    parser = argparse.ArgumentParser(
        description="Train LoRA summarization model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Base model name (default: from config)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (default: from config)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (default: from config)"
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="Max train samples"
    )
    parser.add_argument(
        "--max-test",
        type=int,
        default=None,
        help="Max test samples"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        default=True,
        help="Run evaluation after training"
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=100,
        help="Number of samples for evaluation"
    )
    
    args = parser.parse_args()
    
    try:
        main(args)
    except KeyboardInterrupt:
        logger.warning("\n⚠️  İşlem kullanıcı tarafından durduruldu!")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"❌ Hata: {e}")
        sys.exit(1)

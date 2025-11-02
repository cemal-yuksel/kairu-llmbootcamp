"""
================================================================================
1. DATA PREPARATION - Intelligent Review Summarizer
================================================================================

IMDB dataset'ini yükler, temizler ve RAG sistemi için hazırlar.

Bu script:
1. IMDB dataset'ini Hugging Face'den yükler
2. Review'ları temizler ve filtreler
3. Summarization için training data oluşturur
4. RAG için chunk'lara böler
5. İşlenmiş veriyi kaydeder

KULLANIM:
---------
python 1_data_preparation.py

veya

python 1_data_preparation.py --max-train 5000 --max-test 1000

Yazar: Kairu AI - Build with LLMs Bootcamp
Tarih: 2 Kasım 2025
================================================================================
"""

import sys
from pathlib import Path
import argparse
import json
from loguru import logger
from tqdm import tqdm
import pandas as pd

# Proje root'una path ekle
sys.path.append(str(Path(__file__).parent))

from config import config
from utils.data_loader import DataLoader
from utils.text_processor import TextProcessor


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging():
    """Logging yapılandırması"""
    logger.remove()  # Default handler'ı kaldır
    
    # Console handler (renkli)
    logger.add(
        sys.stderr,
        format=config.LOG_FORMAT,
        level=config.LOG_LEVEL,
        colorize=True
    )
    
    # File handler
    logger.add(
        config.LOG_FILE,
        format=config.LOG_FORMAT,
        level=config.LOG_LEVEL,
        rotation="10 MB"
    )


# ============================================================================
# DATA PREPARATION FUNCTIONS
# ============================================================================

def load_and_filter_imdb(
    max_train: int = None,
    max_test: int = None
):
    """
    IMDB dataset'ini yükle ve filtrele
    
    Args:
        max_train: Maksimum train samples
        max_test: Maksimum test samples
        
    Returns:
        DatasetDict
    """
    logger.info("=" * 80)
    logger.info("🎬 IMDB DATASET YÜKLEME")
    logger.info("=" * 80)
    
    # DataLoader
    loader = DataLoader(config)
    
    # IMDB yükle
    dataset = loader.load_imdb(
        max_train_samples=max_train,
        max_test_samples=max_test,
        cache_dir=config.RAW_DATA_DIR
    )
    
    # Review'ları filtrele
    logger.info("\n🔍 Review'lar filtreleniyor...")
    
    dataset["train"] = loader.filter_reviews(
        dataset["train"],
        min_length=config.MIN_REVIEW_LENGTH,
        max_length=config.MAX_REVIEW_LENGTH
    )
    
    dataset["test"] = loader.filter_reviews(
        dataset["test"],
        min_length=config.MIN_REVIEW_LENGTH,
        max_length=config.MAX_REVIEW_LENGTH
    )
    
    # İstatistikler
    train_stats = loader.get_dataset_stats(dataset["train"])
    test_stats = loader.get_dataset_stats(dataset["test"])
    
    logger.info("\n📊 TRAIN SET STATİSTİKLERİ:")
    logger.info(f"  • Total samples: {train_stats['total_samples']:,}")
    logger.info(f"  • Avg length: {train_stats['text_stats']['avg_length']:.0f} chars")
    logger.info(f"  • Label distribution: {train_stats['label_distribution']}")
    
    logger.info("\n📊 TEST SET STATİSTİKLERİ:")
    logger.info(f"  • Total samples: {test_stats['total_samples']:,}")
    logger.info(f"  • Avg length: {test_stats['text_stats']['avg_length']:.0f} chars")
    
    return dataset


def clean_reviews(dataset):
    """
    Review'ları temizle
    
    Args:
        dataset: DatasetDict
        
    Returns:
        Temizlenmiş DatasetDict
    """
    logger.info("\n" + "=" * 80)
    logger.info("🧹 TEXT CLEANING")
    logger.info("=" * 80)
    
    processor = TextProcessor()
    
    def clean_text_batch(examples):
        """Batch cleaning"""
        cleaned_texts = []
        
        for text in examples["text"]:
            # Temizle
            cleaned = processor.clean_text(
                text,
                lowercase=False,  # Casing'i koru (model için)
                remove_html=True,
                remove_urls=True,
                remove_special_chars=False,  # Noktalama koru
                remove_numbers=False,
                remove_extra_spaces=True
            )
            cleaned_texts.append(cleaned)
        
        return {"text": cleaned_texts}
    
    # Her split'i temizle
    logger.info("📝 Train set temizleniyor...")
    dataset["train"] = dataset["train"].map(
        clean_text_batch,
        batched=True,
        batch_size=1000,
        desc="Cleaning train"
    )
    
    logger.info("📝 Test set temizleniyor...")
    dataset["test"] = dataset["test"].map(
        clean_text_batch,
        batched=True,
        batch_size=1000,
        desc="Cleaning test"
    )
    
    logger.info("✅ Cleaning tamamlandı!")
    
    return dataset


def create_summarization_data(dataset):
    """
    Summarization için training data oluştur
    
    IMDB'de gold summary yok, bu yüzden extractive summary oluşturuyoruz:
    - Uzun review'lar için ilk 2-3 cümle summary olarak kullanılır
    
    Args:
        dataset: DatasetDict
        
    Returns:
        Summarization için hazır DatasetDict
    """
    logger.info("\n" + "=" * 80)
    logger.info("📄 SUMMARIZATION DATA OLUŞTURMA")
    logger.info("=" * 80)
    
    processor = TextProcessor()
    
    def create_summary(examples):
        """Her review için summary oluştur"""
        summaries = []
        
        for text in examples["text"]:
            # Cümlelere böl
            sentences = processor.tokenize_sentences(text)
            
            # İlk 2-3 cümle = summary (basit extractive)
            if len(sentences) > 3:
                summary = " ".join(sentences[:2])
            else:
                summary = " ".join(sentences[:1]) if sentences else text[:100]
            
            summaries.append(summary)
        
        return {"summary": summaries}
    
    logger.info("📝 Summaries oluşturuluyor...")
    
    dataset["train"] = dataset["train"].map(
        create_summary,
        batched=True,
        batch_size=1000,
        desc="Creating summaries"
    )
    
    dataset["test"] = dataset["test"].map(
        create_summary,
        batched=True,
        batch_size=1000,
        desc="Creating summaries"
    )
    
    # Örnek göster
    logger.info("\n📌 ÖRNEK:")
    example = dataset["train"][0]
    logger.info(f"\nFull Review:\n{example['text'][:300]}...")
    logger.info(f"\nSummary:\n{example['summary']}")
    
    return dataset


def create_rag_chunks(dataset):
    """
    RAG için review'ları chunk'lara böl
    
    Args:
        dataset: DatasetDict
        
    Returns:
        Chunk'lanmış data listesi
    """
    logger.info("\n" + "=" * 80)
    logger.info("✂️  RAG CHUNKS OLUŞTURMA")
    logger.info("=" * 80)
    
    loader = DataLoader(config)
    
    # Train ve test'i birleştir (RAG tüm veriyi kullanır)
    from datasets import concatenate_datasets
    all_reviews = concatenate_datasets([dataset["train"], dataset["test"]])
    
    # Chunk'la
    chunks = loader.create_review_chunks(
        all_reviews,
        chunk_size=config.CHUNK_SIZE,
        overlap=config.CHUNK_OVERLAP,
        text_column="text"
    )
    
    return chunks


def save_processed_data(dataset, chunks):
    """
    İşlenmiş veriyi kaydet
    
    Args:
        dataset: Summarization dataset
        chunks: RAG chunks
    """
    logger.info("\n" + "=" * 80)
    logger.info("💾 VERİ KAYDETME")
    logger.info("=" * 80)
    
    # Dizinleri oluştur
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Dataset'leri JSON olarak kaydet
    logger.info("📝 Train set kaydediliyor...")
    train_data = []
    for item in tqdm(dataset["train"], desc="Converting train"):
        train_data.append({
            "text": item["text"],
            "summary": item["summary"],
            "label": item["label"]
        })
    
    train_path = config.PROCESSED_DATA_DIR / "train.json"
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    logger.info(f"  ✅ Kaydedildi: {train_path}")
    
    # Test
    logger.info("📝 Test set kaydediliyor...")
    test_data = []
    for item in tqdm(dataset["test"], desc="Converting test"):
        test_data.append({
            "text": item["text"],
            "summary": item["summary"],
            "label": item["label"]
        })
    
    test_path = config.PROCESSED_DATA_DIR / "test.json"
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    logger.info(f"  ✅ Kaydedildi: {test_path}")
    
    # Chunks
    logger.info("📝 RAG chunks kaydediliyor...")
    chunks_path = config.PROCESSED_DATA_DIR / "rag_chunks.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    logger.info(f"  ✅ Kaydedildi: {chunks_path}")
    
    # Metadata
    metadata = {
        "train_samples": len(train_data),
        "test_samples": len(test_data),
        "rag_chunks": len(chunks),
        "config": {
            "min_review_length": config.MIN_REVIEW_LENGTH,
            "max_review_length": config.MAX_REVIEW_LENGTH,
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP
        }
    }
    
    metadata_path = config.METADATA_FILE
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"  ✅ Metadata kaydedildi: {metadata_path}")
    
    # Summary
    logger.info("\n📊 KAYDETME ÖZETİ:")
    logger.info(f"  • Train: {len(train_data):,} samples")
    logger.info(f"  • Test: {len(test_data):,} samples")
    logger.info(f"  • RAG Chunks: {len(chunks):,} chunks")
    logger.info(f"  • Toplam boyut: ~{(train_path.stat().st_size + test_path.stat().st_size + chunks_path.stat().st_size) / 1024 / 1024:.1f} MB")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main(args):
    """Ana veri hazırlama pipeline'ı"""
    
    logger.info("\n")
    logger.info("╔" + "═" * 78 + "╗")
    logger.info("║" + " " * 20 + "DATA PREPARATION PIPELINE" + " " * 33 + "║")
    logger.info("╚" + "═" * 78 + "╝")
    logger.info("\n")
    
    # Configuration özeti
    config.print_config()
    
    # 1. IMDB Yükle ve Filtrele
    dataset = load_and_filter_imdb(
        max_train=args.max_train,
        max_test=args.max_test
    )
    
    # 2. Text Cleaning
    dataset = clean_reviews(dataset)
    
    # 3. Summarization Data
    dataset = create_summarization_data(dataset)
    
    # 4. RAG Chunks
    chunks = create_rag_chunks(dataset)
    
    # 5. Kaydet
    save_processed_data(dataset, chunks)
    
    # Final
    logger.info("\n")
    logger.info("╔" + "═" * 78 + "╗")
    logger.info("║" + " " * 28 + "✅ BAŞARILI! ✅" + " " * 35 + "║")
    logger.info("╚" + "═" * 78 + "╝")
    logger.info("\n")
    logger.info("📁 İşlenmiş veriler: " + str(config.PROCESSED_DATA_DIR))
    logger.info("\n🚀 Sonraki adım: python 2_embedding_creation.py")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    # Setup
    setup_logging()
    
    # Arguments
    parser = argparse.ArgumentParser(
        description="IMDB Dataset Preparation for Review Summarizer"
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="Maximum train samples (None = all)"
    )
    parser.add_argument(
        "--max-test",
        type=int,
        default=None,
        help="Maximum test samples (None = all)"
    )
    
    args = parser.parse_args()
    
    # Run
    try:
        main(args)
    except KeyboardInterrupt:
        logger.warning("\n⚠️  İşlem kullanıcı tarafından durduruldu!")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"❌ Hata: {e}")
        sys.exit(1)

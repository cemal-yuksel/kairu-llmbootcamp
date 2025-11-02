"""
================================================================================
DATA LOADER - Intelligent Review Summarizer
================================================================================

IMDB ve diğer veri setlerini yüklemek ve işlemek için yardımcı fonksiyonlar.

Yazar: Kairu AI - Build with LLMs Bootcamp
Tarih: 2 Kasım 2025
================================================================================
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datasets import load_dataset, Dataset, DatasetDict
from loguru import logger
import pandas as pd
from tqdm import tqdm


class DataLoader:
    """
    Veri seti yükleme ve yönetim sınıfı
    
    IMDB ve custom dataset'leri yükler, filtreleyip işler.
    """
    
    def __init__(self, config):
        """
        Args:
            config: Configuration objesi
        """
        self.config = config
        self.dataset: Optional[DatasetDict] = None
        
    def load_imdb(
        self, 
        max_train_samples: Optional[int] = None,
        max_test_samples: Optional[int] = None,
        cache_dir: Optional[Path] = None
    ) -> DatasetDict:
        """
        IMDB dataset'ini yükle
        
        IMDB Dataset:
        - 50,000 film yorumu (25k train, 25k test)
        - Binary sentiment (pos/neg)
        - Her split dengeli (12.5k pos, 12.5k neg)
        
        Args:
            max_train_samples: Maksimum train örnekleri (None = hepsi)
            max_test_samples: Maksimum test örnekleri
            cache_dir: Cache directory
            
        Returns:
            DatasetDict: {"train": Dataset, "test": Dataset}
        """
        logger.info(f"📥 IMDB dataset yükleniyor...")
        
        try:
            # Dataset'i Hugging Face'den yükle
            dataset = load_dataset(
                self.config.DATASET_NAME,
                cache_dir=str(cache_dir) if cache_dir else None,
                trust_remote_code=True
            )
            
            # Sampling uygula
            if max_train_samples:
                logger.info(f"  🔹 Train set {max_train_samples} ile sınırlanıyor...")
                dataset["train"] = dataset["train"].select(range(max_train_samples))
                
            if max_test_samples:
                logger.info(f"  🔹 Test set {max_test_samples} ile sınırlanıyor...")
                dataset["test"] = dataset["test"].select(range(max_test_samples))
            
            # İstatistikler
            logger.info(f"✅ IMDB yüklendi:")
            logger.info(f"  📊 Train samples: {len(dataset['train']):,}")
            logger.info(f"  📊 Test samples: {len(dataset['test']):,}")
            
            self.dataset = dataset
            return dataset
            
        except Exception as e:
            logger.error(f"❌ IMDB yükleme hatası: {e}")
            raise
    
    def filter_reviews(
        self,
        dataset: Dataset,
        min_length: int = 50,
        max_length: int = 2000,
        sentiment: Optional[int] = None
    ) -> Dataset:
        """
        Review'ları filtrele
        
        Args:
            dataset: Filtrelenecek dataset
            min_length: Minimum karakter sayısı
            max_length: Maximum karakter sayısı
            sentiment: Sadece belirli sentiment (0=neg, 1=pos)
            
        Returns:
            Filtrelenmiş Dataset
        """
        logger.info(f"🔍 Review'lar filtreleniyor...")
        original_len = len(dataset)
        
        def filter_function(example):
            text = example["text"]
            # Uzunluk kontrolü
            if len(text) < min_length or len(text) > max_length:
                return False
            # Sentiment kontrolü
            if sentiment is not None and example["label"] != sentiment:
                return False
            return True
        
        filtered = dataset.filter(filter_function)
        filtered_len = len(filtered)
        
        logger.info(f"  ✅ {original_len:,} → {filtered_len:,} review kaldı")
        logger.info(f"  📉 {original_len - filtered_len:,} review filtrelendi")
        
        return filtered
    
    def balance_dataset(
        self,
        dataset: Dataset,
        label_column: str = "label",
        samples_per_class: Optional[int] = None
    ) -> Dataset:
        """
        Dataset'i dengele (her sınıftan eşit sayıda örnek)
        
        Args:
            dataset: Dengelenecek dataset
            label_column: Label sütunu adı
            samples_per_class: Her sınıftan kaç örnek (None = minimum)
            
        Returns:
            Dengelenmiş Dataset
        """
        logger.info("⚖️  Dataset dengeleniyor...")
        
        # Her sınıfın örneklerini ayır
        labels = dataset[label_column]
        unique_labels = set(labels)
        
        class_datasets = {}
        for label in unique_labels:
            class_datasets[label] = dataset.filter(
                lambda x: x[label_column] == label
            )
            logger.info(f"  📊 Class {label}: {len(class_datasets[label]):,} samples")
        
        # Kaç örnek alınacağını belirle
        if samples_per_class is None:
            samples_per_class = min(len(ds) for ds in class_datasets.values())
        
        logger.info(f"  🎯 Hedef: {samples_per_class:,} sample per class")
        
        # Her sınıftan eşit sayıda al
        balanced_datasets = []
        for label, ds in class_datasets.items():
            if len(ds) > samples_per_class:
                ds = ds.shuffle(seed=42).select(range(samples_per_class))
            balanced_datasets.append(ds)
        
        # Birleştir
        from datasets import concatenate_datasets
        balanced = concatenate_datasets(balanced_datasets)
        
        logger.info(f"  ✅ Dengelenmiş dataset: {len(balanced):,} samples")
        
        return balanced.shuffle(seed=42)
    
    def get_dataset_stats(self, dataset: Dataset) -> Dict:
        """
        Dataset istatistikleri
        
        Args:
            dataset: Analiz edilecek dataset
            
        Returns:
            İstatistik dictionary'si
        """
        stats = {
            "total_samples": len(dataset),
            "columns": dataset.column_names,
        }
        
        # Text uzunlukları
        if "text" in dataset.column_names:
            texts = dataset["text"]
            lengths = [len(t) for t in texts]
            stats["text_stats"] = {
                "min_length": min(lengths),
                "max_length": max(lengths),
                "avg_length": sum(lengths) / len(lengths),
                "total_chars": sum(lengths)
            }
        
        # Label dağılımı
        if "label" in dataset.column_names:
            labels = dataset["label"]
            from collections import Counter
            label_counts = Counter(labels)
            stats["label_distribution"] = dict(label_counts)
        
        return stats
    
    def save_to_json(
        self,
        dataset: Dataset,
        output_path: Union[str, Path],
        include_labels: bool = True
    ):
        """
        Dataset'i JSON formatında kaydet
        
        Args:
            dataset: Kaydedilecek dataset
            output_path: Çıktı dosya yolu
            include_labels: Label'ları dahil et
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"💾 Dataset kaydediliyor: {output_path}")
        
        data = []
        for item in tqdm(dataset, desc="Converting to JSON"):
            entry = {"text": item["text"]}
            if include_labels and "label" in item:
                entry["label"] = item["label"]
            data.append(entry)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"  ✅ {len(data):,} samples kaydedildi")
    
    def load_from_json(
        self,
        input_path: Union[str, Path]
    ) -> Dataset:
        """
        JSON'dan dataset yükle
        
        Args:
            input_path: JSON dosya yolu
            
        Returns:
            Dataset objesi
        """
        input_path = Path(input_path)
        logger.info(f"📥 JSON yükleniyor: {input_path}")
        
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        dataset = Dataset.from_list(data)
        logger.info(f"  ✅ {len(dataset):,} samples yüklendi")
        
        return dataset
    
    def create_review_chunks(
        self,
        dataset: Dataset,
        chunk_size: int = 512,
        overlap: int = 50,
        text_column: str = "text"
    ) -> List[Dict]:
        """
        Uzun review'ları chunk'lara böl
        
        RAG için review'ları daha küçük parçalara böler.
        
        Args:
            dataset: Bölünecek dataset
            chunk_size: Her chunk'ın max token sayısı
            overlap: Chunk'lar arası overlap
            text_column: Text sütunu adı
            
        Returns:
            Chunk'lar listesi [{"text": ..., "metadata": ...}]
        """
        logger.info(f"✂️  Review'lar chunk'lara bölünüyor...")
        logger.info(f"  📏 Chunk size: {chunk_size}, Overlap: {overlap}")
        
        chunks = []
        
        for idx, item in enumerate(tqdm(dataset, desc="Chunking")):
            text = item[text_column]
            
            # Basit word-based chunking
            words = text.split()
            
            for i in range(0, len(words), chunk_size - overlap):
                chunk_words = words[i:i + chunk_size]
                chunk_text = " ".join(chunk_words)
                
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "review_id": idx,
                        "chunk_id": i // (chunk_size - overlap),
                        "label": item.get("label", None),
                        "word_count": len(chunk_words)
                    }
                })
        
        logger.info(f"  ✅ {len(chunks):,} chunk oluşturuldu")
        logger.info(f"  📊 Ortalama: {len(chunks) / len(dataset):.1f} chunk/review")
        
        return chunks


# ============================================================================
# QUICK TEST
# ============================================================================
if __name__ == "__main__":
    from config import config
    
    logger.info("🧪 DataLoader test başlıyor...")
    
    # DataLoader oluştur
    loader = DataLoader(config)
    
    # IMDB yükle (küçük sample)
    dataset = loader.load_imdb(max_train_samples=100, max_test_samples=50)
    
    # İstatistikler
    stats = loader.get_dataset_stats(dataset["train"])
    logger.info(f"\n📊 Dataset Stats:\n{json.dumps(stats, indent=2)}")
    
    # Filtreleme testi
    filtered = loader.filter_reviews(
        dataset["train"],
        min_length=100,
        max_length=1000
    )
    
    # Chunk testi
    chunks = loader.create_review_chunks(
        filtered,
        chunk_size=100,
        overlap=20
    )
    
    logger.info("\n✅ Tüm testler başarılı!")

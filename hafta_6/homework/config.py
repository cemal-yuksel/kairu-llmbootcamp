"""
================================================================================
CONFIGURATION - Intelligent Review Summarizer with Q&A
================================================================================

Bu dosya tüm proje konfigürasyonlarını merkezi olarak yönetir.
Farklı ortamlar (development, production) için kolayca ayarlanabilir.

KULLANIM:
---------
from config import Config
config = Config()
print(config.MODEL_NAME)

Yazar: Kairu AI - Build with LLMs Bootcamp
Tarih: 2 Kasım 2025
================================================================================
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict


# ============================================================================
# GLOBAL PATHS
# ============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
EVAL_DIR = BASE_DIR / "evaluation" / "results"

# Klasörleri oluştur
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR, EVAL_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Alt klasörleri de oluştur
(DATA_DIR / "raw").mkdir(parents=True, exist_ok=True)
(DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)
(MODELS_DIR / "vector_db").mkdir(parents=True, exist_ok=True)
(MODELS_DIR / "lora_summarizer").mkdir(parents=True, exist_ok=True)


# ============================================================================
# CONFIGURATION CLASS
# ============================================================================

@dataclass
class Config:
    """
    Ana konfigürasyon sınıfı
    
    Tüm hiperparametreler ve ayarlar burada merkezi olarak tutulur.
    Environment variables ile override edilebilir.
    """
    
    # ========================================================================
    # DATASET CONFIGURATION
    # ========================================================================
    DATASET_NAME: str = "imdb"
    DATASET_SPLIT_TRAIN: str = "train"
    DATASET_SPLIT_TEST: str = "test"
    MAX_SAMPLES_TRAIN: Optional[int] = None  # None = tüm veri, yoksa limit
    MAX_SAMPLES_TEST: Optional[int] = None
    
    # Text preprocessing
    MIN_REVIEW_LENGTH: int = 50  # Minimum karakter sayısı
    MAX_REVIEW_LENGTH: int = 2000  # Maximum karakter sayısı
    CHUNK_SIZE: int = 512  # Her chunk'ın maksimum token sayısı
    CHUNK_OVERLAP: int = 50  # Chunk'lar arası overlap
    
    # ========================================================================
    # MODEL CONFIGURATION
    # ========================================================================
    # Base model
    BASE_MODEL_NAME: str = "facebook/bart-base"  # Summarization için BART
    # Alternatifler:
    # - "t5-small" (küçük, hızlı)
    # - "t5-base" (orta)
    # - "facebook/bart-large" (büyük, iyi performans)
    
    # Embedding model
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Alternatifler:
    # - "sentence-transformers/all-mpnet-base-v2" (daha iyi ama yavaş)
    # - "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" (çok dilli)
    
    # ========================================================================
    # LORA CONFIGURATION
    # ========================================================================
    LORA_R: int = 16  # LoRA rank
    LORA_ALPHA: int = 32  # LoRA alpha (scaling factor)
    LORA_DROPOUT: float = 0.1  # Dropout rate
    LORA_TARGET_MODULES: List[str] = None  # None = otomatik, yoksa ["q_proj", "v_proj"]
    LORA_BIAS: str = "none"  # "none", "all", or "lora_only"
    
    # ========================================================================
    # TRAINING CONFIGURATION
    # ========================================================================
    # Training hyperparameters
    NUM_EPOCHS: int = 3
    BATCH_SIZE_TRAIN: int = 4
    BATCH_SIZE_EVAL: int = 8
    GRADIENT_ACCUMULATION_STEPS: int = 4  # Efektif batch = 4 * 4 = 16
    LEARNING_RATE: float = 5e-5
    WARMUP_STEPS: int = 500
    WEIGHT_DECAY: float = 0.01
    MAX_GRAD_NORM: float = 1.0
    
    # Optimization
    FP16: bool = True  # Mixed precision training (GPU'da hızlandırır)
    GRADIENT_CHECKPOINTING: bool = True  # RAM tasarrufu
    
    # Logging and saving
    LOGGING_STEPS: int = 100
    EVAL_STEPS: int = 500
    SAVE_STEPS: int = 1000
    SAVE_TOTAL_LIMIT: int = 3  # Maksimum checkpoint sayısı
    
    # ========================================================================
    # GENERATION CONFIGURATION
    # ========================================================================
    # Summarization parameters
    MAX_SUMMARY_LENGTH: int = 150  # Maximum özet uzunluğu (tokens)
    MIN_SUMMARY_LENGTH: int = 30  # Minimum özet uzunluğu
    NUM_BEAMS: int = 4  # Beam search için beam sayısı
    LENGTH_PENALTY: float = 2.0  # Uzunluk cezası (>1 = daha uzun özetler)
    NO_REPEAT_NGRAM_SIZE: int = 3  # Tekrar eden n-gram'ları engelle
    EARLY_STOPPING: bool = True
    
    # Temperature settings
    TEMPERATURE_DETERMINISTIC: float = 0.3  # Faktüel içerik için
    TEMPERATURE_CREATIVE: float = 0.9  # Yaratıcı içerik için
    TOP_P: float = 0.9  # Nucleus sampling
    TOP_K: int = 50  # Top-K sampling
    
    # ========================================================================
    # RAG CONFIGURATION
    # ========================================================================
    # Vector database
    VECTOR_DB_TYPE: str = "faiss"  # "faiss" veya "chroma"
    EMBEDDING_DIMENSION: int = 384  # all-MiniLM-L6-v2 için
    
    # Retrieval parameters
    TOP_K_RETRIEVAL: int = 5  # Kaç doküman getirilecek
    SIMILARITY_THRESHOLD: float = 0.5  # Minimum similarity score
    
    # Context window
    MAX_CONTEXT_LENGTH: int = 1024  # RAG için maksimum context
    
    # ========================================================================
    # EVALUATION CONFIGURATION
    # ========================================================================
    # ROUGE metrics
    ROUGE_TYPES: List[str] = None  # ["rouge1", "rouge2", "rougeL"]
    USE_STEMMER: bool = True  # ROUGE için stemmer kullan
    
    # Other metrics
    CALCULATE_BERTSCORE: bool = False  # BERTScore (yavaş ama kaliteli)
    CALCULATE_BLEU: bool = False  # BLEU score
    
    # ========================================================================
    # UI CONFIGURATION
    # ========================================================================
    # Streamlit
    APP_TITLE: str = "🎬 Intelligent Movie Review Analyzer"
    APP_DESCRIPTION: str = """
    IMDB film yorumlarını analiz eden akıllı asistan.
    Binlerce yorumu özetleyip sorularınızı yanıtlar! 🤖
    """
    PAGE_ICON: str = "🎬"
    LAYOUT: str = "wide"  # "wide" veya "centered"
    
    # UI features
    SHOW_TECHNICAL_DETAILS: bool = True  # Teknik detayları göster
    ENABLE_CACHING: bool = True  # Streamlit caching
    MAX_DISPLAY_REVIEWS: int = 10  # UI'da gösterilecek max review sayısı
    
    # ========================================================================
    # PATHS
    # ========================================================================
    # Data paths
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    EMBEDDINGS_FILE: Path = PROCESSED_DATA_DIR / "embeddings.npy"
    METADATA_FILE: Path = PROCESSED_DATA_DIR / "metadata.json"
    
    # Model paths
    LORA_MODEL_DIR: Path = MODELS_DIR / "lora_summarizer"
    VECTOR_DB_DIR: Path = MODELS_DIR / "vector_db"
    
    # Logs path
    LOGS_DIR: Path = LOGS_DIR
    
    # Evaluation paths
    EVAL_RESULTS_DIR: Path = EVAL_DIR
    ROUGE_RESULTS_FILE: Path = EVAL_DIR / "rouge_scores.json"
    QA_RESULTS_FILE: Path = EVAL_DIR / "qa_metrics.json"
    
    # ========================================================================
    # LOGGING CONFIGURATION
    # ========================================================================
    LOG_LEVEL: str = "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR"
    LOG_FILE: Path = LOGS_DIR / "app.log"
    LOG_FORMAT: str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    
    # ========================================================================
    # DEVICE CONFIGURATION
    # ========================================================================
    @property
    def DEVICE(self) -> str:
        """GPU varsa cuda, yoksa cpu kullan"""
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    @property
    def N_GPU(self) -> int:
        """Kullanılabilir GPU sayısı"""
        import torch
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    # ========================================================================
    # ENVIRONMENT VARIABLES
    # ========================================================================
    def __post_init__(self):
        """Environment variables'dan override et"""
        
        # ROUGE types default
        if self.ROUGE_TYPES is None:
            self.ROUGE_TYPES = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        
        # Environment variable overrides
        self.DATASET_NAME = os.getenv("DATASET_NAME", self.DATASET_NAME)
        self.BASE_MODEL_NAME = os.getenv("BASE_MODEL_NAME", self.BASE_MODEL_NAME)
        
        # Integer overrides
        if os.getenv("NUM_EPOCHS"):
            self.NUM_EPOCHS = int(os.getenv("NUM_EPOCHS"))
        if os.getenv("BATCH_SIZE_TRAIN"):
            self.BATCH_SIZE_TRAIN = int(os.getenv("BATCH_SIZE_TRAIN"))
        
        # Boolean overrides
        if os.getenv("FP16"):
            self.FP16 = os.getenv("FP16").lower() == "true"
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    def get_model_config(self) -> Dict:
        """Model configuration dict döndür"""
        return {
            "base_model": self.BASE_MODEL_NAME,
            "lora_r": self.LORA_R,
            "lora_alpha": self.LORA_ALPHA,
            "lora_dropout": self.LORA_DROPOUT,
        }
    
    def get_training_config(self) -> Dict:
        """Training configuration dict döndür"""
        return {
            "num_epochs": self.NUM_EPOCHS,
            "batch_size": self.BATCH_SIZE_TRAIN,
            "learning_rate": self.LEARNING_RATE,
            "warmup_steps": self.WARMUP_STEPS,
            "device": self.DEVICE,
        }
    
    def print_config(self):
        """Tüm konfigürasyonu yazdır"""
        from rich.console import Console
        from rich.table import Table
        
        console = Console()
        table = Table(title="📋 Configuration Summary", show_header=True)
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="green")
        
        # Dataset
        table.add_row("Dataset", self.DATASET_NAME)
        table.add_row("Base Model", self.BASE_MODEL_NAME)
        table.add_row("Embedding Model", self.EMBEDDING_MODEL_NAME)
        
        # Training
        table.add_row("Epochs", str(self.NUM_EPOCHS))
        table.add_row("Batch Size", str(self.BATCH_SIZE_TRAIN))
        table.add_row("Learning Rate", f"{self.LEARNING_RATE:.0e}")
        
        # LoRA
        table.add_row("LoRA Rank", str(self.LORA_R))
        table.add_row("LoRA Alpha", str(self.LORA_ALPHA))
        
        # Device
        table.add_row("Device", self.DEVICE)
        table.add_row("GPU Count", str(self.N_GPU))
        
        console.print(table)


# ============================================================================
# GLOBAL CONFIG INSTANCE
# ============================================================================
config = Config()


# ============================================================================
# QUICK TEST
# ============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("CONFIGURATION TEST")
    print("=" * 80)
    
    config.print_config()
    
    print("\n" + "=" * 80)
    print("PATHS:")
    print("=" * 80)
    print(f"Base Dir: {BASE_DIR}")
    print(f"Data Dir: {DATA_DIR}")
    print(f"Models Dir: {MODELS_DIR}")
    print(f"Logs Dir: {LOGS_DIR}")
    
    print("\n✅ Configuration loaded successfully!")

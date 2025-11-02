"""
================================================================================
2. EMBEDDING CREATION - Intelligent Review Summarizer
================================================================================

RAG sistemi için vector database oluşturur.

Bu script:
1. İşlenmiş RAG chunks'ları yükler
2. Sentence embeddings oluşturur
3. FAISS vector database kurar
4. Metadata ile birlikte kaydeder

KULLANIM:
---------
python 2_embedding_creation.py

veya

python 2_embedding_creation.py --embedding-model sentence-transformers/all-mpnet-base-v2

Yazar: Kairu AI - Build with LLMs Bootcamp
Tarih: 2 Kasım 2025
================================================================================
"""

import sys
from pathlib import Path
import argparse
import json
import numpy as np
from loguru import logger
from tqdm import tqdm
import pickle

# Proje root'una path ekle
sys.path.append(str(Path(__file__).parent))

from config import config


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
# EMBEDDING FUNCTIONS
# ============================================================================

def load_embedding_model(model_name: str = None):
    """
    Sentence embedding modelini yükle
    
    Args:
        model_name: Model adı (None = config'den)
        
    Returns:
        SentenceTransformer model
    """
    logger.info("=" * 80)
    logger.info("🤖 EMBEDDING MODEL YÜKLEME")
    logger.info("=" * 80)
    
    if model_name is None:
        model_name = config.EMBEDDING_MODEL_NAME
    
    logger.info(f"📦 Model: {model_name}")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer(model_name)
        
        logger.info(f"✅ Model yüklendi!")
        logger.info(f"  • Max sequence length: {model.max_seq_length}")
        logger.info(f"  • Embedding dimension: {model.get_sentence_embedding_dimension()}")
        
        return model
        
    except ImportError:
        logger.error("❌ sentence-transformers yüklü değil!")
        logger.error("   Kurulum: pip install sentence-transformers")
        raise
    except Exception as e:
        logger.error(f"❌ Model yükleme hatası: {e}")
        raise


def load_chunks():
    """
    RAG chunks'ları yükle
    
    Returns:
        Chunks listesi
    """
    logger.info("\n" + "=" * 80)
    logger.info("📥 CHUNKS YÜKLEME")
    logger.info("=" * 80)
    
    chunks_path = config.PROCESSED_DATA_DIR / "rag_chunks.json"
    
    if not chunks_path.exists():
        logger.error(f"❌ Chunks dosyası bulunamadı: {chunks_path}")
        logger.error("   Önce 1_data_preparation.py çalıştırın!")
        raise FileNotFoundError(chunks_path)
    
    logger.info(f"📂 Yükleniyor: {chunks_path}")
    
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    logger.info(f"✅ {len(chunks):,} chunks yüklendi")
    
    # İstatistikler
    text_lengths = [len(chunk["text"]) for chunk in chunks]
    logger.info(f"\n📊 İSTATİSTİKLER:")
    logger.info(f"  • Ortalama uzunluk: {np.mean(text_lengths):.0f} chars")
    logger.info(f"  • Min uzunluk: {np.min(text_lengths)} chars")
    logger.info(f"  • Max uzunluk: {np.max(text_lengths)} chars")
    
    return chunks


def create_embeddings(model, chunks, batch_size: int = 32):
    """
    Chunk'lar için embeddings oluştur
    
    Args:
        model: SentenceTransformer model
        chunks: Chunk'lar listesi
        batch_size: Batch size
        
    Returns:
        numpy array (N x embedding_dim)
    """
    logger.info("\n" + "=" * 80)
    logger.info("🧮 EMBEDDINGS OLUŞTURMA")
    logger.info("=" * 80)
    
    # Text'leri çıkar
    texts = [chunk["text"] for chunk in chunks]
    
    logger.info(f"📝 {len(texts):,} text encode ediliyor...")
    logger.info(f"  • Batch size: {batch_size}")
    logger.info(f"  • Device: {model.device}")
    
    # Encode (progress bar ile)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # Cosine similarity için normalize
    )
    
    logger.info(f"\n✅ Embeddings oluşturuldu!")
    logger.info(f"  • Shape: {embeddings.shape}")
    logger.info(f"  • Dtype: {embeddings.dtype}")
    logger.info(f"  • Size: {embeddings.nbytes / 1024 / 1024:.1f} MB")
    
    return embeddings


def create_faiss_index(embeddings):
    """
    FAISS index oluştur
    
    FAISS (Facebook AI Similarity Search):
    - Verimli nearest neighbor search
    - Milyonlarca vector için optimize
    
    Args:
        embeddings: numpy array (N x D)
        
    Returns:
        FAISS index
    """
    logger.info("\n" + "=" * 80)
    logger.info("🔍 FAISS INDEX OLUŞTURMA")
    logger.info("=" * 80)
    
    try:
        import faiss
    except ImportError:
        logger.error("❌ faiss yüklü değil!")
        logger.error("   Kurulum: pip install faiss-cpu  (veya faiss-gpu)")
        raise
    
    dimension = embeddings.shape[1]
    n_vectors = embeddings.shape[0]
    
    logger.info(f"📐 Dimension: {dimension}")
    logger.info(f"📊 Vectors: {n_vectors:,}")
    
    # Index tipi seçimi
    # IndexFlatIP: Exact search, Inner Product (cosine similarity için)
    # Alternatif: IndexFlatL2 (L2 distance için)
    
    logger.info("🛠️  IndexFlatIP oluşturuluyor (exact search)...")
    index = faiss.IndexFlatIP(dimension)
    
    # Embeddings'leri ekle
    logger.info("📥 Embeddings ekleniyor...")
    index.add(embeddings.astype(np.float32))
    
    logger.info(f"✅ Index oluşturuldu!")
    logger.info(f"  • Total vectors: {index.ntotal:,}")
    logger.info(f"  • Is trained: {index.is_trained}")
    
    return index


def create_chromadb_collection(chunks, embeddings):
    """
    ChromaDB collection oluştur (alternatif)
    
    ChromaDB:
    - Document-oriented vector database
    - Metadata filtering
    - Kolay kullanım
    
    Args:
        chunks: Chunk'lar
        embeddings: Embeddings array
        
    Returns:
        ChromaDB collection
    """
    logger.info("\n" + "=" * 80)
    logger.info("🗄️  CHROMADB COLLECTION OLUŞTURMA")
    logger.info("=" * 80)
    
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        logger.warning("⚠️  chromadb yüklü değil, atlanıyor...")
        return None
    
    # Client oluştur
    chroma_path = config.MODELS_DIR / "chroma_db"
    chroma_path.mkdir(parents=True, exist_ok=True)
    
    client = chromadb.PersistentClient(
        path=str(chroma_path),
        settings=Settings(anonymized_telemetry=False)
    )
    
    # Collection oluştur (varsa sil)
    collection_name = "imdb_reviews"
    try:
        client.delete_collection(collection_name)
    except:
        pass
    
    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "IMDB movie reviews for RAG"}
    )
    
    # Documents ekle
    logger.info("📥 Documents ekleniyor...")
    
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    
    # Batch ekleme
    batch_size = 1000
    for i in tqdm(range(0, len(chunks), batch_size), desc="Adding to ChromaDB"):
        end_idx = min(i + batch_size, len(chunks))
        
        collection.add(
            ids=ids[i:end_idx],
            embeddings=embeddings[i:end_idx].tolist(),
            documents=documents[i:end_idx],
            metadatas=metadatas[i:end_idx]
        )
    
    logger.info(f"✅ ChromaDB collection oluşturuldu!")
    logger.info(f"  • Path: {chroma_path}")
    logger.info(f"  • Total documents: {collection.count()}")
    
    return collection


def save_vector_db(index, chunks, embeddings):
    """
    Vector database'i kaydet
    
    Args:
        index: FAISS index
        chunks: Chunk'lar
        embeddings: Embeddings array
    """
    logger.info("\n" + "=" * 80)
    logger.info("💾 VECTOR DATABASE KAYDETME")
    logger.info("=" * 80)
    
    # Dizin oluştur (kesinlikle oluşturulmalı!)
    vector_db_dir = config.VECTOR_DB_DIR
    try:
        # Önce parent dizinleri oluştur
        vector_db_dir.parent.mkdir(parents=True, exist_ok=True)
        # Sonra hedef dizini oluştur
        vector_db_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Vector DB dizini oluşturuldu: {vector_db_dir}")
    except Exception as e:
        logger.error(f"❌ Dizin oluşturma hatası: {e}")
        raise
    
    # Dizinin varlığını kontrol et
    if not vector_db_dir.exists():
        logger.error(f"❌ Dizin oluşturulamadı: {vector_db_dir}")
        raise RuntimeError(f"Cannot create directory: {vector_db_dir}")
    
    logger.info(f"📁 Vector DB dizini mevcut: {vector_db_dir}")
    
    # FAISS index kaydet (Türkçe karakter path problemi için workaround)
    logger.info("📝 FAISS index kaydediliyor...")
    import faiss
    
    index_path = vector_db_dir / "faiss_index.bin"
    logger.info(f"  📍 Hedef dosya: {index_path}")
    
    try:
        # FAISS'i serialize et ve binary olarak kaydet
        index_bytes = faiss.serialize_index(index)
        with open(index_path, 'wb') as f:
            f.write(index_bytes)
        logger.info(f"  ✅ Kaydedildi: {index_path}")
    except Exception as e:
        logger.error(f"  ❌ FAISS kaydetme hatası: {e}")
        logger.info(f"  🔄 Index ve embeddings pickle olarak kaydediliyor...")
        # Fallback: tüm veriyi pickle ile kaydet
        pickle_path = vector_db_dir / "faiss_index.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump({'index': index, 'serialized': faiss.serialize_index(index)}, f)
        logger.info(f"  ✅ Pickle olarak kaydedildi: {pickle_path}")
    
    # Embeddings array kaydet
    logger.info("📝 Embeddings array kaydediliyor...")
    embeddings_path = vector_db_dir / "embeddings.npy"
    np.save(embeddings_path, embeddings)
    logger.info(f"  ✅ Kaydedildi: {embeddings_path}")
    
    # Chunks metadata kaydet
    logger.info("📝 Chunks metadata kaydediliyor...")
    chunks_path = vector_db_dir / "chunks.pkl"
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    logger.info(f"  ✅ Kaydedildi: {chunks_path}")
    
    # Config kaydet
    logger.info("📝 Config kaydediliyor...")
    config_data = {
        "embedding_model": config.EMBEDDING_MODEL_NAME,
        "embedding_dimension": embeddings.shape[1],
        "num_chunks": len(chunks),
        "index_type": "IndexFlatIP",
        "normalized": True
    }
    
    config_path = vector_db_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)
    logger.info(f"  ✅ Kaydedildi: {config_path}")
    
    # Özet
    total_size = sum(
        p.stat().st_size for p in vector_db_dir.glob("*") if p.is_file()
    )
    
    logger.info(f"\n📊 KAYDETME ÖZETİ:")
    logger.info(f"  • FAISS index: {index_path.stat().st_size / 1024 / 1024:.1f} MB")
    logger.info(f"  • Embeddings: {embeddings_path.stat().st_size / 1024 / 1024:.1f} MB")
    logger.info(f"  • Chunks: {chunks_path.stat().st_size / 1024 / 1024:.1f} MB")
    logger.info(f"  • Toplam: {total_size / 1024 / 1024:.1f} MB")


def test_search(index, chunks, embeddings, model):
    """
    Vector search testi
    
    Args:
        index: FAISS index
        chunks: Chunks
        embeddings: Embeddings
        model: SentenceTransformer model
    """
    logger.info("\n" + "=" * 80)
    logger.info("🧪 SEARCH TESTI")
    logger.info("=" * 80)
    
    # Test query
    query = "What do you think about the acting in this movie?"
    logger.info(f"\n📝 Query: {query}")
    
    # Query embedding
    query_embedding = model.encode([query], normalize_embeddings=True)
    
    # Search
    k = 5
    distances, indices = index.search(query_embedding.astype(np.float32), k)
    
    logger.info(f"\n🔍 Top {k} sonuç:")
    for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        chunk = chunks[idx]
        logger.info(f"\n[{i+1}] Similarity: {dist:.4f}")
        logger.info(f"    Review ID: {chunk['metadata']['review_id']}")
        logger.info(f"    Text: {chunk['text'][:150]}...")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main(args):
    """Ana embedding oluşturma pipeline'ı"""
    
    logger.info("\n")
    logger.info("╔" + "═" * 78 + "╗")
    logger.info("║" + " " * 20 + "EMBEDDING CREATION PIPELINE" + " " * 31 + "║")
    logger.info("╚" + "═" * 78 + "╝")
    logger.info("\n")
    
    # 1. Model Yükle
    model = load_embedding_model(args.embedding_model)
    
    # 2. Chunks Yükle
    chunks = load_chunks()
    
    # 3. Embeddings Oluştur
    embeddings = create_embeddings(model, chunks, batch_size=args.batch_size)
    
    # 4. FAISS Index
    index = create_faiss_index(embeddings)
    
    # 5. ChromaDB (optional)
    if args.create_chromadb:
        create_chromadb_collection(chunks, embeddings)
    
    # 6. Kaydet
    save_vector_db(index, chunks, embeddings)
    
    # 7. Test
    if args.test_search:
        test_search(index, chunks, embeddings, model)
    
    # Final
    logger.info("\n")
    logger.info("╔" + "═" * 78 + "╗")
    logger.info("║" + " " * 28 + "✅ BAŞARILI! ✅" + " " * 35 + "║")
    logger.info("╚" + "═" * 78 + "╝")
    logger.info("\n")
    logger.info("📁 Vector DB: " + str(config.VECTOR_DB_DIR))
    logger.info("\n🚀 Sonraki adım: python 3_lora_summarizer_training.py")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    setup_logging()
    
    parser = argparse.ArgumentParser(
        description="Create embeddings and vector database for RAG"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Embedding model name (default: from config)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding"
    )
    parser.add_argument(
        "--create-chromadb",
        action="store_true",
        help="Also create ChromaDB collection"
    )
    parser.add_argument(
        "--test-search",
        action="store_true",
        default=True,
        help="Run search test"
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

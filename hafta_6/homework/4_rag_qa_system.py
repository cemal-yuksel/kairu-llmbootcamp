"""
================================================================================
4. RAG Q&A SYSTEM - Intelligent Review Summarizer
================================================================================

RAG (Retrieval Augmented Generation) sistemi ile soru-cevap ve summarization.

Bu script RAG sisteminin kalbidir:
1. FAISS vector DB'den relevant chunks retrieval
2. Fine-tuned summarizer ile generation
3. Context-aware Q&A
4. Multi-review summarization

KULLANIM:
---------
from rag_qa_system import RAGSystem

rag = RAGSystem()
answer = rag.answer_question("What do people say about the acting?")
summary = rag.summarize_reviews(sentiment="positive", top_k=10)

Yazar: Kairu AI - Build with LLMs Bootcamp
Tarih: 2 Kasım 2025
================================================================================
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
from loguru import logger
import pickle
import json

# Proje root
sys.path.append(str(Path(__file__).parent))

from config import config


class RAGSystem:
    """
    RAG-based Q&A ve Summarization Sistemi
    
    Bu sınıf tüm RAG fonksiyonalitesini birleştirir:
    - Vector-based retrieval
    - Context-aware generation
    - Multi-document summarization
    - Question answering
    """
    
    def __init__(
        self,
        vector_db_dir: Path = None,
        model_dir: Path = None,
        embedding_model_name: str = None
    ):
        """
        RAG sistemini başlat
        
        Args:
            vector_db_dir: Vector DB dizini
            model_dir: Fine-tuned model dizini
            embedding_model_name: Embedding model adı
        """
        logger.info("=" * 80)
        logger.info("🚀 RAG SYSTEM İNİTİALİZATION")
        logger.info("=" * 80)
        
        self.vector_db_dir = vector_db_dir or config.VECTOR_DB_DIR
        self.model_dir = model_dir or (config.LORA_MODEL_DIR / "final")
        self.embedding_model_name = embedding_model_name or config.EMBEDDING_MODEL_NAME
        
        # Components
        self.index = None
        self.chunks = None
        self.embedding_model = None
        self.generator_model = None
        self.tokenizer = None
        
        # Load components
        self._load_vector_db()
        self._load_embedding_model()
        self._load_generator_model()
        
        logger.info("✅ RAG System hazır!\n")
    
    def _load_vector_db(self):
        """FAISS index ve chunks'ları yükle"""
        logger.info("📂 Vector DB yükleniyor...")
        
        import faiss
        import numpy as np
        
        # FAISS index - Türkçe karakterli path için serialize kullan
        index_path = self.vector_db_dir / "faiss_index.bin"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index bulunamadı: {index_path}")
        
        # Türkçe karakter sorunu için dosyayı binary olarak oku
        with open(index_path, 'rb') as f:
            index_data = np.frombuffer(f.read(), dtype=np.uint8)
        self.index = faiss.deserialize_index(index_data)
        logger.info(f"  ✅ FAISS index: {self.index.ntotal:,} vectors")
        
        # Chunks
        chunks_path = self.vector_db_dir / "chunks.pkl"
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
        logger.info(f"  ✅ Chunks: {len(self.chunks):,} items")
    
    def _load_embedding_model(self):
        """Embedding modelini yükle"""
        logger.info("🧮 Embedding model yükleniyor...")
        
        from sentence_transformers import SentenceTransformer
        
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        logger.info(f"  ✅ Model: {self.embedding_model_name}")
    
    def _load_generator_model(self):
        """Fine-tuned summarizer modelini yükle"""
        logger.info("🧠 Generator model yükleniyor...")
        
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        from peft import PeftModel
        
        if not self.model_dir.exists():
            logger.warning(f"⚠️  Model bulunamadı: {self.model_dir}")
            logger.warning("  Base model kullanılacak (fine-tuned değil)")
            
            self.tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME)
            self.generator_model = AutoModelForSeq2SeqLM.from_pretrained(config.BASE_MODEL_NAME)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
            self.generator_model = AutoModelForSeq2SeqLM.from_pretrained(str(self.model_dir))
        
        self.generator_model.to(config.DEVICE)
        self.generator_model.eval()
        
        logger.info(f"  ✅ Generator hazır (device: {config.DEVICE})")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_sentiment: Optional[int] = None
    ) -> List[Dict]:
        """
        Query'ye benzer chunks'ları getir
        
        Args:
            query: Arama sorgusu
            top_k: Kaç sonuç
            filter_sentiment: Sentiment filtresi (0=neg, 1=pos, None=all)
            
        Returns:
            Retrieved chunks listesi
        """
        # Query embedding
        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True
        )
        
        # Search
        distances, indices = self.index.search(
            query_embedding.astype(np.float32),
            top_k * 3 if filter_sentiment is not None else top_k
        )
        
        # Sonuçları topla
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            chunk = self.chunks[idx]
            
            # Sentiment filtresi
            if filter_sentiment is not None:
                if chunk["metadata"].get("label") != filter_sentiment:
                    continue
            
            results.append({
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "similarity": float(dist)
            })
            
            if len(results) >= top_k:
                break
        
        return results
    
    def generate_summary(
        self,
        context: str,
        max_length: int = None,
        min_length: int = None
    ) -> str:
        """
        Context'ten summary oluştur
        
        Args:
            context: Özetlenecek text
            max_length: Max summary uzunluğu
            min_length: Min summary uzunluğu
            
        Returns:
            Generated summary
        """
        if max_length is None:
            max_length = config.MAX_SUMMARY_LENGTH
        if min_length is None:
            min_length = config.MIN_SUMMARY_LENGTH
        
        # Tokenize
        inputs = self.tokenizer(
            context,
            max_length=config.MAX_CONTEXT_LENGTH,
            truncation=True,
            return_tensors="pt"
        ).to(config.DEVICE)
        
        # Generate
        with torch.no_grad():
            outputs = self.generator_model.generate(
                **inputs,
                max_new_tokens=max_length,
                min_length=min_length,
                num_beams=config.NUM_BEAMS,
                length_penalty=config.LENGTH_PENALTY,
                no_repeat_ngram_size=config.NO_REPEAT_NGRAM_SIZE,
                early_stopping=config.EARLY_STOPPING
            )
        
        # Decode
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return summary
    
    def answer_question(
        self,
        question: str,
        top_k: int = 5,
        filter_sentiment: Optional[int] = None
    ) -> Dict:
        """
        Soruyu yanıtla (RAG pipeline)
        
        Args:
            question: Soru
            top_k: Kaç chunk retrieve edilecek
            filter_sentiment: Sentiment filtresi
            
        Returns:
            {
                "answer": str,
                "sources": List[Dict],
                "confidence": float
            }
        """
        logger.info(f"❓ Question: {question}")
        
        # 1. Retrieve relevant chunks
        logger.info(f"🔍 Retrieving top-{top_k} chunks...")
        retrieved = self.retrieve(question, top_k=top_k, filter_sentiment=filter_sentiment)
        
        if not retrieved:
            return {
                "answer": "Sorry, I couldn't find relevant information.",
                "sources": [],
                "confidence": 0.0
            }
        
        logger.info(f"  ✅ Retrieved {len(retrieved)} chunks")
        
        # 2. Combine context
        context_parts = []
        for i, chunk in enumerate(retrieved, 1):
            context_parts.append(f"[Review {i}] {chunk['text']}")
        
        context = "\n\n".join(context_parts)
        
        # 3. Generate answer
        logger.info("🤖 Generating answer...")
        prompt = f"Question: {question}\n\nRelevant reviews:\n{context}\n\nAnswer:"
        
        answer = self.generate_summary(
            prompt,
            max_length=200,
            min_length=20
        )
        
        logger.info(f"✅ Answer: {answer[:100]}...")
        
        # Confidence (ortalama similarity)
        confidence = np.mean([r["similarity"] for r in retrieved])
        
        return {
            "answer": answer,
            "sources": retrieved,
            "confidence": float(confidence)
        }
    
    def summarize_reviews(
        self,
        sentiment: Optional[str] = None,
        top_k: int = 10,
        aspect: Optional[str] = None
    ) -> Dict:
        """
        Review'ları özetle
        
        Args:
            sentiment: "positive", "negative", veya None
            top_k: Kaç review
            aspect: Odaklanılacak aspect (örn: "acting", "plot")
            
        Returns:
            {
                "summary": str,
                "num_reviews": int,
                "sentiment_distribution": Dict
            }
        """
        logger.info("📊 Review summarization başlıyor...")
        
        # Sentiment filter
        filter_label = None
        if sentiment == "positive":
            filter_label = 1
        elif sentiment == "negative":
            filter_label = 0
        
        # Query oluştur
        if aspect:
            query = f"Reviews about {aspect}"
        else:
            query = "Overall movie reviews"
        
        # Retrieve
        retrieved = self.retrieve(query, top_k=top_k, filter_sentiment=filter_label)
        
        if not retrieved:
            return {
                "summary": "No reviews found.",
                "num_reviews": 0,
                "sentiment_distribution": {}
            }
        
        # Combine texts
        texts = [r["text"] for r in retrieved]
        combined = "\n\n".join(texts)
        
        # Generate summary
        summary = self.generate_summary(
            combined,
            max_length=300,
            min_length=50
        )
        
        # Sentiment distribution
        sentiments = [r["metadata"].get("label", -1) for r in retrieved]
        from collections import Counter
        sentiment_dist = dict(Counter(sentiments))
        
        return {
            "summary": summary,
            "num_reviews": len(retrieved),
            "sentiment_distribution": sentiment_dist,
            "sources": retrieved
        }


# ============================================================================
# CLI DEMO
# ============================================================================

def main():
    """Demo çalıştırma"""
    from rich.console import Console
    from rich.panel import Panel
    
    console = Console()
    
    console.print("\n")
    console.print(Panel.fit("🎬 RAG Q&A System Demo", style="bold magenta"))
    console.print("\n")
    
    try:
        # RAG sistemi
        rag = RAGSystem()
        
        # Demo sorular
        questions = [
            "What do people say about the acting?",
            "Is this movie suitable for children?",
            "What are the main criticisms?"
        ]
        
        for question in questions:
            console.print(f"\n{'='*80}")
            console.print(f"[bold cyan]Question:[/bold cyan] {question}")
            console.print("="*80)
            
            result = rag.answer_question(question, top_k=3)
            
            console.print(f"\n[bold green]Answer:[/bold green] {result['answer']}")
            console.print(f"\n[dim]Confidence: {result['confidence']:.2f}[/dim]")
            console.print(f"[dim]Sources: {len(result['sources'])} reviews[/dim]")
        
        # Summarization demo
        console.print(f"\n\n{'='*80}")
        console.print("[bold magenta]POSITIVE REVIEWS SUMMARY[/bold magenta]")
        console.print("="*80)
        
        summary_result = rag.summarize_reviews(sentiment="positive", top_k=5)
        console.print(f"\n{summary_result['summary']}")
        console.print(f"\n[dim]Based on {summary_result['num_reviews']} reviews[/dim]")
        
        console.print("\n✅ Demo tamamlandı!")
        
    except Exception as e:
        console.print(f"\n[bold red]❌ Hata:[/bold red] {e}")
        logger.exception(e)


if __name__ == "__main__":
    from loguru import logger
    
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{message}")
    
    main()

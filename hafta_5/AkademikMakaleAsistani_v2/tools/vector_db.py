"""
Enhanced Vector Database Manager with advanced search capabilities
Improved version with LangChain integration and multiple embedding models
"""
import os
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import chromadb
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    print("✅ SentenceTransformers loaded successfully")
except ImportError as e:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print(f"❌ Warning: sentence-transformers not available: {e}")
import numpy as np
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedVectorDB:
    def __init__(self, db_dir: str = None, embedding_model: str = None):
        self.db_dir = Path(db_dir) if db_dir else Path(__file__).parent.parent / 'data' / 'chroma_db'
        self.db_dir.mkdir(parents=True, exist_ok=True)
        
        # Multiple embedding models for different purposes - optimized for speed
        self.primary_model_name = embedding_model or "paraphrase-MiniLM-L3-v2"  # Fastest English model
        self.english_model_name = "paraphrase-MiniLM-L3-v2"  # Much faster than L6-v2
        
        # Initialize models (if available)
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.turkish_model = SentenceTransformer(self.primary_model_name)
            self.english_model = SentenceTransformer(self.english_model_name)
        else:
            self.turkish_model = None
            self.english_model = None
            logger.warning("SentenceTransformer models not available - using fallback embeddings")
        
        # LangChain embeddings
        self.lc_embeddings = HuggingFaceEmbeddings(
            model_name=self.primary_model_name,
            model_kwargs={'device': 'cpu'}
        )
        
        # ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=str(self.db_dir))
        self.collection = self.chroma_client.get_or_create_collection(
            name="academic_papers",
            metadata={"hnsw:space": "cosine"}
        )
        
        # LangChain Vector Store
        self.vector_store = Chroma(
            collection_name="langchain_papers",
            embedding_function=self.lc_embeddings,
            persist_directory=str(self.db_dir / "langchain")
        )
        
        logger.info(f"Vector DB initialized with models: {self.primary_model_name}")
    
    def detect_language(self, text: str) -> str:
        """Detect if text is primarily Turkish or English"""
        try:
            # Simple heuristic - count Turkish specific characters
            turkish_chars = "ğüşıöçĞÜŞİÖÇ"
            turkish_count = sum(1 for char in text if char in turkish_chars)
            return "turkish" if turkish_count > len(text) * 0.01 else "english"
        except:
            return "turkish"  # Default to Turkish
    
    def get_embedding_model(self, language: str = "turkish"):
        """Get appropriate embedding model based on language"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            return None
        return self.turkish_model if language == "turkish" else self.english_model
    
    def add_document(self, text: str, metadata: Dict, pdf_name: str) -> None:
        """Add document with improved chunking and metadata"""
        try:
            # Enhanced text splitting
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " "]
            )
            
            chunks = text_splitter.split_text(text)
            logger.info(f"Split document into {len(chunks)} chunks")
            
            # Detect language
            language = self.detect_language(text)
            embedding_model = self.get_embedding_model(language)
            
            # Prepare data for ChromaDB
            chunk_ids = [f"{pdf_name}_chunk_{i}" for i in range(len(chunks))]
            
            if embedding_model is not None:
                embeddings = embedding_model.encode(chunks).tolist()
            else:
                # Fallback: use simple hash-based embeddings (not ideal but prevents crash)
                logger.warning("Using fallback embeddings - functionality will be limited")
                embeddings = [[hash(chunk) % 1000 / 1000.0] * 384 for chunk in chunks]
            
            # Enhanced metadata for each chunk
            chunk_metadatas = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = {
                    **metadata,
                    "chunk_id": i,
                    "chunk_text_length": len(chunk),
                    "language": language,
                    "pdf_name": pdf_name,
                    "embedding_model": embedding_model.get_sentence_embedding_dimension()
                }
                chunk_metadatas.append(chunk_metadata)
            
            # Add to ChromaDB
            self.collection.add(
                documents=chunks,
                embeddings=embeddings,
                metadatas=chunk_metadatas,
                ids=chunk_ids
            )
            
            # Also add to LangChain vector store
            self.vector_store.add_texts(
                texts=chunks,
                metadatas=chunk_metadatas,
                ids=chunk_ids
            )
            
            logger.info(f"Added {len(chunks)} chunks to vector database for {pdf_name}")
            
        except Exception as e:
            logger.error(f"Error adding document to vector DB: {e}")
    
    def search_documents(self, query: str, pdf_names: List[str] = None, 
                        top_k: int = 5, language: str = None) -> Tuple[List[str], List[Dict]]:
        """Enhanced document search with filtering and language detection"""
        try:
            # Auto-detect query language if not provided
            if language is None:
                language = self.detect_language(query)
            
            # Get appropriate embedding model
            embedding_model = self.get_embedding_model(language)
            
            if embedding_model is not None:
                query_embedding = embedding_model.encode([query])[0].tolist()
            else:
                # Fallback: use simple hash-based embeddings (not ideal but prevents crash)
                logger.warning("Using fallback query embeddings - search results may be poor")
                query_embedding = [hash(query) % 1000 / 1000.0] * 384
            
            # Build filter for specific PDFs
            where_filter = None
            if pdf_names:
                where_filter = {"pdf_name": {"$in": pdf_names}}
            
            # Search using ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter
            )
            
            documents = results['documents'][0] if results['documents'] else []
            metadatas = results['metadatas'][0] if results['metadatas'] else []
            distances = results['distances'][0] if results['distances'] else []
            
            # Add similarity scores to metadata
            for i, metadata in enumerate(metadatas):
                metadata['similarity_score'] = 1 - distances[i]  # Convert distance to similarity
                metadata['search_language'] = language
            
            logger.info(f"Found {len(documents)} relevant documents for query")
            return documents, metadatas
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return [], []
    
    def search_with_langchain(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search using LangChain vector store"""
        try:
            results = self.vector_store.similarity_search_with_score(query, k=top_k)
            
            formatted_results = []
            for doc, score in results:
                result = {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'similarity_score': score
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in LangChain search: {e}")
            return []
    
    def get_document_stats(self) -> Dict:
        """Get statistics about indexed documents"""
        try:
            collection_info = self.collection.get()
            
            # Count unique PDFs
            pdf_names = set()
            languages = {}
            
            for metadata in collection_info['metadatas']:
                pdf_name = metadata.get('pdf_name', 'unknown')
                pdf_names.add(pdf_name)
                
                language = metadata.get('language', 'unknown')
                languages[language] = languages.get(language, 0) + 1
            
            stats = {
                'total_chunks': len(collection_info['documents']),
                'total_pdfs': len(pdf_names),
                'pdf_names': list(pdf_names),
                'language_distribution': languages,
                'collection_name': self.collection.name
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting document stats: {e}")
            return {}
    
    def delete_document(self, pdf_name: str) -> bool:
        """Delete all chunks related to a specific PDF"""
        try:
            # Get all chunk IDs for this PDF
            results = self.collection.get(
                where={"pdf_name": pdf_name}
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} chunks for {pdf_name}")
                return True
            else:
                logger.warning(f"No chunks found for {pdf_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting document {pdf_name}: {e}")
            return False
    
    def update_document(self, pdf_name: str, new_text: str, new_metadata: Dict) -> bool:
        """Update an existing document"""
        try:
            # Delete old version
            self.delete_document(pdf_name)
            
            # Add new version
            self.add_document(new_text, new_metadata, pdf_name)
            
            logger.info(f"Updated document: {pdf_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating document {pdf_name}: {e}")
            return False
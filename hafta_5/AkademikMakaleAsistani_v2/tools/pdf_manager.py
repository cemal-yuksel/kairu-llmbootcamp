"""
Enhanced PDF Manager with advanced text extraction and metadata handling
Combines hafta_4 pdf_manager features with hafta_5 LangChain integration
"""
import os
import json
import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
from PyPDF2 import PdfReader
import openai
from dotenv import load_dotenv
import hashlib
import shutil
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class EnhancedPDFManager:
    """
    Enhanced PDF Manager with comprehensive document processing capabilities
    Includes features from hafta_4 with advanced metadata management
    """
    
    def __init__(self, pdf_dir: str = None, data_dir: str = None):
        self.pdf_dir = Path(pdf_dir) if pdf_dir else Path(__file__).parent.parent / 'pdfs'
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent.parent / 'data'
        self.metadata_file = self.data_dir / 'metadata.json'
        self.library_index_file = self.data_dir / 'library_index.json'
        
        # Create directories if they don't exist
        self.pdf_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize OpenAI client
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.openai_api_key:
            self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
        else:
            logger.warning("OpenAI API key not found - title extraction will be limited")
            self.openai_client = None
    
    def load_metadata(self) -> Dict:
        """Load metadata from JSON file"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            return {}
    
    def save_metadata(self, metadata: Dict) -> None:
        """Save metadata to JSON file"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"Metadata saved to {self.metadata_file}")
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def load_library_index(self) -> Dict:
        """Load library index for advanced search"""
        try:
            if self.library_index_file.exists():
                with open(self.library_index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {"documents": {}, "categories": {}, "authors": {}, "keywords": {}}
        except Exception as e:
            logger.error(f"Error loading library index: {e}")
            return {"documents": {}, "categories": {}, "authors": {}, "keywords": {}}
    
    def save_library_index(self, index: Dict) -> None:
        """Save library index"""
        try:
            with open(self.library_index_file, 'w', encoding='utf-8') as f:
                json.dump(index, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving library index: {e}")
    
    def extract_text(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                reader = PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text()
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def extract_enhanced_metadata(self, pdf_path: str, pdf_text: str) -> Dict[str, Any]:
        """Extract comprehensive metadata from PDF"""
        try:
            # Basic file metadata
            file_stat = os.stat(pdf_path)
            basic_metadata = {
                "file_size": file_stat.st_size,
                "upload_date": datetime.datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                "file_hash": self._calculate_file_hash(pdf_path),
                "text_length": len(pdf_text),
                "page_count": self._get_page_count(pdf_path)
            }
            
            # Enhanced metadata with LLM
            if self.openai_client and pdf_text:
                enhanced_metadata = self._extract_metadata_with_llm(pdf_text)
                basic_metadata.update(enhanced_metadata)
            
            return basic_metadata
            
        except Exception as e:
            logger.error(f"Error extracting enhanced metadata: {e}")
            return {"error": str(e)}
    
    def _extract_metadata_with_llm(self, pdf_text: str) -> Dict[str, Any]:
        """Extract metadata using LLM analysis"""
        try:
            prompt = f"""
Aşağıdaki akademik makale metnini analiz ederek şu bilgileri JSON formatında çıkar:

1. title (makale başlığı)
2. authors (yazarlar - liste halinde)
3. abstract (özet - varsa)
4. keywords (anahtar kelimeler - liste halinde) 
5. research_field (araştırma alanı)
6. research_type (nicel/nitel/karma/teorik)
7. publication_year (yayın yılı - varsa)
8. journal (dergi adı - varsa)
9. doi (DOI - varsa)
10. language (makale dili)

Makale Metni (İlk 3000 karakter):
{pdf_text[:3000]}

JSON formatında yanıt ver:
{{
    "title": "string",
    "authors": ["yazar1", "yazar2"],
    "abstract": "string",
    "keywords": ["kelime1", "kelime2"],
    "research_field": "string",
    "research_type": "string",
    "publication_year": "string", 
    "journal": "string",
    "doi": "string",
    "language": "string"
}}
"""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Sen akademik makale metadatası çıkaran bir uzmansın. JSON formatında yanıt verirsin."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=800,
                timeout=60
            )
            
            response_text = response.choices[0].message.content
            
            # Parse JSON response
            try:
                metadata = json.loads(response_text)
                return metadata
            except json.JSONDecodeError:
                logger.warning("Could not parse LLM metadata response as JSON, trying to extract title manually")
                # Try to extract title from response text
                title_match = re.search(r'"title":\s*"([^"]+)"', response_text)
                if title_match:
                    return {"title": title_match.group(1)}
                return {}
                
        except Exception as e:
            logger.error(f"Error extracting metadata with LLM: {e}")
            return {}
    
    def _clean_filename(self, title: str) -> str:
        """Clean title for use as filename"""
        if not title:
            return ""
        
        # Remove quotes and clean text
        title = title.strip().strip('"\'')
        
        # Replace invalid filename characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            title = title.replace(char, '-')
        
        # Remove multiple spaces and dashes
        title = re.sub(r'\s+', ' ', title)
        title = re.sub(r'-+', '-', title)
        
        # Limit length
        return title[:150].strip()

    def extract_title_with_llm(self, pdf_text: str) -> str:
        """Extract title from PDF text using LLM"""
        try:
            if not self.openai_client:
                return None
                
            prompt = f"""
Aşağıdaki akademik makale metninden SADECE makale başlığını çıkar. 
Sadece başlığı ver, başka hiçbir şey yazma.
Eğer başlık bulamazsan "Başlık bulunamadı" yaz.

Makale Metni (İlk 3000 karakter):
{pdf_text[:3000]}

Başlık:"""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Sen akademik makale başlığı çıkarma uzmanısın. Sadece başlığı ver, başka hiçbir şey yazma."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=200,
                timeout=30
            )
            
            title = response.choices[0].message.content.strip()
            
            # Check for failure indicators
            failure_indicators = ["başlık bulunamadı", "başlık çıkarılamadı", "title not found", "no title"]
            if any(indicator in title.lower() for indicator in failure_indicators):
                return None
            
            return self._clean_filename(title)
            
        except Exception as e:
            logger.error(f"Error extracting title with LLM: {e}")
            return None
    
    def save_pdf(self, uploaded_file) -> Tuple[str, str]:
        """
        Save uploaded PDF with intelligent naming and metadata extraction
        Returns (file_path, final_name)
        """
        try:
            # Save initial file
            original_name = uploaded_file.name
            temp_path = self.pdf_dir / original_name
            
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            # Extract text and metadata
            pdf_text = self.extract_text(str(temp_path))
            
            if not pdf_text:
                logger.warning(f"No text extracted from {original_name}")
                return str(temp_path), original_name
            
            # Enhanced metadata extraction
            enhanced_metadata = self.extract_enhanced_metadata(str(temp_path), pdf_text)
            
            # Intelligent title extraction - Try multiple methods
            extracted_title = enhanced_metadata.get('title')
            if not extracted_title or extracted_title == "Başlık çıkarılamadı":
                extracted_title = self.extract_title_with_llm(pdf_text)
            
            # Clean and validate title
            if extracted_title:
                # Remove common invalid title patterns
                invalid_patterns = ["Başlık çıkarılamadı", "Bilinmeyen Başlık", "Title not found", "No title", ""]
                if extracted_title not in invalid_patterns and len(extracted_title.strip()) > 3:
                    # Clean title for filename
                    clean_title = self._clean_filename(extracted_title)
                    if len(clean_title) > 3:
                        extracted_title = clean_title
                    else:
                        extracted_title = None
                else:
                    extracted_title = None
            
            # Generate new filename if title is meaningful
            if extracted_title and len(extracted_title) > 3:
                new_filename = f"{extracted_title}.pdf"
                new_path = self.pdf_dir / new_filename
                
                # Handle duplicate names
                counter = 1
                while new_path.exists():
                    new_filename = f"{extracted_title}_{counter}.pdf"
                    new_path = self.pdf_dir / new_filename
                    counter += 1
                
                # Rename file
                temp_path.rename(new_path)
                final_name = new_filename
                file_path = str(new_path)
            else:
                final_name = original_name
                file_path = str(temp_path)
            
            # Update metadata store
            metadata_store = self.load_metadata()
            metadata_store[final_name] = {
                'original_name': original_name,
                'extracted_metadata': enhanced_metadata,
                'processing_date': datetime.datetime.now().isoformat(),
                'file_path': file_path
            }
            self.save_metadata(metadata_store)
            
            # Update library index
            self._update_library_index(final_name, enhanced_metadata, pdf_text[:1000])
            
            logger.info(f"PDF saved successfully: {final_name}")
            return file_path, final_name
            
        except Exception as e:
            logger.error(f"Error saving PDF: {e}")
            # Return original file if something goes wrong
            return str(self.pdf_dir / uploaded_file.name), uploaded_file.name
    
    def delete_pdf(self, pdf_name: str) -> bool:
        """Delete PDF and its metadata"""
        try:
            # Remove file
            file_path = self.pdf_dir / pdf_name
            if file_path.exists():
                file_path.unlink()
            
            # Remove from metadata
            metadata_store = self.load_metadata()
            if pdf_name in metadata_store:
                del metadata_store[pdf_name]
                self.save_metadata(metadata_store)
            
            # Remove from library index
            library_index = self.load_library_index()
            if pdf_name in library_index.get("documents", {}):
                del library_index["documents"][pdf_name]
                self.save_library_index(library_index)
            
            logger.info(f"PDF deleted successfully: {pdf_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting PDF {pdf_name}: {e}")
            return False
    
    def get_uploaded_pdfs(self) -> List[str]:
        """Get list of uploaded PDF files"""
        try:
            pdf_files = [f.name for f in self.pdf_dir.glob("*.pdf")]
            return sorted(pdf_files)
        except Exception as e:
            logger.error(f"Error getting uploaded PDFs: {e}")
            return []
    
    def process_pdf(self, pdf_path: str) -> Tuple[str, Dict]:
        """Process PDF and extract text with metadata"""
        try:
            # Extract text
            text = self.extract_text(pdf_path)
            if not text:
                raise ValueError("No text extracted from PDF")
            
            # Extract title using LLM
            title = self.extract_title_with_llm(text)
            
            # Create metadata
            pdf_name = Path(pdf_path).name
            metadata = {
                'filename': pdf_name,
                'title': title,
                'text_length': len(text),
                'processed_date': str(Path(pdf_path).stat().st_mtime),
                'word_count': len(text.split()),
                'character_count': len(text)
            }
            
            # Update global metadata
            all_metadata = self.load_metadata()
            all_metadata[pdf_name] = metadata
            self.save_metadata(all_metadata)
            
            logger.info(f"Processed PDF: {pdf_name} - Title: {title}")
            return text, metadata
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            return "", {}
    
    def get_pdf_info(self, pdf_name: str) -> Optional[Dict]:
        """Get metadata info for a specific PDF"""
        metadata = self.load_metadata()
        return metadata.get(pdf_name)
    
    def chunk_text(self, text: str, chunk_size: int = 1200, chunk_overlap: int = 200) -> List[str]:
        """Enhanced text chunking with overlap"""
        try:
            sentences = text.split('. ')
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                test_chunk = current_chunk + " " + sentence
                
                if len(test_chunk) > chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    
                    # Add overlap
                    words = current_chunk.split()
                    overlap_size = min(chunk_overlap // 5, len(words))
                    current_chunk = ' '.join(words[-overlap_size:]) + " " + sentence
                else:
                    current_chunk = test_chunk
            
            if current_chunk:
                chunks.append(current_chunk.strip())
                
            logger.info(f"Text chunked into {len(chunks)} chunks")
            return [chunk for chunk in chunks if chunk.strip()]
            
        except Exception as e:
            logger.error(f"Error chunking text: {e}")
            return [text]  # Return original text as single chunk if chunking fails

    def get_pdf_library_info(self) -> Dict[str, Any]:
        """Get comprehensive library information"""
        try:
            metadata_store = self.load_metadata()
            library_index = self.load_library_index()
            
            library_stats = {
                "total_documents": len(metadata_store),
                "total_size": sum(meta.get("extracted_metadata", {}).get("file_size", 0) for meta in metadata_store.values()),
                "research_fields": list(library_index.get("categories", {}).keys()),
                "authors": list(library_index.get("authors", {}).keys()),
                "document_types": {},
                "recent_uploads": []
            }
            
            # Analyze document types
            for doc_name, doc_info in metadata_store.items():
                research_type = doc_info.get("extracted_metadata", {}).get("research_type", "Unknown")
                library_stats["document_types"][research_type] = library_stats["document_types"].get(research_type, 0) + 1
            
            # Get recent uploads (last 5)
            recent_docs = sorted(
                metadata_store.items(),
                key=lambda x: x[1].get("processing_date", ""),
                reverse=True
            )[:5]
            
            library_stats["recent_uploads"] = [
                {
                    "name": doc[0],
                    "title": doc[1].get("extracted_metadata", {}).get("title", ""),
                    "date": doc[1].get("processing_date", "")
                }
                for doc in recent_docs
            ]
            
            return library_stats
            
        except Exception as e:
            logger.error(f"Error getting library info: {e}")
            return {}
    
    def search_library(self, query: str, category: str = None) -> List[Dict[str, Any]]:
        """Search PDF library by query and optional category"""
        try:
            metadata_store = self.load_metadata()
            library_index = self.load_library_index()
            
            results = []
            query_lower = query.lower()
            
            for doc_name, doc_info in metadata_store.items():
                metadata = doc_info.get("extracted_metadata", {})
                
                # Search in title, authors, keywords, abstract
                searchable_text = " ".join([
                    str(metadata.get("title", "")),
                    " ".join(metadata.get("authors", [])),
                    " ".join(metadata.get("keywords", [])),
                    str(metadata.get("abstract", "")),
                    str(metadata.get("research_field", ""))
                ]).lower()
                
                # Category filter
                if category and metadata.get("research_field", "").lower() != category.lower():
                    continue
                
                # Text search
                if query_lower in searchable_text:
                    results.append({
                        "name": doc_name,
                        "title": metadata.get("title", doc_name),
                        "authors": metadata.get("authors", []),
                        "research_field": metadata.get("research_field", ""),
                        "relevance_score": self._calculate_relevance(query_lower, searchable_text)
                    })
            
            # Sort by relevance
            results.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching library: {e}")
            return []
    
    def _update_library_index(self, pdf_name: str, metadata: Dict[str, Any], text_sample: str):
        """Update library index for search capabilities"""
        try:
            library_index = self.load_library_index()
            
            # Add document to index
            library_index["documents"][pdf_name] = {
                "title": metadata.get("title", ""),
                "authors": metadata.get("authors", []),
                "keywords": metadata.get("keywords", []),
                "research_field": metadata.get("research_field", ""),
                "research_type": metadata.get("research_type", ""),
                "indexed_date": datetime.datetime.now().isoformat()
            }
            
            # Update categories
            research_field = metadata.get("research_field", "")
            if research_field:
                if research_field not in library_index["categories"]:
                    library_index["categories"][research_field] = []
                library_index["categories"][research_field].append(pdf_name)
            
            # Update authors
            for author in metadata.get("authors", []):
                if author not in library_index["authors"]:
                    library_index["authors"][author] = []
                library_index["authors"][author].append(pdf_name)
            
            # Update keywords
            for keyword in metadata.get("keywords", []):
                if keyword not in library_index["keywords"]:
                    library_index["keywords"][keyword] = []
                library_index["keywords"][keyword].append(pdf_name)
            
            self.save_library_index(library_index)
            
        except Exception as e:
            logger.error(f"Error updating library index: {e}")
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate file hash for duplicate detection"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating file hash: {e}")
            return ""
    
    def _get_page_count(self, pdf_path: str) -> int:
        """Get number of pages in PDF"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PdfReader(file)
                return len(reader.pages)
        except Exception:
            return 0
    
    def _calculate_relevance(self, query: str, text: str) -> float:
        """Calculate relevance score for search results"""
        try:
            query_words = query.split()
            score = 0
            
            for word in query_words:
                if word in text:
                    score += text.count(word)
            
            return score / len(query_words) if query_words else 0
            
        except Exception:
            return 0
    
    def get_pdf_metadata(self, pdf_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific PDF"""
        try:
            metadata = self.load_metadata()
            return metadata.get(pdf_name)
        except Exception as e:
            logger.error(f"Error getting PDF metadata: {e}")
            return None

# Legacy functions for backward compatibility
def extract_text(pdf_path: str) -> str:
    """Legacy function wrapper"""
    manager = EnhancedPDFManager()
    return manager.extract_text(pdf_path)

def save_pdf(uploaded_file):
    """Legacy function wrapper"""
    manager = EnhancedPDFManager()
    return manager.save_pdf(uploaded_file)

def delete_pdf(pdf_name: str):
    """Legacy function wrapper"""
    manager = EnhancedPDFManager()
    return manager.delete_pdf(pdf_name)

def extract_title_with_llm(pdf_text: str) -> str:
    """Legacy function wrapper"""
    manager = EnhancedPDFManager()
    return manager.extract_title_with_llm(pdf_text)
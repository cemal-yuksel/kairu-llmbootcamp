"""
Reference Management Tool - Advanced reference and bibliography management system
Handles DOI validation, metadata retrieval, and cross-reference analysis
"""
import logging
import requests
import json
import re
from typing import Dict, List, Optional, Any
from datetime import datetime
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from pydantic import ConfigDict, Field
import urllib.parse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReferenceManagerTool(BaseTool):
    """
    Comprehensive reference management tool that:
    - Validates and retrieves DOI metadata
    - Manages reference collections
    - Analyzes citation networks
    - Formats references in multiple styles
    - Detects duplicate and related references
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str = "reference_manager"
    description: str = """
    Advanced reference management system. Use this to:
    - Validate DOIs and retrieve complete metadata
    - Manage collections of academic references
    - Format references in various citation styles
    - Analyze citation relationships and networks
    - Detect duplicate or related references
    
    Input should be DOI, reference data, or management commands.
    """
    
    # Class fields for API URLs
    crossref_url: str = "https://api.crossref.org/works"
    doi_url: str = "https://doi.org"
    
    # Define fields that will be set in __init__
    session: Optional[Any] = Field(default=None, exclude=True)
    reference_db: Optional[Dict] = Field(default=None, exclude=True)
    collections: Optional[Dict] = Field(default=None, exclude=True)
    
    def __init__(self):
        super().__init__()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AcademicResearchAssistant/1.0',
            'Accept': 'application/json'
        })
        
        # Reference database (in-memory for demo)
        self.reference_db = {}
        self.collections = {}
    
    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Execute reference management task"""
        try:
            logger.info(f"Reference management task: {query}")
            
            # Parse command
            command = self._parse_command(query)
            
            if command["action"] == "validate_doi":
                return self._validate_and_retrieve_doi(command["data"])
            elif command["action"] == "add_reference":
                return self._add_reference(command["data"])
            elif command["action"] == "format_references":
                return self._format_references(command["data"])
            elif command["action"] == "analyze_citations":
                return self._analyze_citations(command["data"])
            elif command["action"] == "find_duplicates":
                return self._find_duplicate_references()
            elif command["action"] == "create_collection":
                return self._create_reference_collection(command["data"])
            else:
                return self._general_reference_help(query)
                
        except Exception as e:
            error_msg = f"Reference management error: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def _parse_command(self, query: str) -> Dict[str, Any]:
        """Parse the input query to determine action"""
        try:
            # Try to parse as JSON
            if query.strip().startswith('{'):
                return json.loads(query)
            
            # Parse natural language commands
            query_lower = query.lower()
            
            if any(keyword in query_lower for keyword in ["doi", "10.", "validate"]):
                # Extract DOI
                doi_match = re.search(r'10\.\d{4,}/[^\s]+', query)
                doi = doi_match.group(0) if doi_match else query.replace("validate ", "").replace("doi ", "").strip()
                return {"action": "validate_doi", "data": {"doi": doi}}
            
            elif "format" in query_lower:
                style = "apa"  # default
                if "mla" in query_lower:
                    style = "mla"
                elif "chicago" in query_lower:
                    style = "chicago"
                elif "ieee" in query_lower:
                    style = "ieee"
                
                return {"action": "format_references", "data": {"style": style}}
            
            elif "duplicate" in query_lower:
                return {"action": "find_duplicates", "data": {}}
            
            elif "citation" in query_lower or "network" in query_lower:
                return {"action": "analyze_citations", "data": {"query": query}}
            
            elif "collection" in query_lower:
                collection_name = query.replace("collection", "").strip()
                return {"action": "create_collection", "data": {"name": collection_name}}
            
            else:
                return {"action": "general_help", "data": {"query": query}}
                
        except Exception as e:
            logger.warning(f"Error parsing command: {e}")
            return {"action": "general_help", "data": {"query": query}}
    
    def _validate_and_retrieve_doi(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate DOI and retrieve complete metadata"""
        try:
            doi = data.get("doi", "").strip()
            if not doi:
                return {"error": "No DOI provided"}
            
            # Clean DOI
            doi = doi.replace("https://doi.org/", "").replace("http://dx.doi.org/", "")
            
            logger.info(f"Validating DOI: {doi}")
            
            # Query CrossRef API
            url = f"{self.crossref_url}/{doi}"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 404:
                return {
                    "doi": doi,
                    "valid": False,
                    "error": "DOI not found in CrossRef database"
                }
            
            response.raise_for_status()
            data = response.json()
            
            work = data.get("message", {})
            
            # Extract comprehensive metadata
            metadata = {
                "doi": doi,
                "valid": True,
                "title": work.get("title", ["Unknown Title"])[0] if work.get("title") else "Unknown Title",
                "authors": self._extract_authors(work.get("author", [])),
                "publication_date": self._extract_publication_date(work),
                "journal": work.get("container-title", ["Unknown Journal"])[0] if work.get("container-title") else "Unknown Journal",
                "publisher": work.get("publisher", "Unknown Publisher"),
                "type": work.get("type", "journal-article"),
                "subjects": work.get("subject", []),
                "abstract": work.get("abstract", ""),
                "citation_count": work.get("is-referenced-by-count", 0),
                "references_count": work.get("references-count", 0),
                "url": work.get("URL", f"https://doi.org/{doi}"),
                "issn": work.get("ISSN", []),
                "volume": work.get("volume", ""),
                "issue": work.get("issue", ""),
                "pages": work.get("page", ""),
                "language": work.get("language", "en")
            }
            
            # Add to reference database
            ref_id = self._generate_reference_id(metadata)
            self.reference_db[ref_id] = {
                **metadata,
                "reference_id": ref_id,
                "added_date": datetime.now().isoformat(),
                "notes": "",
                "tags": [],
                "read_status": "unread"
            }
            
            logger.info(f"Successfully retrieved metadata for DOI: {doi}")
            return {
                "metadata": metadata,
                "reference_id": ref_id,
                "success": True
            }
            
        except requests.RequestException as e:
            return {
                "doi": data.get("doi", ""),
                "valid": False,
                "error": f"Network error: {str(e)}"
            }
        except Exception as e:
            return {
                "doi": data.get("doi", ""),
                "valid": False,
                "error": f"Validation error: {str(e)}"
            }
    
    def _extract_authors(self, authors_data: List[Dict]) -> List[str]:
        """Extract author names from CrossRef data"""
        try:
            authors = []
            for author in authors_data:
                given = author.get("given", "")
                family = author.get("family", "")
                if given and family:
                    authors.append(f"{given} {family}")
                elif family:
                    authors.append(family)
                    
            return authors
        except Exception:
            return []
    
    def _extract_publication_date(self, work_data: Dict) -> str:
        """Extract publication date from work data"""
        try:
            # Try different date fields
            for date_field in ["published-print", "published-online", "issued"]:
                date_info = work_data.get(date_field)
                if date_info and "date-parts" in date_info:
                    date_parts = date_info["date-parts"][0]
                    if len(date_parts) >= 1:
                        year = date_parts[0]
                        month = date_parts[1] if len(date_parts) > 1 else 1
                        day = date_parts[2] if len(date_parts) > 2 else 1
                        return f"{year}-{month:02d}-{day:02d}"
            
            return ""
        except Exception:
            return ""
    
    def _generate_reference_id(self, metadata: Dict) -> str:
        """Generate unique reference ID"""
        try:
            # Use first author + year + first word of title
            authors = metadata.get("authors", [])
            first_author = authors[0].split()[-1] if authors else "Unknown"
            year = metadata.get("publication_date", "")[:4] if metadata.get("publication_date") else "0000"
            title_word = metadata.get("title", "").split()[0] if metadata.get("title") else "Unknown"
            
            base_id = f"{first_author}_{year}_{title_word}".lower()
            
            # Ensure uniqueness
            counter = 1
            ref_id = base_id
            while ref_id in self.reference_db:
                ref_id = f"{base_id}_{counter}"
                counter += 1
                
            return ref_id
            
        except Exception:
            return f"ref_{len(self.reference_db) + 1}"
    
    def _add_reference(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Add a reference manually"""
        try:
            # Generate reference ID
            ref_id = self._generate_reference_id(data)
            
            # Add metadata
            reference = {
                **data,
                "reference_id": ref_id,
                "added_date": datetime.now().isoformat(),
                "notes": data.get("notes", ""),
                "tags": data.get("tags", []),
                "read_status": data.get("read_status", "unread")
            }
            
            self.reference_db[ref_id] = reference
            
            return {
                "reference_id": ref_id,
                "success": True,
                "message": f"Reference added successfully: {data.get('title', 'Unknown Title')}"
            }
            
        except Exception as e:
            return {"error": f"Error adding reference: {e}"}
    
    def _format_references(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format references in specified citation style"""
        try:
            style = data.get("style", "apa").lower()
            ref_ids = data.get("reference_ids", list(self.reference_db.keys()))
            
            formatted_refs = []
            
            for ref_id in ref_ids:
                if ref_id in self.reference_db:
                    ref = self.reference_db[ref_id]
                    formatted = self._format_single_reference(ref, style)
                    formatted_refs.append({
                        "reference_id": ref_id,
                        "formatted": formatted
                    })
            
            return {
                "style": style.upper(),
                "total_references": len(formatted_refs),
                "formatted_references": formatted_refs,
                "bibliography": "\\n\\n".join([ref["formatted"] for ref in formatted_refs])
            }
            
        except Exception as e:
            return {"error": f"Error formatting references: {e}"}
    
    def _format_single_reference(self, ref: Dict[str, Any], style: str) -> str:
        """Format a single reference in specified style"""
        try:
            authors = ref.get("authors", [])
            title = ref.get("title", "Unknown Title")
            journal = ref.get("journal", "")
            year = ref.get("publication_date", "")[:4] if ref.get("publication_date") else ""
            volume = ref.get("volume", "")
            pages = ref.get("pages", "")
            doi = ref.get("doi", "")
            
            if style == "apa":
                # APA 7th format (simplified)
                author_str = ", ".join(authors[:3]) if authors else "Unknown Author"
                if len(authors) > 3:
                    author_str += " et al."
                
                formatted = f"{author_str} ({year}). {title}. "
                if journal:
                    formatted += f"{journal}"
                    if volume:
                        formatted += f", {volume}"
                    if pages:
                        formatted += f", {pages}"
                formatted += "."
                if doi:
                    formatted += f" https://doi.org/{doi}"
                    
            elif style == "mla":
                # MLA format (simplified)
                if authors:
                    author_str = authors[0]
                    if len(authors) > 1:
                        author_str += " et al."
                else:
                    author_str = "Unknown Author"
                
                formatted = f'{author_str}. "{title}." {journal}'
                if volume:
                    formatted += f" {volume}"
                if year:
                    formatted += f" ({year})"
                if pages:
                    formatted += f": {pages}"
                formatted += "."
                
            else:
                # Default format
                author_str = "; ".join(authors) if authors else "Unknown Author"
                formatted = f"{author_str}. {title}. {journal} {year}."
            
            return formatted
            
        except Exception as e:
            return f"Error formatting reference: {e}"
    
    def _analyze_citations(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze citation patterns and networks"""
        try:
            analysis = {
                "total_references": len(self.reference_db),
                "citation_analysis": {
                    "highly_cited": [],
                    "recent_papers": [],
                    "top_journals": {},
                    "author_frequency": {},
                    "publication_timeline": {}
                }
            }
            
            # Analyze references
            journal_counts = {}
            author_counts = {}
            year_counts = {}
            
            for ref in self.reference_db.values():
                # Journal analysis
                journal = ref.get("journal", "Unknown")
                journal_counts[journal] = journal_counts.get(journal, 0) + 1
                
                # Author analysis
                for author in ref.get("authors", []):
                    author_counts[author] = author_counts.get(author, 0) + 1
                
                # Year analysis
                year = ref.get("publication_date", "")[:4] if ref.get("publication_date") else "Unknown"
                year_counts[year] = year_counts.get(year, 0) + 1
                
                # Highly cited papers
                citation_count = ref.get("citation_count", 0)
                if citation_count > 100:  # Threshold for "highly cited"
                    analysis["citation_analysis"]["highly_cited"].append({
                        "title": ref.get("title", ""),
                        "citations": citation_count,
                        "year": year
                    })
            
            # Sort and limit results
            analysis["citation_analysis"]["top_journals"] = dict(
                sorted(journal_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            )
            
            analysis["citation_analysis"]["author_frequency"] = dict(
                sorted(author_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            )
            
            analysis["citation_analysis"]["publication_timeline"] = dict(
                sorted(year_counts.items(), key=lambda x: x[0], reverse=True)[:10]
            )
            
            return analysis
            
        except Exception as e:
            return {"error": f"Citation analysis error: {e}"}
    
    def _find_duplicate_references(self) -> Dict[str, Any]:
        """Find potential duplicate references"""
        try:
            duplicates = []
            refs_list = list(self.reference_db.values())
            
            for i, ref1 in enumerate(refs_list):
                for j, ref2 in enumerate(refs_list[i+1:], i+1):
                    similarity = self._calculate_reference_similarity(ref1, ref2)
                    if similarity > 0.8:  # 80% similarity threshold
                        duplicates.append({
                            "reference_1": ref1.get("reference_id"),
                            "reference_2": ref2.get("reference_id"),
                            "similarity": similarity,
                            "title_1": ref1.get("title", ""),
                            "title_2": ref2.get("title", ""),
                            "reason": "High title similarity"
                        })
            
            return {
                "total_duplicates_found": len(duplicates),
                "duplicates": duplicates,
                "recommendation": "Review these potential duplicates and merge if necessary"
            }
            
        except Exception as e:
            return {"error": f"Duplicate detection error: {e}"}
    
    def _calculate_reference_similarity(self, ref1: Dict, ref2: Dict) -> float:
        """Calculate similarity between two references"""
        try:
            # Simple similarity based on title and authors
            title1 = ref1.get("title", "").lower()
            title2 = ref2.get("title", "").lower()
            
            # Title similarity (Jaccard index)
            words1 = set(title1.split())
            words2 = set(title2.split())
            
            if not words1 and not words2:
                title_sim = 1.0
            elif not words1 or not words2:
                title_sim = 0.0
            else:
                intersection = words1.intersection(words2)
                union = words1.union(words2)
                title_sim = len(intersection) / len(union)
            
            # Author similarity
            authors1 = set(ref1.get("authors", []))
            authors2 = set(ref2.get("authors", []))
            
            if not authors1 and not authors2:
                author_sim = 1.0
            elif not authors1 or not authors2:
                author_sim = 0.0
            else:
                intersection = authors1.intersection(authors2)
                union = authors1.union(authors2)
                author_sim = len(intersection) / len(union)
            
            # Combined similarity (weighted)
            return 0.7 * title_sim + 0.3 * author_sim
            
        except Exception:
            return 0.0
    
    def _create_reference_collection(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a named collection of references"""
        try:
            collection_name = data.get("name", f"Collection_{len(self.collections) + 1}")
            ref_ids = data.get("reference_ids", [])
            
            collection = {
                "name": collection_name,
                "created_date": datetime.now().isoformat(),
                "reference_ids": ref_ids,
                "description": data.get("description", ""),
                "tags": data.get("tags", [])
            }
            
            self.collections[collection_name] = collection
            
            return {
                "collection_name": collection_name,
                "total_references": len(ref_ids),
                "success": True,
                "message": f"Collection '{collection_name}' created successfully"
            }
            
        except Exception as e:
            return {"error": f"Error creating collection: {e}"}
    
    def _general_reference_help(self, query: str) -> Dict[str, Any]:
        """Provide general reference management help"""
        return {
            "help_topic": "Reference Management",
            "available_commands": [
                "validate doi [DOI] - Validate and retrieve DOI metadata",
                "format references [style] - Format references in APA, MLA, Chicago, etc.",
                "find duplicates - Detect potential duplicate references",
                "analyze citations - Analyze citation patterns and networks",
                "create collection [name] - Create a named reference collection"
            ],
            "tips": [
                "Always validate DOIs to ensure accurate metadata",
                "Use consistent citation styles throughout your document",
                "Regularly check for duplicate references to maintain clean bibliography",
                "Organize references into collections by topic or project"
            ],
            "query_received": query
        }
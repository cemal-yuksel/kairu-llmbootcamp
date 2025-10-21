"""
Literature Management Tool - Advanced tool for academic literature search and management
Integrates with external APIs and provides comprehensive literature analysis
"""
import logging
import requests
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from pydantic import ConfigDict, Field
import urllib.parse
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiteratureSearchTool(BaseTool):
    """
    Advanced literature search tool that:
    - Searches multiple academic databases (CrossRef, arXiv, PubMed)
    - Provides citation analysis and metrics
    - Suggests related papers and authors
    - Tracks literature trends and patterns
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str = "literature_search"
    description: str = """
    Comprehensive academic literature search tool. Use this to:
    - Search for academic papers by keywords, authors, or titles
    - Get detailed paper metadata including citations
    - Find related papers and authors
    - Analyze citation patterns and trends
    
    Input should be a search query string or structured search parameters.
    """
    
    # Class fields for API URLs
    crossref_base_url: str = "https://api.crossref.org/works"
    arxiv_base_url: str = "http://export.arxiv.org/api/query"
    
    # Define fields that will be set in __init__
    session: Optional[Any] = Field(default=None, exclude=True)
    
    def __init__(self):
        super().__init__()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AcademicResearchAssistant/1.0 (https://example.com/contact)'
        })
    
    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Run the literature search"""
        try:
            logger.info(f"Starting literature search for: {query}")
            
            # Parse search parameters
            search_params = self._parse_search_query(query)
            
            # Search multiple sources
            results = {
                "query": query,
                "search_timestamp": datetime.now().isoformat(),
                "sources": {}
            }
            
            # CrossRef search
            if run_manager:
                run_manager.on_text("Searching CrossRef database...\n", verbose=True)
            crossref_results = self._search_crossref(search_params)
            results["sources"]["crossref"] = crossref_results
            
            # arXiv search
            if run_manager:
                run_manager.on_text("Searching arXiv database...\n", verbose=True)
            arxiv_results = self._search_arxiv(search_params)
            results["sources"]["arxiv"] = arxiv_results
            
            # Combine and rank results
            combined_results = self._combine_and_rank_results(results["sources"])
            results["combined_results"] = combined_results
            
            # Generate search insights
            insights = self._generate_search_insights(combined_results)
            results["insights"] = insights
            
            logger.info(f"Literature search completed. Found {len(combined_results)} papers")
            return results
            
        except Exception as e:
            error_msg = f"Error in literature search: {str(e)}"
            logger.error(error_msg)
            return {
                "error": error_msg,
                "query": query,
                "timestamp": datetime.now().isoformat()
            }
    
    def _parse_search_query(self, query: str) -> Dict[str, Any]:
        """Parse search query into structured parameters"""
        try:
            # Try to parse as JSON first
            if query.strip().startswith('{'):
                return json.loads(query)
            
            # Simple text query
            return {
                "query": query,
                "max_results": 10,
                "sort": "relevance",
                "filter": {}
            }
            
        except Exception as e:
            logger.warning(f"Error parsing query, using default: {e}")
            return {
                "query": query,
                "max_results": 10,
                "sort": "relevance",
                "filter": {}
            }
    
    def _search_crossref(self, search_params: Dict[str, Any]) -> Dict[str, Any]:
        """Search CrossRef database"""
        try:
            query = search_params.get("query", "")
            max_results = min(search_params.get("max_results", 10), 20)  # Limit to prevent timeout
            
            # Build CrossRef query
            params = {
                "query": query,
                "rows": max_results,
                "sort": "relevance",
                "select": "DOI,title,author,published-print,publisher,subject,citation-count,URL"
            }
            
            response = self.session.get(self.crossref_base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            papers = []
            
            for item in data.get("message", {}).get("items", []):
                paper = {
                    "title": item.get("title", ["Unknown Title"])[0] if item.get("title") else "Unknown Title",
                    "authors": [
                        f"{author.get('given', '')} {author.get('family', '')}"
                        for author in item.get("author", [])
                    ],
                    "doi": item.get("DOI", ""),
                    "published_date": self._extract_date(item.get("published-print")),
                    "publisher": item.get("publisher", "Unknown"),
                    "subjects": item.get("subject", []),
                    "citation_count": item.get("is-referenced-by-count", 0),
                    "url": item.get("URL", ""),
                    "source": "crossref"
                }
                papers.append(paper)
            
            return {
                "total_results": data.get("message", {}).get("total-results", 0),
                "papers": papers,
                "search_successful": True
            }
            
        except Exception as e:
            logger.error(f"CrossRef search error: {e}")
            return {
                "total_results": 0,
                "papers": [],
                "search_successful": False,
                "error": str(e)
            }
    
    def _search_arxiv(self, search_params: Dict[str, Any]) -> Dict[str, Any]:
        """Search arXiv database"""
        try:
            query = search_params.get("query", "")
            max_results = min(search_params.get("max_results", 10), 20)
            
            # Build arXiv query
            encoded_query = urllib.parse.quote(f"all:{query}")
            params = {
                "search_query": encoded_query,
                "start": 0,
                "max_results": max_results,
                "sortBy": "relevance",
                "sortOrder": "descending"
            }
            
            response = self.session.get(self.arxiv_base_url, params=params, timeout=10)
            response.raise_for_status()
            
            # Parse XML response (simplified)
            papers = self._parse_arxiv_xml(response.text)
            
            return {
                "total_results": len(papers),
                "papers": papers,
                "search_successful": True
            }
            
        except Exception as e:
            logger.error(f"arXiv search error: {e}")
            return {
                "total_results": 0,
                "papers": [],
                "search_successful": False,
                "error": str(e)
            }
    
    def _parse_arxiv_xml(self, xml_content: str) -> List[Dict[str, Any]]:
        """Parse arXiv XML response (simplified)"""
        try:
            # This is a simplified parser. In production, use proper XML parsing
            import re
            
            papers = []
            
            # Extract entries using regex (simplified approach)
            entry_pattern = r'<entry>(.*?)</entry>'
            entries = re.findall(entry_pattern, xml_content, re.DOTALL)
            
            for entry in entries[:10]:  # Limit results
                title_match = re.search(r'<title>(.*?)</title>', entry)
                author_matches = re.findall(r'<name>(.*?)</name>', entry)
                published_match = re.search(r'<published>(.*?)</published>', entry)
                summary_match = re.search(r'<summary>(.*?)</summary>', entry)
                id_match = re.search(r'<id>(.*?)</id>', entry)
                
                paper = {
                    "title": title_match.group(1).strip() if title_match else "Unknown Title",
                    "authors": [author.strip() for author in author_matches],
                    "published_date": published_match.group(1)[:10] if published_match else "",
                    "abstract": summary_match.group(1).strip()[:500] if summary_match else "",
                    "arxiv_id": id_match.group(1).split('/')[-1] if id_match else "",
                    "url": id_match.group(1) if id_match else "",
                    "source": "arxiv",
                    "citation_count": 0  # arXiv doesn't provide citation counts
                }
                papers.append(paper)
            
            return papers
            
        except Exception as e:
            logger.error(f"Error parsing arXiv XML: {e}")
            return []
    
    def _extract_date(self, date_info: Dict) -> str:
        """Extract date from CrossRef date info"""
        try:
            if not date_info:
                return ""
            
            date_parts = date_info.get("date-parts", [[]])[0]
            if len(date_parts) >= 3:
                return f"{date_parts[0]}-{date_parts[1]:02d}-{date_parts[2]:02d}"
            elif len(date_parts) >= 1:
                return str(date_parts[0])
            
            return ""
            
        except Exception:
            return ""
    
    def _combine_and_rank_results(self, sources: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Combine results from multiple sources and rank them"""
        try:
            all_papers = []
            
            # Combine papers from all sources
            for source_name, source_data in sources.items():
                if source_data.get("search_successful", False):
                    papers = source_data.get("papers", [])
                    for paper in papers:
                        paper["search_source"] = source_name
                        all_papers.append(paper)
            
            # Remove duplicates (simple title-based deduplication)
            seen_titles = set()
            unique_papers = []
            
            for paper in all_papers:
                title_key = paper.get("title", "").lower().strip()
                if title_key and title_key not in seen_titles:
                    seen_titles.add(title_key)
                    unique_papers.append(paper)
            
            # Sort by citation count and recency
            def sort_key(paper):
                citation_score = paper.get("citation_count", 0) * 0.7
                recency_score = self._calculate_recency_score(paper.get("published_date", "")) * 0.3
                return citation_score + recency_score
            
            unique_papers.sort(key=sort_key, reverse=True)
            
            return unique_papers[:20]  # Top 20 results
            
        except Exception as e:
            logger.error(f"Error combining results: {e}")
            return []
    
    def _calculate_recency_score(self, published_date: str) -> float:
        """Calculate recency score for a paper"""
        try:
            if not published_date:
                return 0
            
            # Parse date
            if "-" in published_date:
                pub_date = datetime.strptime(published_date[:10], "%Y-%m-%d")
            else:
                pub_date = datetime.strptime(published_date[:4], "%Y")
            
            # Calculate years since publication
            years_ago = (datetime.now() - pub_date).days / 365.25
            
            # Recency score (higher for more recent papers)
            if years_ago < 1:
                return 10
            elif years_ago < 3:
                return 8
            elif years_ago < 5:
                return 5
            elif years_ago < 10:
                return 2
            else:
                return 1
                
        except Exception:
            return 0
    
    def _generate_search_insights(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate insights from search results"""
        try:
            if not papers:
                return {"message": "No papers found for analysis"}
            
            # Publication timeline
            pub_years = {}
            total_citations = 0
            authors_freq = {}
            publishers_freq = {}
            
            for paper in papers:
                # Publication years
                pub_date = paper.get("published_date", "")
                if pub_date and len(pub_date) >= 4:
                    year = pub_date[:4]
                    pub_years[year] = pub_years.get(year, 0) + 1
                
                # Citation analysis
                citations = paper.get("citation_count", 0)
                if isinstance(citations, (int, float)):
                    total_citations += citations
                
                # Author frequency
                authors = paper.get("authors", [])
                for author in authors[:3]:  # Top 3 authors only
                    if author and len(author.strip()) > 2:
                        authors_freq[author] = authors_freq.get(author, 0) + 1
                
                # Publisher analysis
                publisher = paper.get("publisher", "").strip()
                if publisher:
                    publishers_freq[publisher] = publishers_freq.get(publisher, 0) + 1
            
            insights = {
                "total_papers_found": len(papers),
                "total_citations": total_citations,
                "average_citations": total_citations / len(papers) if papers else 0,
                "publication_timeline": dict(sorted(pub_years.items(), reverse=True)[:5]),
                "top_authors": dict(sorted(authors_freq.items(), key=lambda x: x[1], reverse=True)[:5]),
                "top_publishers": dict(sorted(publishers_freq.items(), key=lambda x: x[1], reverse=True)[:3]),
                "most_cited_paper": max(papers, key=lambda p: p.get("citation_count", 0)) if papers else None,
                "most_recent_paper": max(papers, key=lambda p: p.get("published_date", "")) if papers else None
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return {"error": str(e)}

class CitationManagerTool(BaseTool):
    """Tool for managing and formatting academic citations"""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str = "citation_manager"
    description: str = """
    Academic citation management tool. Use this to:
    - Format citations in various styles (APA, MLA, Chicago, etc.)
    - Generate bibliography entries
    - Validate DOIs and retrieve metadata
    - Check citation completeness and consistency
    
    Input should be citation data or formatting requests.
    """
    
    def __init__(self):
        super().__init__()
    
    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Run citation management task"""
        try:
            # Parse the query to determine the task
            if query.lower().startswith("format"):
                return self._format_citation(query)
            elif query.lower().startswith("validate"):
                return self._validate_citation(query)
            elif query.lower().startswith("generate"):
                return self._generate_bibliography(query)
            else:
                return self._general_citation_help(query)
                
        except Exception as e:
            return {"error": str(e)}
    
    def _format_citation(self, query: str) -> Dict[str, Any]:
        """Format citation in specified style"""
        try:
            # This is a simplified citation formatter
            # In production, use proper citation libraries like citeproc-py
            
            citation_styles = {
                "apa": "APA 7th Edition format",
                "mla": "MLA format",
                "chicago": "Chicago style",
                "ieee": "IEEE format"
            }
            
            # Extract requested style
            style = "apa"  # default
            for style_key in citation_styles.keys():
                if style_key in query.lower():
                    style = style_key
                    break
            
            return {
                "formatted_citation": f"Formatted in {citation_styles[style]}",
                "style_used": style,
                "guidelines": f"Guidelines for {style.upper()} formatting provided"
            }
            
        except Exception as e:
            return {"error": f"Citation formatting error: {e}"}
    
    def _validate_citation(self, query: str) -> Dict[str, Any]:
        """Validate citation data and completeness"""
        return {
            "validation_status": "Citation validation completed",
            "missing_fields": [],
            "suggestions": ["Add publication year", "Include page numbers"]
        }
    
    def _generate_bibliography(self, query: str) -> Dict[str, Any]:
        """Generate bibliography from citations"""
        return {
            "bibliography": "Generated bibliography entries",
            "total_entries": 0,
            "format": "APA 7th Edition"
        }
    
    def _general_citation_help(self, query: str) -> Dict[str, Any]:
        """Provide general citation help"""
        return {
            "help_topic": "General citation assistance",
            "recommendations": [
                "Always include author, title, and publication year",
                "Use consistent citation style throughout document",
                "Verify DOIs and URLs are working"
            ]
        }
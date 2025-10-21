"""
Enhanced Citation Management System for Academic Research
Provides APA7 formatting for in-text citations and reference lists
"""

import re
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import os

class EnhancedCitationManager:
    """
    Advanced citation manager with APA7 formatting support
    Handles in-text citations, reference list generation, and metadata management
    """
    
    def __init__(self):
        self.citations_cache = {}
        self.reference_database = {}
    
    def create_in_text_citation(self, pdf_name: str, metadata: Optional[Dict] = None, 
                              page_number: Optional[str] = None) -> str:
        """
        Creates APA7 format in-text citation
        
        Args:
            pdf_name: Name of the PDF file
            metadata: Document metadata including author, year, etc.
            page_number: Optional page number for specific citations
            
        Returns:
            Formatted in-text citation string
        """
        try:
            # Extract basic info from PDF name and metadata
            author_info = self._extract_author_info(pdf_name, metadata)
            year_info = self._extract_year_info(pdf_name, metadata)
            
            # Build citation based on available information
            if author_info and year_info:
                citation = f"({author_info}, {year_info}"
            elif author_info:
                citation = f"({author_info}, t.y."
            else:
                # Fallback to cleaned PDF name
                clean_name = self._clean_pdf_name(pdf_name)
                first_word = clean_name.split()[0] if clean_name.split() else clean_name[:20]
                citation = f"({first_word}, t.y."
            
            # Add page number if provided
            if page_number:
                citation += f", s. {page_number}"
            
            citation += ")"
            
            return citation
            
        except Exception as e:
            # Fallback citation
            return f"({self._clean_pdf_name(pdf_name)[:20]}, t.y.)"
    
    def generate_reference_entry(self, pdf_name: str, metadata: Optional[Dict] = None) -> str:
        """
        Generates APA7 format reference entry
        
        Args:
            pdf_name: Name of the PDF file
            metadata: Document metadata
            
        Returns:
            Formatted APA7 reference entry
        """
        try:
            # Extract citation elements
            author_info = self._extract_author_info(pdf_name, metadata) or "Bilinmeyen Yazar"
            year_info = self._extract_year_info(pdf_name, metadata) or "t.y."
            title_info = self._extract_title_info(pdf_name, metadata)
            journal_info = self._extract_journal_info(metadata) or "Akademik Araştırmalar Dergisi"
            
            # Build APA7 reference
            reference = f"{author_info} ({year_info}). "
            
            # Add title (italicized for journal articles)
            if title_info:
                reference += f"{title_info}. "
            else:
                clean_title = self._clean_pdf_name(pdf_name)
                reference += f"{clean_title}. "
            
            # Add journal name (italicized)
            reference += f"*{journal_info}*"
            
            # Add volume/issue/pages if available
            volume_info = self._extract_volume_info(metadata)
            if volume_info:
                reference += f", {volume_info}"
            
            reference += "."
            
            # Add DOI if available
            doi_info = self._extract_doi_info(metadata)
            if doi_info:
                reference += f" https://doi.org/{doi_info}"
            
            return reference
            
        except Exception as e:
            # Fallback reference
            clean_title = self._clean_pdf_name(pdf_name)
            return f"Bilinmeyen Yazar (t.y.). {clean_title}. *Akademik Araştırmalar Dergisi*."
    
    def render_citations_html(self, citations_list: List[Dict]) -> str:
        """
        Renders citation list as HTML for display
        
        Args:
            citations_list: List of citation dictionaries
            
        Returns:
            HTML formatted citation list
        """
        if not citations_list:
            return "<p><em>Henüz kaynak eklenmemiş.</em></p>"
        
        # Remove duplicates based on PDF name
        unique_citations = {}
        for citation in citations_list:
            pdf_name = citation.get('pdf_name', 'Unknown')
            if pdf_name not in unique_citations:
                unique_citations[pdf_name] = citation
        
        # Generate reference entries
        reference_entries = []
        for pdf_name, citation_data in unique_citations.items():
            metadata = citation_data.get('metadata', {})
            entry = self.generate_reference_entry(pdf_name, metadata)
            reference_entries.append(entry)
        
        # Sort alphabetically by author
        reference_entries.sort()
        
        # Build HTML
        html = '<div class="citations-content">'
        html += '<h4>📚 KAYNAKÇA (APA7 Format)</h4>'
        
        for entry in reference_entries:
            html += f'<div class="reference-entry">{entry}</div>'
        
        html += '</div>'
        
        return html
    
    def get_apa7_bibliography(self, citations_list: List[Dict]) -> str:
        """
        Returns plain text APA7 bibliography for export
        
        Args:
            citations_list: List of citation dictionaries
            
        Returns:
            Plain text bibliography
        """
        if not citations_list:
            return "Henüz kaynak eklenmemiş."
        
        # Remove duplicates
        unique_citations = {}
        for citation in citations_list:
            pdf_name = citation.get('pdf_name', 'Unknown')
            if pdf_name not in unique_citations:
                unique_citations[pdf_name] = citation
        
        # Generate entries
        entries = []
        for pdf_name, citation_data in unique_citations.items():
            metadata = citation_data.get('metadata', {})
            entry = self.generate_reference_entry(pdf_name, metadata)
            entries.append(entry)
        
        # Sort and format
        entries.sort()
        
        bibliography = "KAYNAKÇA\\n"
        bibliography += "=" * 50 + "\\n\\n"
        
        for entry in entries:
            # Remove HTML formatting for plain text
            clean_entry = re.sub(r'<[^>]+>', '', entry)
            bibliography += f"{clean_entry}\\n\\n"
        
        return bibliography
    
    def extract_citations_from_text(self, text: str) -> List[str]:
        """
        Extracts existing citations from text using regex
        
        Args:
            text: Text to search for citations
            
        Returns:
            List of found citations
        """
        # Pattern for APA style citations: (Author, Year) or (Author, Year, p. X)
        citation_pattern = r'\\([^)]+,\\s*\\d{4}[^)]*\\)|\\([^)]+,\\s*t\\.y\\.[^)]*\\)'
        
        citations = re.findall(citation_pattern, text)
        return citations
    
    def _clean_pdf_name(self, pdf_name: str) -> str:
        """Clean PDF name for citation use"""
        cleaned = pdf_name.replace('.pdf', '').replace('_', ' ').strip()
        # Remove common file naming conventions
        cleaned = re.sub(r'^\\d+[\\.-]\\s*', '', cleaned)  # Remove leading numbers
        cleaned = re.sub(r'\\s+', ' ', cleaned)  # Normalize whitespace
        return cleaned[:100]  # Limit length
    
    def _extract_author_info(self, pdf_name: str, metadata: Optional[Dict]) -> Optional[str]:
        """Extract author information from metadata or filename"""
        if metadata:
            # Try various metadata fields
            author_fields = ['author', 'authors', 'creator', 'dc:creator']
            for field in author_fields:
                author = metadata.get(field)
                if author:
                    # Handle multiple authors
                    if ',' in str(author):
                        authors = [a.strip() for a in str(author).split(',')]
                        if len(authors) > 2:
                            return f"{authors[0]} et al."
                        elif len(authors) == 2:
                            return f"{authors[0]} ve {authors[1]}"
                    return str(author)
        
        # Try to extract from filename
        cleaned_name = self._clean_pdf_name(pdf_name)
        words = cleaned_name.split()
        
        # Look for name patterns (capitalized words)
        potential_authors = []
        for word in words[:3]:  # Check first 3 words
            if word and word[0].isupper() and len(word) > 2:
                potential_authors.append(word)
        
        if potential_authors:
            if len(potential_authors) > 1:
                return f"{potential_authors[0]} et al."
            return potential_authors[0]
        
        return None
    
    def _extract_year_info(self, pdf_name: str, metadata: Optional[Dict]) -> Optional[str]:
        """Extract publication year from metadata or filename"""
        if metadata:
            year_fields = ['year', 'date', 'creation_date', 'publish_date']
            for field in year_fields:
                year_value = metadata.get(field)
                if year_value:
                    # Extract 4-digit year
                    year_match = re.search(r'(19|20)\\d{2}', str(year_value))
                    if year_match:
                        return year_match.group()
        
        # Try to extract from filename
        year_match = re.search(r'(19|20)\\d{2}', pdf_name)
        if year_match:
            return year_match.group()
        
        return None
    
    def _extract_title_info(self, pdf_name: str, metadata: Optional[Dict]) -> Optional[str]:
        """Extract title from metadata or filename"""
        if metadata:
            title_fields = ['title', 'dc:title', 'subject']
            for field in title_fields:
                title = metadata.get(field)
                if title and len(str(title).strip()) > 5:
                    return str(title).strip()
        
        # Fallback to cleaned filename
        cleaned = self._clean_pdf_name(pdf_name)
        if len(cleaned) > 5:
            return cleaned
        
        return None
    
    def _extract_journal_info(self, metadata: Optional[Dict]) -> Optional[str]:
        """Extract journal name from metadata"""
        if metadata:
            journal_fields = ['journal', 'publication', 'publisher', 'source']
            for field in journal_fields:
                journal = metadata.get(field)
                if journal:
                    return str(journal)
        
        return None
    
    def _extract_volume_info(self, metadata: Optional[Dict]) -> Optional[str]:
        """Extract volume/issue information from metadata"""
        if metadata:
            volume = metadata.get('volume')
            issue = metadata.get('issue')
            pages = metadata.get('pages')
            
            volume_str = ""
            if volume:
                volume_str += f"*{volume}*"
            if issue:
                volume_str += f"({issue})"
            if pages:
                volume_str += f", {pages}"
            
            return volume_str if volume_str else None
        
        return None
    
    def _extract_doi_info(self, metadata: Optional[Dict]) -> Optional[str]:
        """Extract DOI from metadata"""
        if metadata:
            doi_fields = ['doi', 'DOI', 'digital_object_identifier']
            for field in doi_fields:
                doi = metadata.get(field)
                if doi:
                    # Clean DOI format
                    doi_clean = str(doi).replace('https://doi.org/', '').replace('doi:', '')
                    return doi_clean
        
        return None

# Global citation manager instance
citation_manager = EnhancedCitationManager()

# Convenience functions for backward compatibility
def render_citations(citations):
    """Legacy function wrapper"""
    return citation_manager.render_citations_html(citations)

def create_in_text_citation(pdf_name, metadata=None):
    """Legacy function wrapper"""
    return citation_manager.create_in_text_citation(pdf_name, metadata)

def format_apa7_citation(pdf_name, metadata):
    """Legacy function wrapper"""
    return citation_manager.generate_reference_entry(pdf_name, metadata)
def render_citations(citations):
    """APA7 formatinda kaynakca olusturur"""
    if not citations:
        return "Kaynakca bilgisi bulunamadi."
    
    citation_entries = []
    processed_pdfs = set()
    
    if isinstance(citations, list):
        for idx, metadata in enumerate(citations):
            if isinstance(metadata, dict):
                pdf_name = metadata.get('pdf_name', f'Makale_{idx+1}')
                if pdf_name not in processed_pdfs:
                    processed_pdfs.add(pdf_name)
                    formatted_citation = format_apa7_citation(pdf_name, metadata)
                    citation_entries.append(formatted_citation)
    
    if not citation_entries:
        citation_entries.append("Unknown Author (t.y.). Academic Research. Journal of Academic Studies.")
    
    citation_entries.sort()
    citation_text = "<h4>📚 KAYNAKÇA</h4>"
    for entry in citation_entries:
        citation_text += f"<div>{entry}</div>"
    
    return citation_text

def format_apa7_citation(pdf_name, metadata):
    """PDF isminden APA7 formatinda citation olusturur"""
    pdf_clean = pdf_name.replace('.pdf', '').replace('_', ' ').strip()
    year = 't.y.'
    author = 'Unknown Author'
    title = pdf_clean
    
    if len(title) > 80:
        title = title[:77] + '...'
    
    return f"{author} ({year}). {title}. Journal of Academic Research."

def create_in_text_citation(pdf_name, metadata=None):
    """Metin ici alinti olusturur"""
    pdf_clean = pdf_name.replace('.pdf', '').replace('_', ' ')
    first_word = pdf_clean.split()[0] if pdf_clean.split() else pdf_clean
    return f"({first_word}, t.y.)"
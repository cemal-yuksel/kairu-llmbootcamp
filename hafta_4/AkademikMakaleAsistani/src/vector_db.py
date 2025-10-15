import os
import chromadb
from sentence_transformers import SentenceTransformer
from .pdf_manager import extract_text

DB_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
EMBEDDING_MODEL_NAME = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

client = chromadb.Client()
collection = client.get_or_create_collection(name="makaleler")

# PDF'i chunk'lara ayır ve vektör veritabanına ekle
def index_pdf(pdf_name):
    pdf_path = os.path.join(os.path.dirname(__file__), '..', 'pdfs', pdf_name)
    text = extract_text(pdf_path)
    chunks = chunk_text(text)
    metadatas = [{"pdf_name": pdf_name, "chunk_id": i} for i in range(len(chunks))]
    embeddings = model.encode(chunks)
    collection.add(documents=chunks, embeddings=embeddings, metadatas=metadatas)

# Sorguya göre en alakalı chunk'ları getir
def query_pdf(pdf_name, query, top_k=5):
    pdf_chunks = [md for md in collection.get()['metadatas'] if md['pdf_name'] == pdf_name]
    query_embedding = model.encode([query])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    context_chunks = results['documents'][0]
    metadata = results['metadatas'][0]
    return context_chunks, metadata

# Basit chunking fonksiyonu
def chunk_text(text, chunk_size=1200, chunk_overlap=200):
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        test_chunk = current_chunk + " " + sentence
        if len(test_chunk) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            words = current_chunk.split()
            overlap_size = min(chunk_overlap // 5, len(words))
            current_chunk = ' '.join(words[-overlap_size:])
        else:
            current_chunk = test_chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

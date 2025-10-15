import os
from PyPDF2 import PdfReader
import openai
from dotenv import load_dotenv
import json

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

PDF_DIR = os.path.join(os.path.dirname(__file__), '..', 'pdfs')
METADATA_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'metadata.json')

# Metadata dosyasını yükle veya oluştur
def load_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_metadata(metadata):
    os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, indent=2, fp=f, ensure_ascii=False)

# PDF'den başlık çıkar (LLM ile)
def extract_title_with_llm(pdf_text):
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        prompt = f"""
Aşağıdaki akademik makale metninden SADECE makale başlığını çıkar. 
Sadece başlığı ver, başka hiçbir şey yazma.

Makale Metni (İlk 2000 karakter):
{pdf_text[:2000]}

Başlık:"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100
        )
        
        title = response.choices[0].message.content.strip()
        # Dosya adı için geçersiz karakterleri temizle
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            title = title.replace(char, '')
        return title[:150]  # Max 150 karakter
    except:
        return None

# PDF'i kaydet
def save_pdf(uploaded_file):
    file_path = os.path.join(PDF_DIR, uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    # Metadata oluştur
    metadata = load_metadata()
    pdf_text = extract_text(file_path)
    
    # LLM ile başlık çıkar
    title = extract_title_with_llm(pdf_text)
    
    if title and title != uploaded_file.name:
        # Dosyayı yeni başlıkla yeniden adlandır
        new_filename = f"{title}.pdf"
        new_path = os.path.join(PDF_DIR, new_filename)
        
        # Aynı isimde dosya varsa numara ekle
        counter = 1
        while os.path.exists(new_path):
            new_filename = f"{title}_{counter}.pdf"
            new_path = os.path.join(PDF_DIR, new_filename)
            counter += 1
        
        os.rename(file_path, new_path)
        file_path = new_path
        final_name = new_filename
    else:
        final_name = uploaded_file.name
    
    # Metadata kaydet
    metadata[final_name] = {
        'original_name': uploaded_file.name,
        'title': title or uploaded_file.name,
        'size': uploaded_file.size,
        'upload_date': str(os.path.getctime(file_path))
    }
    save_metadata(metadata)
    
    return file_path, final_name

# PDF'i sil
def delete_pdf(pdf_name):
    file_path = os.path.join(PDF_DIR, pdf_name)
    if os.path.exists(file_path):
        os.remove(file_path)
        
        # Metadata'dan da sil
        metadata = load_metadata()
        if pdf_name in metadata:
            del metadata[pdf_name]
            save_metadata(metadata)
        return True
    return False

# Metadata güncelle
def update_metadata(pdf_name, new_title):
    metadata = load_metadata()
    if pdf_name in metadata:
        metadata[pdf_name]['title'] = new_title
        save_metadata(metadata)
        return True
    return False

# Kütüphanedeki PDF'leri listele
def list_pdfs():
    return [f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]

# PDF'den metin çıkar
def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# Metadata getir
def get_metadata(pdf_name):
    metadata = load_metadata()
    return metadata.get(pdf_name, {})

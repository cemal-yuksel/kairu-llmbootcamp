"""
Few-Shot Prompting Örneği
Few-shot: Modelin öğrenmesi için birkaç örnek vererek prompt yazma
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

# .env dosyasından API anahtarını yükle
load_dotenv()
# OpenAI istemcisini başlat
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def few_shot_classification():
    """Few-shot ile duygusal analiz"""
    # 1. Modelin öğrenmesi için örnek yorumlar ve duygularını prompt'a ekle
    # 2. Sınıflandırılması istenen yeni yorumu prompt'a ekle
    prompt = """Aşağıdaki örnekleri incele ve son yorumun duygusal tonunu belirle:

Örnek 1:
Yorum: "Bu ürün berbat, hiç beğenmedim!"
Duygu: negatif

Örnek 2:
Yorum: "Oldukça iyi bir ürün, memnunum."
Duygu: pozitif

Örnek 3:
Yorum: "Normal bir ürün, ne iyi ne kötü."
Duygu: nötr

Şimdi bu yorumu sınıflandır:
Yorum: "Harika bir deneyimdi! Kesinlikle tekrar alırım."
Duygu:"""

    # 3. OpenAI API'sine isteği gönder
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=50,
        temperature=0
    )
    # 4. Modelin döndürdüğü cevabı al ve döndür
    return response.choices[0].message.content

def few_shot_entity_extraction():
    """Few-shot ile varlık çıkarma"""
    # 1. Modelin öğrenmesi için örnek metinler ve kişi isimlerini prompt'a ekle
    # 2. Analiz edilmesi istenen yeni metni prompt'a ekle
    prompt = """Metinlerden kişi isimlerini çıkar:

Örnek 1:
Metin: "Ahmet Yılmaz bugün Ankara'ya gitti."
Kişi: Ahmet Yılmaz

Örnek 2:
Metin: "Fatma Hanım ve Mehmet Bey toplantıya katıldı."
Kişi: Fatma Hanım, Mehmet Bey

Örnek 3:
Metin: "İstanbul'da güzel bir gün geçirdik."
Kişi: -

Şimdi bu metni analiz et:
Metin: "Prof. Dr. Ayşe Kaya ve Mühendis Ali Demir projeyi tamamladı."
Kişi:"""

    # 3. OpenAI API'sine isteği gönder
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        temperature=0
    )
    # 4. Modelin döndürdüğü cevabı al ve döndür
    return response.choices[0].message.content

def few_shot_text_formatting():
    """Few-shot ile metin formatlama"""
    # 1. Modelin öğrenmesi için örnek cümleler ve başlık formatlarını prompt'a ekle
    # 2. Formatlanacak yeni cümleyi prompt'a ekle
    prompt = """Verilen cümleleri başlık formatına çevir:

Örnek 1:
Girdi: "yapay zeka ve gelecek"
Çıktı: "Yapay Zeka ve Gelecek"

Örnek 2:
Girdi: "makine öğrenmesi algoritmaları"
Çıktı: "Makine Öğrenmesi Algoritmaları"

Örnek 3:
Girdi: "derin öğrenme modelleri"
Çıktı: "Derin Öğrenme Modelleri"

Şimdi bu cümleyi formatla:
Girdi: "doğal dil işleme teknikleri"
Çıktı:"""

    # 3. OpenAI API'sine isteği gönder
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=50,
        temperature=0
    )
    # 4. Modelin döndürdüğü cevabı al ve döndür
    return response.choices[0].message.content

def few_shot_qa_format():
    """Few-shot ile soru-cevap formatı"""
    # 1. Modelin öğrenmesi için örnek bilgi, soru ve cevap çiftlerini prompt'a ekle
    # 2. Soru-cevap oluşturulacak yeni bilgiyi prompt'a ekle
    prompt = """Verilen bilgilerden soru-cevap çiftleri oluştur:

Örnek 1:
Bilgi: "Python 1991 yılında Guido van Rossum tarafından geliştirildi."
Soru: Python ne zaman ve kim tarafından geliştirildi?
Cevap: Python 1991 yılında Guido van Rossum tarafından geliştirildi.

Örnek 2:
Bilgi: "JavaScript web tarayıcılarında çalışan bir programlama dilidir."
Soru: JavaScript nerede çalışır?
Cevap: JavaScript web tarayıcılarında çalışır.

Şimdi bu bilgiden soru-cevap oluştur:
Bilgi: "Machine Learning, verilerdeki kalıpları öğrenen algoritmalar kullanır."
Soru:"""

    # 3. OpenAI API'sine isteği gönder
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0
    )
    # 4. Modelin döndürdüğü cevabı al ve döndür
    return response.choices[0].message.content

if __name__ == "__main__":
    print("=== FEW-SHOT PROMPTING ÖRNEKLERİ ===\n")
    
    # Duygusal analiz
    print("1. Few-Shot Duygusal Analiz:")
    try:
        # 1. few_shot_classification fonksiyonunu çağır
        result = few_shot_classification()
        # 2. Sonucu ekrana yazdır
        print(f"Sonuç: {result}\n")
    except Exception as e:
        # 3. Hata oluşursa ekrana yazdır
        print(f"Hata: {e}\n")
    
    # Varlık çıkarma
    print("2. Few-Shot Varlık Çıkarma:")
    try:
        # 1. few_shot_entity_extraction fonksiyonunu çağır
        result = few_shot_entity_extraction()
        # 2. Sonucu ekrana yazdır
        print(f"Sonuç: {result}\n")
    except Exception as e:
        # 3. Hata oluşursa ekrana yazdır
        print(f"Hata: {e}\n")
    
    # Metin formatlama
    print("3. Few-Shot Metin Formatlama:")
    try:
        # 1. few_shot_text_formatting fonksiyonunu çağır
        result = few_shot_text_formatting()
        # 2. Sonucu ekrana yazdır
        print(f"Sonuç: {result}\n")
    except Exception as e:
        # 3. Hata oluşursa ekrana yazdır
        print(f"Hata: {e}\n")
    
    # Soru-cevap formatı
    print("4. Few-Shot Soru-Cevap:")
    try:
        # 1. few_shot_qa_format fonksiyonunu çağır
        result = few_shot_qa_format()
        # 2. Sonucu ekrana yazdır
        print(f"Sonuç: {result}\n")
    except Exception as e:
        # 3. Hata oluşursa ekrana yazdır
        print(f"Hata: {e}\n")
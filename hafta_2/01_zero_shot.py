"""
Zero-Shot Prompting Örneği
Zero-shot: Herhangi bir örnek vermeden doğrudan görev tanımı ile prompt yazma
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

# .env dosyasından API anahtarını yükle
load_dotenv()

# OpenAI istemcisini başlat
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def zero_shot_classification():
    """Sıfır örnekle metin sınıflandırma"""
    # 1. Kullanıcıdan duygu analizi yapılacak yorumu belirle
    # 2. Yorumu ve görevi içeren prompt'u hazırla
    prompt = """Bu yorumun duygusal tonunu belirle (pozitif, negatif, nötr):

Yorum: "Bu ürün gerçekten harika! Çok memnun kaldım, herkese tavsiye ederim."

Duygusal ton:"""

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

def zero_shot_translation():
    """Sıfır örnekle çeviri"""
    # 1. Çevrilecek Türkçe metni ve görevi içeren prompt'u hazırla
    prompt = """Aşağıdaki Türkçe metni İngilizceye çevir:

"Bugün hava çok güzel. Parkta yürüyüş yapmaya gidiyorum."

İngilizce çeviri:"""

    # 2. OpenAI API'sine isteği gönder
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        temperature=0
    )
    # 3. Modelin döndürdüğü cevabı al ve döndür
    return response.choices[0].message.content

def zero_shot_summary():
    """Sıfır örnekle özet çıkarma"""
    # 1. Özetlenecek metni tanımla
    text = """
    Yapay zeka teknolojisi son yıllarda büyük gelişmeler göstermiştir. Özellikle büyük dil modelleri, 
    doğal dil işleme görevlerinde insan seviyesinde performans gösterebilmektedir. Bu teknolojiler 
    çeviri, özet çıkarma, soru yanıtlama gibi birçok alanda kullanılmaktadır. Ancak bu gelişmeler 
    beraberinde etik endişeler ve iş gücü üzerindeki potansiyel etkiler gibi konuları da gündeme 
    getirmektedir. Gelecekte bu teknolojilerin daha da gelişeceği ve hayatımızın daha fazla alanında 
    yer alacağı öngörülmektedir.
    """
    # 2. Metni ve görevi içeren prompt'u hazırla
    prompt = f"""Aşağıdaki metni 2-3 cümleyle özetle:

{text}

Özet:"""

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
    print("=== ZERO-SHOT PROMPTING ÖRNEKLERİ ===\n")
    
    # Duygusal analiz
    print("1. Duygusal Analiz:")
    try:
        # 1. zero_shot_classification fonksiyonunu çağır
        result = zero_shot_classification()
        # 2. Sonucu ekrana yazdır
        print(f"Sonuç: {result}\n")
    except Exception as e:
        # 3. Hata oluşursa ekrana yazdır
        print(f"Hata: {e}\n")
    
    # Çeviri
    print("2. Çeviri:")
    try:
        # 1. zero_shot_translation fonksiyonunu çağır
        result = zero_shot_translation()
        # 2. Sonucu ekrana yazdır
        print(f"Sonuç: {result}\n")
    except Exception as e:
        # 3. Hata oluşursa ekrana yazdır
        print(f"Hata: {e}\n")
    
    # Özet çıkarma
    print("3. Özet Çıkarma:")
    try:
        # 1. zero_shot_summary fonksiyonunu çağır
        result = zero_shot_summary()
        # 2. Sonucu ekrana yazdır
        print(f"Sonuç: {result}\n")
    except Exception as e:
        # 3. Hata oluşursa ekrana yazdır
        print(f"Hata: {e}\n")
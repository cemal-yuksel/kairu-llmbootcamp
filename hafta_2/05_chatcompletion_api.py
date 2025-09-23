"""
OpenAI ChatCompletion API Detaylı Kullanım Örnekleri
Farklı parametreler, streaming, conversation management
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# .env dosyasından API anahtarını yükle
load_dotenv()
# OpenAI istemcisini başlat
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def basic_chat_completion():
    """Temel ChatCompletion kullanımı"""
    # 1. Kullanıcıdan gelen mesajı belirle
    # 2. OpenAI API'sine mesajı gönder
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Python'da liste ve tuple arasındaki fark nedir?"}
        ]
    )
    # 3. Modelin döndürdüğü cevabı al ve döndür
    return response.choices[0].message.content

def chat_with_system_message():
    """System message ile davranış belirleme"""
    # 1. Modelin rolünü ve davranışını system mesajında belirt
    # 2. Kullanıcıdan gelen isteği user mesajında ekle
    messages = [
        {
            "role": "system", 
            "content": "Sen bir Python öğretmeni olan yardımcısın. Açıklamalarını basit ve kod örnekleriyle destekle."
        },
        {
            "role": "user", 
            "content": "Decorators nasıl çalışır?"
        }
    ]
    # 3. OpenAI API'sine mesajları gönder
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
        max_tokens=300
    )
    # 4. Modelin döndürdüğü cevabı al ve döndür
    return response.choices[0].message.content

def conversation_management():
    """Konuşma geçmişi yönetimi"""
    # 1. Konuşma geçmişini bir liste olarak başlat
    conversation = [
        {"role": "system", "content": "Sen yardımcı bir AI asistanısın."},
        {"role": "user", "content": "Merhaba! Python öğrenmeye yeni başladım."},
    ]
    # 2. İlk yanıtı almak için OpenAI API'sine isteği gönder
    response1 = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=conversation
    )
    # 3. AI yanıtını konuşma geçmişine assistant olarak ekle
    conversation.append({
        "role": "assistant", 
        "content": response1.choices[0].message.content
    })
    # 4. Kullanıcının ikinci sorusunu konuşmaya ekle
    conversation.append({
        "role": "user", 
        "content": "Hangi konulardan başlamalıyım?"
    })
    # 5. İkinci yanıtı almak için OpenAI API'sine isteği gönder (tüm geçmişle birlikte)
    response2 = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=conversation
    )
    # 6. Yanıtları ve konuşma geçmişini döndür
    return {
        "first_response": response1.choices[0].message.content,
        "second_response": response2.choices[0].message.content,
        "full_conversation": conversation
    }

def different_temperature_examples():
    """Farklı temperature değerleri ile yaratıcılık kontrolü"""
    # 1. Aynı prompt ile farklı temperature değerleri dene
    prompt = "Yapay zeka teknolojisinin geleceği hakkında 2 cümle yaz."
    results = {}
    # 2. Düşük temperature (0.1) ile deterministik yanıt al
    response_low = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    results["low_temperature"] = response_low.choices[0].message.content
    # 3. Orta temperature (0.7) ile dengeli yanıt al
    response_mid = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    results["mid_temperature"] = response_mid.choices[0].message.content
    # 4. Yüksek temperature (1.2) ile yaratıcı yanıt al
    response_high = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=1.2
    )
    results["high_temperature"] = response_high.choices[0].message.content
    # 5. Sonuçları döndür
    return results

def streaming_example():
    """Streaming response örneği"""
    # 1. Streaming modunda yanıt almak için stream=True parametresiyle API'ye isteği gönder
    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Machine Learning konusunda kısa bir özet yaz."}
        ],
        stream=True,
        max_tokens=200
    )
    # 2. Yanıtı parça parça (chunk) olarak al ve ekrana yazdır
    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            full_response += content
            print(content, end="", flush=True)
    # 3. Yanıt tamamlandığında yeni satır ekle ve tam yanıtı döndür
    print()  # Yeni satır
    return full_response

def multiple_choices_example():
    """Birden fazla yanıt seçeneği alma"""
    # 1. Aynı prompt için n parametresiyle birden fazla yanıt iste
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Motivasyonel bir söz söyle."}
        ],
        n=3,  # 3 farklı yanıt
        temperature=0.9
    )
    # 2. Her bir yanıtı choices listesine ekle
    choices = []
    for i, choice in enumerate(response.choices):
        choices.append({
            "choice_number": i + 1,
            "content": choice.message.content
        })
    # 3. Yanıtları döndür
    return choices

def token_usage_tracking():
    """Token kullanımını takip etme"""
    # 1. API'ye isteği gönder ve yanıtla birlikte token kullanım bilgisini al
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "E-ticaret sitesi için basit bir kullanıcı kayıt sistemi nasıl tasarlanır?"}
        ],
        max_tokens=250
    )
    # 2. Yanıtı ve token kullanım istatistiklerini döndür
    return {
        "content": response.choices[0].message.content,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
    }

def model_comparison():
    """Farklı modelleri karşılaştırma"""
    # 1. Aynı prompt ile farklı modelleri test et
    prompt = "JavaScript'te async/await nasıl kullanılır?"
    models = ["gpt-3.5-turbo", "gpt-4"]
    results = {}
    # 2. Her model için API'ye isteği gönder ve sonucu kaydet
    for model in models:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200
            )
            results[model] = response.choices[0].message.content
        except Exception as e:
            results[model] = f"Hata: {str(e)}"
    # 3. Sonuçları döndür
    return results

if __name__ == "__main__":
    print("=== OPENAI CHATCOMPLETION API ÖRNEKLERİ ===\n")
    
    # 1. Temel kullanım
    print("1. Temel ChatCompletion:")
    try:
        # a. basic_chat_completion fonksiyonunu çağır
        result = basic_chat_completion()
        # b. Sonucu ekrana yazdır
        print(f"Yanıt: {result}\n")
        print("-" * 60)
    except Exception as e:
        # c. Hata oluşursa ekrana yazdır
        print(f"Hata: {e}\n")
    
    # 2. System message ile
    print("2. System Message ile:")
    try:
        # a. chat_with_system_message fonksiyonunu çağır
        result = chat_with_system_message()
        # b. Sonucu ekrana yazdır
        print(f"Yanıt: {result}\n")
        print("-" * 60)
    except Exception as e:
        print(f"Hata: {e}\n")
    
    # 3. Konuşma yönetimi
    print("3. Konuşma Geçmişi Yönetimi:")
    try:
        # a. conversation_management fonksiyonunu çağır
        result = conversation_management()
        # b. İlk ve ikinci yanıtı ekrana yazdır
        print(f"İlk Yanıt: {result['first_response']}")
        print(f"İkinci Yanıt: {result['second_response']}\n")
        print("-" * 60)
    except Exception as e:
        print(f"Hata: {e}\n")
    
    # 4. Temperature karşılaştırması
    print("4. Farklı Temperature Değerleri:")
    try:
        # a. different_temperature_examples fonksiyonunu çağır
        result = different_temperature_examples()
        # b. Her temperature için sonucu ekrana yazdır
        for temp_type, content in result.items():
            print(f"{temp_type}: {content}")
        print("-" * 60)
    except Exception as e:
        print(f"Hata: {e}\n")
    
    # 5. Streaming
    print("5. Streaming Response:")
    try:
        # a. streaming_example fonksiyonunu çağır
        print("Streaming yanıt:")
        result = streaming_example()
        # b. Tam yanıtı ekrana yazdır
        print(f"\nTam yanıt: {result}")
        print("-" * 60)
    except Exception as e:
        print(f"Hata: {e}\n")
    
    # 6. Birden fazla seçenek
    print("6. Birden Fazla Yanıt Seçeneği:")
    try:
        # a. multiple_choices_example fonksiyonunu çağır
        result = multiple_choices_example()
        # b. Her seçeneği ekrana yazdır
        for choice in result:
            print(f"Seçenek {choice['choice_number']}: {choice['content']}")
        print("-" * 60)
    except Exception as e:
        print(f"Hata: {e}\n")
    
    # 7. Token kullanımı
    print("7. Token Kullanımı Takibi:")
    try:
        # a. token_usage_tracking fonksiyonunu çağır
        result = token_usage_tracking()
        # b. Yanıtı ve token kullanımını ekrana yazdır
        print(f"Yanıt: {result['content']}")
        print(f"Token Kullanımı: {result['usage']}")
        print("-" * 60)
    except Exception as e:
        print(f"Hata: {e}\n")
    
    # 8. Model karşılaştırması
    print("8. Model Karşılaştırması:")
    try:
        # a. model_comparison fonksiyonunu çağır
        result = model_comparison()
        # b. Her model için sonucu ekrana yazdır (ilk 100 karakter)
        for model, response in result.items():
            print(f"{model}: {response[:100]}...")
        print("-" * 60)
    except Exception as e:
        print(f"Hata: {e}\n")
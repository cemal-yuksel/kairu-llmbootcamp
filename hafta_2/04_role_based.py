"""
Role-Based Prompting Örneği
Role-based: Modele belirli bir rol vererek daha spesifik ve tutarlı yanıtlar alma
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

# .env dosyasından API anahtarını yükle
load_dotenv()
# OpenAI istemcisini başlat
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def marketing_expert_role():
    """Pazarlama uzmanı rolünde danışmanlık"""
    # 1. Modelin rolünü ve uzmanlık alanını system mesajında belirt
    # 2. Kullanıcıdan gelen isteği user mesajında ekle
    messages = [
        {
            "role": "system",
            "content": """Sen deneyimli bir dijital pazarlama uzmanısın. 10 yıllık tecriben var. 
            E-ticaret, sosyal medya pazarlama ve marka stratejileri konularında uzmansın. 
            Pratik, uygulanabilir ve ölçülebilir öneriler verirsin."""
        },
        {
            "role": "user",
            "content": """Yeni açılan kahve dükkanım için sosyal medya stratejisi önerir misin? 
            Hedef kitlem 25-40 yaş arası, kahve seven profesyoneller."""
        }
    ]
    # 3. OpenAI API'sine isteği gönder
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=300,
        temperature=0.7
    )
    # 4. Modelin döndürdüğü cevabı al ve döndür
    return response.choices[0].message.content

def technical_writer_role():
    """Teknik yazar rolünde dokümantasyon yazma"""
    # 1. Modelin rolünü ve uzmanlık alanını system mesajında belirt
    # 2. Kullanıcıdan gelen isteği user mesajında ekle
    messages = [
        {
            "role": "system",
            "content": """Sen profesyonel bir teknik yazarsın. Karmaşık teknik konuları 
            basit, anlaşılır dilde açıklama konusunda uzmansın. API dokümantasyonları, 
            kullanıcı kılavuzları ve teknik makaleler yazarsın."""
        },
        {
            "role": "user", 
            "content": """REST API'nin temel kavramlarını yeni başlayan geliştiriciler 
            için açıklar mısın? HTTP metodları, endpoint'ler ve response kodları dahil."""
        }
    ]
    # 3. OpenAI API'sine isteği gönder
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=400,
        temperature=0.5
    )
    # 4. Modelin döndürdüğü cevabı al ve döndür
    return response.choices[0].message.content

def financial_advisor_role():
    """Mali müşavir rolünde finansal analiz"""
    # 1. Modelin rolünü ve uzmanlık alanını system mesajında belirt
    # 2. Kullanıcıdan gelen isteği user mesajında ekle
    messages = [
        {
            "role": "system",
            "content": """Sen sertifikalı bir mali müşavirsin. Şirket finansmanı, 
            yatırım analizi ve risk yönetimi konularında 15 yıllık deneyimin var. 
            Türkiye'deki vergi mevzuatını ve finansal düzenlemeleri çok iyi bilirsin."""
        },
        {
            "role": "user",
            "content": """Startup'ım için yatırımcı bulmak istiyorum. Mali tablolarımda 
            hangi metriklere odaklanmalıyım ve nasıl sunmalıyım?"""
        }
    ]
    # 3. OpenAI API'sine isteği gönder
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=350,
        temperature=0.6
    )
    # 4. Modelin döndürdüğü cevabı al ve döndür
    return response.choices[0].message.content

def teacher_role():
    """Öğretmen rolünde eğitim içeriği"""
    # 1. Modelin rolünü ve uzmanlık alanını system mesajında belirt
    # 2. Kullanıcıdan gelen isteği user mesajında ekle
    messages = [
        {
            "role": "system",
            "content": """Sen deneyimli bir matematik öğretmenisin. Öğrencilerin 
            seviyesine uygun açıklamalar yapar, örneklerle desteklersin. 
            Karmaşık konuları basit adımlarla öğretme konusunda çok başarılısın."""
        },
        {
            "role": "user",
            "content": """Lise öğrencilerine logaritma konusunu nasıl açıklayabilirim? 
            Günlük hayattan örneklerle anlatabiliir misin?"""
        }
    ]
    # 3. OpenAI API'sine isteği gönder
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=350,
        temperature=0.7
    )
    # 4. Modelin döndürdüğü cevabı al ve döndür
    return response.choices[0].message.content

def psychologist_role():
    """Psikolog rolünde danışmanlık"""
    # 1. Modelin rolünü ve uzmanlık alanını system mesajında belirt
    # 2. Kullanıcıdan gelen isteği user mesajında ekle
    messages = [
        {
            "role": "system",
            "content": """Sen lisanslı bir klinik psikologsun. Stres yönetimi, 
            iletişim becerileri ve kişisel gelişim konularında uzmansın. 
            Empati kurarak, destekleyici ve yapıcı tavsiyelerde bulunursun."""
        },
        {
            "role": "user",
            "content": """İş hayatında stresle başa çıkma konusunda zorlanıyorum. 
            Hangi teknikleri uygulayabilirim?"""
        }
    ]
    # 3. OpenAI API'sine isteği gönder
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=300,
        temperature=0.8
    )
    # 4. Modelin döndürdüğü cevabı al ve döndür
    return response.choices[0].message.content

def chef_role():
    """Şef rolünde yemek tarifi"""
    # 1. Modelin rolünü ve uzmanlık alanını system mesajında belirt
    # 2. Kullanıcıdan gelen isteği user mesajında ekle
    messages = [
        {
            "role": "system",
            "content": """Sen Michelin yıldızlı restoran deneyimi olan profesyonel 
            bir şefsin. Hem geleneksel hem modern mutfak tekniklerini bilirsin. 
            Tariflerini detaylı ve uygulanabilir şekilde verirsin."""
        },
        {
            "role": "user",
            "content": """Evde kolayca yapabileceğim, misafirlere ikram edebileceğim 
            şık bir tatlı tarifi önerir misin?"""
        }
    ]
    # 3. OpenAI API'sine isteği gönder
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=300,
        temperature=0.7
    )
    # 4. Modelin döndürdüğü cevabı al ve döndür
    return response.choices[0].message.content

if __name__ == "__main__":
    print("=== ROLE-BASED PROMPTING ÖRNEKLERİ ===\n")
    
    # 1. Rol ve fonksiyon çiftlerini bir listeye ekle
    roles = [
        ("Pazarlama Uzmanı", marketing_expert_role),
        ("Teknik Yazar", technical_writer_role),
        ("Mali Müşavir", financial_advisor_role),
        ("Matematik Öğretmeni", teacher_role),
        ("Psikolog", psychologist_role),
        ("Profesyonel Şef", chef_role)
    ]
    
    # 2. Her rol için sırayla ilgili fonksiyonu çağır ve sonucu ekrana yazdır
    for i, (role_name, role_function) in enumerate(roles, 1):
        print(f"{i}. {role_name} Rolü:")
        try:
            # a. Rol fonksiyonunu çağır
            result = role_function()
            # b. Sonucu ekrana yazdır
            print(f"Yanıt:\n{result}\n")
            print("-" * 60)
        except Exception as e:
            # c. Hata oluşursa ekrana yazdır
            print(f"Hata: {e}\n")
            print("-" * 60)
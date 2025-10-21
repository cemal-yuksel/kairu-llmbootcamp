"""
Hafta 5 - Bölüm 3: Tools ve Agents
LangChain ile tool kullanımı ve agent oluşturma
"""

# Gerekli kütüphaneleri içe aktar
import os
import requests
import json
from datetime import datetime

# LangChain kütüphanelerini içe aktar
from langchain_openai import OpenAI  # OpenAI LLM'ini kullanmak için
from langchain.agents import Tool, AgentType, initialize_agent, create_react_agent  # Agent oluşturma araçları
from langchain.tools import BaseTool  # Özel tool sınıfı oluşturmak için
from langchain import hub  # Hazır prompt'ları yüklemek için
from langchain.agents import AgentExecutor  # Agent çalıştırıcısı
from langchain.memory import ConversationBufferMemory  # Konuşma geçmişini saklamak için
from typing import Optional, Type  # Tip belirtimi için

# Callback yöneticileri (tool çalışma durumunu izlemek için)
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from pydantic import BaseModel, Field  # Veri doğrulama için

# Çevre değişkenlerini yükle (.env dosyasından API anahtarları vs.)
from dotenv import load_dotenv
load_dotenv()

# OpenAI LLM'ini başlat
# temperature=0: Deterministic (tutarlı) cevaplar için
# max_tokens=200: Maksimum token sayısı sınırı
# request_timeout=15: 15 saniye zaman aşımı
llm = OpenAI(temperature=0, max_tokens=200, request_timeout=15)

# =============================================================================
# BASIT TOOL ÖRNEKLERİ - Temel fonksiyonlar olarak tanımlanan tool'lar
# =============================================================================

def get_current_time(query: str) -> str:
    """
    Şu anki tarih ve saati döndüren basit tool
    Agent bu tool'u zaman bilgisi istediğinde kullanacak
    """
    now = datetime.now()  # Şu anki zaman
    return f"Şu anki tarih ve saat: {now.strftime('%Y-%m-%d %H:%M:%S')}"

def simple_calculator(expression: str) -> str:
    """
    Temel matematik işlemlerini yapan güvenli hesap makinesi tool'u
    Güvenlik için sadece belirli karakterlere izin verir
    """
    try:
        # Güvenlik kontrolü: Sadece matematik karakterlerine izin ver
        allowed_chars = "0123456789+-*/.() "
        if all(c in allowed_chars for c in expression):
            result = eval(expression)  # Matematiksel ifadeyi hesapla
            return f"Sonuç: {result}"
        else:
            return "Sadece temel matematik işlemleri desteklenir."
    except:
        return "Geçersiz matematik ifadesi."

def text_length_counter(text: str) -> str:
    """
    Bir metnin kelime ve karakter sayısını hesaplayan tool
    """
    word_count = len(text.split())  # Boşluklara göre ayırarak kelime say
    char_count = len(text)  # Toplam karakter sayısı
    return f"Kelime sayısı: {word_count}, Karakter sayısı: {char_count}"

def basic_tools_example():
    """
    Basit tool'ların nasıl kullanıldığını gösteren örnek
    """
    print("=" * 60)
    print("1. BASIT TOOLS KULLANIMI")
    print("=" * 60)
    
    # Tool'ları LangChain Tool nesneleri olarak oluştur
    tools = [
        Tool(
            name="get_current_time",  # Tool'un adı (agent bu isimle çağıracak)
            func=get_current_time,    # Çalıştırılacak fonksiyon
            description="Şu anki tarih ve saati öğrenmek için kullanın"  # Agent'a tool'u ne zaman kullanacağını söyler
        ),
        Tool(
            name="calculator",
            func=simple_calculator,
            description="Matematik işlemleri yapmak için kullanın. Örnek: 2+2 veya 10*5"
        ),
        Tool(
            name="text_counter",
            func=text_length_counter,
            description="Bir metnin kelime ve karakter sayısını bulmak için kullanın"
        )
    ]
    
    # Zero-shot ReAct agent oluştur
    # Bu agent type: Tek seferde karar verir, önceki adımları hatırlamaz
    agent = initialize_agent(
        tools=tools,  # Kullanılabilir tool'lar
        llm=llm,      # Karar verici LLM
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Agent türü
        verbose=False,     # Debug çıktıları gösterme
        max_iterations=3   # Maksimum deneme sayısı
    )
    
    # Test sorularını hazırla ve çalıştır
    questions = [
        "Şu anki saat kaç?",           # get_current_time tool'unu kullanacak
        "25 çarpı 4 kaç eder?",       # calculator tool'unu kullanacak
        "Bu metin kaç kelimeden oluşuyor: 'LangChain ile agent geliştiriyoruz'?"  # text_counter tool'unu kullanacak
    ]
    
    # Her soruyu agent'a sor ve cevabını al
    for question in questions:
        print(f"\nSoru: {question}")
        try:
            response = agent.run(question)  # Agent soruyu çözer ve cevap döner
            print(f"Cevap: {response}")
        except Exception as e:
            print(f"Hata: {e}")
    
    return agent

# =============================================================================
# ÖZEL TOOL SINIFI - Pydantic BaseModel ile güçlü tool'lar
# =============================================================================

class WeatherToolInput(BaseModel):
    """
    Hava durumu tool'u için giriş parametrelerini tanımlayan sınıf
    Pydantic ile tip kontrolü ve doğrulama sağlar
    """
    city: str = Field(description="Hava durumunu öğrenmek istediğiniz şehir adı")

class WeatherTool(BaseTool):
    """
    Hava durumu bilgisi veren özel tool sınıfı
    BaseTool'dan türetilmiş, daha gelişmiş özellikler sunar
    """
    name: str = "weather_tool"  # Tool'un benzersiz adı
    description: str = "Herhangi bir şehrin hava durumunu öğrenmek için kullanın"  # Agent'ın ne zaman kullanacağını belirler
    args_schema: Type[BaseModel] = WeatherToolInput  # Girdi parametrelerinin şeması

    def _run(
        self, 
        city: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        Tool'un ana çalışma fonksiyonu
        Gerçek API çağrısı yerine simüle edilmiş veri döndürür
        """
        # Simüle edilmiş hava durumu veritabanı
        weather_data = {
            "istanbul": "İstanbul: 15°C, Parçalı bulutlu",
            "ankara": "Ankara: 12°C, Açık", 
            "izmir": "İzmir: 18°C, Güneşli",
            "bursa": "Bursa: 14°C, Yağmurlu"
        }
        
        city_lower = city.lower()  # Büyük/küçük harf duyarsız karşılaştırma
        if city_lower in weather_data:
            return weather_data[city_lower]
        else:
            return f"{city} şehri için hava durumu bilgisi bulunamadı. Mevcut şehirler: İstanbul, Ankara, İzmir, Bursa"

    async def _arun(
        self,
        city: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """
        Tool'un asenkron versiyonu (şu an desteklenmiyor)
        """
        raise NotImplementedError("Bu tool async desteklemez")

class NewsToolInput(BaseModel):
    """
    Haber tool'u için giriş parametrelerini tanımlayan sınıf
    """
    topic: str = Field(description="Haber konusu")

class NewsTool(BaseTool):
    """
    Belirli konularda haber getiren özel tool sınıfı
    """
    name: str = "news_tool"
    description: str = "Belirli bir konu hakkında güncel haberleri getirir"
    args_schema: Type[BaseModel] = NewsToolInput
    
    def _run(
        self, 
        topic: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        Konu bazında simüle edilmiş haber verisi döndürür
        """
        # Konulara göre simüle edilmiş haber veritabanı
        news_data = {
            "teknoloji": [
                "Yapay zeka alanında yeni gelişmeler",
                "Quantum bilgisayarlarda büyük adım",
                "5G teknolojisi yaygınlaşıyor"
            ],
            "spor": [
                "Futbol liginde heyecanlı maçlar",
                "Olimpiyat hazırlıkları devam ediyor",
                "Yeni spor kompleksi açıldı"
            ],
            "ekonomi": [
                "Borsa günü yükselişle kapattı",
                "Yeni yatırım teşvikleri açıklandı",
                "Dijital para birimlerinde hareket"
            ]
        }
        
        topic_lower = topic.lower()
        if topic_lower in news_data:
            news = news_data[topic_lower]
            # Haberleri madde madde listele
            return f"{topic} konusundaki güncel haberler:\n" + "\n".join([f"• {n}" for n in news])
        else:
            return f"{topic} konusunda haber bulunamadı. Mevcut konular: teknoloji, spor, ekonomi"

def custom_tools_example():
    """
    Özel tool sınıflarının nasıl kullanıldığını gösteren örnek
    """
    print("\n" + "=" * 60)
    print("2. ÖZEL TOOL SINIFLARI")
    print("=" * 60)
    
    # Özel tool sınıflarının örneklerini oluştur
    tools = [
        WeatherTool(),  # Hava durumu tool'u
        NewsTool(),     # Haber tool'u
        Tool(           # Basit hesap makinesi tool'u
            name="calculator",
            func=simple_calculator,
            description="Matematik işlemleri yapmak için kullanın"
        )
    ]
    
    # Agent'ı oluştur (aynı tip: zero-shot ReAct)
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False
    )
    
    # Özel tool'ları test edecek sorular
    questions = [
        "İstanbul'da hava nasıl?",                    # WeatherTool'u kullanacak
        "Teknoloji konusunda güncel haberler neler?", # NewsTool'u kullanacak
        "50 bölü 5 kaç eder?"                         # Calculator tool'unu kullanacak
    ]
    
    # Her soruyu test et
    for question in questions:
        print(f"\nSoru: {question}")
        try:
            response = agent.run(question)
            print(f"Cevap: {response}")
        except Exception as e:
            print(f"Hata: {e}")
    
    return agent

# =============================================================================
# MEMORY İLE AGENT - Konuşma geçmişini hatırlayan agent
# =============================================================================

def memory_agent_example():
    """
    Konuşma geçmişini hatırlayan agent örneği
    Bu agent önceki soruları ve cevapları hatırlar
    """
    print("\n" + "=" * 60)
    print("3. MEMORY İLE AGENT")
    print("=" * 60)
    
    # Konuşma geçmişini saklamak için memory oluştur
    memory = ConversationBufferMemory(
        memory_key="chat_history",  # Memory'de konuşma geçmişinin saklanacağı anahtar
        return_messages=True        # Mesajları döndür
    )
    
    # Tool'ları tanımla
    tools = [
        Tool(
            name="get_time",
            func=get_current_time,
            description="Şu anki zamanı öğrenmek için kullanın"
        ),
        Tool(
            name="calculator",
            func=simple_calculator,
            description="Matematik işlemleri yapmak için kullanın"
        ),
        WeatherTool()  # Hava durumu tool'u
    ]
    
    # Memory destekli conversational agent oluştur
    # Bu agent türü önceki konuşmaları hatırlar ve bağlam kurabilir
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,  # Konuşma destekli agent türü
        memory=memory,    # Memory'yi agent'a bağla
        verbose=False
    )
    
    # Birbiriyle ilişkili sorular dizisi (önceki cevapları hatırlaması gerekecek)
    conversation = [
        "Merhaba! İstanbul'da hava nasıl?",          # İlk soru: hava durumu
        "Bu sıcaklığı Fahrenheit'a çevirir misin?",  # İkinci soru: önceki cevaptaki sıcaklığı kullan
        "İlk sorumda hangi şehri sormuştum?",        # Üçüncü soru: memory test
        "Şu anki saat kaç?"                          # Dördüncü soru: farklı tool
    ]
    
    # Konuşmayı sırayla yürüt
    for question in conversation:
        print(f"\nKullanıcı: {question}")
        try:
            response = agent.run(question)
            print(f"Agent: {response}")
        except Exception as e:
            print(f"Hata: {e}")
    
    return agent

# =============================================================================
# REACTİVE AGENT - Modern ReAct yapısı ile gelişmiş agent
# =============================================================================

def create_reactive_agent_example():
    """
    Modern ReAct (Reasoning + Acting) agent yapısını kullanan örnek
    Bu agent daha gelişmiş düşünme ve hareket etme yetisine sahip
    """
    print("\n" + "=" * 60)
    print("4. REACT AGENT YAPISI")
    print("=" * 60)
    
    # Gelişmiş tool'ları tanımla
    tools = [
        Tool(
            name="search",
            func=lambda query: f"'{query}' konusunda arama yapıldı. İlgili bilgiler bulundu.",
            description="İnternet araması yapmak için kullanın"
        ),
        Tool(
            name="calculator",
            func=simple_calculator,
            description="Matematik işlemleri yapmak için kullanın"
        ),
        WeatherTool()
    ]
    
    try:
        # LangChain Hub'dan önceden hazırlanmış ReAct prompt'unu yükle
        # Bu prompt agent'ın nasıl düşüneceğini ve hareket edeceğini belirler
        prompt = hub.pull("hwchase17/react")
        
        # Modern ReAct agent'ı oluştur
        agent = create_react_agent(llm, tools, prompt)
        
        # Agent'ı çalıştırmak için executor oluştur
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,                # Debug çıktılarını gösterme
            handle_parsing_errors=True    # Parsing hatalarını otomatik handle et
        )
        
        # Karmaşık, çok adımlı sorular
        complex_questions = [
            "İstanbul'da hava 15 derece. Bu sıcaklığı Fahrenheit'a çevir ve sonucu açıkla.",  # Hava + matematik
            "Python programlama dili hakkında arama yap."                                    # Arama tool'u
        ]
        
        # Karmaşık soruları test et
        for question in complex_questions:
            print(f"\nKarmaşık Soru: {question}")
            try:
                # invoke() metodunu kullanarak agent'ı çalıştır
                response = agent_executor.invoke({"input": question})
                print(f"Detaylı Cevap: {response['output']}")
            except Exception as e:
                print(f"Hata: {e}")
                
    except Exception as e:
        print(f"ReAct agent oluşturulurken hata: {e}")
        print("Basit agent ile devam ediliyor...")

# =============================================================================
# PERFORMANS ANALİZİ - Agent türleri ve optimizasyon ipuçları
# =============================================================================

def agent_performance_analysis():
    """
    Farklı agent türlerinin performansını ve kullanım alanlarını analiz eder
    """
    print("\n" + "=" * 60)
    print("5. AGENT PERFORMANS ANALİZİ")
    print("=" * 60)
    
    print("""
    AGENT TÜRLERİ VE KULLANIM ALANLARI:
    
    1. ZERO_SHOT_REACT_DESCRIPTION
       ✓ En basit agent türü - hızlı başlangıç için ideal
       ✓ Tool açıklamalarını kullanarak karar verir
       ✓ Tek adımda sorunu çözmeye çalışır
       ✗ Karmaşık, çok adımlı görevlerde yetersiz kalabilir
       
    2. CONVERSATIONAL_REACT_DESCRIPTION  
       ✓ Memory destekli - önceki konuşmaları hatırlar
       ✓ Bağlamsal sohbet imkanı sunar
       ✓ İnsan benzeri etkileşim sağlar
       ✓ Uzun konuşmalarda tutarlılık
       
    3. REACT_DOCSTORE
       ✓ Dokuman arama ve bilgi çıkarma için optimize edilmiş
       ✓ Büyük bilgi tabanlarında arama yapar
       ✓ Akademik/araştırma görevleri için ideal
       
    4. SELF_ASK_WITH_SEARCH
       ✓ Karmaşık soruları alt sorulara böler
       ✓ Adım adım problemi çözer
       ✓ Analitik düşünme gerektirenler için
       
    PERFORMANS İPUÇLARI:
    - Tool açıklamalarını mümkün olduğunca net ve spesifik yazın
    - Verbose=True ile debug yaparak agent'ın düşünce sürecini izleyin
    - Max_iterations ile sonsuz döngülere karşı korunun
    - Her tool için error handling ekleyin
    - Memory kullanımını ihtiyaca göre ayarlayın (uzun konuşmalarda bellek şişer)
    - Tool sayısını çok artırmayın (agent kararsız kalabilir)
    """)

# =============================================================================
# ANA ÇALIŞTIRMA BLOĞU
# =============================================================================

if __name__ == "__main__":
    print("LANGCHAIN TOOLS VE AGENTS ÖRNEKLERİ")
    print("Bu örneklerde tool ve agent kullanımını öğreneceksiniz.\n")
    
    try:
        # Tüm örnekleri sırayla çalıştır
        print("🔧 Basit tool'lar çalıştırılıyor...")
        basic_tools_example()
        
        print("\n🛠️ Özel tool sınıfları test ediliyor...")
        custom_tools_example()
        
        print("\n🧠 Memory'li agent deneniyor...")
        memory_agent_example()
        
        print("\n⚡ Modern ReAct agent oluşturuluyor...")
        create_reactive_agent_example()
        
        print("\n📊 Performans analizi gösteriliyor...")
        agent_performance_analysis()
        
        print("\n" + "=" * 60)
        print("✅ TÜM TOOL VE AGENT ÖRNEKLERİ BAŞARIYLA TAMAMLANDI!")
        print("🎯 Artık kendi tool'larınızı ve agent'larınızı oluşturabilirsiniz.")
        print("💡 Gerçek projelerinizde bu örnekleri temel alabilirsiniz.")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Genel hata: {e}")
        print("🔍 OpenAI API anahtarınızı ve bağımlılıkları kontrol edin!")
        print("📋 Gerekli paketler: langchain, langchain-openai, python-dotenv")
"""
Hafta 5 - Bölüm 5: Streaming Output ve Canlı Veri Akışı
LangChain ile streaming ve real-time uygulamalar
"""

# Gerekli kütüphaneleri içe aktarıyoruz
import os
import time
import asyncio
from typing import Any, Dict, List, Optional

# LangChain kütüphanelerini içe aktarıyoruz
from langchain_openai import OpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory

# .env dosyasından çevre değişkenlerini yüklüyoruz (API anahtarları için)
from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# ÖZEL CALLBACK HANDLER'LAR (ÖZEL YANIT YAKALAYICILARI)
# =============================================================================

class CustomStreamingHandler(BaseCallbackHandler):
    """
    Özelleştirilmiş streaming handler sınıfı
    Bu sınıf AI'ın yanıt üretme sürecini adım adım yakalar ve özel işlemler yapar
    """
    
    def __init__(self):
        """
        Sınıf başlatıcısı - ilk değerleri ayarlıyoruz
        """
        self.tokens = []  # Gelen token'ları (kelime parçalarını) saklayacak liste
        self.current_response = ""  # Şu anki yanıtın tamamını tutacak değişken
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> Any:
        """
        LLM (Dil Modeli) yanıt üretmeye başladığında çağrılır
        Kullanıcıya işlemin başladığını bildirir
        """
        print("🤖 AI yanıt oluşturuyor...\n")
        print("📝 Canlı Yanıt: ", end="", flush=True)  # Satır sonu yazmadan bekle
    
    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """
        Her yeni token (kelime parçası) geldiğinde çağrılır
        Bu sayede yanıt kelime kelime ekranda görünür (typing efekti)
        """
        print(token, end="", flush=True)  # Token'ı hemen yazdır
        self.tokens.append(token)  # Token'ı listeye ekle
        self.current_response += token  # Toplam yanıta ekle
        time.sleep(0.05)  # Daktilo efekti için kısa bekle
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """
        LLM yanıt üretmeyi bitirdiğinde çağrılır
        İstatistikleri gösterir
        """
        print("\n\n✅ Yanıt tamamlandı!")
        print(f"📊 Toplam token sayısı: {len(self.tokens)}")
    
    def on_llm_error(self, error: Exception, **kwargs: Any) -> Any:
        """
        Hata oluştuğunda çağrılır
        Hata mesajını kullanıcıya gösterir
        """
        print(f"\n❌ Hata: {error}")

class ProgressHandler(BaseCallbackHandler):
    """
    İlerleme gösterici handler sınıfı
    AI'ın yanıt üretme sürecini görsel olarak takip etmemizi sağlar
    """
    
    def __init__(self):
        """
        İlerleme göstergesi için gerekli değişkenleri başlatır
        """
        self.step_count = 0  # Hangi adımda olduğumuzu tutar
        # İlerleme adımlarının listesi
        self.steps = ["🔍 Analiz", "💭 Düşünme", "✍️ Yazma", "🎯 Tamamlama"]
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> Any:
        """
        LLM başladığında ilerleme çubuğunu başlatır
        """
        print("📈 İşlem Başlıyor:")
        self.show_progress()
    
    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """
        Her yeni token geldiğinde ilerleme durumunu günceller
        Noktalama işaretlerini adım geçişi olarak algılar
        """
        if len(token.strip()) > 0 and self.step_count < len(self.steps) - 1:
            # Cümle sonu işaretlerinde bir sonraki adıma geç
            if token in ['.', '!', '?', '\n']:
                self.step_count += 1
                self.show_progress()
    
    def show_progress(self):
        """
        İlerleme çubuğunu ekranda günceller
        Tamamlanan adımları ✅ ile, bekleyenleri ⏳ ile gösterir
        """
        progress = "["
        for i, step in enumerate(self.steps):
            if i <= self.step_count:
                progress += f"✅ {step} "  # Tamamlanan adım
            else:
                progress += f"⏳ {step} "  # Bekleyen adım
        progress += "]"
        print(f"\r{progress}", end="", flush=True)  # Aynı satırda güncelle
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """
        LLM bittiğinde tüm adımları tamamlanmış olarak gösterir
        """
        self.step_count = len(self.steps) - 1
        self.show_progress()
        print("\n🎉 İşlem Tamamlandı!\n")

# =============================================================================
# STREAMING ÖRNEKLERİ (CANLI AKIŞ ÖRNEKLERİ)
# =============================================================================

def basic_streaming_example():
    """
    Temel streaming örneği
    Normal LLM ile streaming LLM arasındaki farkı gösterir
    """
    print("=" * 60)
    print("1. TEMEL STREAMING OUTPUT")
    print("=" * 60)
    
    # Streaming özelliği aktif olan LLM oluştur
    # StreamingStdOutCallbackHandler: Yanıtları otomatik olarak ekrana yazdırır
    llm = OpenAI(
        temperature=0.7,  # Yaratıcılık seviyesi (0-1 arası)
        streaming=True,   # Streaming özelliğini aktifleştir
        callbacks=[StreamingStdOutCallbackHandler()]  # Otomatik yazdırma handler'ı
    )
    
    # Normal (streaming olmayan) LLM ile karşılaştırma
    print("Normal yanıt (streaming yok):")
    normal_llm = OpenAI(temperature=0.7)  # Streaming kapalı
    normal_response = normal_llm("Python hakkında kısa bir açıklama yaz.")
    print(normal_response)  # Yanıt tek seferde gelir
    
    print("\n" + "-" * 40)
    print("Streaming yanıt:")
    # Streaming LLM ile aynı soruyu sor - yanıt kelime kelime gelecek
    streaming_response = llm("Python hakkında kısa bir açıklama yaz.")
    print("\n")
    
    return streaming_response

def custom_streaming_example():
    """
    Özel streaming handler örneği
    Kendi yazdığımız CustomStreamingHandler'ı kullanır
    """
    print("\n" + "=" * 60)
    print("2. ÖZEL STREAMING HANDLER")
    print("=" * 60)
    
    # Özel handler'ımızı oluştur
    custom_handler = CustomStreamingHandler()
    
    # Özel handler ile LLM oluştur
    llm = OpenAI(
        temperature=0.7,
        streaming=True,
        callbacks=[custom_handler]  # Özel handler'ımızı kullan
    )
    
    # Uzun bir prompt ile test - daha çok token üretilecek
    prompt = """
    Yapay zeka teknolojisinin gelecekte topluma etkilerini detaylı olarak açıkla.
    Pozitif ve negatif etkileri ayrı ayrı ele al.
    """
    
    # LLM'den yanıt al - custom handler otomatik çalışacak
    response = llm(prompt)
    
    # Handler'dan toplanan bilgileri göster
    print(f"\n📋 Handler Bilgileri:")
    print(f"- Toplanan token sayısı: {len(custom_handler.tokens)}")
    print(f"- İlk 5 token: {custom_handler.tokens[:5]}")
    print(f"- Son 5 token: {custom_handler.tokens[-5:]}")
    
    return custom_handler

def progress_streaming_example():
    """
    İlerleme gösterici ile streaming örneği
    AI'ın yanıt üretme sürecini görsel olarak takip eder
    """
    print("\n" + "=" * 60)
    print("3. İLERLEME GÖSTERİCİLİ STREAMING")
    print("=" * 60)
    
    # İlerleme gösterici handler'ı oluştur
    progress_handler = ProgressHandler()
    
    # İlerleme göstericili LLM oluştur
    llm = OpenAI(
        temperature=0.7,
        streaming=True,
        callbacks=[progress_handler]  # İlerleme handler'ını kullan
    )
    
    # Uzun yanıt gerektiren prompt
    prompt = """
    Bir startup'ın başarılı olması için gerekli 5 temel unsuru açıkla.
    Her unsur için detaylı açıklama yap.
    """
    
    # LLM'den yanıt al - ilerleme çubuğu otomatik gösterilecek
    response = llm(prompt)
    print(response)
    
    return progress_handler

# =============================================================================
# STREAMING CHAIN ÖRNEKLERİ (ZİNCİRLİ İŞLEMLER)
# =============================================================================

def streaming_chain_example():
    """
    Chain (zincir) ile streaming örneği
    Template kullanarak daha karmaşık işlemler yapar
    """
    print("\n" + "=" * 60)
    print("4. STREAMING CHAIN KULLANIMI")
    print("=" * 60)
    
    # Streaming özellikli LLM oluştur
    streaming_llm = OpenAI(
        temperature=0.7,
        streaming=True,
        callbacks=[CustomStreamingHandler()]  # Özel handler kullan
    )
    
    # Prompt template oluştur - değişken yerler için {topic} kullan
    prompt = PromptTemplate(
        input_variables=["topic"],  # Girdi değişkenleri
        template="""
        Bu konu hakkında yaratıcı bir hikaye yaz: {topic}
        Hikaye en az 200 kelime olsun ve heyecanlı detaylar içersin.
        """
    )
    
    # LLM ve prompt'u birleştirerek chain (zincir) oluştur
    chain = LLMChain(
        llm=streaming_llm,  # Streaming LLM kullan
        prompt=prompt       # Template'i bağla
    )
    
    # Chain'i çalıştır - {topic} yerine değer geçir
    print("Hikaye konusu: 'Uzayda kaybolmuş bir robot'")
    result = chain.run("uzayda kaybolmuş bir robot")
    
    return result

# =============================================================================
# REAL-TIME SOHBET SİMÜLASYONU (CANLI SOHBET)
# =============================================================================

class RealTimeChatBot:
    """
    Gerçek zamanlı sohbet botu sınıfı
    Kullanıcıyla sürekli sohbet edebilen, geçmişi hatırlayan bot
    """
    
    def __init__(self):
        """
        Sohbet botunu başlatır ve gerekli bileşenleri oluşturur
        """
        # Konuşma geçmişini saklamak için hafıza oluştur
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",  # Hafıza anahtarı
            return_messages=True        # Mesajları döndür
        )
        
        # Özel streaming handler oluştur
        self.streaming_handler = CustomStreamingHandler()
        
        # Streaming özellikli LLM oluştur
        self.llm = OpenAI(
            temperature=0.8,  # Daha yaratıcı yanıtlar için yüksek değer
            streaming=True,
            callbacks=[self.streaming_handler]
        )
        
        # Sohbet için prompt template oluştur
        self.prompt = PromptTemplate(
            input_variables=["chat_history", "user_input"],
            template="""
            Sen arkadaş canlısı bir sohbet botusun. Kullanıcıyla doğal bir sohbet et.
            
            Önceki konuşma:
            {chat_history}
            
            Kullanıcı: {user_input}
            
            Bot:
            """
        )
        
        # LLM, prompt ve hafızayı birleştiren chain oluştur
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory  # Hafızayı bağla
        )
    
    def chat(self, user_input: str):
        """
        Kullanıcı girişini işler ve yanıt üretir
        
        Args:
            user_input (str): Kullanıcının yazdığı mesaj
            
        Returns:
            str: Botun yanıtı
        """
        print(f"\n👤 Kullanıcı: {user_input}")
        
        # "Bot yazıyor" efekti oluştur
        print("⌨️  Bot yazıyor", end="", flush=True)
        for i in range(3):
            time.sleep(0.5)  # Yarım saniye bekle
            print(".", end="", flush=True)  # Nokta ekle
        print("\n")
        
        # Chain'i çalıştırarak yanıt üret (streaming ile)
        response = self.chain.run(user_input=user_input)
        return response

def realtime_chat_example():
    """
    Gerçek zamanlı sohbet örneği
    Sohbet botunu test eder
    """
    print("\n" + "=" * 60)
    print("5. REAL-TIME SOHBET BOT'U")
    print("=" * 60)
    
    # Sohbet botunu oluştur
    chatbot = RealTimeChatBot()
    
    # Test için simüle edilmiş konuşma
    conversation = [
        "Merhaba! Nasılsın?",
        "Bugün hava çok güzel, sen ne yapıyorsun?",
        "Yapay zeka hakkında ne düşünüyorsun?",
        "Bana bir şaka anlatır mısın?"
    ]
    
    # Her mesajı sırayla bota gönder
    for message in conversation:
        try:
            response = chatbot.chat(message)  # Mesajı gönder
            time.sleep(1)  # Sohbet akışı için bekle
        except Exception as e:
            print(f"Sohbet hatası: {e}")
            break  # Hata durumunda döngüden çık
    
    return chatbot

# =============================================================================
# ASYNC STREAMING ÖRNEKLERİ (ASENKRON CANLI AKIŞ)
# =============================================================================

async def async_streaming_example():
    """
    Asynchronous (eşzamansız) streaming örneği
    Birden fazla işlemi aynı anda yapabilir
    """
    print("\n" + "=" * 60)
    print("6. ASYNC STREAMING (Simüle Edilmiş)")
    print("=" * 60)
    
    # Farklı yanıt parçalarını simüle et
    responses = [
        "Python çok güçlü bir programlama dilidir.",
        "Web geliştirme, veri analizi, yapay zeka alanlarında kullanılır.",
        "Söz dizimi basit ve okunabilir olduğu için öğrenmesi kolaydır.",
        "Geniş kütüphane ekosistemi sayesinde hızlı geliştirme sağlar."
    ]
    
    print("🚀 Async streaming başlıyor...\n")
    
    # Her yanıt parçasını işle
    for i, response in enumerate(responses, 1):
        print(f"📦 Chunk {i}: ", end="", flush=True)
        
        # Her karakteri ayrı ayrı yazdır (typing efekti)
        for char in response:
            print(char, end="", flush=True)
            await asyncio.sleep(0.03)  # Asenkron bekle
        
        print()  # Yeni satır
        await asyncio.sleep(0.5)  # Chunk'lar arası bekle
    
    print("\n✅ Async streaming tamamlandı!")

# =============================================================================
# PERFORMANS KARŞILAŞTIRMASI
# =============================================================================

def streaming_performance_comparison():
    """
    Streaming vs Normal LLM performans karşılaştırması
    Hangi yöntemin ne zaman kullanılacağını anlamaya yardımcı olur
    """
    print("\n" + "=" * 60)
    print("7. PERFORMANS KARŞILAŞTIRMASI")
    print("=" * 60)
    
    # Test için ortak prompt
    prompt = "Python programlama dilinin avantajlarını listele ve açıkla."
    
    # Normal LLM testini yap ve süresini ölç
    print("⏱️  Normal LLM testi...")
    normal_llm = OpenAI(temperature=0.7)  # Streaming kapalı
    start_time = time.time()  # Başlangıç zamanı
    normal_response = normal_llm(prompt)
    normal_time = time.time() - start_time  # Geçen süre
    
    # Normal LLM sonuçlarını göster
    print(f"Normal LLM süresi: {normal_time:.2f} saniye")
    print(f"Normal yanıt uzunluğu: {len(normal_response)} karakter\n")
    
    # Streaming LLM testini yap ve süresini ölç
    print("⏱️  Streaming LLM testi...")
    streaming_llm = OpenAI(
        temperature=0.7,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]  # Canlı yazdırma
    )
    start_time = time.time()  # Başlangıç zamanı
    streaming_response = streaming_llm(prompt)
    streaming_time = time.time() - start_time  # Geçen süre
    
    # Streaming LLM sonuçlarını göster
    print(f"\nStreaming LLM süresi: {streaming_time:.2f} saniye")
    print(f"Streaming yanıt uzunluğu: {len(streaming_response)} karakter")
    
    # Karşılaştırma analizi
    print(f"\n📊 Analiz:")
    print(f"- Süre farkı: {abs(normal_time - streaming_time):.2f} saniye")
    print(f"- Streaming kullanıcı deneyimi: Daha iyi (canlı feedback)")
    print(f"- Normal LLM: Daha hızlı işlem (tek seferde)")

# =============================================================================
# ANA FONKSİYON
# =============================================================================

def main():
    """
    Ana program fonksiyonu
    Tüm örnekleri sırayla çalıştırır
    """
    print("LANGCHAIN STREAMING VE CANLI VERİ AKIŞI ÖRNEKLERİ")
    print("Bu örneklerde streaming output ve real-time uygulamaları öğreneceksiniz.\n")
    
    try:
        # Tüm streaming örneklerini sırayla çalıştır
        basic_streaming_example()          # 1. Temel streaming
        custom_streaming_example()         # 2. Özel handler
        progress_streaming_example()       # 3. İlerleme göstergesi
        streaming_chain_example()          # 4. Chain ile streaming
        realtime_chat_example()            # 5. Canlı sohbet
        
        # Asenkron örneği çalıştır
        asyncio.run(async_streaming_example())  # 6. Async streaming
        
        # Performans karşılaştırması
        streaming_performance_comparison()  # 7. Performans testi
        
        # Bitirme mesajı
        print("\n" + "=" * 60)
        print("TÜM STREAMING ÖRNEKLERİ TAMAMLANDI!")
        print("Artık kendi real-time uygulamalarınızı geliştirebilirsiniz.")
        print("=" * 60)
        
        # Kullanıcı için pratik ipuçları
        print("\n🎯 STREAMING İPUÇLARI:")
        print("1. Uzun yanıtlar için streaming kullanın")      # Kullanıcı beklemez
        print("2. Kullanıcı deneyimini iyileştirir")           # Canlı feedback
        print("3. Custom handler'lar ile özelleştirin")        # Kendi ihtiyaçlarınıza göre
        print("4. Progress indicator'lar ekleyin")             # İlerleme göstergesi
        print("5. Error handling'i unutmayın")                 # Hata yönetimi önemli
        
    except Exception as e:
        # Genel hata yakalama
        print(f"Genel hata: {e}")
        print("OpenAI API anahtarınızı kontrol edin!")

# Program başlangıç noktası
if __name__ == "__main__":
    main()  # Ana fonksiyonu çalıştır
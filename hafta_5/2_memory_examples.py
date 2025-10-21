"""
Hafta 5 - Bölüm 2: Memory Kullanımı
LangChain ile farklı memory türlerini öğrenme
"""

import os
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    ConversationSummaryBufferMemory,
    ConversationTokenBufferMemory
)

from dotenv import load_dotenv
load_dotenv()

# OpenAI LLM'ini başlat
# temperature=0.5: Orta seviye yaratıcılık
# max_tokens=50: Kısa cevaplar (döngü önlemek için)
# request_timeout=10: 10 saniye zaman aşımı
llm = OpenAI(temperature=0.5, max_tokens=50, request_timeout=10)

def buffer_memory_example():
    """
    ConversationBufferMemory Örneği
    - Tüm konuşma geçmişini bellekte tutar
    - Hiçbir bilgi kaybetmez
    - Uzun konuşmalarda bellek kullanımı artar
    """
    print("=" * 60)
    print("1. BUFFER MEMORY - Tüm konuşmayı hatırlar")
    print("=" * 60)
    
    # Buffer memory oluştur - tüm sohbeti saklar
    memory = ConversationBufferMemory()
    
    # ConversationChain oluştur - memory ile birlikte
    conversation = ConversationChain(
        llm=llm,           # Kullanılacak LLM
        memory=memory,     # Bellek sistemi
        verbose=False      # Detaylı çıktı kapalı (döngü önlemek için)
    )
    
    # Sohbet simülasyonu başlat
    print("Sohbet başlıyor...")
    
    # İlk mesaj - kullanıcı kendini tanıtıyor
    response1 = conversation.predict(input="Merhaba! Benim adım Ahmet.")
    print(f"AI: {response1}")
    
    # İkinci mesaj - AI'ın hatırlayıp hatırlamadığını test et
    response2 = conversation.predict(input="Sen benim adımı hatırlıyor musun?")
    print(f"AI: {response2}")
    
    # Üçüncü mesaj - daha detaylı bilgi sorgusu
    response3 = conversation.predict(input="Peki ben hangi konularda ilgileniyorum?")
    print(f"AI: {response3}")
    
    # Memory içeriğini göster - tüm konuşma geçmişi
    print("\nMemory içeriği:")
    print(memory.buffer)
    
    return memory

def window_memory_example():
    """
    ConversationBufferWindowMemory Örneği
    - Sadece son k sayıda mesajı hatırlar
    - Sabit bellek kullanımı
    - Eski mesajlar otomatik silinir
    """
    print("\n" + "=" * 60)
    print("2. WINDOW MEMORY - Son 2 mesajı hatırlar")
    print("=" * 60)
    
    # Window memory oluştur - sadece son 2 etkileşimi tutar
    memory = ConversationBufferWindowMemory(k=2)  # k=2: son 2 mesaj çifti
    
    # Konversasyon chain'i oluştur
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False
    )
    
    # Test için birden fazla mesaj hazırla
    messages = [
        "İlk mesaj: Merhaba!",
        "İkinci mesaj: Bugün hava nasıl?",
        "Üçüncü mesaj: Spor hakkında konuşalım.",
        "Dördüncü mesaj: İlk mesajımı hatırlıyor musun?"  # Bu soruda ilk mesaj unutulmuş olmalı
    ]
    
    # Her mesajı gönder ve yanıtı al
    for i, msg in enumerate(messages, 1):
        print(f"\n--- Mesaj {i} ---")
        response = conversation.predict(input=msg)
        print(f"AI: {response}")
    
    # Son durumda sadece son 2 etkileşim kalmalı
    print("\nSon memory içeriği (sadece son 2 etkileşim):")
    print(memory.buffer)
    
    return memory

def summary_memory_example():
    """
    ConversationSummaryMemory Örneği
    - Konuşmayı sürekli özetler
    - LLM kullanarak özet çıkarır
    - Uzun konuşmalarda verimli
    """
    print("\n" + "=" * 60)
    print("3. SUMMARY MEMORY - Konuşmayı özetler")
    print("=" * 60)
    
    # Summary memory oluştur - konuşmayı özetlemek için LLM kullanır
    memory = ConversationSummaryMemory(llm=llm)
    
    # Konversasyon chain'i oluştur
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False
    )
    
    # Uzun sohbet simülasyonu - kişisel bilgiler içeren mesajlar
    long_conversation = [
        "Merhaba! Ben bir yazılım geliştiricisiyim.",      # Meslek bilgisi
        "Python ve JavaScript kullanıyorum.",              # Teknoloji tercihleri
        "Yapay zeka konusunda çok ilgiliyim.",             # İlgi alanları
        "Son zamanlarda LangChain öğreniyorum.",           # Güncel aktiviteler
        "Peki benim hakkımda neler hatırlıyorsun?"         # Özet kontrolü
    ]
    
    # Her mesajı gönder ve yanıtı al
    for msg in long_conversation:
        response = conversation.predict(input=msg)
        print(f"AI: {response}\n")
    
    # Memory özeti - LLM tarafından oluşturulan özet
    print("Memory özeti:")
    print(memory.buffer)
    
    return memory

def summary_buffer_memory_example():
    """
    ConversationSummaryBufferMemory Örneği
    - Summary ve Buffer memory'nin kombinasyonu
    - Token limitine kadar buffer, sonra özet
    - En verimli hibrit yaklaşım
    """
    print("\n" + "=" * 60)
    print("4. SUMMARY BUFFER MEMORY - Hibrit yaklaşım")
    print("=" * 60)
    
    # Summary buffer memory - 100 token limiti ile
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=100  # 100 token geçince özetleme başlar
    )
    
    # Konversasyon chain'i oluştur
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False
    )
    
    # Teknik konularda uzun mesajlar
    messages = [
        "Merhaba! Bugün makine öğrenimi hakkında konuşmak istiyorum.",
        "Özellikle neural network'ler ilgimi çekiyor.",
        "TensorFlow ve PyTorch arasındaki farkları merak ediyorum.",
        "Hangi projelerde hangi framework'ü kullanmalıyım?",
        "Peki başlangıçta nelerden bahsetmiştik?"  # Başlangıç özetlenmiş olmalı
    ]
    
    # Her mesajı gönder ve yanıtı al
    for msg in messages:
        response = conversation.predict(input=msg)
        print(f"AI: {response}\n")
    
    # Memory durumunu göster - hem özet hem de son mesajlar
    print("Memory durumu:")
    print(f"Moving summary: {memory.moving_summary_buffer}")  # Özetlenen kısım
    print(f"Chat memory: {memory.chat_memory.messages}")      # Son mesajlar
    
    return memory

def token_buffer_memory_example():
    """
    ConversationTokenBufferMemory Örneği
    - Token sayısına göre bellek yönetimi
    - Tam kontrol ve maliyet optimizasyonu
    - Belirli token limitini aşmaz
    """
    print("\n" + "=" * 60)
    print("5. TOKEN BUFFER MEMORY - Token limiti ile")
    print("=" * 60)
    
    # Token buffer memory - maksimum 50 token
    memory = ConversationTokenBufferMemory(
        llm=llm,
        max_token_limit=50  # 50 token geçince eski mesajlar silinir
    )
    
    # Konversasyon chain'i oluştur
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False
    )
    
    # Kısa mesajlar ile test
    short_messages = [
        "Merhaba!",                                # Kısa selamlama
        "Nasılsın?",                              # Hal hatır
        "Bugün güzel bir gün.",                   # Gözlem
        "Programlama öğreniyorum.",               # Aktivite
        "İlk mesajımı hatırlıyor musun?"          # Token limiti kontrolü
    ]
    
    # Her mesajı gönder ve yanıtı al
    for msg in short_messages:
        response = conversation.predict(input=msg)
        print(f"AI: {response}\n")
    
    # Token buffer içeriği - sadece limit dahilindeki mesajlar
    print("Token buffer içeriği:")
    print(memory.buffer)
    
    return memory

def custom_memory_with_chain():
    """
    Özel Memory Implementasyonu
    - Kişiselleştirilmiş prompt template
    - Message formatında memory
    - Daha kontrollü konuşma akışı
    """
    print("\n" + "=" * 60)
    print("6. ÖZEL MEMORY KULLANIMI - Kişiselleştirilmiş")
    print("=" * 60)
    
    # Özel memory konfigürasyonu
    memory = ConversationBufferMemory(
        memory_key="chat_history",    # Memory anahtarı
        return_messages=True          # Message formatında döndür
    )
    
    # Özel prompt template - Türkçe ve kişiselleştirilmiş
    template = """
    Sen yardımsever bir asistansın. Kullanıcının önceki mesajlarını hatırla.
    
    Önceki konuşma:
    {chat_history}
    
    Kullanıcı: {input}
    
    Asistan:
    """
    
    # Prompt template oluştur
    prompt = PromptTemplate(
        input_variables=["chat_history", "input"],  # Gerekli değişkenler
        template=template
    )
    
    # LLMChain oluştur - özel prompt ve memory ile
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory
    )
    
    # Etkileşim senaryosu
    interactions = [
        "Bana Python hakkında bilgi ver.",                    # Bilgi talebi
        "Peki bu bilgileri nasıl uygulayabilirim?",          # Uygulama sorusu
        "İlk sorumda ne sormuştum?"                          # Memory testi
    ]
    
    # Her etkileşimi çalıştır
    for interaction in interactions:
        response = chain.run(input=interaction)
        print(f"Kullanıcı: {interaction}")
        print(f"Asistan: {response}\n")
    
    return chain

def memory_comparison():
    """
    Memory Türlerinin Detaylı Karşılaştırması
    - Her memory türünün avantaj ve dezavantajları
    - Kullanım senaryoları
    - Seçim kriterleri
    """
    print("\n" + "=" * 60)
    print("7. MEMORY TÜRLERİ KARŞILAŞTIRMASI")
    print("=" * 60)
    
    print("""
    MEMORY TÜRLERİ VE ÖZELLİKLERİ:
    
    1. ConversationBufferMemory (Tam Bellek)
       ✓ Tüm konuşmayı eksiksiz hatırlar
       ✓ Hiçbir bilgi kaybı olmaz
       ✗ Uzun konuşmalarda çok bellek kullanır
       ✗ Token maliyeti yüksek olabilir
       → Kısa sohbetler için ideal
       
    2. ConversationBufferWindowMemory (Pencere Bellek)
       ✓ Sabit boyutta bellek kullanır
       ✓ Performans öngörülebilir
       ✗ Eski bilgileri tamamen kaybeder
       ✗ Bağlam kopuklukları olabilir
       → Sürekli sohbetler için uygun
       
    3. ConversationSummaryMemory (Özet Bellek)
       ✓ Uzun konuşmaları verimli özetler
       ✓ Önemli bilgileri korur
       ✗ Detay kaybı olabilir
       ✗ LLM kullanımı ek maliyet
       → Uzun dönemli sohbetler için ideal
       
    4. ConversationSummaryBufferMemory (Hibrit Bellek)
       ✓ En iyi özelliklerini birleştirir
       ✓ Esnek ve verimli
       ✓ Hem detay hem özet korur
       ✗ Karmaşık konfigürasyon
       → Çoğu senaryo için en iyi seçim
       
    5. ConversationTokenBufferMemory (Token Bellek)
       ✓ Token limiti ile tam kontrol
       ✓ Maliyet optimizasyonu mükemmel
       ✓ Öngörülebilir performans
       ✗ Manuel token yönetimi gerekli
       → Maliyet kritik projeler için ideal
    
    SEÇIM KRİTERLERİ:
    - Kısa sohbetler → BufferMemory
    - Uzun sohbetler → SummaryMemory
    - Maliyet hassasiyeti → TokenBufferMemory
    - En iyi performans → SummaryBufferMemory
    - Basit kullanım → WindowMemory
    """)

if __name__ == "__main__":
    print("LANGCHAIN MEMORY ÖRNEKLERİ")
    print("Bu örneklerde farklı memory türlerini detaylı olarak öğreneceksiniz.\n")
    
    try:
        # Tüm memory örneklerini sırasıyla çalıştır
        print("🚀 Memory örnekleri başlatılıyor...")
        
        # 1. Temel buffer memory
        buffer_memory_example()
        
        # 2. Pencere tabanlı memory
        window_memory_example()
        
        # 3. Özet tabanlı memory
        summary_memory_example()
        
        # 4. Hibrit memory yaklaşımı
        summary_buffer_memory_example()
        
        # 5. Token tabanlı memory
        token_buffer_example()
        
        # 6. Özel memory implementasyonu
        custom_memory_with_chain()
        
        # 7. Karşılaştırma ve değerlendirme
        memory_comparison()
        
        print("\n" + "=" * 60)
        print("🎉 TÜM MEMORY ÖRNEKLERİ BAŞARIYLA TAMAMLANDI!")
        print("📝 Projelerinizde hangi memory türünü kullanacağınıza artık karar verebilirsiniz.")
        print("💡 Her memory türünün kendine özgü avantajları ve kullanım alanları var.")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Hata oluştu: {e}")
        print("🔑 OpenAI API anahtarınızı kontrol edin!")
        print("🌐 İnternet bağlantınızı kontrol edin!")
        print("📋 .env dosyasında OPENAI_API_KEY ayarlandığından emin olun!")
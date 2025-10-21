"""
Hafta 5 - Bölüm 4: Senaryo Bazlı Uygulamalar
Gerçek hayat senaryoları ile LangChain kullanımı
"""

# Gerekli kütüphaneleri içe aktarma
import os
import json
from datetime import datetime, timedelta
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, AgentType, initialize_agent
from langchain.tools import BaseTool
from typing import Optional, Type, List
from pydantic import BaseModel, Field

# Çevre değişkenlerini yükle (.env dosyasından API anahtarları)
from dotenv import load_dotenv
load_dotenv()

# OpenAI LLM'ini başlat (temperature=0.7 yaratıcılık için)
llm = OpenAI(temperature=0.7)

# =============================================================================
# SENARYO 1: MÜŞTERİ HİZMETLERİ BOT'U
# Müşteri sorularını otomatik yanıtlayan akıllı bot sistemi
# =============================================================================

class CustomerServiceBot:
    """
    Müşteri hizmetleri bot'u sınıfı
    - Müşteri bilgilerini sorgulama
    - Sipariş durumu takibi
    - Destek bileti oluşturma
    """
    
    def __init__(self):
        """Bot'u başlatır ve gerekli bileşenleri kurar"""
        # Konuşma geçmişini saklayacak bellek
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",  # Bellek anahtarı
            return_messages=True        # Mesajları geri döndür
        )
        
        # Simüle edilmiş müşteri veri tabanı
        # Gerçek uygulamada bu bir SQL veritabanı olurdu
        self.customer_db = {
            "12345": {
                "name": "Ahmet Yılmaz",
                "email": "ahmet@email.com",
                "orders": ["ORD001", "ORD002"],  # Bu müşterinin siparişleri
                "status": "Premium"              # Müşteri durumu
            },
            "67890": {
                "name": "Elif Kaya",
                "email": "elif@email.com", 
                "orders": ["ORD003"],
                "status": "Standard"
            }
        }
        
        # Simüle edilmiş sipariş veri tabanı
        self.order_db = {
            "ORD001": {"product": "Laptop", "status": "Delivered", "date": "2024-01-15"},
            "ORD002": {"product": "Mouse", "status": "Shipped", "date": "2024-01-20"},
            "ORD003": {"product": "Keyboard", "status": "Processing", "date": "2024-01-18"}
        }
        
        # Bot'un kullanacağı araçları kur
        self.setup_tools()
        # Agent'ı (otonom çalışan bot) kur
        self.setup_agent()
    
    def get_customer_info(self, customer_id: str) -> str:
        """
        Müşteri ID'sine göre müşteri bilgilerini getirir
        Args: customer_id - Müşteri kimlik numarası
        Returns: Müşteri bilgileri string formatında
        """
        if customer_id in self.customer_db:
            customer = self.customer_db[customer_id]
            return f"Müşteri: {customer['name']}, Durum: {customer['status']}, Email: {customer['email']}"
        return "Müşteri bulunamadı."
    
    def get_order_status(self, order_id: str) -> str:
        """
        Sipariş ID'sine göre sipariş durumunu kontrol eder
        Args: order_id - Sipariş numarası
        Returns: Sipariş durumu bilgisi
        """
        if order_id in self.order_db:
            order = self.order_db[order_id]
            return f"Sipariş {order_id}: {order['product']}, Durum: {order['status']}, Tarih: {order['date']}"
        return "Sipariş bulunamadı."
    
    def create_support_ticket(self, issue: str) -> str:
        """
        Müşteri sorunu için yeni destek bileti oluşturur
        Args: issue - Müşteri sorunu açıklaması
        Returns: Oluşturulan bilet bilgisi
        """
        # Benzersiz bilet numarası oluştur (tarih-saat bazlı)
        ticket_id = f"TKT{datetime.now().strftime('%Y%m%d%H%M%S')}"
        return f"Destek biletiniz oluşturuldu. Bilet No: {ticket_id}. Konunuz: {issue}"
    
    def setup_tools(self):
        """Bot'un kullanacağı araçları tanımlar"""
        self.tools = [
            # Müşteri bilgileri sorgulama aracı
            Tool(
                name="get_customer_info",
                func=self.get_customer_info,
                description="Müşteri ID'si ile müşteri bilgilerini getirmek için kullanın"
            ),
            # Sipariş durumu kontrol aracı
            Tool(
                name="get_order_status", 
                func=self.get_order_status,
                description="Sipariş ID'si ile sipariş durumunu kontrol etmek için kullanın"
            ),
            # Destek bileti oluşturma aracı
            Tool(
                name="create_support_ticket",
                func=self.create_support_ticket,
                description="Müşteri sorunu için destek bileti oluşturmak için kullanın"
            )
        ]
    
    def setup_agent(self):
        """Konuşmalı ReAct agent'ını kurar"""
        self.agent = initialize_agent(
            tools=self.tools,                                    # Kullanılacak araçlar
            llm=llm,                                             # Dil modeli
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,    # Agent tipi (konuşmalı)
            memory=self.memory,                                  # Konuşma belleği
            verbose=True                                         # Detaylı çıktı
        )
    
    def handle_customer_query(self, query: str) -> str:
        """
        Müşteri sorgusunu işler ve uygun yanıt verir
        Args: query - Müşteri sorusu
        Returns: Bot'un yanıtı
        """
        # Sistem promptu: Bot'un davranış şeklini belirler
        system_prompt = f"""
        Sen yardımsever bir müşteri hizmetleri temsilcisisin. 
        Müşterilere nazik ve profesyonel şekilde yardım et.
        
        Mevcut araçlar:
        - Müşteri bilgileri sorgulama
        - Sipariş durumu kontrolü  
        - Destek bileti oluşturma
        
        Müşteri sorusu: {query}
        """
        
        # Agent'ı çalıştır ve yanıt al
        return self.agent.run(system_prompt)

def customer_service_scenario():
    """Müşteri hizmetleri senaryosunu test eder"""
    print("=" * 60)
    print("SENARYO 1: MÜŞTERİ HİZMETLERİ BOT'U")
    print("=" * 60)
    
    # Müşteri hizmetleri bot'unu oluştur
    bot = CustomerServiceBot()
    
    # Test edilecek müşteri senaryoları
    scenarios = [
        "Merhaba, 12345 ID'li müşteri olarak hesap bilgilerimi öğrenebilir miyim?",
        "ORD001 numaralı siparişimin durumu nedir?",
        "Aldığım ürün bozuk geldi, ne yapabilirim?",
        "67890 müşteri ID'mle son siparişlerimi görebilir miyim?"
    ]
    
    # Her senaryoyu test et
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n--- Test Senaryosu {i} ---")
        print(f"Müşteri: {scenario}")
        
        try:
            # Bot'tan yanıt al
            response = bot.handle_customer_query(scenario)
            print(f"Bot: {response}")
        except Exception as e:
            print(f"Hata: {e}")
    
    return bot

# =============================================================================
# SENARYO 2: İÇERİK OLUŞTURMA ASISTANI
# Blog yazısı, makale gibi içerikleri otomatik oluşturan sistem
# =============================================================================

class ContentCreationAssistant:
    """
    İçerik oluşturma asistan sınıfı
    3 aşamalı süreç: Araştırma -> Planlama -> Yazım
    """
    
    def __init__(self):
        """İçerik oluşturma chain'lerini kurar"""
        self.setup_chains()
    
    def setup_chains(self):
        """İçerik oluşturma sürecinin chain'lerini tanımlar"""
        
        # AŞAMA 1: Konu araştırması prompt'u
        self.research_prompt = PromptTemplate(
            input_variables=["topic"],  # Giriş değişkeni: konu
            template="""
            Bu konu hakkında detaylı araştırma yapın: {topic}
            
            Şunları içeren bir araştırma raporu hazırlayın:
            - Ana konuların özeti
            - Hedef kitle analizi
            - Trend analizi
            - Anahtar kelimeler
            """
        )
        # Araştırma chain'i oluştur
        self.research_chain = LLMChain(
            llm=llm,
            prompt=self.research_prompt,
            output_key="research"  # Çıktı anahtarı
        )
        
        # AŞAMA 2: İçerik planı oluşturma prompt'u
        self.planning_prompt = PromptTemplate(
            input_variables=["topic", "research"],  # Konu ve araştırma sonucu
            template="""
            Konu: {topic}
            Araştırma: {research}
            
            Bu bilgilere dayanarak detaylı bir içerik planı oluşturun:
            - Ana başlıklar
            - Alt başlıklar  
            - İçerik akışı
            - Call-to-action önerileri
            """
        )
        # Planlama chain'i oluştur
        self.planning_chain = LLMChain(
            llm=llm,
            prompt=self.planning_prompt,
            output_key="content_plan"
        )
        
        # AŞAMA 3: İçerik yazımı prompt'u
        self.writing_prompt = PromptTemplate(
            input_variables=["topic", "research", "content_plan"],  # Tüm önceki çıktılar
            template="""
            Konu: {topic}
            Araştırma: {research}
            İçerik Planı: {content_plan}
            
            Bu plan doğrultusunda SEO-friendly, ilgi çekici ve bilgilendirici bir blog yazısı yazın.
            Yazı 800-1000 kelime olsun.
            """
        )
        # Yazım chain'i oluştur
        self.writing_chain = LLMChain(
            llm=llm,
            prompt=self.writing_prompt,
            output_key="final_content"
        )
        
        # AŞAMA 4: Tüm chain'leri sıralı olarak birleştir
        # Her chain'in çıktısı bir sonrakinin girdisi olur
        self.overall_chain = SequentialChain(
            chains=[self.research_chain, self.planning_chain, self.writing_chain],
            input_variables=["topic"],                                              # Ana girdi
            output_variables=["research", "content_plan", "final_content"]          # Tüm çıktılar
        )
    
    def create_content(self, topic: str):
        """
        Verilen konu için tam içerik oluşturma süreci
        Args: topic - İçerik konusu
        Returns: Tüm aşamaların sonuçları
        """
        print(f"\n'{topic}' konusunda içerik oluşturuluyor...\n")
        
        # Tüm chain'i çalıştır
        result = self.overall_chain({"topic": topic})
        
        # Sonuçları formatla ve göster
        print("🔍 ARAŞTIRMA RAPORU:")
        print("-" * 40)
        print(result["research"])
        
        print("\n📋 İÇERİK PLANI:")
        print("-" * 40)
        print(result["content_plan"])
        
        print("\n✍️ FINAL İÇERİK:")
        print("-" * 40)
        print(result["final_content"])
        
        return result

def content_creation_scenario():
    """İçerik oluşturma senaryosunu test eder"""
    print("\n" + "=" * 60)
    print("SENARYO 2: İÇERİK OLUŞTURMA ASISTANI")
    print("=" * 60)
    
    # İçerik asistanını oluştur
    assistant = ContentCreationAssistant()
    
    # Test edilecek konular
    topics = [
        "Sürdürülebilir yaşam tarzı",
        "Uzaktan çalışmanın geleceği"
    ]
    
    # Her konu için içerik oluştur
    for topic in topics:
        print(f"\n{'='*20} {topic.upper()} {'='*20}")
        try:
            assistant.create_content(topic)
        except Exception as e:
            print(f"İçerik oluşturma hatası: {e}")
    
    return assistant

# =============================================================================
# SENARYO 3: EĞİTİM PLANLAMA ASISTANI  
# Kişiselleştirilmiş öğrenim planları oluşturan sistem
# =============================================================================

class EducationPlannerBot:
    """
    Eğitim planlama bot'u sınıfı
    Öğrenci profiline göre kişiselleştirilmiş öğrenim planı oluşturur
    """
    
    def __init__(self):
        """Bot'u başlatır ve kurs veri tabanını kurar"""
        # Mevcut kursların veri tabanı
        self.courses_db = {
            "python": {
                "duration": "8 hafta", 
                "level": "Başlangıç", 
                "topics": ["Değişkenler", "Fonksiyonlar", "OOP"]
            },
            "javascript": {
                "duration": "10 hafta", 
                "level": "Başlangıç", 
                "topics": ["DOM", "ES6", "React"]
            },
            "machine_learning": {
                "duration": "12 hafta", 
                "level": "İleri", 
                "topics": ["Algoritma", "Neural Networks", "Deep Learning"]
            },
            "data_science": {
                "duration": "16 hafta", 
                "level": "Orta", 
                "topics": ["Pandas", "Visualization", "Statistics"]
            }
        }
        # Planlama chain'lerini kur
        self.setup_chains()
    
    def get_course_info(self, course: str) -> str:
        """
        Kurs bilgilerini veri tabanından getirir
        Args: course - Kurs adı
        Returns: Kurs detayları
        """
        # Kurs adını veri tabanı formatına çevir
        course_key = course.lower().replace(" ", "_")
        if course_key in self.courses_db:
            info = self.courses_db[course_key]
            return f"Kurs: {course}, Süre: {info['duration']}, Seviye: {info['level']}, Konular: {', '.join(info['topics'])}"
        return f"'{course}' kursu bulunamadı."
    
    def setup_chains(self):
        """Eğitim planlama sürecinin chain'lerini kurar"""
        
        # AŞAMA 1: Öğrenci seviye değerlendirmesi
        self.assessment_prompt = PromptTemplate(
            input_variables=["student_background", "goals"],
            template="""
            Öğrenci Geçmişi: {student_background}
            Hedefler: {goals}
            
            Bu bilgilere dayanarak öğrencinin seviyesini değerlendirin ve uygun başlangıç noktasını önerin.
            - Mevcut seviye analizi
            - Güçlü ve zayıf yönler
            - Önerilen başlangıç seviyesi
            """
        )
        self.assessment_chain = LLMChain(
            llm=llm,
            prompt=self.assessment_prompt,
            output_key="assessment"
        )
        
        # AŞAMA 2: Kişiselleştirilmiş öğrenim planı oluşturma
        self.planning_prompt = PromptTemplate(
            input_variables=["student_background", "goals", "assessment"],
            template="""
            Geçmiş: {student_background}
            Hedefler: {goals}
            Değerlendirme: {assessment}
            
            Kişiselleştirilmiş 12 haftalık öğrenim planı oluşturun:
            - Haftalık konular ve hedefler
            - Pratik projeler ve uygulamalar
            - Değerlendirme kriterleri
            - Kaynak önerileri (kitap, video, kurs)
            """
        )
        self.planning_chain = LLMChain(
            llm=llm,
            prompt=self.planning_prompt,
            output_key="learning_plan"
        )
        
        # AŞAMA 3: Motivasyon ve takip stratejileri
        self.motivation_prompt = PromptTemplate(
            input_variables=["learning_plan"],
            template="""
            Öğrenim Planı: {learning_plan}
            
            Bu plan için motivasyon stratejileri ve ilerleme takip yöntemleri önerin:
            - Günlük öğrenim rutinleri
            - Haftalık milestone'lar (ara hedefler)
            - Ödül sistemi önerileri
            - Zorluk anlarında yapılacaklar
            - İlerleme ölçüm yöntemleri
            """
        )
        self.motivation_chain = LLMChain(
            llm=llm,
            prompt=self.motivation_prompt,
            output_key="motivation_plan"
        )
        
        # AŞAMA 4: Tüm chain'leri birleştir
        self.overall_chain = SequentialChain(
            chains=[self.assessment_chain, self.planning_chain, self.motivation_chain],
            input_variables=["student_background", "goals"],
            output_variables=["assessment", "learning_plan", "motivation_plan"]
        )
    
    def create_learning_plan(self, background: str, goals: str):
        """
        Öğrenci profili için kişisel öğrenim planı oluşturur
        Args: 
            background - Öğrenci geçmişi
            goals - Öğrenci hedefleri
        Returns: Kapsamlı öğrenim planı
        """
        # Tüm planlama sürecini çalıştır
        result = self.overall_chain({
            "student_background": background,
            "goals": goals
        })
        
        # Sonuçları formatla ve göster
        print("📊 SEVİYE DEĞERLENDİRMESİ:")
        print("-" * 40)
        print(result["assessment"])
        
        print("\n📚 KİŞİSEL ÖĞRENME PLANI:")
        print("-" * 40)  
        print(result["learning_plan"])
        
        print("\n💪 MOTİVASYON STRATEJİLERİ:")
        print("-" * 40)
        print(result["motivation_plan"])
        
        return result

def education_planning_scenario():
    """Eğitim planlama senaryosunu test eder"""
    print("\n" + "=" * 60)
    print("SENARYO 3: EĞİTİM PLANLAMA ASISTANI") 
    print("=" * 60)
    
    # Eğitim planlama bot'unu oluştur
    planner = EducationPlannerBot()
    
    # Test öğrenci profilleri
    students = [
        {
            "background": "Bilgisayar mühendisliği mezunu, 2 yıl web geliştirme deneyimi",
            "goals": "Veri bilimci olmak ve makine öğrenimi projelerinde çalışmak"
        },
        {
            "background": "İşletme mezunu, programlama deneyimi yok",
            "goals": "Mobil uygulama geliştirici olmak"
        }
    ]
    
    # Her öğrenci profili için plan oluştur
    for i, student in enumerate(students, 1):
        print(f"\n{'='*15} ÖĞRENCİ {i} {'='*15}")
        try:
            planner.create_learning_plan(
                student["background"], 
                student["goals"]
            )
        except Exception as e:
            print(f"Plan oluşturma hatası: {e}")
    
    return planner

# =============================================================================
# ANA FONKSİYON - Tüm senaryoları çalıştırır
# =============================================================================

if __name__ == "__main__":
    print("LANGCHAIN SENARYO BAZLI UYGULAMALAR")
    print("Gerçek hayat senaryoları ile LangChain kullanımı\n")
    
    try:
        # Tüm senaryoları sırayla çalıştır
        print("🤖 Müşteri hizmetleri bot'u test ediliyor...")
        customer_service_scenario()
        
        print("\n✍️ İçerik oluşturma asistanı test ediliyor...")
        content_creation_scenario()
        
        print("\n📚 Eğitim planlama asistanı test ediliyor...")
        education_planning_scenario()
        
        # Başarı mesajı
        print("\n" + "=" * 60)
        print("✅ TÜM SENARYOLAR BAŞARIYLA TAMAMLANDI!")
        print("Bu örnekleri kendi projelerinizde referans olarak kullanabilirsiniz.")
        print("Her senaryo farklı LangChain bileşenlerini göstermektedir:")
        print("- Agent'lar ve Tool'lar (Müşteri Hizmetleri)")
        print("- Sequential Chain'ler (İçerik Oluşturma)")
        print("- Prompt Engineering (Eğitim Planlama)")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Genel hata: {e}")
        print("🔧 Çözüm önerileri:")
        print("1. OpenAI API anahtarınızı .env dosyasında kontrol edin")
        print("2. İnternet bağlantınızı kontrol edin")
        print("3. Gerekli kütüphanelerin yüklü olduğundan emin olun")
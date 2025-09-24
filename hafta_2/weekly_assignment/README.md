# 🚀 BANÜ MIS CONNECT: Akıllı YBS Asistanı  
*Bandırma Onyedi Eylül Üniversitesi Yönetim Bilişim Sistemleri öğrencileri için tasarlanmış, yapay zeka destekli akıllı kampüs asistanı.*

---

## 📜 Proje Hakkında
**BANÜ MIS CONNECT**, Bandırma Onyedi Eylül Üniversitesi (BANÜ) Yönetim Bilişim Sistemleri (YBS) öğrencilerinin akademik, sosyal ve kişisel süreçlerini desteklemek amacıyla **LLM tabanlı fonksiyonlarla geliştirilmiş yapay zeka destekli etkileşim odaklı bir web platformudur**  

Proje, öğrencilerin kampüs hayatıyla ilgili dağınık bilgilere (ders programları, akademik takvim, etkinlikler, kampüs mekanları vb.) **tek bir arayüzden, doğal dil sorgularıyla anında erişimini** sağlamayı hedefler.  

> **Vizyon:** Her YBS öğrencisine kişisel bir kampüs asistanı sunarak bilgiye erişimi zahmetsizleştirmek, kampüs hayatını daha verimli, bağlantılı ve akıllı hale getirmek.

---

## 🎯 Çözülen Problem
- **Dağınık Bilgi Mimarisi:** Ders programları ÖBS’de, akademik takvim web sitesinde PDF olarak, etkinlikler ise sosyal medyada.  
- **Verimsiz Zaman Yönetimi:** Basit bir soruya yanıt bulmak için çoklu kaynaklara bakmak gerekiyor.  
- **Anlık Yanıt Eksikliği:** “UBF nerede?” veya “Final sınavları ne zaman?” gibi sorulara hızlı çözüm mevcut değil.  

👉 BANÜ MIS CONNECT, bu parçalı yapıyı **tek bir merkezde birleştirerek** öğrencilere güvenilir bir bilgi ve verimlilik platformu sunar.

---

## ✨ Temel Yetenekler ve Fonksiyonlar

### 🧠 Akıllı Bilgi Erişimi ve Kampüs Entegrasyonu
- **Dinamik Ders Programı:** `get_course_schedule`  
- **Akademik Takvim:** `get_academic_calendar`  
- **Kampüs Navigasyonu:** Fakülte, yemekhane, kütüphane konumları  
- **Gerçek Zamanlı Bilgiler:** Hava durumu (`get_weather_info`), yemekhane menüsü (`get_cafeteria_menu`)
- **Etkinlikler:** Sosyal etkinlikler, üniversite toplulukları, konser, seminer, atölye bilgileri...

---

## 🎨 "UI Hero" Konsept Arayüzü
- **Glassmorphism Tasarımı:** Gradyan arka plan + buzlu cam efektli paneller  
- **Dinamik Yan Panel:** İçeriğe göre harita, takvim veya kart görselleri  
- **Kişiselleştirme:** Özel harf avatarları (Ö/B) ile daha okunabilir sohbet arayüzü  

---

## 🛠️ Kullanılan Teknolojiler

| Kategori         | Teknoloji                |
|------------------|--------------------------|
| **Dil**          | Python 3.10+             |
| **Web Framework**| Flask                    |
| **Yapay Zeka**   | OpenAI API (gpt-4-turbo) |
| **API İstemcisi**| openai Python Kütüphanesi|
| **Konfigürasyon**| python-dotenv            |
| **Arayüz**       | HTML5, CSS3, JavaScript  |

---

## 🏗️ Teknik Mimari

```mermaid
graph TD
    subgraph "Tarayıcı (Kullanıcı Arayüzü)"
        A["HTML/CSS/JS Arayüzü"]
    end

    subgraph "Sunucu (app.py)"
        B["Flask Sunucusu (API Endpoint: /chat)"]
        C{"Chatbot Orkestratörü\n(CampusAssistant Sınıfı)"}
        D["OpenAI API İstemcisi"]
        E["Alet Çantası (Tools/Functions)"]
        F["Veri Kaynağı (Python Dictionaries)"]
    end

    A -- "POST /chat isteği (Kullanıcı Mesajı)" --> B
    B --> C
    C -- "Mesajı ve fonksiyonları LLM'e gönder" --> D
    D -- "Yanıt: 'Bu fonksiyonu çağır' -> get_course_schedule" --> C
    C -- "İlgili aracı çalıştır" --> E
    E -- "Veri kullan" --> F
    E -- "Fonksiyon sonucunu döndür (JSON)" --> C
    C -- "Fonksiyon sonucunu nihai cevap için tekrar LLM'e gönder" --> D
    D -- "Yanıt: 'İşte ders programın...' (Doğal Dil Metni)" --> C
    C -- "JSON yanıtı oluştur" --> B
    B -- "Arayüzü dinamik olarak güncelle" --> A

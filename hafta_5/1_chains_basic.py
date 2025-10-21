"""
Hafta 5 - Bölüm 1: Temel Chain Yapıları
LangChain ile Chain oluşturma ve kullanma
"""

import os
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.schema import BaseOutputParser

# OpenAI API anahtarını yükle
from dotenv import load_dotenv
load_dotenv()

# LLM'i başlat - temperature 0.7 ile orta seviye yaratıcılık
llm = OpenAI(temperature=0.7)

def basic_chain_example():
    """
    Temel LLMChain kullanımı
    Bu fonksiyon en basit chain yapısını gösterir:
    - Bir prompt template oluşturma
    - LLM ile bağlama
    - Tek adımda sonuç alma
    """
    print("=" * 50)
    print("1. TEMEL CHAIN KULLANIMI")
    print("=" * 50)
    
    # Prompt template oluştur - {topic} değişkeni kullanıcı girdisi alacak
    prompt = PromptTemplate(
        input_variables=["topic"],  # Girdi değişkenlerini tanımla
        template="Bu konu hakkında 3 cümlelik bir açıklama yaz: {topic}"  # Şablon metni
    )
    
    # Chain oluştur - LLM ve prompt'u birleştir
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Chain'i çalıştır - "Yapay zeka" konusu ile
    result = chain.run("Yapay zeka")
    print(f"Sonuç: {result}")
    return result

def sequential_chain_example():
    """
    Sıralı chain kullanımı - SimpleSequentialChain
    Bu fonksiyon iki chain'i sırayla çalıştırır:
    1. İlk chain: Hikaye başlangıcı oluşturur
    2. İkinci chain: Bu başlangıcı alıp hikayeyi tamamlar
    
    SimpleSequentialChain: Her chain'in çıktısı bir sonrakinin girdisi olur
    """
    print("\n" + "=" * 50)
    print("2. SİRALI CHAIN KULLANIMI")
    print("=" * 50)
    
    # İlk chain: Hikaye başlangıcı oluştur
    first_prompt = PromptTemplate(
        input_variables=["theme"],  # Tema girdi olarak alınacak
        template="Bu tema ile başlayan kısa bir hikaye başlangıcı yaz: {theme}"
    )
    first_chain = LLMChain(llm=llm, prompt=first_prompt)
    
    # İkinci chain: Hikaye sonunu ekle
    # Bu chain, ilk chain'in çıktısını girdi olarak alacak
    second_prompt = PromptTemplate(
        input_variables=["story_beginning"],  # İlk chain'in çıktısı buraya gelir
        template="Bu hikaye başlangıcını heyecanlı bir sonla tamamla:\n{story_beginning}"
    )
    second_chain = LLMChain(llm=llm, prompt=second_prompt)
    
    # Chainleri sırayla birleştir - çıktı otomatik olarak bir sonrakinin girdisi olur
    overall_chain = SimpleSequentialChain(
        chains=[first_chain, second_chain]  # Sıra önemli: önce first, sonra second
    )
    
    # Çalıştır - sadece ilk chain'in girdisini ver, gerisini otomatik halleder
    result = overall_chain.run("uzay macerası")
    print(f"Tamamlanmış hikaye:\n{result}")
    return result

def complex_sequential_chain_example():
    """
    Karmaşık sıralı chain - SequentialChain
    Bu fonksiyon üç farklı chain'i sırayla çalıştırır:
    1. Ürün analizi yapar
    2. Bu analize dayanarak pazarlama stratejisi geliştirir  
    3. Strateji için bütçe planı oluşturur
    
    SequentialChain: Çoklu girdi/çıktı destekler ve değişken adları ile kontrol sağlar
    """
    print("\n" + "=" * 50)
    print("3. KARMAŞIK SİRALI CHAIN")
    print("=" * 50)
    
    # İlk chain: Ürün özelliklerini ve hedef kitlesini analiz et
    product_analysis_prompt = PromptTemplate(
        input_variables=["product_name"],  # Ürün adı girdi olarak alınır
        template="Bu ürünün özelliklerini ve hedef kitlesini analiz et: {product_name}"
    )
    product_analysis_chain = LLMChain(
        llm=llm, 
        prompt=product_analysis_prompt,
        output_key="analysis"  # Bu chain'in çıktısı "analysis" adıyla saklanır
    )
    
    # İkinci chain: Analiz sonucuna göre pazarlama stratejisi geliştir
    marketing_prompt = PromptTemplate(
        # Bu chain hem ürün adını hem de analiz sonucunu kullanır
        input_variables=["product_name", "analysis"],
        template=""":
        Ürün: {product_name}
        Analiz: {analysis}
        
        Bu analiz temelinde 3 pazarlama stratejisi öner:
        """
    )
    marketing_chain = LLMChain(
        llm=llm,
        prompt=marketing_prompt,
        output_key="marketing_strategy"  # Çıktı "marketing_strategy" olarak saklanır
    )
    
    # Üçüncü chain: Pazarlama stratejisi için bütçe planı oluştur
    budget_prompt = PromptTemplate(
        # Ürün adı ve pazarlama stratejisini girdi olarak kullanır
        input_variables=["product_name", "marketing_strategy"],
        template=""":
        Ürün: {product_name}
        Pazarlama Stratejisi: {marketing_strategy}
        
        Bu stratejiler için aylık bütçe dağılımı öner:
        """
    )
    budget_chain = LLMChain(
        llm=llm,
        prompt=budget_prompt,
        output_key="budget_plan"  # Son çıktı "budget_plan" olarak saklanır
    )
    
    # Tüm chainleri sırayla birleştir - karmaşık girdi/çıktı yönetimi
    overall_chain = SequentialChain(
        chains=[product_analysis_chain, marketing_chain, budget_chain],
        input_variables=["product_name"],  # Başlangıç girdisi sadece ürün adı
        output_variables=["analysis", "marketing_strategy", "budget_plan"]  # Tüm çıktılar saklanır
    )
    
    # Çalıştır - dictionary formatında girdi ver, tüm çıktıları dictionary olarak al
    result = overall_chain({"product_name": "Akıllı saat"})
    
    # Sonuçları düzenli şekilde göster
    print("ÜRÜN ANALİZİ:")
    print(result["analysis"])
    print("\nPAZARLAMA STRATEJİSİ:")
    print(result["marketing_strategy"])
    print("\nBÜTÇE PLANI:")
    print(result["budget_plan"])
    
    return result

class JsonOutputParser(BaseOutputParser):
    """
    Özel output parser sınıfı
    LLM çıktısını istediğimiz formata dönüştürmek için kullanılır
    Bu örnek, düz metni basit bir dictionary formatına çevirir
    """
    
    def parse(self, text: str):
        """
        Metni parse etme fonksiyonu
        LLM'den gelen metni alır ve dictionary formatına çevirir
        """
        try:
            import json
            # Basit JSON parse denemesi - her satırı key:value çifti olarak işle
            lines = text.strip().split('\n')  # Metni satırlara böl
            result = {}
            for line in lines:
                if ':' in line:  # Eğer satırda ':' varsa key:value çifti olarak kabul et
                    key, value = line.split(':', 1)  # İlk ':' den böl
                    result[key.strip()] = value.strip()  # Boşlukları temizle ve dictionary'e ekle
            return result
        except:
            # Parse başarısız olursa ham metni döndür
            return {"raw_output": text}

def custom_output_parser_example():
    """
    Özel output parser kullanımı
    Bu fonksiyon LLM çıktısını özel bir parser ile işler:
    - LLM'den yapılandırılmış veri ister
    - Parser bu veriyi dictionary formatına çevirir
    - Sonucu daha düzenli şekilde gösterir
    """
    print("\n" + "=" * 50)
    print("4. ÖZEL OUTPUT PARSER")
    print("=" * 50)
    
    # Yapılandırılmış çıktı için özel prompt template
    prompt = PromptTemplate(
        input_variables=["city"],
        template=""":
        Bu şehir hakkında bilgileri şu formatta ver:
        Nüfus: [nüfus bilgisi]
        İklim: [iklim bilgisi]
        Meşhur yemek: [yemek bilgisi]
        
        Şehir: {city}
        """
    )
    
    # Özel parser'ı oluştur ve chain ile birleştir
    parser = JsonOutputParser()
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        output_parser=parser  # LLM çıktısı otomatik olarak parser'dan geçer
    )
    
    # Chain'i çalıştır - çıktı otomatik olarak parse edilir
    result = chain.run("İstanbul")
    print("Parsed result:")
    # Dictionary formatındaki sonucu düzenli şekilde göster
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    return result

if __name__ == "__main__":
    print("LANGCHAIN CHAIN ÖRNEKLERİ")
    print("Bu örneklerde farklı chain türlerini öğreneceksiniz.\n")
    
    # Örnekleri sırasıyla çalıştır - basit'ten karmaşığa doğru
    basic_chain_example()              # 1. Temel chain kullanımı
    sequential_chain_example()         # 2. İki chain'i sıralama
    complex_sequential_chain_example() # 3. Çoklu chain ve değişken yönetimi
    custom_output_parser_example()     # 4. Çıktı formatını özelleştirme
    
    print("\n" + "=" * 50)
    print("TÜM ÖRNEKLER TAMAMLANDI!")
    print("Chain'ler ile daha karmaşık iş akışları oluşturabilirsiniz.")
    print("=" * 50)
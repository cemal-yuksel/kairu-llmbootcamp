"""
OpenAI Function Calling Örnekleri - Chatbot Sınıfı
Function calling ile AI'ın dış araçları kullanmasını sağlama
"""

import os
import json
import math
import re
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# .env dosyasından API anahtarını yükle
load_dotenv()
# OpenAI istemcisini başlat
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class FunctionCallingChatbot:
    def __init__(self):
        # 1. Konuşma geçmişini başlat (system mesajı ile)
        self.conversation_history = [
            {
                "role": "system",
                "content": """Sen çok yetenekli bir AI asistanısın. Kullanıcılara yardım etmek için 
                çeşitli matematiksel, bilgi ve hava durumu araçları kullanabilirsin. 
                Her zaman dostça, yardımcı ve profesyonel ol."""
            }
        ]
        
        # 2. Kullanılabilir fonksiyonları bir sözlükte tanımla
        self.available_functions = {
            "calculate_area": self.calculate_area,
            "get_weather": self.get_weather,
            "convert_currency": self.convert_currency,
            "get_current_time": self.get_current_time,
            "validate_email": self.validate_email
        }

    def calculate_area(self, shape, **kwargs):
        """Geometrik şekillerin alanını hesaplar"""
        # 1. Şekil türüne göre parametreleri kontrol et ve alanı hesapla
        try:
            if shape == "rectangle":
                width = kwargs.get("width")
                height = kwargs.get("height")
                if width and height:
                    result = width * height
                    return {
                        "shape": shape,
                        "width": width,
                        "height": height,
                        "area": result,
                        "unit": "square units",
                        "status": "success"
                    }
            elif shape == "circle":
                radius = kwargs.get("radius")
                if radius:
                    result = math.pi * (radius ** 2)
                    return {
                        "shape": shape,
                        "radius": radius,
                        "area": round(result, 2),
                        "unit": "square units",
                        "status": "success"
                    }
            elif shape == "triangle":
                base = kwargs.get("base")
                height = kwargs.get("height")
                if base and height:
                    result = 0.5 * base * height
                    return {
                        "shape": shape,
                        "base": base,
                        "height": height,
                        "area": result,
                        "unit": "square units",
                        "status": "success"
                    }
            # 2. Geçersiz parametrelerde hata döndür
            return {"status": "error", "message": "Geçersiz parametreler"}
        except Exception as e:
            # 3. Hesaplama hatası durumunda hata mesajı döndür
            return {"status": "error", "message": f"Hesaplama hatası: {str(e)}"}

    def get_weather(self, city):
        """Hava durumu bilgisi alır (demo)"""
        # 1. Demo şehir ve hava durumu verilerini tanımla
        weather_data = {
            "istanbul": {
                "temperature": 22, 
                "condition": "Parçalı bulutlu", 
                "humidity": 65,
                "wind_speed": "15 km/h",
                "feels_like": 24
            },
            "ankara": {
                "temperature": 18, 
                "condition": "Güneşli", 
                "humidity": 45,
                "wind_speed": "10 km/h",
                "feels_like": 19
            },
            "izmir": {
                "temperature": 25, 
                "condition": "Açık", 
                "humidity": 70,
                "wind_speed": "20 km/h",
                "feels_like": 27
            },
            "bursa": {
                "temperature": 20, 
                "condition": "Yağmurlu", 
                "humidity": 80,
                "wind_speed": "12 km/h",
                "feels_like": 18
            }
        }
        # 2. Şehir adını küçük harfe çevir ve Türkçe karakter düzelt
        city_lower = city.lower().replace("ı", "i")
        # 3. Şehir veri tabanında varsa bilgileri döndür, yoksa hata mesajı döndür
        if city_lower in weather_data:
            data = weather_data[city_lower]
            return {
                "city": city,
                "current_weather": data,
                "status": "success",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        else:
            return {
                "status": "error", 
                "message": f"{city} için hava durumu bulunamadı",
                "available_cities": list(weather_data.keys())
            }

    def convert_currency(self, amount, from_currency, to_currency):
        """Para birimi dönüştürür (demo kurlar)"""
        # 1. Demo döviz kurlarını tanımla
        rates = {
            "USD": {"TRY": 27.5, "EUR": 0.92, "GBP": 0.79, "JPY": 149.50},
            "TRY": {"USD": 0.036, "EUR": 0.033, "GBP": 0.029, "JPY": 5.42},
            "EUR": {"USD": 1.08, "TRY": 30.0, "GBP": 0.86, "JPY": 161.20},
            "GBP": {"USD": 1.26, "TRY": 34.5, "EUR": 1.16, "JPY": 187.80}
        }
        # 2. Para birimlerini büyük harfe çevir
        from_currency = from_currency.upper()
        to_currency = to_currency.upper()
        # 3. Kur varsa dönüşümü yap, yoksa hata döndür
        if from_currency in rates and to_currency in rates[from_currency]:
            rate = rates[from_currency][to_currency]
            converted_amount = round(amount * rate, 2)
            return {
                "original_amount": amount,
                "from_currency": from_currency,
                "to_currency": to_currency,
                "converted_amount": converted_amount,
                "exchange_rate": rate,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "success",
                "note": "Demo kurlar kullanılmıştır"
            }
        else:
            return {
                "status": "error", 
                "message": "Desteklenmeyen döviz çifti",
                "supported_currencies": list(rates.keys())
            }

    def get_current_time(self):
        """Şu anki zamanı döndürür"""
        # 1. Şu anki zamanı al ve formatla
        now = datetime.now()
        return {
            "current_time": now.strftime("%H:%M:%S"),
            "current_date": now.strftime("%Y-%m-%d"),
            "day_of_week": now.strftime("%A"),
            "month": now.strftime("%B"),
            "year": now.year,
            "timestamp": now.timestamp(),
            "timezone": "Local",
            "status": "success"
        }

    def validate_email(self, email):
        """E-posta formatını doğrular"""
        # 1. E-posta regex ile kontrol et
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        is_valid = bool(re.match(pattern, email))
        # 2. Sonucu ve kontrol zamanını döndür
        return {
            "email": email,
            "is_valid": is_valid,
            "message": "Geçerli e-posta formatı" if is_valid else "Geçersiz e-posta formatı",
            "checked_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "success"
        }

    def get_function_definitions(self):
        """OpenAI için fonksiyon tanımlarını döndürür"""
        # 1. Her fonksiyon için isim, açıklama ve parametre şemasını döndür
        return [
            {
                "name": "calculate_area",
                "description": "Geometrik şekillerin alanını hesaplar",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "shape": {
                            "type": "string",
                            "enum": ["rectangle", "circle", "triangle"],
                            "description": "Şeklin türü"
                        },
                        "width": {
                            "type": "number",
                            "description": "Dikdörtgen genişliği"
                        },
                        "height": {
                            "type": "number", 
                            "description": "Dikdörtgen veya üçgen yüksekliği"
                        },
                        "radius": {
                            "type": "number",
                            "description": "Daire yarıçapı"
                        },
                        "base": {
                            "type": "number",
                            "description": "Üçgen tabanı"
                        }
                    },
                    "required": ["shape"]
                }
            },
            {
                "name": "get_weather",
                "description": "Belirtilen şehir için hava durumu bilgisi alır",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "Şehir adı"
                        }
                    },
                    "required": ["city"]
                }
            },
            {
                "name": "convert_currency",
                "description": "Para birimi dönüştürür",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "amount": {
                            "type": "number",
                            "description": "Dönüştürülecek miktar"
                        },
                        "from_currency": {
                            "type": "string",
                            "description": "Kaynak para birimi"
                        },
                        "to_currency": {
                            "type": "string", 
                            "description": "Hedef para birimi"
                        }
                    },
                    "required": ["amount", "from_currency", "to_currency"]
                }
            },
            {
                "name": "get_current_time",
                "description": "Şu anki zaman ve tarih bilgisini verir",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "validate_email",
                "description": "E-posta adresinin formatını doğrular",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "email": {
                            "type": "string",
                            "description": "Doğrulanacak e-posta adresi"
                        }
                    },
                    "required": ["email"]
                }
            }
        ]

    def chat(self, user_message):
        """Ana chatbot fonksiyonu"""
        # 1. Kullanıcı mesajını konuşma geçmişine ekle
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        try:
            # 2. OpenAI API'sine fonksiyon tanımları ile isteği gönder
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=self.conversation_history,
                functions=self.get_function_definitions(),
                function_call="auto",
                temperature=0.7
            )
            response_message = response.choices[0].message
            # 3. Yanıtta function_call var mı kontrol et
            if response_message.function_call:
                function_name = response_message.function_call.name
                function_args = json.loads(response_message.function_call.arguments)
                print(f"🔧 {function_name} fonksiyonu çağırılıyor...")
                # 4. Fonksiyon mevcutsa çağır ve sonucu al
                if function_name in self.available_functions:
                    function_result = self.available_functions[function_name](**function_args)
                    # 5. AI'nın function_call isteğini konuşma geçmişine ekle
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": None,
                        "function_call": response_message.function_call
                    })
                    # 6. Fonksiyon sonucunu function rolüyle ekle
                    self.conversation_history.append({
                        "role": "function",
                        "name": function_name,
                        "content": json.dumps(function_result, ensure_ascii=False)
                    })
                    # 7. Son olarak, fonksiyon sonucu ile final yanıtı al
                    final_response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=self.conversation_history,
                        temperature=0.7
                    )
                    final_message = final_response.choices[0].message.content
                    # 8. Final yanıtı konuşma geçmişine ekle
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": final_message
                    })
                    return final_message
                else:
                    # 9. Fonksiyon bulunamazsa hata döndür
                    return f"Üzgünüm, {function_name} fonksiyonu mevcut değil."
            else:
                # 10. Function call yoksa normal yanıtı döndür
                assistant_message = response_message.content
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_message
                })
                return assistant_message
        except Exception as e:
            # 11. Hata durumunda hata mesajı döndür
            return f"Üzgünüm, bir hata oluştu: {str(e)}"

    def get_conversation_summary(self):
        """Konuşma özetini döndürür"""
        # 1. Fonksiyon çağrısı sayısını say
        function_calls = 0
        for msg in self.conversation_history:
            if msg.get("function_call"):
                function_calls += 1
        # 2. Toplam mesaj ve fonksiyon çağrısı sayısını döndür
        return {
            "total_messages": len(self.conversation_history),
            "function_calls_made": function_calls,
            "available_functions": list(self.available_functions.keys())
        }

def main():
    """Ana demo fonksiyonu"""
    # 1. Kullanıcıya chatbot fonksiyonlarını tanıt
    print("🤖 Function Calling Chatbot'a Hoş Geldiniz!")
    print("Şu fonksiyonları kullanabilirim:")
    print("• Alan hesaplama (dikdörtgen, daire, üçgen)")
    print("• Hava durumu sorgulama")
    print("• Döviz dönüştürme")
    print("• Zaman bilgisi")
    print("• E-posta doğrulama\n")
    print("Çıkmak için 'quit' yazın.\n")
    # 2. Chatbot nesnesini oluştur
    chatbot = FunctionCallingChatbot()
    # 3. Kullanıcıdan giriş al ve yanıt döngüsünü başlat
    while True:
        try:
            user_input = input("Sen: ").strip()
            if user_input.lower() in ['quit', 'exit', 'çık']:
                # 4. Çıkışta konuşma özetini göster
                summary = chatbot.get_conversation_summary()
                print(f"\n📊 Konuşma Özeti:")
                print(f"Toplam mesaj: {summary['total_messages']}")
                print(f"Fonksiyon çağırımları: {summary['function_calls_made']}")
                print("Görüşmek üzere! 👋")
                break
            if not user_input:
                continue
            # 5. Chatbot yanıtı al ve ekrana yazdır
            print("\nBot: ", end="", flush=True)
            response = chatbot.chat(user_input)
            print(response)
            print()
        except KeyboardInterrupt:
            print("\n\nGörüşmek üzere! 👋")
            break
        except Exception as e:
            print(f"\nHata: {e}\n")

if __name__ == "__main__":
    # Demo kullanım
    print("=== FUNCTION CALLING CHATBOT ===\n")
    # 1. Demo chatbot nesnesi oluştur
    demo_bot = FunctionCallingChatbot()
    # 2. Demo soruları sırayla gönder ve yanıtları yazdır
    demo_questions = [
        "Merhaba! Nasılsın?",
        "Yarıçapı 10 olan bir dairenin alanını hesapla",
        "İstanbul'un hava durumu nasıl?",
        "100 USD kaç TL eder?", 
        "Saat kaç?",
        "test@example.com geçerli bir e-posta mı?",
        "5x8 dikdörtgenin alanı nedir?"
    ]
    for question in demo_questions:
        print(f"🗣️ Kullanıcı: {question}")
        response = demo_bot.chat(question)
        print(f"🤖 Bot: {response}\n")
        print("-" * 70)
    print("\n🎯 İnteraktif moda geçmek için main() fonksiyonunu çalıştırın!")
    # İnteraktif mod için uncomment edin:
    # main()
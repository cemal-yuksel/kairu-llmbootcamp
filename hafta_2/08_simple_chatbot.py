"""
Basit Function Calling Chatbot
Temel fonksiyon çağırımı ile kolay anlaşılır chatbot örneği
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# .env dosyasından API anahtarını yükle
load_dotenv()
# OpenAI istemcisini başlat
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class SimpleChatbot:
    def __init__(self):
        # 1. Sistem mesajı ile mesaj geçmişini başlat
        self.messages = [
            {
                "role": "system",
                "content": "Sen yardımcı bir AI asistanısın. Hesap makinesi ve not alma fonksiyonlarını kullanabilirsin."
            }
        ]
        # 2. Notlar için boş bir liste oluştur
        self.notes = []
        # 3. Kullanılabilir fonksiyonları bir sözlükte tanımla
        self.available_functions = {
            "calculator": self.calculator,
            "save_note": self.save_note,
            "list_notes": self.list_notes
        }
    
    def calculator(self, operation, num1, num2):
        """Basit hesap makinesi"""
        # 1. İşlem türüne göre matematiksel işlemi uygula
        try:
            if operation == "toplama":
                result = num1 + num2
            elif operation == "çıkarma":
                result = num1 - num2
            elif operation == "çarpma":
                result = num1 * num2
            elif operation == "bölme":
                if num2 == 0:
                    return {"error": "Sıfıra bölme hatası"}
                result = num1 / num2
            else:
                return {"error": "Geçersiz işlem"}
            # 2. Sonucu ve parametreleri döndür
            return {
                "operation": operation,
                "num1": num1,
                "num2": num2,
                "result": result
            }
        except Exception as e:
            # 3. Hata durumunda hata mesajı döndür
            return {"error": str(e)}
    
    def save_note(self, title, content):
        """Not kaydetme"""
        # 1. Not nesnesini oluştur ve listeye ekle
        note = {
            "id": len(self.notes) + 1,
            "title": title,
            "content": content,
            "timestamp": "şimdi"
        }
        self.notes.append(note)
        # 2. Sonucu döndür
        return {
            "message": f"Not kaydedildi: '{title}'",
            "note_id": note["id"],
            "total_notes": len(self.notes)
        }
    
    def list_notes(self):
        """Notları listeler"""
        # 1. Not yoksa bilgilendirici mesaj döndür
        if not self.notes:
            return {
                "message": "Henüz kaydedilmiş not bulunmuyor.",
                "notes": [],
                "total_count": 0
            }
        # 2. Notlar varsa listeyi ve sayıyı döndür
        return {
            "message": f"Toplam {len(self.notes)} not bulundu:",
            "notes": self.notes,
            "total_count": len(self.notes)
        }
    
    def get_functions(self):
        """Fonksiyon tanımları"""
        # 1. OpenAI'ya bildirilecek fonksiyon şemalarını döndür
        return [
            {
                "name": "calculator",
                "description": "Temel matematik işlemleri yapar",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["toplama", "çıkarma", "çarpma", "bölme"],
                            "description": "Yapılacak işlem"
                        },
                        "num1": {
                            "type": "number",
                            "description": "İlk sayı"
                        },
                        "num2": {
                            "type": "number", 
                            "description": "İkinci sayı"
                        }
                    },
                    "required": ["operation", "num1", "num2"]
                }
            },
            {
                "name": "save_note",
                "description": "Not kaydeder",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Not başlığı"
                        },
                        "content": {
                            "type": "string",
                            "description": "Not içeriği"
                        }
                    },
                    "required": ["title", "content"]
                }
            },
            {
                "name": "list_notes",
                "description": "Kayıtlı tüm notları listeler",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]
    
    def chat(self, user_message):
        """Chat fonksiyonu"""
        # 1. Kullanıcı mesajını mesaj geçmişine ekle
        self.messages.append({"role": "user", "content": user_message})
        try:
            # 2. OpenAI API çağrısı ile yanıt al (fonksiyon tanımları ile)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=self.messages,
                functions=self.get_functions(),
                function_call="auto"
            )
            message = response.choices[0].message
            # 3. Yanıtta function_call var mı kontrol et
            if message.function_call:
                function_name = message.function_call.name
                function_args = json.loads(message.function_call.arguments)
                # 4. Fonksiyon mevcutsa çağır, yoksa hata döndür
                if function_name in self.available_functions:
                    result = self.available_functions[function_name](**function_args)
                else:
                    result = {"error": "Bilinmeyen fonksiyon"}
                # 5. Fonksiyon çağrısını ve sonucunu mesaj geçmişine ekle
                self.messages.append({
                    "role": "assistant",
                    "content": None,
                    "function_call": message.function_call
                })
                self.messages.append({
                    "role": "function",
                    "name": function_name,
                    "content": json.dumps(result, ensure_ascii=False)
                })
                # 6. Fonksiyon sonucuna göre final yanıtı tekrar OpenAI'dan al
                final_response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=self.messages
                )
                final_answer = final_response.choices[0].message.content
                self.messages.append({"role": "assistant", "content": final_answer})
                return final_answer
            else:
                # 7. Function call yoksa normal yanıtı döndür
                answer = message.content
                self.messages.append({"role": "assistant", "content": answer})
                return answer
        except Exception as e:
            # 8. Hata durumunda hata mesajı döndür
            return f"Hata: {str(e)}"

def demo():
    """Demo kullanım"""
    # 1. Kullanıcıya demo özelliklerini tanıt
    print("🤖 Basit Chatbot Demo")
    print("Hesap makinesi ve not alma özelliklerim var!\n")
    # 2. Chatbot nesnesi oluştur
    bot = SimpleChatbot()
    # 3. Test mesajlarını sırayla gönder ve yanıtları yazdır
    test_messages = [
        "Merhaba!",
        "25 ile 17'yi topla",
        "120'yi 8'e böl", 
        "Bugün market listesi: süt, ekmek, yumurta - bu notu 'market' başlığıyla kaydet",
        "45 çarpı 3 kaç eder?",
        "Proje toplantısı: Yarın saat 14:00'da ofiste - bunu 'toplantı' başlığıyla kaydet",
        "Notlarımı göster",
        "Kaç notum var?"
    ]
    for msg in test_messages:
        print(f"👤 Sen: {msg}")
        response = bot.chat(msg)
        print(f"🤖 Bot: {response}\n")
        print("-" * 50)

if __name__ == "__main__":
    demo()
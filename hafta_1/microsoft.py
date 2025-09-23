import os  # İşletim sistemi ile ilgili işlemler için os modülünü içe aktarır.
from dotenv import load_dotenv  # Ortam değişkenlerini .env dosyasından yüklemek için kullanılır.
from transformers import AutoTokenizer, pipeline  # Hugging Face Transformers kütüphanesinden gerekli fonksiyonlar.

# Hugging Face erişim token'ınızı buradan alabilirsiniz: https://huggingface.co/settings/tokens

load_dotenv()  # .env dosyasındaki ortam değişkenlerini yükler.
HF_TOKEN = os.getenv("HF_TOKEN")  # .env dosyasından HF_TOKEN değerini alır.

MODEL_ID = "microsoft/DialoGPT-medium"  # Kullanılacak olan önceden eğitilmiş modelin adı.

# Belirtilen model için tokenizer nesnesini oluşturur ve erişim token'ını kullanır.
tok = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)

# Metin üretimi için bir pipeline (iş hattı) oluşturur.
gen = pipeline(
    "text-generation",        # Görev türü: metin üretimi
    model=MODEL_ID,           # Kullanılacak modelin kimliği
    tokenizer=tok,            # Yukarıda oluşturulan tokenizer nesnesi
    device_map="auto",        # Donanım (CPU/GPU) otomatik olarak seçilir
    torch_dtype="auto",       # PyTorch veri tipi otomatik seçilir (ör: float16)
    max_new_tokens=50,        # Üretilecek maksimum yeni token sayısı
    do_sample=True,           # Örnekleme yapılacak mı? (çeşitlilik için True)
    temperature=0.7,          # Çıktının çeşitliliğini belirleyen sıcaklık parametresi
    token=HF_TOKEN,           # Hugging Face erişim token'ı (kritik: pipeline'a da iletilmeli)
)

# Modele gönderilecek olan istem (prompt) metni.
prompt = "Human: Hello! Bot: Hi! How can I help you? Human: Tell me about language models. Bot:"

# Pipeline aracılığıyla modele prompt gönderilir ve yanıt alınır, ardından ekrana yazdırılır.
print(gen(prompt)[0]["generated_text"])

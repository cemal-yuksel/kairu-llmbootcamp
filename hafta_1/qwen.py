# Hugging Face Transformers kütüphanesinden AutoTokenizer ve pipeline fonksiyonlarını içe aktarır.
from transformers import AutoTokenizer, pipeline

# Kullanılacak olan önceden eğitilmiş modelin adını belirtir.
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# Belirtilen model için uygun olan tokenizer'ı indirir ve yükler.
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Metin üretimi (text-generation) için bir pipeline (iş hattı) oluşturur.
generator = pipeline(
    "text-generation",           # Kullanılacak görev türü: metin üretimi
    model=MODEL_ID,              # Kullanılacak modelin kimliği
    tokenizer=tokenizer,         # Yukarıda oluşturulan tokenizer nesnesi
    device_map="auto",           # Donanım (CPU/GPU) otomatik olarak seçilir
    max_new_tokens=250,          # Üretilecek maksimum yeni token (kelime/parça) sayısı
    temperature=0.7              # Çıktının çeşitliliğini belirleyen sıcaklık parametresi
)

# Modelin cevaplaması için kullanılacak olan istem (prompt) metni.
prompt = "Tüketici aktivizmi nedir?"

# Pipeline aracılığıyla modele prompt gönderilir ve yanıt alınır.
response = generator(prompt)

# Modelin ürettiği metni ekrana yazdırır.
print(response[0]["generated_text"])
"""
AutoTokenizer & AutoModel Yapısı + Pipeline ile Hızlı Model Çağırma

Bu modül AutoTokenizer ve AutoModel sınıflarının kullanımını ve 
pipeline ile hızlı model çağırma yöntemlerini gösterir.
Her adımda detaylı Türkçe açıklamalar eklenmiştir.
"""

import sys
import os

class SuppressStderr:
    def __enter__(self):
        self.stderr = sys.stderr
        self.devnull = open(os.devnull, 'w')
        sys.stderr = self.devnull
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr = self.stderr
        self.devnull.close()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # TensorFlow için uyarı bastırma (varsa)
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
import time

# Eğer GPU kullanmak istemiyorsanız, torch'u CPU'ya zorlayın:
device = torch.device("cpu")

def tokenizer_example(): 
    """
    AutoTokenizer kullanım örneği
    Bu fonksiyonda bir metnin nasıl token'lara ayrıldığı, 
    bu token'ların sayısal id'lere (input_ids) dönüştürülmesi,
    attention mask üretimi ve tekrar metne çevrilmesi adım adım gösterilir.
    """
    # 1. Model adı belirlenir ve tokenizer yüklenir
    print("=== AutoTokenizer Örneği ===")
    # Kullanmak istediğimiz modelin adını belirliyoruz.
    model_name = "bert-base-uncased"
    # Modelin tokenizer'ını indiriyoruz. Bu tokenizer, metni modele uygun şekilde işler.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 2. Tokenize edilecek örnek metin
    text = "Hello, this is a sample text for tokenization."
    
    # 3. Metni token'lara ayırıyoruz (kelime ve alt-kelimelere bölme işlemi)
    tokens = tokenizer.tokenize(text)
    print(f"Tokens: {tokens}")

    # 4. Token'ları sayısal id'lere (input_ids) dönüştürüyoruz
    # encode fonksiyonu, metni modele uygun sayısal vektöre çevirir
    encoded = tokenizer.encode(text, return_tensors="pt")
    print(f"Encoded: {encoded}")
    
    # 5. Tokenizer ile input_ids ve attention mask birlikte alınır
    # Attention mask, modelin hangi tokenlara dikkat etmesi gerektiğini belirtir (1: dikkate al, 0: padding)
    encoded_dict = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    print(f"Input IDs: {encoded_dict['input_ids']}")
    print(f"Attention Mask: {encoded_dict['attention_mask']}")
    
    # 6. Sayısal id'leri tekrar metne çeviriyoruz (decode işlemi)
    decoded = tokenizer.decode(encoded[0], skip_special_tokens=True)
    print(f"Decoded: {decoded}")
    print()

def automodel_example():
    """
    AutoModel kullanım örneği
    Bu fonksiyonda bir transformer modelin çıktısına erişim,
    .last_hidden_state üzerinden token embedding'lerin alınması
    ve ilk token ([CLS]) embedding'inin örneklenmesi gösterilir.
    """
    print("=== AutoModel Örneği ===")
    
    # 1. Model ve tokenizer yüklenir
    # Modelin adı belirlenir ve hem tokenizer hem de model indirilir
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    # 2. Örnek bir metin belirlenir
    text = "This is an example sentence."
    
    # 3. Metin tokenize edilir ve tensöre dönüştürülür
    # Modelin anlayacağı şekilde input hazırlanır
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # 4. Model ile tahmin yapılır (inference)
    # torch.no_grad() ile gradyan hesabı kapatılır, sadece ileri besleme yapılır
    with torch.no_grad():
        outputs = model(**{k: v.to(device) for k, v in inputs.items()})

    # 5. Son katmandaki gizli durumlar (embeddingler) alınır
    # last_hidden_state: Her token için modelin ürettiği vektörler
    last_hidden_states = outputs.last_hidden_state

    print(f"Output shape: {last_hidden_states.shape}")  # (batch_size, sequence_length, hidden_size)
    print(f"CLS token embedding: {last_hidden_states[0][0][:5]}")  # İlk 5 değer
    print()

def pipeline_examples():
    """
    Pipeline ile hızlı model çağırma örnekleri
    pipeline() fonksiyonu ile sadece model adı vererek hızlı demo yapılır.
    Her pipeline görev türü (sentiment-analysis, text-generation, question-answering, fill-mask)
    Hugging Face tarafından ön tanımlı olarak gelir.
    """
    print("=== Pipeline Örnekleri ===")
    
    # 1. Duygu Analizi Pipeline'ı: cümlenin olumlu mu olumsuz mu olduğunu tespit eder
    print("1. Sentiment Analysis:")
    # pipeline fonksiyonu ile duygu analizi için hazır bir pipeline oluşturulur
    sentiment_pipeline = pipeline("sentiment-analysis", device=-1)
    text = "I love this product! It's amazing."
    result = sentiment_pipeline(text)
    print(f"Text: {text}")
    print(f"Result: {result}")
    print()
    
    # 2. Metin Üretimi Pipeline'ı: gpt2 ile verilen başlangıca metin üretir
    print("2. Text Generation:")
    # text-generation pipeline'ı ile verilen prompt'a devam eden metin üretilir
    text_generator = pipeline("text-generation", model="gpt2", device=-1)
    prompt = "The future of artificial intelligence is"
    result = text_generator(prompt, max_length=50, num_return_sequences=1)
    print(f"Prompt: {prompt}")
    print(f"Generated: {result[0]['generated_text']}")
    print()
    
    # 3. Soru-Cevap Pipeline'ı: verilen bağlamdan sorunun cevabını bulur
    print("3. Question Answering:")
    # question-answering pipeline'ı ile bir bağlamdan sorunun cevabı bulunur
    qa_pipeline = pipeline("question-answering", device=-1)
    context = "The capital of France is Paris. It is known for the Eiffel Tower."
    question = "What is the capital of France?"
    result = qa_pipeline(question=question, context=context)
    print(f"Context: {context}")
    print(f"Question: {question}")
    print(f"Answer: {result['answer']} (score: {result['score']:.4f})")
    print()
    
    # 4. Maskeli Kelime Doldurma Pipeline'ı: [MASK] yerine en olası kelimeleri üretir
    print("4. Fill Mask:") 
    # fill-mask pipeline'ı ile cümledeki maskeli kelimenin yerine en olası kelimeler tahmin edilir
    fill_mask = pipeline("fill-mask", device=-1)
    mask_token = fill_mask.tokenizer.mask_token  # modelin kullandığı mask token'ı alınır
    text = f"The food was{mask_token}."
    result = fill_mask(text)
    print(f"Text: {text}")
    print(f"Top predictions:")
    for i, pred in enumerate(result[:3]):
        print(f"  {i+1}. {pred['token_str']} (score: {pred['score']:.4f})")
    print()

def performance_comparison():
    """
    Manuel tokenization vs Pipeline performans karşılaştırması
    Aynı metinler üzerinde iki farklı yöntemle (manuel ve pipeline) işlem yapılır ve süreler karşılaştırılır.
    """
    print("=== Performans Karşılaştırması ===")
    
    # 1. Test için kullanılacak örnek metinler
    texts = [
        "This is a test sentence.",
        "Another example for testing.",
        "Performance comparison between methods.",
        "Pipeline vs manual approach.",
        "Speed and efficiency analysis."
    ]
    
    # 2. Manuel yaklaşım: Her metin için tokenizer ve model ayrı ayrı çalıştırılır
    start_time = time.time()
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    
    for text in texts:
        # Her metin için ayrı ayrı tokenize ve modelden geçirme işlemi
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**{k: v.to(device) for k, v in inputs.items()})
    
    manual_time = time.time() - start_time
    
    # 3. Pipeline yaklaşımı: pipeline ile topluca özellik çıkarımı yapılır
    start_time = time.time()
    feature_extractor = pipeline("feature-extraction", model=model_name, device=-1)
    
    for text in texts:
        # pipeline ile her metin için özellik çıkarımı
        features = feature_extractor(text)
    
    pipeline_time = time.time() - start_time
    
    # 4. Sonuçların karşılaştırılması
    print(f"Manuel yaklaşım süresi: {manual_time:.4f} saniye")
    print(f"Pipeline yaklaşımı süresi: {pipeline_time:.4f} saniye")
    print(f"Hız farkı: {manual_time/pipeline_time:.2f}x")
    print()

def custom_pipeline_example():
    """
    Özelleştirilmiş pipeline örneği
    Belirli bir model ve tokenizer ile pipeline oluşturulur.
    Modelin çıktılarının nasıl okunacağı (label, score) ve
    farklı duygusal tonlardaki metinlere verilen cevaplar gösterilir.
    """
    print("=== Özelleştirilmiş Pipeline ===")
    
    # 1. Belirli bir model ve tokenizer ile pipeline oluşturulur
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=model_name,
        tokenizer=model_name,
        device=-1
    )
    
    # 2. Analiz edilecek örnek metinler
    texts = [
        "I'm very happy today!",
        "This is terrible.",
        "The weather is okay.",
        "Amazing product, highly recommended!",
        "Not sure about this decision."
    ]
    
    print("Sentiment analysis sonuçları:")
    # 3. Her metin için duygu analizi yapılır ve sonuçlar yazdırılır
    for text in texts:
        result = sentiment_pipeline(text)
        label = result[0]['label']
        score = result[0]['score']
        print(f"'{text}' -> {label} ({score:.4f})")
    print()

if __name__ == "__main__":
    with SuppressStderr():
        # Ana program: tüm örnek fonksiyonlar sırasıyla çalıştırılır
        print("AutoTokenizer & AutoModel + Pipeline Örnekleri\n")
        tokenizer_example()
        automodel_example()
        pipeline_examples()
        performance_comparison()
        custom_pipeline_example()
        print("Tüm örnekler tamamlandı!")
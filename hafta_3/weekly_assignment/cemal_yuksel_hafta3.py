import torch
import time
import psutil
from transformers import pipeline, Pipeline
import matplotlib.pyplot as plt
from tabulate import tabulate
import pandas as pd
import logging
import os
import warnings

# CUDA kernel ve diğer uyarıları bastır
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
warnings.filterwarnings("ignore")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

# Log dosyası oluştur
logging.basicConfig(filename="model_performance.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Test metinleri (çeşitli uzunluk ve duyguda)
TEST_TEXTS = [
    "I love programming with Python. It's amazing for AI development and machine learning projects!",
    "The weather is terrible today and I feel so sad.",
    "Artificial intelligence is transforming the world in unprecedented ways.",
    "Kairu Bootcamp is a great opportunity to learn about LLMs and their real-world applications.",
    "Despite the challenges, the project was a huge success and everyone was happy.",
    "This is the worst movie I have ever seen. The plot was boring and the acting was terrible.",
    "Python is a versatile language used for web development, data science, and more.",
    "The food was delicious and the service was excellent.",
    "I am not sure if I can finish this assignment on time.",
    "The new policy will have a significant impact on the company's future."
]

# Model isimleri ve görevleri
MODEL_CONFIGS = [
    {"name": "GPT-2", "pipeline": "text-generation", "model": "gpt2", "task": "text_generation"},
    {"name": "BERT", "pipeline": "sentiment-analysis", "model": "distilbert-base-uncased-finetuned-sst-2-english", "task": "sentiment_analysis"},
    {"name": "T5", "pipeline": "summarization", "model": "t5-small", "task": "summarization"}
]

# Çıktı kalitesi için örnek skor fonksiyonları
def score_gpt2_output(output, input_text):
    # Basit örnek: Çıktı input'tan anlamlı şekilde farklı mı ve uzun mu?
    gen = output.get("generated_text", "")
    if len(gen) > len(input_text) + 10:
        return 1.0
    elif len(gen) > len(input_text):
        return 0.7
    else:
        return 0.3

def score_bert_output(output):
    # Pozitif/negatif ve güven skoruna göre puan
    label = output.get("label", "")
    score = output.get("score", 0)
    if label in ["POSITIVE", "NEGATIVE"] and score > 0.9:
        return 1.0
    elif score > 0.7:
        return 0.7
    else:
        return 0.4

def score_t5_output(output, input_text):
    # Özet gerçekten kısalmış mı ve anlamlı mı (örnek)
    summary = output.get("summary_text", "")
    if len(summary) < len(input_text) * 0.7:
        return 1.0
    elif len(summary) < len(input_text):
        return 0.7
    else:
        return 0.4

# Model yükleme ve yükleme süresi ölçümü
def load_pipeline_safe(pipe_type, model_name):
    start = time.time()
    try:
        # GPU uyumsuzluğunu önlemek için force_cpu parametresi
        pipe = pipeline(pipe_type, model=model_name, device=-1)
        elapsed = time.time() - start
        logging.info(f"{model_name} loaded in {elapsed:.2f} seconds.")
        return pipe, elapsed
    except Exception as e:
        logging.error(f"Error loading {model_name}: {e}")
        return None, None

# Model performansını ölçen fonksiyon (process bazlı memory)
def measure_performance(model_pipeline: Pipeline, text: str, task_name: str):
    """
    Modeli verilen görevde çalıştırır, süre ve bellek kullanımını ölçer.
    Çıktı ile birlikte performans metriklerini döndürür.
    """
    process = psutil.Process(os.getpid())
    start_time = time.time()
    start_memory = process.memory_info().rss / 1e9  # GB

    try:
        if task_name == "text_generation":
            result = model_pipeline(text, max_length=max(len(text.split()) + 10, 30), num_return_sequences=1, do_sample=False)
        elif task_name == "sentiment_analysis":
            result = model_pipeline(text)
        elif task_name == "summarization":
            # max_length input_length'ten küçükse uyarı çıkmasın diye ayarlanıyor
            input_len = len(text.split())
            max_len = max(10, int(input_len * 0.7))
            min_len = min(10, max_len - 1) if max_len > 10 else 5
            result = model_pipeline(text, max_length=max_len, min_length=min_len, do_sample=False)
        else:
            result = model_pipeline(text)
    except Exception as e:
        result = [{"error": str(e)}]

    end_time = time.time()
    end_memory = process.memory_info().rss / 1e9

    return {
        'task': task_name,
        'time': round(end_time - start_time, 3),
        'memory': round(end_memory - start_memory, 2),
        'result': result
    }

# Sonuçları tablo olarak yazdır
def print_results_table(results_df):
    print("\n=== Sonuçlar Tablosu ===")
    print(tabulate(results_df, headers='keys', tablefmt='psql', showindex=False))

# Sonuçları CSV'ye kaydet
def save_results_csv(results_df, filename="model_performance_results.csv"):
    results_df.to_csv(filename, index=False)
    print(f"\nSonuçlar CSV olarak kaydedildi: {filename}")

# Her modelin çıktısını ve skorunu analiz et
def analyze_outputs(model_name, outputs, input_texts):
    scores = []
    for i, out in enumerate(outputs):
        if model_name == "GPT-2":
            score = score_gpt2_output(out, input_texts[i])
        elif model_name == "BERT":
            score = score_bert_output(out)
        elif model_name == "T5":
            score = score_t5_output(out, input_texts[i])
        else:
            score = 0.0
        scores.append(score)
    return scores

# Gelişmiş görselleştirme
def plot_performance_advanced(results_df):
    plt.figure(figsize=(10,5))
    for model in results_df['Model'].unique():
        subset = results_df[results_df['Model']==model]
        plt.plot(subset['TextID'], subset['Time (s)'], marker='o', label=f"{model} Time")
    plt.xlabel("Text ID")
    plt.ylabel("Time (s)")
    plt.title("Model Bazında Çalışma Süreleri")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10,5))
    for model in results_df['Model'].unique():
        subset = results_df[results_df['Model']==model]
        plt.plot(subset['TextID'], subset['Memory (GB)'], marker='x', label=f"{model} Memory")
    plt.xlabel("Text ID")
    plt.ylabel("Memory (GB)")
    plt.title("Model Bazında Bellek Kullanımı")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10,5))
    for model in results_df['Model'].unique():
        subset = results_df[results_df['Model']==model]
        plt.plot(subset['TextID'], subset['Quality Score'], marker='s', label=f"{model} Quality")
    plt.xlabel("Text ID")
    plt.ylabel("Quality Score")
    plt.title("Model Bazında Çıktı Kalitesi")
    plt.legend()
    plt.show()

# Model çıktılarının örneklerini yazdır
def print_sample_outputs(model_name, outputs, input_texts):
    print(f"\n--- {model_name} Çıktı Örnekleri ---")
    for i, out in enumerate(outputs[:2]):
        print(f"Text {i+1}: {input_texts[i][:60]}...")
        if model_name == "GPT-2":
            print("Generated:", out.get("generated_text", "")[:120])
        elif model_name == "BERT":
            print("Sentiment:", out.get("label", ""), "Confidence:", round(out.get("score", 0),2))
        elif model_name == "T5":
            print("Summary:", out.get("summary_text", "")[:120])
        print("-"*40)

# Genel analiz ve öneriler
def print_overall_analysis(results_df):
    print("\n=== Genel Analiz ve Öneriler ===")
    avg = results_df.groupby("Model").mean(numeric_only=True)
    print(tabulate(avg, headers='keys', tablefmt='github'))
    fastest = avg['Time (s)'].idxmin()
    efficient = avg['Memory (GB)'].idxmin()
    best_quality = avg['Quality Score'].idxmax()
    print(f"\nEn hızlı model: {fastest}")
    print(f"En az bellek kullanan model: {efficient}")
    print(f"Çıktı kalitesi en yüksek model: {best_quality}")
    print("\nÖneriler:")
    print("- Hız ve verimlilik için BERT öne çıkıyor.")
    print("- Yaratıcı metin üretimi için GPT-2, özetleme için T5 tercih edilmeli.")
    print("- Uzun metinlerde T5 özetleri oldukça anlamlı ve kısa.")
    print("- Sentiment analizi için BERT çok hızlı ve güvenilir.")

def main():
    # 1. Modelleri yükle ve yükleme sürelerini ölç
    print("📦 Modeller yükleniyor...")
    pipelines = {}
    load_times = {}
    for cfg in MODEL_CONFIGS:
        pipe, elapsed = load_pipeline_safe(cfg["pipeline"], cfg["model"])
        if pipe is None:
            print(f"{cfg['name']} yüklenemedi!")
            return
        pipelines[cfg["name"]] = pipe
        load_times[cfg["name"]] = elapsed
        print(f"{cfg['name']} yüklendi ({elapsed:.2f} sn)")

    print("\n✅ Tüm modeller başarıyla yüklendi!\n")

    # 2. Her model ve her metin için test yap
    results = []
    outputs_dict = {m["name"]: [] for m in MODEL_CONFIGS}
    for text_id, text in enumerate(TEST_TEXTS):
        print(f"\n🧪 Test metni {text_id+1}/{len(TEST_TEXTS)}: {text[:60]}...")
        for cfg in MODEL_CONFIGS:
            model_name = cfg["name"]
            task = cfg["task"]
            pipe = pipelines[model_name]
            perf = measure_performance(pipe, text, task)
            # Çıktıdan ilk sonucu al
            out = perf['result'][0] if perf['result'] else {}
            outputs_dict[model_name].append(out)
            # Kalite skorunu hesapla
            if model_name == "GPT-2":
                quality = score_gpt2_output(out, text)
            elif model_name == "BERT":
                quality = score_bert_output(out)
            elif model_name == "T5":
                quality = score_t5_output(out, text)
            else:
                quality = 0.0
            # Sonuçları kaydet
            results.append({
                "TextID": text_id+1,
                "Model": model_name,
                "Task": task,
                "Time (s)": perf['time'],
                "Memory (GB)": perf['memory'],
                "Quality Score": quality,
                "Output": str(out)[:120]
            })

    # 3. Sonuçları DataFrame'e aktar
    results_df = pd.DataFrame(results)

    # 4. Sonuçları tablo olarak yazdır
    print_results_table(results_df)

    # 5. Sonuçları CSV'ye kaydet
    save_results_csv(results_df)

    # 6. Her model için çıktı örneklerini yazdır
    for cfg in MODEL_CONFIGS:
        print_sample_outputs(cfg["name"], outputs_dict[cfg["name"]], TEST_TEXTS)

    # 7. Gelişmiş görselleştirme
    plot_performance_advanced(results_df)

    # 8. Genel analiz ve öneriler
    print_overall_analysis(results_df)

    print("\n✅ Tüm analizler tamamlandı!")

if __name__ == "__main__":
    main()

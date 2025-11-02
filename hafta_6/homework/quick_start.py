"""
================================================================================
QUICK START - Intelligent Review Summarizer
================================================================================

Tüm pipeline'ı otomatik çalıştıran quick start scripti.

KULLANIM:
---------
python quick_start.py

veya

python quick_start.py --quick-test  # Küçük dataset ile hızlı test

Yazar: Kairu AI - Build with LLMs Bootcamp
Tarih: 2 Kasım 2025
================================================================================
"""

import sys
import argparse
import subprocess
import os
from pathlib import Path
from loguru import logger

# Script'in bulunduğu dizine geç
SCRIPT_DIR = Path(__file__).parent
os.chdir(SCRIPT_DIR)


def run_command(cmd, description):
    """Komutu çalıştır ve sonucu göster"""
    logger.info(f"\n{'='*80}")
    logger.info(f"▶️  {description}")
    logger.info(f"{'='*80}\n")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            text=True,
            capture_output=False,
            cwd=SCRIPT_DIR  # Script dizininde çalıştır
        )
        logger.success(f"✅ {description} - BAŞARILI!\n")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} - BAŞARISIZ!")
        logger.error(f"Hata: {e}")
        return False
    except KeyboardInterrupt:
        logger.warning(f"\n⚠️  {description} - KULLANICI TARAFINDAN DURDURULDU!")
        return False


def main():
    parser = argparse.ArgumentParser(description="Quick start the entire pipeline")
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Küçük dataset ile hızlı test (1000 train, 200 test)"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Model eğitimini atla (sadece data prep ve embedding)"
    )
    
    args = parser.parse_args()
    
    logger.info("\n")
    logger.info("╔" + "═"*78 + "╗")
    logger.info("║" + " "*20 + "🚀 QUICK START PIPELINE 🚀" + " "*32 + "║")
    logger.info("╚" + "═"*78 + "╝")
    logger.info("\n")
    
    # Pipeline steps
    steps = []
    
    # Step 1: Data Preparation
    if args.quick_test:
        cmd1 = "python 1_data_preparation.py --max-train 1000 --max-test 200"
    else:
        cmd1 = "python 1_data_preparation.py"
    
    steps.append((cmd1, "1️⃣  Veri Hazırlama"))
    
    # Step 2: Embedding Creation
    cmd2 = "python 2_embedding_creation.py --test-search"
    steps.append((cmd2, "2️⃣  Embedding ve Vector DB Oluşturma"))
    
    # Step 3: Model Training (optional)
    if not args.skip_training:
        if args.quick_test:
            cmd3 = "python 3_lora_summarizer_training.py --epochs 1 --max-train 1000 --max-test 200"
        else:
            cmd3 = "python 3_lora_summarizer_training.py"
        
        steps.append((cmd3, "3️⃣  LoRA Model Eğitimi"))
    
    # Execute steps
    total_steps = len(steps)
    for i, (cmd, description) in enumerate(steps, 1):
        logger.info(f"\n📍 Adım {i}/{total_steps}")
        
        success = run_command(cmd, description)
        
        if not success:
            logger.error("\n❌ Pipeline başarısız oldu!")
            logger.info("💡 Sorunu düzeltip tekrar çalıştırabilirsiniz.")
            sys.exit(1)
    
    # Final message
    logger.info("\n")
    logger.info("╔" + "═"*78 + "╗")
    logger.info("║" + " "*25 + "🎉 PIPELINE TAMAMLANDI! 🎉" + " "*28 + "║")
    logger.info("╚" + "═"*78 + "╝")
    logger.info("\n")
    
    logger.info("📊 Sonuçlar:")
    logger.info("  ✅ Veri hazırlama tamamlandı")
    logger.info("  ✅ Vector database oluşturuldu")
    if not args.skip_training:
        logger.info("  ✅ Model eğitimi tamamlandı")
    logger.info("\n")
    
    logger.info("🚀 Şimdi ne yapabilirsiniz?")
    logger.info("\n1️⃣  RAG Sistemini Test Edin:")
    logger.info("     python 4_rag_qa_system.py")
    logger.info("\n2️⃣  Web Arayüzünü Başlatın:")
    logger.info("     streamlit run 5_interactive_app.py")
    logger.info("\n")
    
    logger.success("✨ Tebrikler! Sistem kullanıma hazır!")


if __name__ == "__main__":
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
        colorize=True
    )
    
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n\n⚠️  Quick start kullanıcı tarafından durduruldu!")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"\n❌ Beklenmeyen hata: {e}")
        sys.exit(1)

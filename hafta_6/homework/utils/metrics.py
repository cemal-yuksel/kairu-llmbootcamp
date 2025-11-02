"""
================================================================================
METRICS - Intelligent Review Summarizer
================================================================================

Model değerlendirme metrikleri: ROUGE, BLEU, BERTScore ve custom metrics.

Yazar: Kairu AI - Build with LLMs Bootcamp
Tarih: 2 Kasım 2025
================================================================================
"""

from typing import List, Dict, Optional, Tuple, Union
import numpy as np
from loguru import logger
from collections import defaultdict
import json


class MetricsCalculator:
    """
    Model performans metriklerini hesaplayan sınıf
    
    Summarization ve Q&A için çeşitli metrikler sağlar.
    """
    
    def __init__(self, config):
        """
        Args:
            config: Configuration objesi
        """
        self.config = config
        self._initialize_scorers()
    
    def _initialize_scorers(self):
        """Metric scorer'larını başlat"""
        try:
            from rouge_score import rouge_scorer
            self.rouge_scorer = rouge_scorer.RougeScorer(
                self.config.ROUGE_TYPES,
                use_stemmer=self.config.USE_STEMMER
            )
            logger.info("✅ ROUGE scorer hazır")
        except ImportError:
            logger.warning("⚠️  rouge-score yüklenemedi, ROUGE metrikleri kullanılamayacak")
            self.rouge_scorer = None
        
        # BERTScore (optional, yavaş)
        self.bertscore_scorer = None
        if self.config.CALCULATE_BERTSCORE:
            try:
                import bert_score
                self.bertscore_scorer = bert_score
                logger.info("✅ BERTScore scorer hazır")
            except ImportError:
                logger.warning("⚠️  bert-score yüklenemedi")
    
    # ========================================================================
    # ROUGE METRICS (Summarization)
    # ========================================================================
    
    def calculate_rouge(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        ROUGE skorlarını hesapla
        
        ROUGE (Recall-Oriented Understudy for Gisting Evaluation):
        - Özetleme kalitesi için standart metrik
        - N-gram overlap'e bakar
        
        ROUGE Tipleri:
        - ROUGE-1: Unigram overlap
        - ROUGE-2: Bigram overlap
        - ROUGE-L: Longest Common Subsequence
        - ROUGE-Lsum: Multi-sentence summary için ROUGE-L
        
        Args:
            predictions: Model'in ürettiği özetler
            references: Gerçek (gold) özetler
            
        Returns:
            ROUGE skorları dictionary
        """
        if not self.rouge_scorer:
            logger.error("❌ ROUGE scorer yok!")
            return {}
        
        if len(predictions) != len(references):
            raise ValueError(f"Prediction ve reference sayısı eşleşmiyor: {len(predictions)} vs {len(references)}")
        
        logger.info(f"📊 ROUGE hesaplanıyor... ({len(predictions)} sample)")
        
        # Her sample için ROUGE hesapla
        all_scores = defaultdict(list)
        
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            
            for rouge_type, score in scores.items():
                all_scores[f"{rouge_type}_precision"].append(score.precision)
                all_scores[f"{rouge_type}_recall"].append(score.recall)
                all_scores[f"{rouge_type}_fmeasure"].append(score.fmeasure)
        
        # Ortalama al
        avg_scores = {}
        for key, values in all_scores.items():
            avg_scores[key] = np.mean(values)
        
        # Sadece F1 skorlarını logla
        logger.info("✅ ROUGE Scores (F1):")
        for rouge_type in self.config.ROUGE_TYPES:
            f1_key = f"{rouge_type}_fmeasure"
            if f1_key in avg_scores:
                logger.info(f"  • {rouge_type}: {avg_scores[f1_key]:.4f}")
        
        return avg_scores
    
    # ========================================================================
    # BLEU SCORE (Translation/Generation)
    # ========================================================================
    
    def calculate_bleu(
        self,
        predictions: List[str],
        references: List[List[str]]  # Her prediction için multiple reference olabilir
    ) -> Dict[str, float]:
        """
        BLEU skorunu hesapla
        
        BLEU (Bilingual Evaluation Understudy):
        - Makine çevirisi için geliştirilmiş
        - Generation quality için de kullanılır
        - N-gram precision'a odaklanır
        
        Args:
            predictions: Model çıktıları
            references: Reference'lar (her biri liste)
            
        Returns:
            BLEU skorları
        """
        try:
            from sacrebleu import corpus_bleu
            
            # Her prediction için tek reference varsa, nested list yap
            if references and not isinstance(references[0], list):
                references = [[ref] for ref in references]
            
            # sacrebleu references'ı transpose ister
            references_transposed = list(zip(*references))
            
            bleu = corpus_bleu(predictions, references_transposed)
            
            return {
                "bleu": bleu.score,
                "bleu_precisions": bleu.precisions,
                "bleu_bp": bleu.bp,
                "bleu_ratio": bleu.sys_len / bleu.ref_len
            }
        
        except ImportError:
            logger.warning("⚠️  sacrebleu yüklenemedi")
            return {}
    
    # ========================================================================
    # BERTSCORE (Semantic Similarity)
    # ========================================================================
    
    def calculate_bertscore(
        self,
        predictions: List[str],
        references: List[str],
        lang: str = "en"
    ) -> Dict[str, float]:
        """
        BERTScore hesapla
        
        BERTScore:
        - BERT embeddings kullanarak semantic similarity ölçer
        - N-gram'lardan daha sofistike
        - Daha yavaş ama daha kaliteli
        
        Args:
            predictions: Model çıktıları
            references: Gold references
            lang: Dil kodu
            
        Returns:
            BERTScore metrikleri
        """
        if not self.bertscore_scorer:
            logger.warning("⚠️  BERTScore hesaplanamıyor")
            return {}
        
        logger.info("📊 BERTScore hesaplanıyor... (bu biraz sürebilir)")
        
        P, R, F1 = self.bertscore_scorer.score(
            predictions,
            references,
            lang=lang,
            verbose=False
        )
        
        return {
            "bertscore_precision": P.mean().item(),
            "bertscore_recall": R.mean().item(),
            "bertscore_f1": F1.mean().item()
        }
    
    # ========================================================================
    # Q&A METRICS
    # ========================================================================
    
    def calculate_exact_match(
        self,
        predictions: List[str],
        references: List[str],
        normalize: bool = True
    ) -> float:
        """
        Exact Match (EM) hesapla
        
        Prediction ve reference'ın tam olarak eşleşip eşleşmediğini kontrol eder.
        
        Args:
            predictions: Model cevapları
            references: Doğru cevaplar
            normalize: Lowercase ve whitespace normalizasyonu
            
        Returns:
            EM skoru (0-1 arası)
        """
        def normalize_answer(s):
            """Answer normalizasyonu"""
            import re
            import string
            
            # Lowercase
            s = s.lower()
            
            # Punctuation kaldır
            s = s.translate(str.maketrans('', '', string.punctuation))
            
            # Fazla whitespace
            s = re.sub(r'\s+', ' ', s).strip()
            
            return s
        
        matches = 0
        for pred, ref in zip(predictions, references):
            if normalize:
                pred = normalize_answer(pred)
                ref = normalize_answer(ref)
            
            if pred == ref:
                matches += 1
        
        em = matches / len(predictions) if predictions else 0
        logger.info(f"  📊 Exact Match: {em:.4f} ({matches}/{len(predictions)})")
        
        return em
    
    def calculate_f1_score(
        self,
        predictions: List[str],
        references: List[str]
    ) -> float:
        """
        Token-level F1 score hesapla (Q&A için)
        
        Args:
            predictions: Model cevapları
            references: Doğru cevaplar
            
        Returns:
            Ortalama F1 skoru
        """
        def compute_f1(pred_tokens, ref_tokens):
            """Tek bir örnek için F1"""
            common = set(pred_tokens) & set(ref_tokens)
            
            if len(common) == 0:
                return 0.0
            
            precision = len(common) / len(pred_tokens) if pred_tokens else 0
            recall = len(common) / len(ref_tokens) if ref_tokens else 0
            
            if precision + recall == 0:
                return 0.0
            
            f1 = 2 * (precision * recall) / (precision + recall)
            return f1
        
        f1_scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()
            f1 = compute_f1(pred_tokens, ref_tokens)
            f1_scores.append(f1)
        
        avg_f1 = np.mean(f1_scores)
        logger.info(f"  📊 F1 Score: {avg_f1:.4f}")
        
        return avg_f1
    
    # ========================================================================
    # CUSTOM METRICS
    # ========================================================================
    
    def calculate_length_stats(
        self,
        predictions: List[str],
        references: Optional[List[str]] = None
    ) -> Dict:
        """
        Uzunluk istatistikleri
        
        Args:
            predictions: Model çıktıları
            references: Reference'lar (optional)
            
        Returns:
            Uzunluk metrikleri
        """
        pred_lengths = [len(p.split()) for p in predictions]
        
        stats = {
            "pred_avg_length": np.mean(pred_lengths),
            "pred_min_length": np.min(pred_lengths),
            "pred_max_length": np.max(pred_lengths),
            "pred_std_length": np.std(pred_lengths)
        }
        
        if references:
            ref_lengths = [len(r.split()) for r in references]
            stats.update({
                "ref_avg_length": np.mean(ref_lengths),
                "length_ratio": np.mean(pred_lengths) / np.mean(ref_lengths)
            })
        
        return stats
    
    def calculate_diversity_metrics(
        self,
        predictions: List[str]
    ) -> Dict:
        """
        Üretilen text'lerin çeşitliliğini ölç
        
        Args:
            predictions: Model çıktıları
            
        Returns:
            Diversity metrikleri
        """
        all_tokens = []
        for pred in predictions:
            all_tokens.extend(pred.lower().split())
        
        unique_tokens = set(all_tokens)
        
        # Distinct-1, Distinct-2 (unique unigram/bigram oranı)
        bigrams = []
        for pred in predictions:
            tokens = pred.lower().split()
            bigrams.extend([f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens)-1)])
        
        unique_bigrams = set(bigrams)
        
        return {
            "distinct_1": len(unique_tokens) / len(all_tokens) if all_tokens else 0,
            "distinct_2": len(unique_bigrams) / len(bigrams) if bigrams else 0,
            "vocab_size": len(unique_tokens)
        }
    
    # ========================================================================
    # COMPREHENSIVE EVALUATION
    # ========================================================================
    
    def evaluate_summarization(
        self,
        predictions: List[str],
        references: List[str],
        include_bertscore: bool = False
    ) -> Dict:
        """
        Kapsamlı summarization değerlendirmesi
        
        Args:
            predictions: Model özetleri
            references: Gold özetler
            include_bertscore: BERTScore ekle (yavaş)
            
        Returns:
            Tüm metrikler
        """
        logger.info("=" * 80)
        logger.info("📊 SUMMARIZATION EVALUATION")
        logger.info("=" * 80)
        
        results = {}
        
        # ROUGE
        if self.rouge_scorer:
            rouge_scores = self.calculate_rouge(predictions, references)
            results.update(rouge_scores)
        
        # Length stats
        length_stats = self.calculate_length_stats(predictions, references)
        results.update(length_stats)
        
        # Diversity
        diversity = self.calculate_diversity_metrics(predictions)
        results.update(diversity)
        
        # BERTScore (optional)
        if include_bertscore and self.bertscore_scorer:
            bertscore = self.calculate_bertscore(predictions, references)
            results.update(bertscore)
        
        logger.info("=" * 80)
        
        return results
    
    def evaluate_qa(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict:
        """
        Q&A değerlendirmesi
        
        Args:
            predictions: Model cevapları
            references: Doğru cevaplar
            
        Returns:
            Q&A metrikleri
        """
        logger.info("=" * 80)
        logger.info("📊 Q&A EVALUATION")
        logger.info("=" * 80)
        
        results = {}
        
        # Exact Match
        em = self.calculate_exact_match(predictions, references)
        results["exact_match"] = em
        
        # F1 Score
        f1 = self.calculate_f1_score(predictions, references)
        results["f1_score"] = f1
        
        logger.info("=" * 80)
        
        return results
    
    def save_results(
        self,
        results: Dict,
        output_path: str
    ):
        """
        Sonuçları kaydet
        
        Args:
            results: Metrik sonuçları
            output_path: JSON çıktı yolu
        """
        import json
        from pathlib import Path
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Sonuçlar kaydedildi: {output_path}")


# ============================================================================
# QUICK TEST
# ============================================================================
if __name__ == "__main__":
    from config import config
    
    logger.info("🧪 MetricsCalculator test başlıyor...")
    
    calculator = MetricsCalculator(config)
    
    # Test data
    predictions = [
        "This is a great movie with excellent acting.",
        "The film was boring and too long."
    ]
    
    references = [
        "This movie is excellent with great performances.",
        "The movie was dull and overly lengthy."
    ]
    
    # ROUGE test
    if calculator.rouge_scorer:
        rouge = calculator.calculate_rouge(predictions, references)
        logger.info(f"\n📊 ROUGE: {rouge}")
    
    # Q&A test
    qa_preds = ["Paris", "1998"]
    qa_refs = ["Paris", "1998"]
    
    em = calculator.calculate_exact_match(qa_preds, qa_refs)
    f1 = calculator.calculate_f1_score(qa_preds, qa_refs)
    
    logger.info("\n✅ Tüm testler başarılı!")

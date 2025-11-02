"""
================================================================================
TEXT PROCESSOR - Intelligent Review Summarizer
================================================================================

Text preprocessing, cleaning ve transformation fonksiyonları.

Yazar: Kairu AI - Build with LLMs Bootcamp
Tarih: 2 Kasım 2025
================================================================================
"""

import re
import string
from typing import List, Dict, Optional, Tuple
from loguru import logger
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


class TextProcessor:
    """
    Text işleme ve temizleme sınıfı
    
    IMDB review'larını temizler, normalize eder ve işler.
    """
    
    def __init__(self, language: str = "english"):
        """
        Args:
            language: Dil ayarı (stopwords için)
        """
        self.language = language
        self._download_nltk_resources()
        
        # NLTK tools
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        try:
            self.stop_words = set(stopwords.words(language))
        except:
            logger.warning("⚠️  Stopwords yüklenemedi, boş set kullanılıyor")
            self.stop_words = set()
    
    def _download_nltk_resources(self):
        """Gerekli NLTK resource'larını indir"""
        resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
        
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                try:
                    logger.info(f"📥 NLTK {resource} indiriliyor...")
                    nltk.download(resource, quiet=True)
                except:
                    logger.warning(f"⚠️  {resource} indirilemedi")
    
    def clean_text(
        self,
        text: str,
        lowercase: bool = True,
        remove_html: bool = True,
        remove_urls: bool = True,
        remove_special_chars: bool = False,
        remove_numbers: bool = False,
        remove_extra_spaces: bool = True
    ) -> str:
        """
        Text'i temizle
        
        Args:
            text: Temizlenecek text
            lowercase: Küçük harfe çevir
            remove_html: HTML taglarını kaldır
            remove_urls: URL'leri kaldır
            remove_special_chars: Özel karakterleri kaldır
            remove_numbers: Sayıları kaldır
            remove_extra_spaces: Fazla boşlukları kaldır
            
        Returns:
            Temizlenmiş text
        """
        if not text:
            return ""
        
        # HTML tagları
        if remove_html:
            text = re.sub(r'<.*?>', ' ', text)
            text = re.sub(r'&[a-z]+;', ' ', text)  # &nbsp; gibi
        
        # URL'ler
        if remove_urls:
            text = re.sub(r'http\S+|www.\S+', ' ', text)
        
        # Email adresleri
        text = re.sub(r'\S+@\S+', ' ', text)
        
        # Lowercase
        if lowercase:
            text = text.lower()
        
        # Sayılar
        if remove_numbers:
            text = re.sub(r'\d+', ' ', text)
        
        # Özel karakterler (sadece alfanumerik ve space)
        if remove_special_chars:
            text = re.sub(f'[^a-zA-Z0-9\s]', ' ', text)
        
        # Fazla boşluklar
        if remove_extra_spaces:
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
        
        return text
    
    def remove_stopwords(
        self,
        text: str,
        custom_stopwords: Optional[List[str]] = None
    ) -> str:
        """
        Stopword'leri kaldır
        
        Args:
            text: Text
            custom_stopwords: Ekstra stopwords
            
        Returns:
            Stopword'siz text
        """
        words = text.split()
        
        stopwords_set = self.stop_words.copy()
        if custom_stopwords:
            stopwords_set.update(custom_stopwords)
        
        filtered_words = [w for w in words if w not in stopwords_set]
        
        return ' '.join(filtered_words)
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Text'i cümlelere böl
        
        Args:
            text: Bölünecek text
            
        Returns:
            Cümle listesi
        """
        try:
            sentences = sent_tokenize(text)
            return sentences
        except:
            # Fallback: basit nokta ile bölme
            return text.split('. ')
    
    def tokenize_words(self, text: str) -> List[str]:
        """
        Text'i kelimelere böl
        
        Args:
            text: Bölünecek text
            
        Returns:
            Kelime listesi
        """
        try:
            words = word_tokenize(text)
            return words
        except:
            # Fallback: basit split
            return text.split()
    
    def stem_text(self, text: str) -> str:
        """
        Text'i stem'le (kök bulma)
        
        Örnek: "running" -> "run"
        
        Args:
            text: Stemlenecek text
            
        Returns:
            Stemlenmiş text
        """
        words = self.tokenize_words(text)
        stemmed = [self.stemmer.stem(word) for word in words]
        return ' '.join(stemmed)
    
    def lemmatize_text(self, text: str) -> str:
        """
        Text'i lemmatize et
        
        Stemming'den daha sofistike, gerçek kelimeler döndürür.
        
        Args:
            text: Lemmatize edilecek text
            
        Returns:
            Lemmatize edilmiş text
        """
        words = self.tokenize_words(text)
        lemmatized = [self.lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized)
    
    def extract_keywords(
        self,
        text: str,
        top_k: int = 10,
        min_word_length: int = 3
    ) -> List[Tuple[str, int]]:
        """
        Text'ten keyword'leri çıkar (frequency-based)
        
        Args:
            text: Analiz edilecek text
            top_k: Kaç keyword döndürülecek
            min_word_length: Minimum kelime uzunluğu
            
        Returns:
            [(keyword, frequency), ...] listesi
        """
        # Temizle
        cleaned = self.clean_text(
            text,
            lowercase=True,
            remove_special_chars=True,
            remove_numbers=True
        )
        
        # Stopword'leri kaldır
        no_stopwords = self.remove_stopwords(cleaned)
        
        # Kelime frekansları
        words = no_stopwords.split()
        words = [w for w in words if len(w) >= min_word_length]
        
        from collections import Counter
        word_freq = Counter(words)
        
        return word_freq.most_common(top_k)
    
    def get_text_stats(self, text: str) -> Dict:
        """
        Text istatistikleri
        
        Args:
            text: Analiz edilecek text
            
        Returns:
            İstatistik dictionary
        """
        sentences = self.tokenize_sentences(text)
        words = self.tokenize_words(text)
        
        # Karakter sayısı
        char_count = len(text)
        char_count_no_spaces = len(text.replace(' ', ''))
        
        # Kelime sayısı
        word_count = len(words)
        
        # Cümle sayısı
        sentence_count = len(sentences)
        
        # Ortalamalar
        avg_word_length = char_count_no_spaces / word_count if word_count > 0 else 0
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        return {
            "char_count": char_count,
            "char_count_no_spaces": char_count_no_spaces,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_word_length": round(avg_word_length, 2),
            "avg_sentence_length": round(avg_sentence_length, 2)
        }
    
    def truncate_text(
        self,
        text: str,
        max_length: int,
        strategy: str = "end"
    ) -> str:
        """
        Text'i belirli uzunlukta kes
        
        Args:
            text: Kesilecek text
            max_length: Maksimum uzunluk (karakter)
            strategy: "start", "end", "middle"
            
        Returns:
            Kesilmiş text
        """
        if len(text) <= max_length:
            return text
        
        if strategy == "end":
            return text[:max_length-3] + "..."
        elif strategy == "start":
            return "..." + text[-(max_length-3):]
        elif strategy == "middle":
            half = (max_length - 3) // 2
            return text[:half] + "..." + text[-half:]
        else:
            return text[:max_length]
    
    def highlight_keywords(
        self,
        text: str,
        keywords: List[str],
        tag: str = "**"
    ) -> str:
        """
        Text içinde keyword'leri vurgula
        
        Args:
            text: Orijinal text
            keywords: Vurgulanacak keyword'ler
            tag: Vurgulama tagi (Markdown için **)
            
        Returns:
            Vurgulanmış text
        """
        result = text
        for keyword in keywords:
            # Case-insensitive replacement
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            result = pattern.sub(f"{tag}{keyword}{tag}", result)
        
        return result
    
    def preprocess_for_model(
        self,
        text: str,
        max_length: Optional[int] = None
    ) -> str:
        """
        Text'i model için hazırla (tam pipeline)
        
        Args:
            text: İşlenecek text
            max_length: Maksimum uzunluk
            
        Returns:
            İşlenmiş text
        """
        # Temizle
        cleaned = self.clean_text(
            text,
            lowercase=True,
            remove_html=True,
            remove_urls=True,
            remove_special_chars=False,  # Noktalama işaretlerini koru
            remove_numbers=False,
            remove_extra_spaces=True
        )
        
        # Truncate
        if max_length:
            cleaned = self.truncate_text(cleaned, max_length, strategy="end")
        
        return cleaned


# ============================================================================
# QUICK TEST
# ============================================================================
if __name__ == "__main__":
    logger.info("🧪 TextProcessor test başlıyor...")
    
    processor = TextProcessor()
    
    # Test text
    test_text = """
    <p>This is an AMAZING movie! I loved it so much! 🎬</p>
    Visit our website at http://example.com for more reviews.
    The acting was superb, the directing was brilliant, and the 
    cinematography was absolutely stunning. 10/10 would recommend!
    """
    
    logger.info(f"\n📝 Original text:\n{test_text}")
    
    # Temizleme
    cleaned = processor.clean_text(test_text)
    logger.info(f"\n🧹 Cleaned text:\n{cleaned}")
    
    # Stopword removal
    no_stopwords = processor.remove_stopwords(cleaned)
    logger.info(f"\n🚫 No stopwords:\n{no_stopwords}")
    
    # Keywords
    keywords = processor.extract_keywords(test_text, top_k=5)
    logger.info(f"\n🔑 Keywords: {keywords}")
    
    # Stats
    stats = processor.get_text_stats(test_text)
    logger.info(f"\n📊 Stats: {stats}")
    
    logger.info("\n✅ Tüm testler başarılı!")

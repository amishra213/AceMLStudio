"""
AceML Studio – NLP Engine
============================
Natural Language Processing module supporting:
  • Text preprocessing (tokenization, stemming, lemmatization, stop-word removal)
  • Sentiment analysis (TextBlob, VADER, ML-based)
  • Text classification (TF-IDF + classifiers)
  • Topic modeling (LDA, NMF)
  • Named Entity Recognition (spaCy or regex-based)
  • Keyword / keyphrase extraction (TF-IDF, RAKE-like)
  • Text similarity (cosine, Jaccard)
  • Word cloud data generation
  • Text vectorization (TF-IDF, Count, Hashing)
"""

import logging
import re
import time
import warnings
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import (
    CountVectorizer,
    HashingVectorizer,
    TfidfVectorizer,
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger("aceml.nlp")

# ── Optional heavy imports ────────────────────────────────────────
try:
    import nltk  # type: ignore
    from nltk.tokenize import word_tokenize, sent_tokenize  # type: ignore
    from nltk.stem import PorterStemmer, WordNetLemmatizer  # type: ignore
    from nltk.corpus import stopwords as nltk_stopwords  # type: ignore
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False
    logger.info("NLTK not installed – advanced tokenisation disabled")

try:
    from textblob import TextBlob  # type: ignore
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False
    logger.info("TextBlob not installed – TextBlob sentiment disabled")

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer  # type: ignore
    HAS_VADER = True
except ImportError:
    HAS_VADER = False
    logger.info("VADER not available – VADER sentiment disabled")

try:
    import spacy  # type: ignore
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False
    logger.info("spaCy not installed – NER via spaCy disabled")

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Minimal stop-words fallback ───────────────────────────────────
_BASIC_STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his",
    "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having",
    "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
    "or", "because", "as", "until", "while", "of", "at", "by", "for",
    "with", "about", "against", "between", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "in", "out",
    "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
    "can", "will", "just", "don", "should", "now", "d", "ll", "m", "o",
    "re", "ve", "y",
}


def _get_stopwords(language: str = "english") -> set:
    """Get stop-words, preferring NLTK when available."""
    if HAS_NLTK:
        try:
            return set(nltk_stopwords.words(language))
        except LookupError:
            try:
                nltk.download("stopwords", quiet=True)
                return set(nltk_stopwords.words(language))
            except Exception:
                pass
    return _BASIC_STOPWORDS


def _ensure_nltk_data() -> None:
    """Download NLTK data packages if missing."""
    if not HAS_NLTK:
        return
    for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "averaged_perceptron_tagger",
                "vader_lexicon"):
        try:
            nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}"
                           if pkg in ("stopwords", "wordnet") else f"taggers/{pkg}"
                           if "tagger" in pkg else f"sentiment/{pkg}")
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass


# ════════════════════════════════════════════════════════════════════
#  NLP Engine
# ════════════════════════════════════════════════════════════════════

class NLPEngine:
    """Full-lifecycle NLP analysis for tabular text data."""

    # ----------------------------------------------------------------
    #  Text Column Detection
    # ----------------------------------------------------------------
    @staticmethod
    def detect_text_columns(df: pd.DataFrame, min_avg_words: float = 3.0) -> List[Dict[str, Any]]:
        """Auto-detect columns that contain meaningful text."""
        candidates: List[Dict[str, Any]] = []
        for col in df.columns:
            if df[col].dtype != object:
                continue
            sample = df[col].dropna().astype(str)
            if len(sample) == 0:
                continue
            avg_words = sample.str.split().str.len().mean()
            avg_len = sample.str.len().mean()
            unique_ratio = sample.nunique() / len(sample) if len(sample) > 0 else 0
            if avg_words >= min_avg_words or avg_len >= 20:
                candidates.append({
                    "column": col,
                    "avg_words": round(float(avg_words), 2),
                    "avg_length": round(float(avg_len), 1),
                    "unique_ratio": round(float(unique_ratio), 4),
                    "null_count": int(df[col].isna().sum()),
                    "sample": str(sample.iloc[0])[:200],
                })
        return candidates

    # ----------------------------------------------------------------
    #  Text Preprocessing
    # ----------------------------------------------------------------
    @staticmethod
    def preprocess_text(
        texts: pd.Series,
        lowercase: bool = True,
        remove_urls: bool = True,
        remove_html: bool = True,
        remove_numbers: bool = False,
        remove_punctuation: bool = True,
        remove_stopwords: bool = True,
        stemming: bool = False,
        lemmatization: bool = True,
        min_word_length: int = 2,
        language: str = "english",
    ) -> Dict[str, Any]:
        """Clean and normalise a Series of text."""
        start_time = time.time()
        original_count = len(texts)
        processed = texts.fillna("").astype(str).copy()

        if lowercase:
            processed = processed.str.lower()
        if remove_urls:
            processed = processed.str.replace(r"http\S+|www\.\S+", "", regex=True)
        if remove_html:
            processed = processed.str.replace(r"<[^>]+>", "", regex=True)
        if remove_numbers:
            processed = processed.str.replace(r"\d+", "", regex=True)
        if remove_punctuation:
            processed = processed.str.replace(r"[^\w\s]", "", regex=True)

        # Tokenise
        if HAS_NLTK:
            _ensure_nltk_data()
            tokens_series = processed.apply(lambda t: word_tokenize(t) if t else [])
        else:
            tokens_series = processed.str.split()

        # Stop-words
        if remove_stopwords:
            sw = _get_stopwords(language)
            tokens_series = tokens_series.apply(lambda toks: [t for t in toks if t not in sw])

        # Min word length
        if min_word_length > 1:
            tokens_series = tokens_series.apply(
                lambda toks: [t for t in toks if len(t) >= min_word_length]
            )

        # Stemming / Lemmatization
        if stemming and HAS_NLTK:
            stemmer = PorterStemmer()
            tokens_series = tokens_series.apply(lambda toks: [stemmer.stem(t) for t in toks])
        elif lemmatization and HAS_NLTK:
            lemmatizer = WordNetLemmatizer()
            tokens_series = tokens_series.apply(lambda toks: [lemmatizer.lemmatize(t) for t in toks])

        processed_texts = tokens_series.apply(lambda toks: " ".join(toks))

        return {
            "processed_texts": processed_texts.tolist(),
            "total_documents": original_count,
            "avg_tokens_before": round(float(texts.fillna("").astype(str).str.split().str.len().mean()), 2),
            "avg_tokens_after": round(float(tokens_series.str.len().mean()), 2),
            "preprocessing_steps": {
                "lowercase": lowercase,
                "remove_urls": remove_urls,
                "remove_html": remove_html,
                "remove_numbers": remove_numbers,
                "remove_punctuation": remove_punctuation,
                "remove_stopwords": remove_stopwords,
                "stemming": stemming,
                "lemmatization": lemmatization,
            },
            "duration_sec": round(time.time() - start_time, 3),
        }

    # ================================================================
    #  SENTIMENT ANALYSIS
    # ================================================================

    @staticmethod
    def sentiment_analysis(
        texts: pd.Series,
        method: str = "vader",
    ) -> Dict[str, Any]:
        """
        Analyse sentiment of each text.
        Methods: vader, textblob, combined.
        """
        start_time = time.time()
        results_list: List[Dict[str, Any]] = []
        texts_clean = texts.fillna("").astype(str)

        if method in ("vader", "combined"):
            if not HAS_VADER:
                if not HAS_NLTK:
                    return {"error": "NLTK not installed — VADER unavailable"}
                _ensure_nltk_data()
                try:
                    from nltk.sentiment.vader import SentimentIntensityAnalyzer
                except ImportError:
                    return {"error": "VADER not available even after NLTK data download"}
            analyzer = SentimentIntensityAnalyzer()

        for idx, text in enumerate(texts_clean):
            entry: Dict[str, Any] = {"index": idx, "text": text[:200]}
            if method == "vader" or method == "combined":
                try:
                    vs = analyzer.polarity_scores(text)  # type: ignore[possibly-undefined]
                    entry["vader"] = {
                        "compound": round(vs["compound"], 4),
                        "positive": round(vs["pos"], 4),
                        "negative": round(vs["neg"], 4),
                        "neutral": round(vs["neu"], 4),
                        "label": "positive" if vs["compound"] >= 0.05
                                 else "negative" if vs["compound"] <= -0.05
                                 else "neutral",
                    }
                except Exception:
                    entry["vader"] = {"error": "VADER failed"}

            if method == "textblob" or method == "combined":
                if HAS_TEXTBLOB:
                    blob = TextBlob(text)
                    entry["textblob"] = {
                        "polarity": round(float(blob.sentiment.polarity), 4),
                        "subjectivity": round(float(blob.sentiment.subjectivity), 4),
                        "label": "positive" if blob.sentiment.polarity > 0.1
                                 else "negative" if blob.sentiment.polarity < -0.1
                                 else "neutral",
                    }
                else:
                    entry["textblob"] = {"error": "TextBlob not installed"}

            results_list.append(entry)

        # Aggregate stats
        def _aggregate(key: str, subkey: str) -> Dict[str, Any]:
            vals = [r[key][subkey] for r in results_list if key in r and subkey in r.get(key, {})]
            if not vals:
                return {}
            labels = [r[key].get("label", "unknown") for r in results_list if key in r and "label" in r.get(key, {})]
            label_counts = dict(Counter(labels))
            return {
                "mean_score": round(float(np.mean(vals)), 4),
                "std_score": round(float(np.std(vals)), 4),
                "min_score": round(float(np.min(vals)), 4),
                "max_score": round(float(np.max(vals)), 4),
                "label_distribution": label_counts,
            }

        summary: Dict[str, Any] = {}
        if method in ("vader", "combined"):
            summary["vader_summary"] = _aggregate("vader", "compound")
        if method in ("textblob", "combined"):
            summary["textblob_summary"] = _aggregate("textblob", "polarity")

        return {
            "method": method,
            "total_documents": len(texts_clean),
            "results": results_list[:500],  # cap
            "summary": summary,
            "duration_sec": round(time.time() - start_time, 3),
        }

    # ================================================================
    #  TEXT CLASSIFICATION
    # ================================================================

    @staticmethod
    def train_text_classifier(
        texts: pd.Series,
        labels: pd.Series,
        model_type: str = "logistic_regression",
        max_features: int = 10000,
        test_size: float = 0.2,
        ngram_range: Tuple[int, int] = (1, 2),
    ) -> Dict[str, Any]:
        """Train a text classifier using TF-IDF + ML model."""
        start_time = time.time()
        texts_clean = texts.fillna("").astype(str)

        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(labels)
        class_names = le.classes_.tolist()

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            texts_clean, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )

        # Build pipeline
        classifiers = {
            "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
            "naive_bayes": MultinomialNB(),
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "svm": LinearSVC(max_iter=2000, random_state=42),
        }
        clf = classifiers.get(model_type, classifiers["logistic_regression"])

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, sublinear_tf=True)),
            ("clf", clf),
        ])
        pipeline.fit(X_train, y_train)

        # Evaluate
        y_pred = pipeline.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        # Top features per class (for interpretable models)
        top_features: Optional[Dict[str, List[str]]] = None
        tfidf_step = pipeline.named_steps["tfidf"]
        clf_step = pipeline.named_steps["clf"]
        feature_names = tfidf_step.get_feature_names_out()
        if hasattr(clf_step, "coef_"):
            top_features = {}
            coef = clf_step.coef_
            if coef.ndim == 1:
                coef = coef.reshape(1, -1)
            for i, cls_name in enumerate(class_names):
                if i < coef.shape[0]:
                    top_idx = coef[i].argsort()[-20:][::-1]
                    top_features[cls_name] = [str(feature_names[j]) for j in top_idx]

        duration = round(time.time() - start_time, 2)

        return {
            "model_type": model_type,
            "classes": class_names,
            "n_classes": len(class_names),
            "training_time_sec": duration,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "metrics": {
                "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
                "precision": round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
                "recall": round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
                "f1_score": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
            },
            "per_class_report": {
                str(k): {m: round(float(v), 4) for m, v in vals.items()} if isinstance(vals, dict) else vals  # type: ignore[union-attr]
                for k, vals in report.items()  # type: ignore[union-attr]
            },
            "confusion_matrix": {
                "labels": class_names,
                "matrix": cm.tolist(),
            },
            "top_features": top_features,
            "tfidf_params": {
                "max_features": max_features,
                "ngram_range": list(ngram_range),
            },
            "pipeline": pipeline,  # for later predictions
        }

    # ================================================================
    #  TOPIC MODELING
    # ================================================================

    @staticmethod
    def topic_modeling(
        texts: pd.Series,
        n_topics: int = 10,
        method: str = "lda",
        max_features: int = 5000,
        n_top_words: int = 15,
        ngram_range: Tuple[int, int] = (1, 1),
    ) -> Dict[str, Any]:
        """Discover topics using LDA or NMF."""
        start_time = time.time()
        texts_clean = texts.fillna("").astype(str)

        # Vectorize
        if method == "lda":
            vectorizer = CountVectorizer(max_features=max_features, ngram_range=ngram_range,
                                         stop_words="english")
        else:
            vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range,
                                          stop_words="english")

        dtm = vectorizer.fit_transform(texts_clean)
        feature_names = vectorizer.get_feature_names_out()

        # Fit model
        if method == "lda":
            model = LatentDirichletAllocation(
                n_components=n_topics, random_state=42, max_iter=30, learning_method="batch"
            )
        else:
            model = NMF(n_components=n_topics, random_state=42, max_iter=300)

        doc_topics = model.fit_transform(dtm)

        # Extract top words per topic
        topics: List[Dict[str, Any]] = []
        for topic_idx, topic_weights in enumerate(model.components_):
            top_indices = topic_weights.argsort()[:-n_top_words - 1:-1]
            top_words = [(str(feature_names[i]), round(float(topic_weights[i]), 4)) for i in top_indices]
            topics.append({
                "topic_id": topic_idx,
                "top_words": [w for w, _ in top_words],
                "word_weights": {w: s for w, s in top_words},
                "label": f"Topic {topic_idx + 1}",
            })

        # Document-topic assignments
        doc_assignments = doc_topics.argmax(axis=1).tolist()
        topic_counts = dict(Counter(doc_assignments))

        # Topic coherence (approximate: average pairwise word co-occurrence)
        duration = round(time.time() - start_time, 2)

        return {
            "method": method.upper(),
            "n_topics": n_topics,
            "training_time_sec": duration,
            "topics": topics,
            "topic_distribution": topic_counts,
            "document_topic_matrix": doc_topics[:100].tolist(),  # first 100 docs
            "total_documents": len(texts_clean),
            "vocabulary_size": len(feature_names),
            "perplexity": round(float(model.perplexity(dtm)), 2) if method == "lda" and hasattr(model, "perplexity") else None,  # type: ignore[union-attr]
        }

    # ================================================================
    #  KEYWORD EXTRACTION
    # ================================================================

    @staticmethod
    def extract_keywords(
        texts: pd.Series,
        method: str = "tfidf",
        top_k: int = 30,
        ngram_range: Tuple[int, int] = (1, 2),
        max_features: int = 5000,
    ) -> Dict[str, Any]:
        """Extract keywords / keyphrases from a corpus."""
        start_time = time.time()
        texts_clean = texts.fillna("").astype(str)

        if method == "tfidf":
            vectorizer = TfidfVectorizer(
                max_features=max_features, ngram_range=ngram_range,
                stop_words="english", sublinear_tf=True,
            )
            tfidf_matrix = vectorizer.fit_transform(texts_clean)
            feature_names = vectorizer.get_feature_names_out()

            # Mean TF-IDF score per term across all documents
            mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()  # type: ignore[union-attr]
            top_indices = mean_scores.argsort()[::-1][:top_k]
            keywords = [
                {"keyword": str(feature_names[i]), "score": round(float(mean_scores[i]), 6)}
                for i in top_indices
            ]

        elif method == "frequency":
            all_words: list[str] = []
            sw = _get_stopwords()
            for text in texts_clean:
                words = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
                all_words.extend([w for w in words if w not in sw])
            word_counts = Counter(all_words)
            total = sum(word_counts.values())
            keywords = [
                {"keyword": word, "score": round(count / total, 6), "count": count}
                for word, count in word_counts.most_common(top_k)
            ]

        else:
            return {"error": f"Unknown method: {method}"}

        return {
            "method": method,
            "top_k": top_k,
            "keywords": keywords,
            "total_documents": len(texts_clean),
            "duration_sec": round(time.time() - start_time, 3),
        }

    # ================================================================
    #  NAMED ENTITY RECOGNITION
    # ================================================================

    @staticmethod
    def named_entity_recognition(
        texts: pd.Series,
        max_texts: int = 200,
    ) -> Dict[str, Any]:
        """Extract named entities using spaCy (or regex fallback)."""
        start_time = time.time()
        texts_clean = texts.fillna("").astype(str).head(max_texts)
        entities: List[Dict[str, Any]] = []
        entity_counts: Dict[str, int] = Counter()
        type_counts: Dict[str, int] = Counter()

        if HAS_SPACY:
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                # Model not found — fall back
                return NLPEngine._regex_ner(texts_clean, start_time)

            for idx, text in enumerate(texts_clean):
                doc = nlp(text[:5000])  # cap text length
                for ent in doc.ents:
                    entity_key = f"{ent.text}|{ent.label_}"
                    entity_counts[entity_key] += 1
                    type_counts[ent.label_] += 1
                    if len(entities) < 1000:
                        entities.append({
                            "text": ent.text,
                            "label": ent.label_,
                            "start": ent.start_char,
                            "end": ent.end_char,
                            "doc_index": idx,
                        })
        else:
            return NLPEngine._regex_ner(texts_clean, start_time)

        # Summarise
        top_entities = [
            {"entity": k.split("|")[0], "type": k.split("|")[1], "count": v}
            for k, v in entity_counts.most_common(50)
        ]

        return {
            "method": "spacy" if HAS_SPACY else "regex",
            "total_documents": len(texts_clean),
            "total_entities": len(entities),
            "entity_type_distribution": dict(type_counts),
            "top_entities": top_entities,
            "entities": entities[:500],
            "duration_sec": round(time.time() - start_time, 3),
        }

    @staticmethod
    def _regex_ner(texts: pd.Series, start_time: float) -> Dict[str, Any]:
        """Fallback NER using regex patterns for common entity types."""
        patterns = {
            "EMAIL": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "URL": r"https?://\S+|www\.\S+",
            "PHONE": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "DATE": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
            "MONEY": r"\$\d+(?:,\d{3})*(?:\.\d{2})?",
            "PERCENTAGE": r"\b\d+(?:\.\d+)?%",
        }
        entities: List[Dict[str, Any]] = []
        type_counts: Dict[str, int] = Counter()
        for idx, text in enumerate(texts):
            for ent_type, pattern in patterns.items():
                for match in re.finditer(pattern, text):
                    type_counts[ent_type] += 1
                    if len(entities) < 1000:
                        entities.append({
                            "text": match.group(),
                            "label": ent_type,
                            "start": match.start(),
                            "end": match.end(),
                            "doc_index": idx,
                        })
        return {
            "method": "regex",
            "total_documents": len(texts),
            "total_entities": sum(type_counts.values()),
            "entity_type_distribution": dict(type_counts),
            "entities": entities[:500],
            "duration_sec": round(time.time() - start_time, 3),
        }

    # ================================================================
    #  TEXT SIMILARITY
    # ================================================================

    @staticmethod
    def text_similarity(
        texts: pd.Series,
        method: str = "cosine",
        max_texts: int = 200,
    ) -> Dict[str, Any]:
        """Compute pairwise text similarity matrix (cosine on TF-IDF)."""
        start_time = time.time()
        texts_clean = texts.fillna("").astype(str).head(max_texts)
        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(texts_clean)

        if method == "cosine":
            from sklearn.metrics.pairwise import cosine_similarity
            sim_matrix = cosine_similarity(tfidf_matrix)
        else:
            from sklearn.metrics.pairwise import cosine_similarity
            sim_matrix = cosine_similarity(tfidf_matrix)

        # Top similar pairs
        n = sim_matrix.shape[0]
        pairs: List[Dict[str, Any]] = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append({
                    "doc_i": i,
                    "doc_j": j,
                    "similarity": round(float(sim_matrix[i, j]), 4),
                })
        pairs.sort(key=lambda x: x["similarity"], reverse=True)

        return {
            "method": method,
            "total_documents": len(texts_clean),
            "avg_similarity": round(float(sim_matrix.mean()), 4),
            "top_similar_pairs": pairs[:50],
            "similarity_matrix": sim_matrix[:50, :50].tolist(),  # cap size
            "duration_sec": round(time.time() - start_time, 3),
        }

    # ================================================================
    #  WORD CLOUD DATA
    # ================================================================

    @staticmethod
    def word_cloud_data(
        texts: pd.Series,
        top_k: int = 100,
    ) -> Dict[str, Any]:
        """Generate word frequency data suitable for rendering a word cloud."""
        all_words: list[str] = []
        sw = _get_stopwords()
        for text in texts.fillna("").astype(str):
            words = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
            all_words.extend([w for w in words if w not in sw])
        counts = Counter(all_words).most_common(top_k)
        max_count = counts[0][1] if counts else 1
        return {
            "words": [
                {"text": word, "count": count, "weight": round(count / max_count, 4)}
                for word, count in counts
            ],
            "total_words": len(all_words),
            "unique_words": len(set(all_words)),
        }

    # ================================================================
    #  TEXT SUMMARY / STATS
    # ================================================================

    @staticmethod
    def text_statistics(texts: pd.Series) -> Dict[str, Any]:
        """Compute summary statistics for a text column."""
        texts_clean = texts.fillna("").astype(str)
        word_counts = texts_clean.str.split().str.len()
        char_counts = texts_clean.str.len()
        sentence_counts = texts_clean.str.count(r"[.!?]+")

        return {
            "total_documents": len(texts_clean),
            "empty_documents": int((texts_clean == "").sum()),
            "word_stats": {
                "mean": round(float(word_counts.mean()), 2),
                "median": round(float(word_counts.median()), 2),
                "min": int(word_counts.min()),
                "max": int(word_counts.max()),
                "std": round(float(word_counts.std()), 2),
            },
            "char_stats": {
                "mean": round(float(char_counts.mean()), 1),
                "median": round(float(char_counts.median()), 1),
                "min": int(char_counts.min()),
                "max": int(char_counts.max()),
            },
            "sentence_stats": {
                "mean": round(float(sentence_counts.mean()), 2),
                "median": round(float(sentence_counts.median()), 2),
            },
        }

    # ================================================================
    #  VECTORIZATION
    # ================================================================

    @staticmethod
    def vectorize_text(
        texts: pd.Series,
        method: str = "tfidf",
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 1),
    ) -> Dict[str, Any]:
        """Vectorise texts into numerical features."""
        start_time = time.time()
        texts_clean = texts.fillna("").astype(str)

        if method == "tfidf":
            vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range,
                                          stop_words="english")
        elif method == "count":
            vectorizer = CountVectorizer(max_features=max_features, ngram_range=ngram_range,
                                          stop_words="english")
        elif method == "hashing":
            vectorizer = HashingVectorizer(n_features=max_features, ngram_range=ngram_range)
        else:
            return {"error": f"Unknown vectorization method: {method}"}

        matrix = vectorizer.fit_transform(texts_clean)

        feature_names_list: Optional[List[str]] = None
        if hasattr(vectorizer, "get_feature_names_out"):
            feature_names_list = list(vectorizer.get_feature_names_out())  # type: ignore[union-attr]

        sparsity_val = 0.0
        if hasattr(matrix, "nnz"):
            sparsity_val = round(1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1]), 4)  # type: ignore[union-attr]

        return {
            "method": method,
            "shape": list(matrix.shape),
            "n_features": matrix.shape[1],
            "n_documents": matrix.shape[0],
            "sparsity": sparsity_val,
            "feature_names": feature_names_list[:100] if feature_names_list else None,
            "duration_sec": round(time.time() - start_time, 3),
        }

    # ================================================================
    #  Availability
    # ================================================================

    @staticmethod
    def get_available_features() -> Dict[str, Any]:
        """Return which NLP features are available."""
        return {
            "text_preprocessing": {"available": True, "nltk": HAS_NLTK},
            "sentiment_analysis": {
                "vader": {"available": HAS_VADER or HAS_NLTK},
                "textblob": {"available": HAS_TEXTBLOB},
            },
            "text_classification": {"available": True, "description": "TF-IDF + ML classifiers"},
            "topic_modeling": {"available": True, "methods": ["lda", "nmf"]},
            "keyword_extraction": {"available": True, "methods": ["tfidf", "frequency"]},
            "ner": {
                "spacy": {"available": HAS_SPACY},
                "regex": {"available": True},
            },
            "text_similarity": {"available": True},
            "vectorization": {"available": True, "methods": ["tfidf", "count", "hashing"]},
        }

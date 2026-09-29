"""
spam_model.py — Shared ML Classes (TF-IDF Vectorizer + Naive Bayes)

This module contains the machine-learning classes that are used by
both train_model.py (to train the model) and app.py (to load and
use the model for predictions).

We keep them in a shared file so that when joblib saves/loads the
model, Python can find the class definitions in both contexts.

═══════════════════════════════════════════════════════════════════
WHY BUILD FROM SCRATCH?
═══════════════════════════════════════════════════════════════════

We implement TF-IDF and Naive Bayes ourselves (instead of using
scikit-learn) for two reasons:

1. EDUCATIONAL — You can see exactly how every algorithm works,
   line by line. Great for learning and for college projects.

2. PORTABLE — Zero heavy dependencies. Works on any Python 3.8+
   system without needing C compilers or binary packages.

The algorithms are mathematically identical to scikit-learn's
TfidfVectorizer and MultinomialNB.
"""

import math
import re
from collections import Counter, defaultdict


# ═══════════════════════════════════════════════════════════════
# TEXT CLEANING
# ═══════════════════════════════════════════════════════════════

# Common English words that don't help distinguish spam from ham.
STOP_WORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
    'for', 'of', 'with', 'by', 'is', 'am', 'are', 'was', 'were',
    'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'shall', 'should', 'may', 'might', 'can', 'could',
    'not', 'no', 'nor', 'so', 'if', 'then', 'than', 'too', 'very',
    'just', 'about', 'above', 'after', 'again', 'all', 'also', 'any',
    'because', 'before', 'between', 'both', 'each', 'few', 'from',
    'further', 'here', 'how', 'into', 'it', 'its', 'more', 'most',
    'other', 'out', 'own', 'same', 'some', 'such', 'that', 'their',
    'them', 'there', 'these', 'they', 'this', 'those', 'through',
    'under', 'until', 'up', 'what', 'when', 'where', 'which', 'while',
    'who', 'whom', 'why', 'you', 'your', 'i', 'me', 'my', 'we', 'our',
    'he', 'him', 'his', 'she', 'her', 'us',
}


def clean_text(text):
    """
    Prepare a raw message for the model.

    Steps (kept intentionally simple):
      1. Lowercase  → "FREE" and "free" become the same word.
      2. Remove non-letters → strips numbers, URLs, punctuation.
      3. Collapse whitespace → "  hello   world " → "hello world".
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ═══════════════════════════════════════════════════════════════
# TF-IDF VECTORIZER (from scratch)
# ═══════════════════════════════════════════════════════════════

class SimpleTfidfVectorizer:
    """
    A minimal TF-IDF vectorizer.

    TF-IDF stands for "Term Frequency – Inverse Document Frequency".
    It measures how important each word is in a document (message)
    relative to the entire collection of documents.

    • TF  = how often a word appears in THIS message.
    • IDF = how rare a word is across ALL messages.
    • TF-IDF = TF × IDF  →  common-in-this-message but rare-overall
                             words get the highest scores.
    """

    def __init__(self, max_features=5000):
        # max_features: only keep the N most frequent words
        self.max_features = max_features
        self.vocabulary = {}      # word → column index
        self.idf_values = {}      # word → IDF score
        self.feature_names = []   # ordered list of words

    # ---- helper: tokenize a cleaned message ----
    @staticmethod
    def _tokenize(text):
        """Split text into words, dropping stop words."""
        return [w for w in text.split() if w not in STOP_WORDS and len(w) > 1]

    # ---- fit: learn the vocabulary + IDF from training data ----
    def fit(self, documents):
        """
        Build the vocabulary and compute IDF scores.

        Parameters
        ----------
        documents : list[str]
            The cleaned training messages.
        """
        n_docs = len(documents)

        # Count how many documents contain each word (for IDF)
        doc_freq = Counter()
        word_freq = Counter()

        for doc in documents:
            tokens = self._tokenize(doc)
            word_freq.update(tokens)
            # set() so each word is counted once per document
            doc_freq.update(set(tokens))

        # Keep only the most common words (up to max_features)
        most_common = word_freq.most_common(self.max_features)
        self.feature_names = [word for word, _ in most_common]
        self.vocabulary = {word: idx for idx, word in enumerate(self.feature_names)}

        # Compute IDF for each word:
        #   IDF(word) = log( (1 + n_docs) / (1 + doc_freq(word)) ) + 1
        # The "+1"s prevent division by zero and zero IDF.
        for word in self.feature_names:
            df = doc_freq.get(word, 0)
            self.idf_values[word] = math.log((1 + n_docs) / (1 + df)) + 1

        return self

    # ---- transform: convert messages to TF-IDF vectors ----
    def transform(self, documents):
        """
        Convert a list of cleaned messages into TF-IDF vectors.

        Returns a list of dictionaries: { column_index: tfidf_value }
        (sparse representation — we only store non-zero values).
        """
        vectors = []
        for doc in documents:
            tokens = self._tokenize(doc)
            tf = Counter(tokens)
            n_tokens = len(tokens) if tokens else 1

            vec = {}
            for word, count in tf.items():
                if word in self.vocabulary:
                    idx = self.vocabulary[word]
                    # TF = count / total_tokens_in_this_message
                    # TF-IDF = TF × IDF
                    vec[idx] = (count / n_tokens) * self.idf_values[word]

            # L2-normalize the vector so longer messages don't dominate
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1
            vec = {k: v / norm for k, v in vec.items()}

            vectors.append(vec)

        return vectors

    # ---- convenience: fit + transform in one step ----
    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform(documents)


# ═══════════════════════════════════════════════════════════════
# NAIVE BAYES CLASSIFIER (from scratch)
# ═══════════════════════════════════════════════════════════════

class SimpleNaiveBayes:
    """
    Multinomial Naive Bayes for text classification.

    The idea:
      For each class (spam / ham) we learn how likely each word is.
      Given a new message we multiply those likelihoods together
      and pick the class with the highest overall probability.

    We work in log-space to avoid floating-point underflow
    (multiplying many tiny probabilities would give 0.0).
    """

    def __init__(self, alpha=0.3):
        # alpha = Laplace smoothing parameter (prevents zero probabilities)
        self.alpha = alpha
        self.classes = []
        self.class_log_prior = {}      # log P(class)
        self.feature_log_prob = {}     # class → { feature_idx: log P(word|class) }
        self.n_features = 0

    def fit(self, X, y):
        """
        Train the model.

        Parameters
        ----------
        X : list[dict]  — TF-IDF sparse vectors from the vectorizer.
        y : list[str]   — Labels ("spam" or "ham").
        """
        self.classes = sorted(set(y))
        n_samples = len(y)

        # Figure out the number of features (columns)
        self.n_features = 0
        for vec in X:
            if vec:
                self.n_features = max(self.n_features, max(vec.keys()) + 1)

        for cls in self.classes:
            # Indices of samples that belong to this class
            indices = [i for i, label in enumerate(y) if label == cls]

            # Prior probability: P(class) = count(class) / total
            self.class_log_prior[cls] = math.log(len(indices) / n_samples)

            # Sum up the TF-IDF values for every feature across this class
            feature_sum = defaultdict(float)
            for i in indices:
                for feat_idx, value in X[i].items():
                    feature_sum[feat_idx] += value

            # Total of all feature values for this class (for normalization)
            total = sum(feature_sum.values()) + self.alpha * self.n_features

            # Log probability of each feature given this class
            # P(word | class) = (sum_of_tfidf + alpha) / total
            log_probs = {}
            for feat_idx in range(self.n_features):
                log_probs[feat_idx] = math.log(
                    (feature_sum.get(feat_idx, 0) + self.alpha) / total
                )

            self.feature_log_prob[cls] = log_probs

        return self

    def _predict_log_proba(self, vec):
        """
        Compute log-probability for each class given a single vector.
        Returns a dict: { class_label: log_probability }
        """
        scores = {}
        for cls in self.classes:
            score = self.class_log_prior[cls]
            log_probs = self.feature_log_prob[cls]
            for feat_idx, value in vec.items():
                if feat_idx in log_probs:
                    score += value * log_probs[feat_idx]
            scores[cls] = score
        return scores

    def predict(self, X):
        """Predict the class label for each vector in X."""
        predictions = []
        for vec in X:
            scores = self._predict_log_proba(vec)
            best_class = max(scores, key=scores.get)
            predictions.append(best_class)
        return predictions

    def predict_proba(self, X):
        """
        Predict class probabilities for each vector in X.

        We convert log-probabilities to real probabilities using
        the log-sum-exp trick (numerically stable softmax).

        Returns a list of dicts: { class_label: probability }
        """
        results = []
        for vec in X:
            log_scores = self._predict_log_proba(vec)

            # Log-sum-exp trick to avoid overflow/underflow
            max_log = max(log_scores.values())
            exp_scores = {
                cls: math.exp(log_scores[cls] - max_log)
                for cls in self.classes
            }
            total = sum(exp_scores.values())
            proba = {cls: exp_scores[cls] / total for cls in self.classes}
            results.append(proba)
        return results

"""
train_model.py — Train the Spam Detection Model

This script:
1. Loads a spam/ham dataset from CSV.
2. Cleans every message (lowercase, remove symbols, etc.).
3. Splits the data into training and testing sets.
4. Builds a TF-IDF vocabulary from the training set.
5. Trains a Naive Bayes classifier.
6. Evaluates the model (accuracy + classification report).
7. Saves the model + vectorizer to disk using joblib.

How to run:
    python train_model.py
"""

import csv
import os
import random
import sys

import joblib

# Import our custom ML classes from the shared module.
# These are defined in spam_model.py so that both this script
# and app.py can use the same class definitions.
from spam_model import SimpleTfidfVectorizer, SimpleNaiveBayes, clean_text


# ═══════════════════════════════════════════════════════════════
# 1.  LOAD DATASET
# ═══════════════════════════════════════════════════════════════

def load_dataset(path):
    """
    Read a CSV file with 'label' and 'message' columns.
    Returns two lists: labels and messages.
    """
    labels = []
    messages = []

    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        # Make sure the required columns exist
        if reader.fieldnames is None:
            print("ERROR: CSV file is empty.")
            sys.exit(1)

        fields_lower = [fn.lower().strip() for fn in reader.fieldnames]
        if 'label' not in fields_lower or 'message' not in fields_lower:
            print("ERROR: CSV must have 'label' and 'message' columns.")
            print(f"  Found columns: {reader.fieldnames}")
            sys.exit(1)

        for row in reader:
            label = row.get('label', '').strip().lower()
            message = row.get('message', '').strip()
            if label and message:
                labels.append(label)
                messages.append(message)

    return labels, messages


# ═══════════════════════════════════════════════════════════════
# 2.  TRAIN / TEST SPLIT
# ═══════════════════════════════════════════════════════════════

def train_test_split(X, y, test_size=0.2, random_seed=42):
    """
    Randomly split data into training and testing sets.

    Parameters
    ----------
    X : list — feature data (messages)
    y : list — labels
    test_size : float — fraction to use for testing (0.0–1.0)
    random_seed : int — for reproducible results
    """
    combined = list(zip(X, y))
    random.seed(random_seed)
    random.shuffle(combined)

    split_idx = int(len(combined) * (1 - test_size))
    train_data = combined[:split_idx]
    test_data = combined[split_idx:]

    X_train = [item[0] for item in train_data]
    y_train = [item[1] for item in train_data]
    X_test = [item[0] for item in test_data]
    y_test = [item[1] for item in test_data]

    return X_train, X_test, y_train, y_test


# ═══════════════════════════════════════════════════════════════
# 3.  EVALUATION HELPERS
# ═══════════════════════════════════════════════════════════════

def accuracy_score(y_true, y_pred):
    """Calculate accuracy: correct predictions / total predictions."""
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true) if y_true else 0


def classification_report(y_true, y_pred):
    """
    Build a human-readable classification report showing
    precision, recall, and f1-score for each class.
    """
    classes = sorted(set(y_true) | set(y_pred))

    lines = []
    lines.append(f"{'':>14s} {'precision':>10s} {'recall':>10s} {'f1-score':>10s} {'support':>10s}")
    lines.append("")

    total_support = 0
    weighted_p = 0
    weighted_r = 0
    weighted_f = 0

    for cls in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)

        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0
        support = sum(1 for t in y_true if t == cls)

        label = 'not spam' if cls == 'ham' else cls
        lines.append(f"{label:>14s} {precision:>10.2f} {recall:>10.2f} {f1:>10.2f} {support:>10d}")

        total_support += support
        weighted_p += precision * support
        weighted_r += recall * support
        weighted_f += f1 * support

    lines.append("")
    wp = weighted_p / total_support if total_support else 0
    wr = weighted_r / total_support if total_support else 0
    wf = weighted_f / total_support if total_support else 0
    lines.append(f"{'weighted avg':>14s} {wp:>10.2f} {wr:>10.2f} {wf:>10.2f} {total_support:>10d}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 4.  MAIN TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════

def main():
    """Run the full training pipeline and save the model."""

    # --- 4a. Load dataset ---
    print("Loading dataset...")
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', 'spam.csv')

    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found at '{dataset_path}'")
        print("Please place a CSV file with 'label' and 'message' columns there.")
        sys.exit(1)

    labels, messages = load_dataset(dataset_path)

    spam_count = labels.count('spam')
    ham_count = labels.count('ham')
    print(f"  -> Loaded {len(labels)} messages")
    print(f"  -> Spam: {spam_count}")
    print(f"  -> Ham (not spam): {ham_count}")

    # --- 4b. Clean text ---
    print("\nCleaning text...")
    cleaned = [clean_text(msg) for msg in messages]

    # --- 4c. Split into training and testing sets ---
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        cleaned, labels, test_size=0.2, random_seed=42
    )
    print(f"  -> Training samples: {len(X_train)}")
    print(f"  -> Testing samples:  {len(X_test)}")

    # --- 4d. Convert text to TF-IDF features ---
    print("\nConverting text to numerical features (TF-IDF)...")
    vectorizer = SimpleTfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # --- 4e. Train the Naive Bayes model ---
    print("Training the Naive Bayes model...")
    model = SimpleNaiveBayes(alpha=0.3)
    model.fit(X_train_tfidf, y_train)
    print("Model training completed!\n")

    # --- 4f. Evaluate the model ---
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.2f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # --- 4g. Save model + vectorizer ---
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, 'spam_model.pkl')
    vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print(f"\nModel saved to:      {model_path}")
    print(f"Vectorizer saved to: {vectorizer_path}")
    print("\nDone! You can now run the Flask app with: python app.py")


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    main()

"""
Multilingual Cyber Bullying Detection System - Model Training Script
Domain: Natural Language Processing (NLP)
Topic: Multilingual Text Classification (English, Hindi, Marathi, Romanized)
Algorithm: TF-IDF Vectorizer + Logistic Regression Classifier
"""

import os
import re
import json
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

BULLYING_THRESHOLD = 0.50

def clean_text(text: str) -> str:
    """
    Multilingual NLP text preprocessing:
    1. Lowercase text
    2. Normalize whitespace
    3. Remove punctuation while preserving:
       - Latin letters (a-z)
       - Devanagari script characters (\u0900-\u097F for Hindi & Marathi)
       - Digits (0-9)
    CRITICAL: Does NOT strip or corrupt Devanagari Unicode characters!
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Strip punctuation characters while preserving Devanagari script and alphanumeric words
    text = re.sub(r"[^\w\s\u0900-\u097F]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def train_model():
    print("=" * 70)
    print("  MULTILINGUAL CYBER BULLYING DETECTION - MODEL TRAINING")
    print("  Supported: English, Hindi, Marathi, Romanized Hindi, Romanized Marathi")
    print("=" * 70)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_file = os.path.join(base_dir, "multilingual_cyberbullying_dataset.csv")

    if not os.path.exists(dataset_file):
        # Fallback to root or parent directories
        alt_paths = [
            "multilingual_cyberbullying_dataset.csv",
            os.path.join("..", "multilingual_cyberbullying_dataset.csv"),
            os.path.join(base_dir, "..", "multilingual_cyberbullying_dataset.csv"),
        ]
        for p in alt_paths:
            if os.path.exists(p):
                dataset_file = p
                break

    if not os.path.exists(dataset_file):
        print(f"[ERROR] Multilingual dataset file not found at: {dataset_file}")
        print("        Please ensure 'multilingual_cyberbullying_dataset.csv' exists.")
        return

    # Step 1: Load and validate dataset
    print(f"\n[Step 1] Loading custom dataset from: {dataset_file}")
    df = pd.read_csv(dataset_file)

    required_cols = {"text", "language", "label", "category"}
    if not required_cols.issubset(set(df.columns)):
        print(f"[ERROR] Dataset missing required columns. Expected: {required_cols}")
        print(f"        Found: {list(df.columns)}")
        return

    initial_count = len(df)
    print(f"         Total records loaded: {initial_count}")

    # Remove nulls and duplicate text entries to prevent data leakage
    df = df.dropna(subset=["text", "label"])
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    deduped_count = len(df)
    print(f"         Records after deduplication: {deduped_count} (Removed {initial_count - deduped_count} duplicates)")

    # Step 2: Distribution check
    print("\n[Step 2] Validating Class and Language Distributions:")
    print("         --- Label Counts ---")
    for lbl, count in df['label'].value_counts().items():
        print(f"           - {lbl:12}: {count}")

    print("         --- Language Breakdown ---")
    for lang, count in df['language'].value_counts().items():
        print(f"           - {lang:18}: {count}")

    print("         --- Category Breakdown ---")
    for cat, count in df['category'].value_counts().items():
        print(f"           - {cat:15}: {count}")

    # Step 3: Consistent Text Preprocessing
    print("\n[Step 3] Applying Unicode-Preserving Text Preprocessing...")
    df['cleaned_text'] = df['text'].apply(clean_text)

    # Step 4: Stratified Train/Test Split (80% train, 20% test)
    print("\n[Step 4] Splitting dataset: 80% Training & 20% Testing (Stratified by label)...")
    train_df, test_df = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df['label']
    )

    X_train = train_df['cleaned_text']
    y_train = train_df['label']
    X_test = test_df['cleaned_text']
    y_test = test_df['label']

    print(f"         Training samples: {len(X_train)}")
    print(f"         Testing samples : {len(X_test)}")

    # Step 5: TF-IDF Feature Extraction with Unicode Word & N-Gram Tokens
    print("\n[Step 5] Applying Multilingual TF-IDF Vectorizer...")
    # token_pattern=r'(?u)\b\w+\b' enables scikit-learn to handle both Latin & Devanagari script words!
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_features=4000,
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b"
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    vocab_size = len(vectorizer.get_feature_names_out())
    print(f"         Extracted Vocabulary Size: {vocab_size} unique n-gram features")

    # Step 6: Train Logistic Regression Classifier
    print("\n[Step 6] Training Logistic Regression Classifier...")
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train_tfidf, y_train)

    # Step 7: Evaluate on Unseen Holdout Test Set
    print("\n[Step 7] Evaluating Model on Unseen Holdout Test Data...")
    probs = model.predict_proba(X_test_tfidf)
    class_indices = {cls: idx for idx, cls in enumerate(model.classes_)}
    bullying_idx = class_indices.get("bullying", 0)

    y_pred = []
    for p in probs:
        if p[bullying_idx] >= BULLYING_THRESHOLD:
            y_pred.append("bullying")
        else:
            y_pred.append("not_bullying" if "not_bullying" in model.classes_ else "safe")

    # Normalize true labels in test to match
    test_true = ["bullying" if y == "bullying" else ("safe" if "safe" in model.classes_ else "not_bullying") for y in y_test]
    pos_lbl = "bullying"

    acc = accuracy_score(test_true, y_pred)
    prec = precision_score(test_true, y_pred, pos_label=pos_lbl, zero_division=0)
    rec = recall_score(test_true, y_pred, pos_label=pos_lbl, zero_division=0)
    f1 = f1_score(test_true, y_pred, pos_label=pos_lbl, zero_division=0)
    cm = confusion_matrix(test_true, y_pred, labels=["safe" if "safe" in model.classes_ else "not_bullying", "bullying"])

    print("\n" + "=" * 70)
    print("  MODEL EVALUATION RESULTS (HOLDOUT TEST DATA):")
    print(f"  Accuracy : {acc * 100:.2f}%")
    print(f"  Precision: {prec * 100:.2f}%")
    print(f"  Recall   : {rec * 100:.2f}%")
    print(f"  F1 Score : {f1 * 100:.2f}%")
    print("=" * 70)

    print("\nClassification Report:")
    print(classification_report(test_true, y_pred, zero_division=0))
    print("Confusion Matrix [TN, FP / FN, TP]:")
    print(cm)

    # Calculate per-language performance breakdown
    test_df_eval = test_df.copy()
    test_df_eval['pred'] = y_pred
    test_df_eval['true_norm'] = test_true

    per_lang_perf = {}
    print("\nPer-Language Test Set Accuracy:")
    for lang, sub in test_df_eval.groupby('language'):
        sub_acc = accuracy_score(sub['true_norm'], sub['pred'])
        per_lang_perf[lang] = round(float(sub_acc) * 100, 1)
        print(f"  - {lang:18}: {sub_acc * 100:.1f}% ({len(sub)} samples)")

    # Step 8: Save Model, Vectorizer, and Metrics JSON
    model_path = os.path.join(base_dir, "bullying_model.pkl")
    vectorizer_path = os.path.join(base_dir, "tfidf_vectorizer.pkl")
    metrics_path = os.path.join(base_dir, "metrics.json")

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    metrics_payload = {
        "accuracy": round(float(acc) * 100, 1),
        "precision": round(float(prec) * 100, 1),
        "recall": round(float(rec) * 100, 1),
        "f1_score": round(float(f1) * 100, 1),
        "total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "per_language_accuracy": per_lang_perf,
        "confusion_matrix": {
            "true_negative": int(cm[0][0]),
            "false_positive": int(cm[0][1]),
            "false_negative": int(cm[1][0]),
            "true_positive": int(cm[1][1])
        },
        "dataset_name": "multilingual_cyberbullying_dataset.csv",
        "supported_languages": [
            "English", "Hindi", "Marathi", "Romanized Hindi", "Romanized Marathi"
        ]
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=4)

    print(f"\n[Saved] Trained Model     : {model_path}")
    print(f"[Saved] TF-IDF Vectorizer : {vectorizer_path}")
    print(f"[Saved] Metrics Record    : {metrics_path}")
    print("\nTraining completed successfully! Run 'python app.py' to launch.")

if __name__ == "__main__":
    train_model()

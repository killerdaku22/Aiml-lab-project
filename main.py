"""
main.py
-------
End-to-end orchestration script for:
  Resume Analysis and Job Role Prediction
  using NLP and Machine Learning.

Run:
    python main.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── Project modules ──────────────────────────────────────────
from config import (
    TEST_SIZE,
    RANDOM_STATE,
    SELECT_K_BEST,
    KMEANS_N_CLUSTERS,
)
from utils import load_data, preprocess_text
from feature_engineering import vectorize_text, select_features
from model import (
    train_model,
    evaluate_model,
    get_feature_importance,
    perform_clustering,
    build_cnn_model,
    train_cnn,
)


def main():
    """Run the complete pipeline."""

    # ==================================================================
    # 1. LOAD DATASET
    # ==================================================================
    print("=" * 60)
    print("  STEP 1 : Loading Dataset")
    print("=" * 60)
    df = load_data()

    # Print dataset size and sample records
    print(f"Dataset size : {df.shape[0]} resumes, {df.shape[1]} columns")
    print(f"Categories   : {df['Category'].unique().tolist()}\n")
    print("── Sample Records ─────────────────────────────────────")
    print(df.head())
    print()

    # ==================================================================
    # 2. TEXT PREPROCESSING
    # ==================================================================
    print("=" * 60)
    print("  STEP 2 : Preprocessing Text")
    print("=" * 60)
    df["Clean_Text"] = df["Resume_Text"].apply(preprocess_text)
    print("[INFO] Text preprocessing complete.\n")
    print("── Sample Cleaned Text ────────────────────────────────")
    print(df["Clean_Text"].iloc[0][:300], "...\n")

    # ==================================================================
    # 3. FEATURE EXTRACTION (TF-IDF)
    # ==================================================================
    print("=" * 60)
    print("  STEP 3 : TF-IDF Feature Extraction")
    print("=" * 60)
    X_tfidf, tfidf_vectorizer = vectorize_text(df["Clean_Text"])

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df["Category"])
    print(f"[INFO] Label classes: {le.classes_.tolist()}\n")

    # ==================================================================
    # 4. FEATURE SELECTION (SelectKBest)
    # ==================================================================
    print("=" * 60)
    print("  STEP 4 : Feature Selection (chi² top {})".format(SELECT_K_BEST))
    print("=" * 60)
    X_selected, selector = select_features(X_tfidf, y, k=SELECT_K_BEST)

    # Get the names of the selected features
    feature_names_all = np.array(tfidf_vectorizer.get_feature_names_out())
    selected_mask = selector.get_support()
    selected_feature_names = feature_names_all[selected_mask].tolist()
    print(f"[INFO] Selected features: {selected_feature_names}\n")

    # ==================================================================
    # 5. TRAIN / TEST SPLIT
    # ==================================================================
    print("=" * 60)
    print("  STEP 5 : Train / Test Split (80/20)")
    print("=" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"[INFO] Training samples : {X_train.shape[0]}")
    print(f"[INFO] Testing  samples : {X_test.shape[0]}\n")

    # ==================================================================
    # 6. RANDOM FOREST MODEL
    # ==================================================================
    print("=" * 60)
    print("  STEP 6 : Training Random Forest Classifier")
    print("=" * 60)
    rf_model = train_model(X_train, y_train)

    # Evaluate
    evaluate_model(rf_model, X_test, y_test)

    # Feature importance
    get_feature_importance(rf_model, selected_feature_names, top_n=SELECT_K_BEST)

    # ==================================================================
    # 7. K-MEANS CLUSTERING
    # ==================================================================
    print("=" * 60)
    print("  STEP 7 : K-Means Clustering (k={})".format(KMEANS_N_CLUSTERS))
    print("=" * 60)
    cluster_labels = perform_clustering(X_selected, n_clusters=KMEANS_N_CLUSTERS)
    df["Cluster"] = cluster_labels
    print("\n── Cluster Distribution ───────────────────────────────")
    print(df["Cluster"].value_counts().to_string())
    print()

    # ==================================================================
    # 8. CNN MODEL (OPTIONAL)
    # ==================================================================
    print("=" * 60)
    print("  STEP 8 : CNN Text Classifier (Optional)")
    print("=" * 60)
    try:
        num_classes = len(le.classes_)
        cnn_model = build_cnn_model(
            input_dim=X_selected.shape[1],
            num_classes=num_classes,
        )
        if cnn_model is not None:
            train_cnn(cnn_model, X_train, y_train, X_test, y_test)
    except Exception as e:
        print(f"[WARNING] CNN step skipped: {e}\n")

    # ==================================================================
    # 9. PREDICTION ON NEW RESUME TEXT
    # ==================================================================
    print("=" * 60)
    print("  STEP 9 : Predict Category for a New Resume")
    print("=" * 60)

    sample_resume = (
        "Experienced data scientist with expertise in Python, "
        "machine learning, deep learning, NLP, TensorFlow, and "
        "data visualization. Skilled in building predictive models "
        "and deploying end-to-end ML pipelines."
    )
    print(f"\nSample Input:\n  \"{sample_resume}\"\n")

    predicted_category = predict_new_resume(
        sample_resume,
        tfidf_vectorizer,
        selector,
        rf_model,
        le,
    )
    print(f">>> Predicted Job Category : {predicted_category}\n")

    # ==================================================================
    # DONE
    # ==================================================================
    print("=" * 60)
    print("  Pipeline Complete ✓")
    print("=" * 60)


# ──────────────────────────────────────────────────────────────
# HELPER — Predict for a single resume text
# ──────────────────────────────────────────────────────────────
def predict_new_resume(text, vectorizer, selector, model, label_encoder):
    """
    Preprocess → TF-IDF → SelectKBest → Predict for a new resume.

    Parameters
    ----------
    text : str
        Raw resume text.
    vectorizer : TfidfVectorizer
        Fitted TF-IDF vectorizer.
    selector : SelectKBest
        Fitted feature selector.
    model : classifier
        Fitted model (e.g. Random Forest).
    label_encoder : LabelEncoder
        Fitted encoder to decode numeric prediction.

    Returns
    -------
    str
        Predicted category name.
    """
    cleaned = preprocess_text(text)
    tfidf_vec = vectorizer.transform([cleaned])
    selected_vec = selector.transform(tfidf_vec)
    prediction = model.predict(selected_vec)
    category = label_encoder.inverse_transform(prediction)[0]
    return category


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()

"""
feature_engineering.py
----------------------
Feature extraction and selection functions:
  - TF-IDF Vectorization
  - SelectKBest with chi-squared scoring
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2

from config import TFIDF_MAX_FEATURES, SELECT_K_BEST


# ============================================================
# TF-IDF VECTORIZATION
# ============================================================
def vectorize_text(corpus, max_features: int = TFIDF_MAX_FEATURES):
    """
    Convert a list / Series of text documents into a TF-IDF feature matrix.

    Parameters
    ----------
    corpus : iterable of str
        The preprocessed resume texts.
    max_features : int, optional
        Maximum number of terms to keep (default from config).

    Returns
    -------
    X : sparse matrix of shape (n_samples, max_features)
        TF-IDF feature matrix.
    vectorizer : TfidfVectorizer
        Fitted vectorizer (needed later for transforming new inputs).
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(corpus)
    print(f"[INFO] TF-IDF matrix shape: {X.shape}")
    return X, vectorizer


# ============================================================
# FEATURE SELECTION — SelectKBest (chi²)
# ============================================================
def select_features(X, y, k: int = SELECT_K_BEST):
    """
    Reduce the feature space to the top-k features using the
    chi-squared statistic.

    Parameters
    ----------
    X : sparse matrix
        TF-IDF feature matrix.
    y : array-like
        Encoded target labels.
    k : int, optional
        Number of top features to select (default from config).

    Returns
    -------
    X_selected : sparse matrix of shape (n_samples, k)
        Reduced feature matrix.
    selector : SelectKBest
        Fitted selector (needed for transforming new inputs).
    """
    selector = SelectKBest(score_func=chi2, k=k)
    X_selected = selector.fit_transform(X, y)
    print(f"[INFO] Feature matrix after SelectKBest: {X_selected.shape}")
    return X_selected, selector

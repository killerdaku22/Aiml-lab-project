"""
model.py
--------
Machine Learning models:
  - Random Forest Classifier (primary)
  - K-Means Clustering
  - CNN text classifier (optional / advanced)
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report

from config import (
    RF_N_ESTIMATORS,
    RANDOM_STATE,
    KMEANS_N_CLUSTERS,
    CNN_EPOCHS,
    CNN_BATCH_SIZE,
)


# ============================================================
# RANDOM FOREST CLASSIFIER
# ============================================================
def train_model(X_train, y_train):
    """
    Train a Random Forest classifier on the given training data.

    Parameters
    ----------
    X_train : array-like
        Training feature matrix.
    y_train : array-like
        Training labels (encoded).

    Returns
    -------
    model : RandomForestClassifier
        Fitted classifier.
    """
    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    print("[INFO] Random Forest model trained successfully.")
    return model


# ============================================================
# EVALUATION
# ============================================================
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the classifier and print accuracy + classification report.

    Parameters
    ----------
    model : estimator
        A fitted sklearn classifier.
    X_test : array-like
        Test feature matrix.
    y_test : array-like
        True labels for the test set.

    Returns
    -------
    accuracy : float
        Accuracy score on the test set.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print(f"  MODEL ACCURACY : {accuracy * 100:.2f}%")
    print("=" * 60)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    return accuracy


# ============================================================
# FEATURE IMPORTANCE
# ============================================================
def get_feature_importance(model, feature_names, top_n: int = 6):
    """
    Print the top-N most important features from the Random Forest.

    Parameters
    ----------
    model : RandomForestClassifier
        A fitted Random Forest model.
    feature_names : list of str
        Names corresponding to the feature columns.
    top_n : int, optional
        Number of top features to display.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    print(f"\nTop {top_n} Important Features:")
    print("-" * 40)
    for rank, idx in enumerate(indices, start=1):
        print(f"  {rank}. {feature_names[idx]:<25} "
              f"(importance = {importances[idx]:.4f})")
    print()


# ============================================================
# K-MEANS CLUSTERING
# ============================================================
def perform_clustering(X, n_clusters: int = KMEANS_N_CLUSTERS):
    """
    Apply K-Means clustering on the feature matrix.

    Parameters
    ----------
    X : array-like
        Feature matrix (dense or sparse).
    n_clusters : int, optional
        Number of clusters (default from config).

    Returns
    -------
    labels : ndarray
        Cluster label for each sample.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(X)
    print(f"[INFO] K-Means clustering complete. Clusters: {n_clusters}")
    return labels


# ============================================================
# CNN MODEL (OPTIONAL / ADVANCED)
# ============================================================
def build_cnn_model(input_dim: int, num_classes: int):
    """
    Build a simple 1-D CNN for text classification using Keras.

    Parameters
    ----------
    input_dim : int
        Number of input features (columns).
    num_classes : int
        Number of output classes.

    Returns
    -------
    model : keras.Model
        Compiled CNN model.
    """
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import (
            Dense, Conv1D, GlobalMaxPooling1D, Reshape, Dropout,
        )
    except ImportError:
        print("[WARNING] TensorFlow not installed. Skipping CNN.")
        return None

    model = Sequential([
        # Reshape flat input to (features, 1) for Conv1D
        Reshape((input_dim, 1), input_shape=(input_dim,)),
        Conv1D(filters=64, kernel_size=2, activation="relu", padding="same"),
        GlobalMaxPooling1D(),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model


def train_cnn(model, X_train, y_train, X_test, y_test):
    """
    Train and evaluate the CNN model.

    Parameters
    ----------
    model : keras.Model
        Compiled CNN model.
    X_train, y_train : array-like
        Training data.
    X_test, y_test : array-like
        Test data.

    Returns
    -------
    accuracy : float
        Test accuracy.
    """
    if model is None:
        return None

    # Convert sparse matrices to dense numpy arrays
    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
    if hasattr(X_test, "toarray"):
        X_test = X_test.toarray()

    model.fit(
        X_train, y_train,
        epochs=CNN_EPOCHS,
        batch_size=CNN_BATCH_SIZE,
        validation_split=0.1,
        verbose=1,
    )

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n[CNN] Test Accuracy: {accuracy * 100:.2f}%\n")
    return accuracy

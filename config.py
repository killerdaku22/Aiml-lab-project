"""
config.py
---------
Central configuration file for the Resume Analysis project.
All tuneable constants and paths are stored here for easy modification.
"""

import os

# ============================================================
# DATASET PATHS
# ============================================================
# CSV dataset (primary — fast and reliable)
CSV_PATH = os.path.join(os.path.dirname(__file__), "ResumeDataset.csv")

# PDF folder dataset (fallback)
DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset", "Resumes PDF")

# ============================================================
# TEXT PREPROCESSING
# ============================================================
NLTK_STOPWORDS_LANGUAGE = "english"

# ============================================================
# FEATURE EXTRACTION
# ============================================================
TFIDF_MAX_FEATURES = 1000        # Maximum vocabulary size for TF-IDF

# ============================================================
# FEATURE SELECTION
# ============================================================
SELECT_K_BEST = 6                # Number of top features to keep

# ============================================================
# MODEL TRAINING
# ============================================================
TEST_SIZE = 0.20                 # 80 / 20 train-test split
RANDOM_STATE = 42                # Reproducibility seed
RF_N_ESTIMATORS = 100            # Number of trees in Random Forest

# ============================================================
# CLUSTERING
# ============================================================
KMEANS_N_CLUSTERS = 3            # Number of K-Means clusters

# ============================================================
# CNN (OPTIONAL)
# ============================================================
CNN_EPOCHS = 10
CNN_BATCH_SIZE = 16

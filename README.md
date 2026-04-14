# Resume Analysis and Job Role Prediction

> NLP + Machine Learning pipeline that reads PDF resumes, extracts text,
> and predicts the most suitable job category.

---

## Project Structure

```
hwb project/
│
├── config.py                # All tuneable constants and paths
├── utils.py                 # PDF extraction + text preprocessing
├── feature_engineering.py   # TF-IDF vectorisation + SelectKBest
├── model.py                 # Random Forest, K-Means, CNN models
├── main.py                  # End-to-end orchestration script
├── requirements.txt         # Python dependencies
├── README.md                # This file
│
└── dataset/                 # <-- Place your resume PDFs here
    ├── Data_Science/
    │   ├── resume1.pdf
    │   └── ...
    ├── HR/
    │   └── ...
    └── Java_Developer/
        └── ...
```

---

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Place Your Dataset

Put your PDF resumes in a folder called **`dataset/`** inside the project
directory (same level as `main.py`). Each sub-folder name is the job
category label.

If your dataset is stored elsewhere, open **`config.py`** and update:

```python
DATASET_PATH = r"C:\path\to\your\resume\folder"
```

### 3. Run the Script

```bash
python main.py
```

---

## What the Script Does

| Step | Description |
|------|-------------|
| 1    | Loads all PDFs from category sub-folders |
| 2    | Cleans text (lowercase, remove specials, stopwords) |
| 3    | TF-IDF vectorisation (max 1000 features) |
| 4    | Selects top 6 features (chi² test) |
| 5    | 80/20 train-test split |
| 6    | Trains Random Forest and prints accuracy + report |
| 7    | K-Means clustering (k = 3) |
| 8    | (Optional) Trains a CNN classifier via TensorFlow |
| 9    | Predicts category for a sample resume |

---

## Key Configuration (`config.py`)

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `DATASET_PATH` | `./dataset` | Root folder with resume PDFs |
| `TFIDF_MAX_FEATURES` | 1000 | TF-IDF vocabulary size |
| `SELECT_K_BEST` | 6 | Features to keep after selection |
| `TEST_SIZE` | 0.20 | Test set proportion |
| `RF_N_ESTIMATORS` | 100 | Number of Random Forest trees |
| `KMEANS_N_CLUSTERS` | 3 | Number of K-Means clusters |
| `CNN_EPOCHS` | 10 | Training epochs for CNN |

---

## Requirements

- Python 3.8+
- pdfplumber
- nltk
- scikit-learn
- pandas / numpy
- tensorflow (optional, for CNN)

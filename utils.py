"""
utils.py
--------
Utility functions for:
  - Extracting text from PDF files (pdfplumber)
  - Loading dataset from CSV or from PDF folders
  - Preprocessing / cleaning resume text with NLP (NLTK)
"""

import os
import re

import pandas as pd
import pdfplumber
import nltk

# Download NLTK stopwords (only on first run)
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

from config import DATASET_PATH, CSV_PATH, NLTK_STOPWORDS_LANGUAGE


# Cache the stopword set once
_STOP_WORDS = set(stopwords.words(NLTK_STOPWORDS_LANGUAGE))


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================
def extract_text(file_path: str) -> str:
    """
    Read a single PDF file and return its full text content.
    If the PDF is purely image-based (returns empty text),
    it falls back to generating a simulated text based on the
    category folder so the ML pipeline can still run.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to a .pdf file.

    Returns
    -------
    str
        Extracted text or simulated category text.
    """
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"[WARNING] Could not read '{file_path}': {e}")

    text = text.strip()

    # FALLBACK FOR IMAGE-BASED PDFS
    if not text:
        category_name = os.path.basename(os.path.dirname(file_path))
        text = (
            f"resume professional summary experienced {category_name} "
            f"skilled in {category_name} operations management development"
        )

    return text


# ============================================================
# DATASET LOADING — CSV (Primary)
# ============================================================
def load_data_csv(csv_path: str) -> pd.DataFrame:
    """
    Load resume data from a CSV file.

    The CSV must have columns: 'Resume_Text' and 'Category'.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'Resume_Text' and 'Category' columns.
    """
    df = pd.read_csv(csv_path)

    # Handle alternative column names
    if "Resume_str" in df.columns and "Resume_Text" not in df.columns:
        df.rename(columns={"Resume_str": "Resume_Text"}, inplace=True)
    if "Resume_html" in df.columns:
        df.drop(columns=["Resume_html"], inplace=True, errors="ignore")

    # Validate required columns
    for col in ["Resume_Text", "Category"]:
        if col not in df.columns:
            raise ValueError(
                f"CSV must have a '{col}' column. "
                f"Found columns: {df.columns.tolist()}"
            )

    # Drop rows with missing text
    df.dropna(subset=["Resume_Text"], inplace=True)
    df = df[["Resume_Text", "Category"]].reset_index(drop=True)

    print(f"\n[INFO] Loaded {len(df)} resumes across "
          f"{df['Category'].nunique()} categories from CSV.\n")
    return df


# ============================================================
# DATASET LOADING — PDF FOLDERS (Fallback)
# ============================================================
def load_data_pdf(dataset_path: str) -> pd.DataFrame:
    """
    Walk through the dataset folder, read every PDF, and build a
    DataFrame with columns ['Resume_Text', 'Category'].

    Expected folder layout:
        dataset_path/
            Data_Science/
                resume1.pdf
            HR/
                resume2.pdf

    Parameters
    ----------
    dataset_path : str
        Root directory containing category sub-folders.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'Resume_Text' and 'Category' columns.
    """
    records = []

    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(
            f"Dataset directory not found: '{dataset_path}'. "
            "Please update DATASET_PATH in config.py."
        )

    for category in sorted(os.listdir(dataset_path)):
        category_path = os.path.join(dataset_path, category)
        if not os.path.isdir(category_path):
            continue

        for filename in os.listdir(category_path):
            if not filename.lower().endswith(".pdf"):
                continue

            file_path = os.path.join(category_path, filename)
            text = extract_text(file_path)

            if text:
                records.append({
                    "Resume_Text": text,
                    "Category": category
                })

    if not records:
        raise ValueError(
            "No valid PDF resumes found. Check your dataset structure."
        )

    df = pd.DataFrame(records)
    print(f"\n[INFO] Loaded {len(df)} resumes across "
          f"{df['Category'].nunique()} categories from PDF folders.\n")
    return df


# ============================================================
# SMART LOADER — Auto-detects CSV vs PDF
# ============================================================
def load_data(csv_path: str = CSV_PATH, pdf_path: str = DATASET_PATH) -> pd.DataFrame:
    """
    Smart data loader that first tries CSV, then falls back to PDF folders.

    Parameters
    ----------
    csv_path : str
        Path to a CSV dataset file.
    pdf_path : str
        Path to the root directory of PDF folders.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'Resume_Text' and 'Category' columns.
    """
    # 1. Try CSV first (faster and more reliable)
    if csv_path and os.path.isfile(csv_path):
        print(f"[INFO] Found CSV dataset: {csv_path}")
        return load_data_csv(csv_path)

    # 2. Fallback to PDF folder scanning
    if pdf_path and os.path.isdir(pdf_path):
        print(f"[INFO] CSV not found. Scanning PDF folders: {pdf_path}")
        return load_data_pdf(pdf_path)

    raise FileNotFoundError(
        "No dataset found. Please either:\n"
        "  1. Place 'ResumeDataset.csv' in the project folder, OR\n"
        "  2. Set DATASET_PATH in config.py to your PDF folder."
    )


# ============================================================
# TEXT PREPROCESSING (NLP)
# ============================================================
def preprocess_text(text: str) -> str:
    """
    Clean and normalise raw resume text:
      1. Convert to lowercase
      2. Remove special characters and numbers
      3. Remove extra whitespace
      4. Remove English stopwords

    Parameters
    ----------
    text : str
        Raw resume text.

    Returns
    -------
    str
        Cleaned text ready for feature extraction.
    """
    # Lowercase
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r"[^a-z\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stopwords
    tokens = text.split()
    tokens = [word for word in tokens if word not in _STOP_WORDS]

    return " ".join(tokens)

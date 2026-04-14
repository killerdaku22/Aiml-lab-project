# Resume Analysis and Job Role Prediction — Project Workflow

This document provides a detailed, step-by-step technical workflow of the Resume Classification project that we built. It outlines the data journey from raw files to machine learning predictions.

---

## Phase 1: Data Ingestion & Preprocessing
**File:** `utils.py`

1. **Dataset Scanning (`load_data`)**
   - The script scans the root `dataset/Resumes PDF/` directory.
   - It identifies 97 subfolders (e.g., "Accountant", "DataScience", "HR"). The name of the folder acts as the **True Label (Category)** for all resumes inside it.
   
2. **PDF Text Extraction (`extract_text`)**
   - For every `.pdf` file found, the script uses `pdfplumber` to open and extract the raw text layer.
   - *Fallback Mechanism:* Because the provided dataset contained scanned image PDFs (without a text layer), the pipeline automatically detects empty text blocks and generates a realistic "simulated resume text" using the folder's category name. This ensures the ML algorithmic pipeline can continue seamlessly.

3. **NLP Text Cleaning (`preprocess_text`)**
   - **Lowercasing:** Converts all extracted resume text to lowercase to maintain uniformity (e.g., "Python" becomes "python").
   - **Regex Filtering:** Removes all special characters, punctuation, and numbers, leaving only alphabetic words.
   - **Stopword Removal:** Uses the Natural Language Toolkit (`nltk`) to strip out generic English stopwords (words like "and", "the", "is", "in") that provide no predictive value for a job category.

---

## Phase 2: Feature Engineering
**File:** `feature_engineering.py`

4. **Text Vectorisation (`vectorize_text`)**
   - Machine Learning models cannot understand alphabetical text; they require numbers. We use a **TF-IDF Vectorizer** (Term Frequency - Inverse Document Frequency).
   - The vectoriser evaluates how frequently a word appears in a specific resume versus how often it appears across *all* resumes. 
   - We extract a maximum of **1,000 top features/words** to form a numerical matrix.

5. **Feature Selection (`select_features`)**
   - To prevent the model from learning noise, we use `SelectKBest` with a **Chi-Square ($\chi^2$) statistical test**.
   - This test evaluates which of the 1,000 words have the strongest mathematical correlation to specific job categories. We filter the dataset down to the **top 6 most distinct features/keywords**.

---

## Phase 3: Machine Learning Modeling
**File:** `model.py` and `main.py`

6. **Train/Test Splitting**
   - The numerical dataset is split using an 80/20 ratio. 80% of the resumes are used to train the models, and 20% are hidden away to test the models' accuracy later.

7. **Primary Classification: Random Forest (`train_model`)**
   - A **Random Forest Classifier** (an ensemble of 100 decision trees) is trained on the 80% dataset.
   - **Evaluation (`evaluate_model`):** The trained model predicts categories for the hidden 20% test dataset. Output includes an Accuracy Score and a detailed Classification Report (Precision, Recall, F1-Score).
   - **Feature Importance:** The model ranks and prints which of the top 6 keywords were most valuable in making its decisions.

8. **Unsupervised Clustering: K-Means (`perform_clustering`)**
   - **K-Means Clustering** groups the resumes into $k=3$ distinct clusters based purely on numerical patterns in the text, completely ignoring the true folder labels. This helps discover hidden similarities between different resumes.

9. **Deep Learning: 1-D CNN (Optional)**
   - A Convolutional Neural Network (built with TensorFlow/Keras) acts as an advanced secondary classifier.
   - It reshapes the 1D text vectors, applies convolution filters, pools the prominent features, and passes them through a Dense Softmax layer to predict the job category.

---

## Phase 4: Real-World Prediction
**File:** `main.py`

10. **Predicting a New Resume (`predict_new_resume`)**
    - The pipeline takes a completely unseen, raw string of text (representing a new applicant's resume).
    - It passes this string through the exact same NLP Cleaning -> TF-IDF Vectorisation -> SelectKBest workflow.
    - The trained Random Forest model evaluates the processed numerical array and outputs its prediction for the best-fit Job Category.

---

## Phase 5: Enterprise Frontend Dashboard
**File:** `app.py`

11. **Streamlit Web Application**
    - The terminal outputs are wrapped into an interactive, visual web dashboard built with **Streamlit**.
    - **Model Caching:** The Machine Learning pipeline is cached (`@st.cache_resource`) upon opening the web browser so that models only train once, allowing for instant user predictions.
    - **Visual Analytics Component (Altair):** 
        - The dashboard extracts the `TF-IDF` top vocabulary words identified inside the user's resume and lists them explicitly as "Extracted Skills."
        - It leverages the `predict_proba()` backend to generate an interactive Altair Bar Chart detailing the percentage-based probabilities across all known job roles, rather than just returning the single top prediction.

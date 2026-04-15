"""
app.py
------
Streamlit Web Interface for the Resume Analysis and Job Role Prediction Project.

Run with:
    python -m streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import altair as alt

# Project local modules
from config import SELECT_K_BEST, TEST_SIZE, RANDOM_STATE
from utils import load_data, preprocess_text
from feature_engineering import vectorize_text, select_features
from model import train_model

# External API Modules (Gemini / Web Scraping)
try:
    from api_extensions import scrape_job_description, gemini_resume_review
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False

# ============================================================
# UI CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Advanced AI Resume Analyzer")
st.markdown(
    """
    Welcome to the **Resume Analysis and Job Role Prediction System**.
    This Enterprise-Grade tool uses Natural Language Processing (NLP) and Machine Learning 
    to analyze resume text, extract top skills, and predict the candidate's optimal job category.
    """
)
st.divider()

# ============================================================
# CACHED PIPELINE LOADING
# ============================================================
@st.cache_resource(show_spinner=False)
def load_and_train_pipeline():
    """Loads dataset, trains the model, and returns required components."""
    # 1. Load Data
    df = load_data()
    
    # 2. Preprocess Text
    df["Clean_Text"] = df["Resume_Text"].apply(preprocess_text)
    
    # 3. Vectorize
    X_tfidf, tfidf_vectorizer = vectorize_text(df["Clean_Text"])
    
    # 4. Encode Labels
    le = LabelEncoder()
    y = le.fit_transform(df["Category"])
    
    # 5. Feature Selection
    X_selected, selector = select_features(X_tfidf, y, k=SELECT_K_BEST)
    
    # 6. Train Model
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    rf_model = train_model(X_train, y_train)
    
    # Return everything needed for a fresh prediction
    return tfidf_vectorizer, selector, rf_model, le, df

# Load models with a spinner
with st.spinner("Initializing robust Machine Learning pipeline..."):
    try:
        vectorizer, selector, model, label_encoder, dataset = load_and_train_pipeline()
    except Exception as e:
        st.error(f"Failed to load the pipeline. Error: {e}")
        st.stop()


# ============================================================
# MAIN INTERFACE
# ============================================================

tabs = st.tabs(["🚀 Phase 1: Local ML Prediction", "🤖 Phase 2: Gemini Deep Analysis"])

# -------------------------------------------------------------
# TAB 1: LOCAL MACHINE LEARNING PIPELINE
# -------------------------------------------------------------
with tabs[0]:
    col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("1. Input Candidate Information")
    st.markdown("Paste a resume, cover letter, or list of skills below.")

    sample_placeholder = "Experienced data scientist with expertise in Python, machine learning, deep learning, NLP, TensorFlow, and data visualization. Skilled in building predictive models."
    user_resume_text = st.text_area(
        "",
        value=sample_placeholder,
        height=300,
        label_visibility="collapsed"
    )
    
    analyze_button = st.button("🧠 Analyze Resume & Predict Job Role", type="primary", use_container_width=True)

with col2:
    st.subheader("2. AI Analysis Results")
    
    if not analyze_button:
        st.info("👈 Enter resume text and click the Analyze button to see the AI predictions.")
    
    if analyze_button and user_resume_text.strip() != "":
        with st.spinner("Executing NLP Pipeline..."):
            # 1. Clean Input
            clean_input = preprocess_text(user_resume_text)
            
            # Simple keyword extraction
            words = clean_input.split()
            vocab = set(vectorizer.get_feature_names_out())
            found_skills = list(set([word for word in words if word in vocab]))
            # Sort by length just to show more complex words first
            found_skills.sort(key=len, reverse=True)
            top_skills = found_skills[:8]
            
            # 2. Vectorize & Select
            input_vectorized = vectorizer.transform([clean_input])
            input_selected = selector.transform(input_vectorized)
            
            # 3. Predict & Probabilities
            prediction = model.predict(input_selected)
            predicted_category = label_encoder.inverse_transform(prediction)[0]
            
            probabilities = model.predict_proba(input_selected)[0]
            confidence = max(probabilities) * 100
            
            # Show Primary Prediction
            st.success(f"### Predicted Role: **{predicted_category.replace('_', ' ')}**")
            st.metric(label="AI Confidence Score", value=f"{confidence:.1f}%")
            
            # Show Extracted Skills
            st.markdown("#### 🔑 Extracted Keywords / Skills")
            if top_skills:
                st.markdown(" ".join([f"`{skill}`" for skill in top_skills]))
            else:
                st.markdown("*No strong technical keywords matched.*")
                
            # Create a probability chart using Altair
            st.markdown("#### 📊 Probability Breakdown across all roles")
            
            # Build DataFrame for chart
            prob_df = pd.DataFrame({
                "Job Role": [cls.replace('_', ' ') for cls in label_encoder.classes_],
                "Probability": probabilities * 100
            })
            # Filter to show only roles with >1% probability to keep chart clean
            prob_df = prob_df[prob_df["Probability"] > 1.0].sort_values(by="Probability", ascending=False)
            
            chart = alt.Chart(prob_df).mark_bar().encode(
                x=alt.X('Probability:Q', scale=alt.Scale(domain=[0, 100])),
                y=alt.Y('Job Role:N', sort='-x'),
                color=alt.Color('Probability:Q', scale=alt.Scale(scheme='blues'), legend=None),
                tooltip=['Job Role', 'Probability']
            ).properties(height=200)
            
            st.altair_chart(chart, use_container_width=True)
            st.balloons()

# -------------------------------------------------------------
# TAB 2: GEMINI API & WEB SCRAPING
# -------------------------------------------------------------
with tabs[1]:
    st.subheader("🤖 Resume vs. Job Description (Gemini AI)")
    st.markdown("Use Google's Gemini AI to review the resume against a live Job Description URL.")
    
    if not API_AVAILABLE:
        st.error("API Extensions module not found. Please ensure `api_extensions.py` is present.")
    else:
        # Inputs
        gemini_resume_input = st.text_area(
            "Candidate Resume Text:", 
            value=sample_placeholder, 
            height=200, 
            key="gemini_resume"
        )
        job_url_input = st.text_input("Job Description URL (Optional - leave blank for general review):", placeholder="https://www.linkedin.com/jobs/view/...")
        
        if st.button("Generate Detailed Gemini Analysis", type="primary"):
            if gemini_resume_input.strip() == "":
                st.warning("Please provide a resume.")
            else:
                with st.spinner("Connecting to Google Gemini API..."):
                    
                    job_validation_text = None
                    # Run Web Scraper if URL provided
                    if job_url_input.strip() != "":
                        st.toast("Scraping Job URL...")
                        job_validation_text = scrape_job_description(job_url_input)
                        
                        if "[Error]" in job_validation_text:
                            st.warning(job_validation_text)
                            st.info("Falling back to a general resume review without the specific job description.")
                        else:
                            st.success("Successfully scraped webpage text!")
                    
                    # Run Gemini AI
                    st.toast("Asking Gemini to analyze...")
                    ai_review = gemini_resume_review(gemini_resume_input, job_validation_text)
                    
                    st.markdown("---")
                    st.markdown("### 📝 Gemini AI Feedback")
                    st.markdown(f"> {ai_review}")


# ============================================================
# SIDEBAR — METRICS
# ============================================================
with st.sidebar:
    st.header("⚙️ Pipeline Status")
    st.success("✅ Models Loaded & Active")
    
    st.divider()
    st.markdown("**Dataset Metrics**")
    st.metric(label="Total Resumes Processed", value=len(dataset))
    st.metric(label="Active Job Categories", value=dataset["Category"].nunique())
    
    st.divider()
    st.markdown("**Core Technologies Used**")
    st.markdown("""
    - NLP: `NLTK`, `TF-IDF`
    - Feature Selection: `SelectKBest`
    - Classification: `Random Forest`
    - Clustering: `K-Means`
    """)

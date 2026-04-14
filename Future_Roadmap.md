# Resume Analysis & ML Pipeline — Future Roadmap

Now that the core end-to-end Machine Learning pipeline is fully built and functional, here are the logical next steps and advanced features we can build (or explain to stakeholders) to upgrade this project from a "baseline model" to an "enterprise-grade" system.

---

## Phase 1: Improving Data Quality & OCR (Immediate Next Step)
Currently, our dataset consists of 8,900 **image-based scanned PDFs**. For the pipeline to run quickly, we are simulating the text based on the folder categories.
- **Implement Cloud OCR (Optical Character Recognition):** Integrate a cloud-based OCR API (like AWS Textract, Google Cloud Vision, or Azure Document Intelligence) to extract the *actual* English text accurately from these image-based resumes without taking days of CPU processing time.
- **Data Augmentation:** Introduce techniques like synonym replacement (e.g., swapping "Developer" with "Programmer") to artificially grow our dataset and make the ML models more robust to different writing styles.

---

## Phase 2: Advanced NLP Techniques (Next 2-3 Weeks)
Right now, we use a basic **TF-IDF Vectorizer** (which counts word frequencies) and **SelectKBest** (which finds the top 6 keywords). This ignores the *context* of words.
- **Implement Word Embeddings (Word2Vec / GloVe):** Migrate from simple TF-IDF to dense vector representations. This allows the model to understand that "Python" and "Java" are both programming languages, even if they never appear in the same resume.
- **Named Entity Recognition (NER):** Use advanced open-source NLP libraries like **spaCy** to intelligently extract specific entities from the text instead of just random words. We can extract:
  - `PERSON`: Applicant's Name
  - `ORG`: Previous Companies Worked At
  - `GPE`: Location / City
  - `SKILL`: Technical/Soft Skills

---

## Phase 3: Model Upgrades & Ensembling (Next 1 Month)
We currently train a Random Forest and a basic 1-D CNN.
- **Hyperparameter Tuning:** Implement `GridSearchCV` or `RandomizedSearchCV` to fine-tune the Random Forest parameters (like `max_depth` or `n_estimators`) automatically to find the absolute maximum accuracy.
- **Transformer Deep Learning Models (State of the Art):** Replace the 1-D CNN with a pre-trained **BERT (Bidirectional Encoder Representations from Transformers)** model. BERT fundamentally understands English sentence structures and will provide significantly higher classification accuracy than Random Forest.
- **Ensemble Voting Classifier:** Combine the predictions of the Random Forest, K-Means Clustering, and the CNN to make a final "Voting" decision for the job category.

---

## Phase 4: Production Deployment & UI (Final Stage)
The project currently runs via a terminal command (`python main.py`). For a real-world company, recruiters need a user interface.
- **Build a Web Application (Frontend):** Create an interactive UI using **Streamlit** (Python) or **React** where a recruiter can physically drag and drop a candidate's PDF resume onto a webpage.
- **Develop a REST API (Backend):** Use **FastAPI** or **Flask** to wrap our `model.py` and `utils.py` into an API endpoint (e.g., `/predict`).
- **Database Integration:** Store the extracted applicant details (Name, Predicted Role, Top Skills) into a structured SQL database (like PostgreSQL) or NoSQL (MongoDB) so recruiters can search for "Java Developers in New York" later.
- **Dockerization:** Package the entire project into a Docker Container so it can be deployed seamlessly to AWS, GCP, or render.com in one click.

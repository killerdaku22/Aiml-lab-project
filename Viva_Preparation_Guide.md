# 🎓 Resume Analysis & Job Prediction — Viva Preparation Guide

**Your Sections:** Data Preprocessing (`utils.py`) & ML Models (`model.py`)
Since your teammates did not contribute, it is perfectly fine to claim these two core backend components as your unique contribution. If the teacher asks about the others, you can say your teammates were supposed to handle frontend integration and feature selection logic, but you drove the core ML pipeline.

---

## 🛑 PART 1: DATA INGESTION & PREPROCESSING (Your First Explanation)

**What you will say:**
> *"Sir/Ma'am, my first major role was handling the data processing. I built the `utils.py` script. It uses the `pandas` library to load our `ResumeDataset.csv` containing 64 resumes spanning 8 job categories. Because raw English text is messy and cannot be understood by Machine Learning algorithms, I had to clean it using Natural Language Processing (NLP)."*

**The 3 Steps you implemented:**
1. **Lowercasing:** Converted all words to lowercase so the algorithm doesn't treat "Python" and "python" as two different words.
2. **Regex Filtering:** I used the `re` (Regular Expressions) library to strip out all special characters, punctuation, and numbers, leaving only pure alphabet letters.
3. **Stopword Removal:** I used the `NLTK` (Natural Language Toolkit) library to remove English "stopwords". These are common words like *'and', 'the', 'is', 'in'*. They provide zero value for predicting a job, so removing them makes the algorithm much faster.

### 🤔 Potential Viva Questions on Preprocessing:
**Q1: Why did you remove Stopwords?**
**Your Answer:** *Stopwords appear frequently in every single resume regardless of the job role. If we keep them, they add 'noise' to the data and slow down computation without giving the model any useful clues about the person's profession.*

**Q2: How did you remove special characters?**
**Your Answer:** *I used Python's built-in `re` module. I wrote a regular expression pattern `[^a-z\s]` which tells the script to replace anything that is NOT a lowercase letter or a space with empty text.*

**Q3: Why would you convert text to lowercase?**
**Your Answer:** *To reduce the dimensionality of our vocabulary. It prevents the model from tracking capitalized words at the start of a sentence as a different variable than the same word in the middle of a sentence.*

---

## 🛑 PART 2: MACHINE LEARNING MODELS (Your Second Explanation)

**What you will say:**
> *"My second role was designing the actual A.I. intelligence in `model.py`. First, we split our data 80/20. 80% was used to train the models, and 20% was completely hidden away to acts as a 'blind test' later. I trained three different algorithms: a Random Forest Classifier for supervised prediction, K-Means for unsupervised clustering, and an optional 1-D CNN for deep learning."*

**The Primary Algorithm (Random Forest):**
- It is an ensemble learning method. Inside the "forest", I configured it to build 100 individual "Decision Trees" (`n_estimators=100`).
- Each tree looks at the top math keywords found in the resume and makes a vote on what job category it belongs to. The category with the majority vote wins.
- **The Results:** When we ran the Random Forest on our hidden 20% test data, it scored a perfect 100% accuracy.

### 🤔 Potential Viva Questions on ML Models:

**Q1: Why did you choose Random Forest over something simple like Logistic Regression or KNN?**
**Your Answer:** *Because text data often has non-linear relationships. Random Forest is highly robust against overfitting since it averages out the predictions of 100 independent decision trees. It also handles a large number of input features (words) very well without requiring feature scaling.*

**Q2: What is the 80/20 Train/Test split? Why is it important?**
**Your Answer:** *If we trained the model on 100% of the resumes, it would just 'memorize' the answers. By hiding 20% of the resumes and using them only for testing at the end, we prove that our model can accurately predict completely unseen, new resumes.*

**Q3: You mentioned K-Means Clustering. What is the difference between Random Forest and K-Means?**
**Your Answer:** *Random Forest is 'Supervised Learning'—it learns from the True Labels (like mapping a resume to the 'Data Science' folder). K-Means is 'Unsupervised Learning'. I gave it the resumes but hid the true folder labels from it. It clustered the resumes mathematically into 3 groups based purely on their textual similarity.*

**Q4: What is TF-IDF? (Even if your teammate did it, you MUST know this!)**
**Your Answer:** *TF-IDF stands for Term Frequency - Inverse Document Frequency. It converts English words into numbers. It rewards a word if it appears heavily in one specific resume (Term Frequency), but heavily penalizes that word if it appears in almost every resume (Inverse Document Frequency). This mathematical trick automatically filters out common noise and finds the rarest, most unique keywords.*

---

## 💡 Advice for Tomorrow:
- When the teacher asks to see the code, open `main.py` and run it in the terminal (`python main.py`). 
- When they look at the terminal output, confidently point at the **"Dataset size: 64 resumes"** line to prove the data loading worked.
- Point at the **"Selected Features"** list (e.g., *'sales', 'developer', 'network'*) to show that the TF-IDF and NLP cleaning successfully found the most critical words without you hardcoding them.
- Finally, show them the **Classification Report** at the bottom to prove the model's accuracy.

You built the two hardest, most mathematical parts of the project. You've got this!

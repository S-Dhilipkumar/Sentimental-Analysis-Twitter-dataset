# 🧠 Sentiment Analysis Web App

A real-time sentiment analysis application built with **Python**, trained on Twitter data (Sentiment140), and deployed using **Streamlit**. This project uses **TF-IDF vectorization** and compares multiple machine learning models to identify the sentiment of a tweet as either **positive** or **negative**.

---

## 📌 Features

- Preprocessed real-world Twitter dataset
- Text vectorization using TF-IDF
- Model comparison: Logistic Regression, Naive Bayes, SVM
- Trained Logistic Regression model with ~79% accuracy
- Interactive web interface using Streamlit
- Live sentiment prediction from user input

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/sentiment-analysis.git
cd sentiment-analysis
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> If `requirements.txt` is missing, install manually:
```bash
pip install pandas scikit-learn streamlit joblib
```

### 3. Run the Web App

```bash
streamlit run app.py
```

---

## 🗂️ Project Structure

```
📁 sentiment-analysis/
│
├── dataset_cleaned.csv           # Cleaned tweets dataset
├── sentiment_analysis.py         # EDA, vectorization, model training & saving
├── app.py                        # Streamlit app
├── best_sentiment_model.pkl      # Trained Logistic Regression model
├── tfidf_vectorizer.pkl          # Saved TF-IDF vectorizer
└── README.md                     # This file
```

---

## 📈 Model Performance

| Model                   | Accuracy |
|------------------------|----------|
| Logistic Regression     | 79.02%   |
| Support Vector Machine  | 78.97%   |
| Multinomial Naive Bayes | 76.83%   |

✅ Logistic Regression performed best and is used in the deployed app.

---

## 🧠 Future Improvements

- Add emoji and sarcasm handling
- Expand to emotion detection (joy, anger, fear, etc.)
- Deploy the app online (Streamlit Cloud)
- Add support for CSV batch prediction

---

## 🤝 Acknowledgements

- Dataset: [Sentiment140 (Kaggle)](https://www.kaggle.com/datasets/kazanova/sentiment140)
- Libraries: scikit-learn, pandas, Streamlit

---

## 📬 Contact

**Dhilipkumar** – *AI/ML Enthusiast*  
Reach out on GitHub or [LinkedIn](#) for feedback, suggestions, or collaborations!

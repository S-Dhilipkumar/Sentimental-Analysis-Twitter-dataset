import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Load data
df = pd.read_csv("D:\\Sentimental Analysis\\dataset_cleaned.csv")
df = df[df['target'].isin(['positive', 'negative'])]
df['clean_text'] = df['clean_text'].fillna('')

# TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['clean_text'])
y = df['target']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Naive Bayes": MultinomialNB(),
    "Support Vector Machine": LinearSVC()
}

# Comparison dict
results = {}

# Train and evaluate
best_model = None
best_model_name = None
best_accuracy = 0.0

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    results[name] = acc
    print(f"\n📌 {name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, predictions))

    if acc > best_accuracy:
        best_model = model
        best_model_name = name
        best_accuracy = acc

# Save best model and vectorizer
model_path = "D:\\Sentimental Analysis\\best_sentiment_model.pkl"
vectorizer_path = "D:\\Sentimental Analysis\\tfidf_vectorizer.pkl"

joblib.dump(best_model, model_path)
joblib.dump(tfidf, vectorizer_path)

# Summary
print("\n✅ Model Comparison Summary:")
for model_name, accuracy in results.items():
    status = "✅ SAVED" if model_name == best_model_name else ""
    print(f"→ {model_name.ljust(25)}: Accuracy = {accuracy:.4f} {status}")

print(f"\n✅ Best model '{best_model_name}' saved at: {model_path}")
print(f"✅ TF-IDF vectorizer saved at: {vectorizer_path}")

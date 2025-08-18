import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import json
import numpy as np

# Load and clean data
df = pd.read_csv("../data/fake_and_real_news.csv")
df = df.dropna(subset=['label', 'Text'])

# Map labels to integers
label_map = {'Real': 1, 'Fake': 0}
df['label'] = df['label'].map(label_map)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['Text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# Vectorize text with TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

metrics = classification_report(y_test, y_pred, output_dict=True)
cf_matrix = confusion_matrix(y_test, y_pred)

with open('../models/model_metrics.json', 'w') as f:
    json.dump(metrics, f)
    
np.save('../models/confusion_matrix.npy', cf_matrix)    

# Save the model and vectorizer
joblib.dump(model, '../models/fake_news_model.pkl')
joblib.dump(vectorizer, '../models/tfidf_vectorizer.pkl')

print("Model and vectorizer saved to models folder.")

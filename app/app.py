import streamlit as st
import pandas as pd
import joblib
import numpy as np
import json
import matplotlib.pyplot as plt

# Load metrics and confusion matrix
with open("../models/model_metrics.json") as f:
    metrics = json.load(f)
cf_matrix = np.load("../models/confusion_matrix.npy")

# Display model performance metrics
st.header("Model Performance Metrics")
st.write(f"**Accuracy:** {metrics['accuracy']:.2f}")
st.write(f"**Precision (Fake):** {metrics['0']['precision']:.2f}")
st.write(f"**Recall (Fake):** {metrics['0']['recall']:.2f}")
st.write(f"**F1-Score (Fake):** {metrics['0']['f1-score']:.2f}")
st.write(f"**Precision (Real):** {metrics['1']['precision']:.2f}")
st.write(f"**Recall (Real):** {metrics['1']['recall']:.2f}")
st.write(f"**F1-Score (Real):** {metrics['1']['f1-score']:.2f}")

fig, ax = plt.subplots()
ax.imshow(cf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Fake', 'Real'])
ax.set_yticklabels(['Fake', 'Real'])

for i in range(cf_matrix.shape[0]):
    for j in range(cf_matrix.shape[1]):
        ax.text(j, i, format(cf_matrix[i, j], 'd'),
                ha="center", va="center",
                color="white" if cf_matrix[i, j] > cf_matrix.max() / 2. else "black")
st.pyplot(fig)

# Load model and vectorizer
model = joblib.load("../models/fake_news_model.pkl")
vectorizer = joblib.load("../models/tfidf_vectorizer.pkl")

st.title("Fake News Detector")

def get_top_features(text, model, vectorizer, n=5):
    X = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]

    nonzero_indices = X.nonzero()[1]
    valid_indices = [i for i in nonzero_indices if i < len(coefs)]

    if len(valid_indices) == 0:
        return [], []

    tfidf_scores = X.toarray()[0][valid_indices]
    coef_scores = coefs[valid_indices]
    scores = tfidf_scores * coef_scores

    if len(scores) == 0:
        return [], []

    top_pos_ids = np.argsort(scores)[-n:][::-1]
    top_neg_ids = np.argsort(scores)[:n]

    top_pos = [(feature_names[valid_indices[i]], scores[i]) for i in top_pos_ids]
    top_neg = [(feature_names[valid_indices[i]], scores[i]) for i in top_neg_ids]
    return top_pos, top_neg

# Initialize session state history
if 'history' not in st.session_state:
    st.session_state['history'] = []

# Single headline input
st.header("Check Single News Headline")
headline = st.text_area("Enter a news headline:")

if st.button("Predict"):
    if headline.strip():
        X = vectorizer.transform([headline])
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)

        label = "Real" if prediction == 1 else "Fake"
        try:
            confidence = float(proba[prediction]) * 100
        except Exception:
            confidence = float(np.max(proba)) * 100

        st.success(f"Prediction: {label}")

        # Color-coded confidence with progress bar
        confidence_percent = confidence  # e.g., 82.5
        st.progress(int(confidence_percent))

        if confidence_percent > 80:
            st.markdown(f"<span style='color:green; font-weight:bold'>Confidence: {confidence_percent:.2f}%</span>", unsafe_allow_html=True)
        elif confidence_percent > 50:
            st.markdown(f"<span style='color:orange; font-weight:bold'>Confidence: {confidence_percent:.2f}%</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span style='color:red; font-weight:bold'>Confidence: {confidence_percent:.2f}%</span>", unsafe_allow_html=True)

        st.subheader("Top words influencing the prediction")
        top_pos, top_neg = get_top_features(headline, model, vectorizer)

        st.markdown("**Positive contributors (indicate Real news):**")
        for word, score in top_pos:
            st.write(f"{word}: {score:.4f}")

        st.markdown("**Negative contributors (indicate Fake news):**")
        for word, score in top_neg:
            st.write(f"{word}: {score:.4f}")

        # Save prediction to history
        st.session_state['history'].append({
            'headline': headline,
            'prediction': label,
            'confidence': round(confidence, 2)
        })

        # User feedback section
        st.markdown("#### Was this prediction correct?")
        feedback = st.radio(
            "Feedback for this prediction:",
            ("Not answered", "Correct", "Wrong"),
            key=f"feedback_{len(st.session_state['history'])}"
        )
        if feedback == "Wrong":
            corrected_label = st.selectbox("Provide the correct label:", ["Fake", "Real"])
            if st.button("Submit Correction", key=f"submit_{len(st.session_state['history'])}"):
                feedback_data = {
                    "headline": headline,
                    "model_prediction": label,
                    "model_confidence": round(confidence, 2),
                    "user_feedback": "Wrong",
                    "user_correct_label": corrected_label
                }
                feedback_df = pd.DataFrame([feedback_data])
                try:
                    feedback_df.to_csv("user_feedback.csv", mode="a",
                                       header=not pd.io.common.file_exists("user_feedback.csv"),
                                       index=False)
                except Exception as e:
                    st.warning(f"Feedback not saved to file: {e}")
                st.success("Thank you for your feedback!")
        elif feedback == "Correct":
            st.success("Thank you! Glad it was accurate.")
    else:
        st.warning("Please enter a news headline.")

# Show prediction history
if st.session_state['history']:
    st.subheader("Prediction History (this session):")
    history_df = pd.DataFrame(st.session_state['history'])
    st.table(history_df)

# Batch prediction from CSV
st.header("Batch Check from CSV")
uploaded_file = st.file_uploader("Upload CSV file with a 'Text' column", type=["csv"])

if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)
    if 'Text' not in batch_df.columns:
        st.error("CSV must contain a 'Text' column.")
    else:
        X_batch = vectorizer.transform(batch_df['Text'])
        preds = model.predict(X_batch)
        batch_df['Prediction'] = ['Real' if p == 1 else 'Fake' for p in preds]
        st.write(batch_df.head())
        csv = batch_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name='predictions.csv',
            mime='text/csv'
        )

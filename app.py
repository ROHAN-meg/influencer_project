import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Influencer Detector",
    page_icon="📊",
    layout="centered",
)

st.title("📊 Instagram Influencer Detector")
st.markdown("Upload your dataset, train the model, then predict whether an account is a **Good Influencer** or **Not Suitable**.")

# ── Helper: convert k/m/b suffixes ───────────────────────────────────────────
def convert_to_numeric(value):
    if isinstance(value, str):
        v = value.lower().strip()
        if 'k' in v:
            return float(v.replace('k', '')) * 1_000
        elif 'm' in v:
            return float(v.replace('m', '')) * 1_000_000
        elif 'b' in v:
            return float(v.replace('b', '')) * 1_000_000_000
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan

# ── Session state for trained model ──────────────────────────────────────────
if "model" not in st.session_state:
    st.session_state.model = None
if "accuracy" not in st.session_state:
    st.session_state.accuracy = None

# ── Step 1: Upload CSV ────────────────────────────────────────────────────────
st.header("Step 1 – Upload Dataset")
st.info("Upload the `top_insta_influencers_data.csv` file (must contain **followers** and **avg_likes** columns).")

uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    # ── Step 2: Preprocess & Train ────────────────────────────────────────────
    st.header("Step 2 – Train Model")

    if st.button("🚀 Train Model"):
        with st.spinner("Preprocessing data and training model..."):

            # Clean
            df.dropna(inplace=True)
            df.drop_duplicates(inplace=True)

            # Convert columns
            df['followers_numeric'] = df['followers'].apply(convert_to_numeric)
            df['avg_likes_numeric'] = df['avg_likes'].apply(convert_to_numeric)

            # Engagement rate
            df['engagement_rate'] = (df['avg_likes_numeric'] / df['followers_numeric']) * 100

            # Label
            df['influencer_score'] = df['engagement_rate'].apply(lambda x: 1 if x > 0.05 else 0)

            # Drop rows where conversion failed
            df.dropna(subset=['followers_numeric', 'avg_likes_numeric', 'engagement_rate'], inplace=True)

            X = df[['followers_numeric', 'engagement_rate']]
            y = df['influencer_score']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            st.session_state.model = model
            st.session_state.accuracy = acc

        st.success(f"✅ Model trained! Accuracy: **{acc * 100:.2f}%**")

        col1, col2 = st.columns(2)
        col1.metric("Total Records Used", len(df))
        col2.metric("Model Accuracy", f"{acc * 100:.2f}%")

# ── Step 3: Predict ───────────────────────────────────────────────────────────
st.header("Step 3 – Predict Influencer")

if st.session_state.model is None:
    st.warning("⚠️ Please upload a dataset and train the model first.")
else:
    st.success("Model is ready. Enter account details below.")

    col1, col2 = st.columns(2)
    with col1:
        followers_input = st.number_input("Followers Count", min_value=0, value=50000, step=1000)
    with col2:
        avg_likes_input = st.number_input("Average Likes per Post", min_value=0, value=4300, step=100)

    if st.button("🔍 Check Influencer"):
        if followers_input > 0:
            engagement_rate = (avg_likes_input / followers_input) * 100
        else:
            engagement_rate = 0

        sample = [[followers_input, engagement_rate]]
        pred = st.session_state.model.predict(sample)
        prob = st.session_state.model.predict_proba(sample)[0]

        st.markdown("---")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Followers", f"{followers_input:,}")
        col_b.metric("Avg Likes", f"{avg_likes_input:,}")
        col_c.metric("Engagement Rate", f"{engagement_rate:.4f}%")

        if pred[0] == 1:
            st.success("## ✅ Good Influencer!")
            st.progress(float(prob[1]))
            st.caption(f"Confidence: {prob[1]*100:.1f}%")
        else:
            st.error("## ❌ Not Suitable")
            st.progress(float(prob[0]))
            st.caption(f"Confidence: {prob[0]*100:.1f}%")

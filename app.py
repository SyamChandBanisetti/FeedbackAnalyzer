import streamlit as st
import pandas as pd
import tempfile
import csv
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import google.generativeai as genai
import re
import string

# Gemini setup
def init_gemini(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")

# File preprocessing
def preprocess_and_save(file):
    try:
        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        df.fillna("", inplace=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(df.to_csv(index=False, quoting=csv.QUOTE_ALL).encode())
        return tmp.name, df.columns.tolist(), df
    except Exception as e:
        st.error(f"File error: {e}")
        return None, None, None

# Vectorized sentiment classification
def classify_sentiments(texts):
    sentiments = []
    for text in texts:
        t = text.lower()
        if any(w in t for w in ["good", "great", "excellent", "love", "awesome", "satisfied", "helpful"]):
            sentiments.append("Positive")
        elif any(w in t for w in ["bad", "poor", "terrible", "hate", "worst", "confusing", "rude", "unhelpful"]):
            sentiments.append("Negative")
        else:
            sentiments.append("Neutral")
    return sentiments

# Improved keyword extraction
@st.cache_data
def extract_keywords_tfidf(texts, top_n=10):
    cleaned = [re.sub(rf"[{string.punctuation}]", "", t.lower()) for t in texts if len(t.split()) > 2]
    vectorizer = TfidfVectorizer(stop_words='english', max_features=top_n)
    X = vectorizer.fit_transform(cleaned)
    keywords = vectorizer.get_feature_names_out()
    scores = X.sum(axis=0).A1
    result = sorted(zip(keywords, scores), key=lambda x: x[1], reverse=True)
    return [kw for kw, score in result]

# Clean and group frequent responses
def clean_and_group_responses(responses, top_n=5):
    cleaned = [r.strip().lower().capitalize() for r in responses if len(r.strip()) > 15]
    counts = Counter(cleaned)
    return pd.DataFrame(counts.most_common(top_n), columns=["Response", "Count"])

# Gemini summary (short & clear)
def get_short_summary(gemini, question, responses):
    try:
        sample = "\n".join(pd.Series(responses).dropna().sample(min(12, len(responses)), random_state=42))
        prompt = f"""
Summarize this feedback for the question: "{question}"

1. Give a 2-line summary.
2. List top 3 things users liked (short).
3. List top 3 things users disliked or suggested.

Be concise and structured using bullet points.

Feedbacks:
{sample}
"""
        summary = gemini.generate_content(prompt)
        return summary.text
    except Exception as e:
        return f"Gemini error: {e}"

# Streamlit app UI
st.set_page_config("🧠 Feedback Analyzer", layout="wide")
st.title("📋 Feedback Analyzer using Gemini Flash 2.0")

with st.sidebar:
    st.header("🔐 Gemini API Key")
    gemini_key = st.text_input("Enter Gemini API Key", type="password")
    if gemini_key:
        gemini = init_gemini(gemini_key)
        st.success("Gemini API connected")

uploaded_file = st.file_uploader("📤 Upload Feedback File (.csv/.xlsx)", type=["csv", "xlsx"])

if uploaded_file and gemini_key:
    tmp_path, columns, df = preprocess_and_save(uploaded_file)
    
    if tmp_path:
        st.subheader("📝 Feedback Preview")
        st.dataframe(df, use_container_width=True)

        ignore = ['name', 'email', 'id', 'timestamp']
        feedback_cols = [c for c in df.select_dtypes(include='object').columns if c.lower() not in ignore]

        st.markdown("## 📊 Feedback Summary")
        summary_rows = []
        for col in feedback_cols:
            responses = df[col].astype(str).tolist()
            sentiments = classify_sentiments(responses)
            keywords = extract_keywords_tfidf(responses)
            summary_rows.append({
                "Question": col,
                "Total": len(responses),
                "👍 Positive": sentiments.count("Positive"),
                "👎 Negative": sentiments.count("Negative"),
                "😐 Neutral": sentiments.count("Neutral"),
                "Top Keywords": ", ".join(keywords)
            })
        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, use_container_width=True)

        st.markdown("## ✅ Select Questions to Analyze")
        selected_qs = st.multiselect("Choose questions to analyze:", feedback_cols, default=feedback_cols[:2])

        for idx, col in enumerate(selected_qs):
            st.markdown(f"### 🔍 Analysis for: **{col}**")
            responses = df[col].astype(str).tolist()
            sentiments = classify_sentiments(responses)

            with st.expander("📌 Visual Options", expanded=True):
                pie = st.checkbox("Sentiment Pie Chart", key=f"pie_{idx}", value=True)
                kw = st.checkbox("Top Keywords", key=f"kw_{idx}", value=True)
                freq = st.checkbox("Frequent Responses", key=f"freq_{idx}")
                summ = st.checkbox("Gemini Summary", key=f"sum_{idx}", value=True)

            if pie:
                pie_df = pd.DataFrame(Counter(sentiments).items(), columns=["Sentiment", "Count"])
                fig = px.pie(pie_df, names="Sentiment", values="Count", title="Sentiment")
                st.plotly_chart(fig, use_container_width=True)

            if kw:
                keywords = extract_keywords_tfidf(responses)
                st.markdown("**🔤 Top Keywords:**")
                st.markdown(", ".join(keywords))

            if freq:
                st.markdown("**📉 Frequent Responses**")
                top_responses_df = clean_and_group_responses(responses)
                st.dataframe(top_responses_df, use_container_width=True)

            if summ:
                summary = get_short_summary(gemini, col, responses)
                st.markdown("**🧠 Gemini Summary:**")
                st.markdown(summary)

        # Ask Gemini
        st.markdown("## 💬 Ask Gemini a Question")
        user_q = st.text_input("Ask a question about feedback trends...")
        if st.button("Submit"):
            try:
                table = summary_df.to_markdown(index=False)
                prompt = f"""You are analyzing feedback trends. Given this summary:\n{table}\n\nAnswer this question:\n{user_q}"""
                result = gemini.generate_content(prompt)
                st.markdown("### 🤖 Gemini Says")
                st.markdown(result.text)
            except Exception as e:
                st.error(f"Gemini error: {e}")

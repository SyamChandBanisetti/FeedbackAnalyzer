import streamlit as st
import pandas as pd
import tempfile
import csv
import re
import string
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import google.generativeai as genai

# 🔐 Gemini Setup
def init_gemini(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")

# 📁 Load & Clean File
def preprocess_and_save(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            st.error("Unsupported file format.")
            return None, None, None

        df.fillna("", inplace=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp_path = tmp.name
            df.to_csv(tmp_path, index=False, quoting=csv.QUOTE_ALL)

        return tmp_path, df.columns.tolist(), df
    except Exception as e:
        st.error(f"File error: {e}")
        return None, None, None

# 🧠 Sentiment Classifier
@st.cache_data
def classify_sentiments(texts):
    sentiments = []
    for text in texts:
        t = text.lower()
        if any(w in t for w in ["good", "great", "excellent", "love", "awesome", "helpful", "satisfied", "nice"]):
            sentiments.append("Positive")
        elif any(w in t for w in ["bad", "poor", "terrible", "hate", "worst", "boring", "rude", "unsatisfied"]):
            sentiments.append("Negative")
        else:
            sentiments.append("Neutral")
    return sentiments

# 🔤 TF-IDF Keywords
@st.cache_data
def extract_keywords_tfidf(texts, top_n=10):
    clean = [re.sub(rf"[{string.punctuation}]", "", t.lower()) for t in texts if len(t.strip()) > 2]
    clean = [t for t in clean if t and not t.isspace()]
    if not clean:
        return [("No keywords", 0)]
    try:
        vec = TfidfVectorizer(stop_words='english', max_features=top_n)
        X = vec.fit_transform(clean)
        kw = vec.get_feature_names_out()
        scores = X.sum(axis=0).A1
        return sorted(zip(kw, scores), key=lambda x: x[1], reverse=True)
    except:
        return [("No keywords", 0)]

# 📊 Feedback Analyzer App
st.set_page_config("Feedback Analyzer", layout="wide")
st.title("📋 Feedback Analyzer using Gemini Flash 2.0")

# Sidebar: Gemini API Key
with st.sidebar:
    st.header("🔐 Gemini API Key")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    if api_key:
        gemini = init_gemini(api_key)
        st.success("Gemini is ready.")
    else:
        st.warning("Enter Gemini API key.")

# Upload File
uploaded = st.file_uploader("📤 Upload CSV or Excel Feedback File", type=["csv", "xlsx"])

if uploaded and api_key:
    path, cols, df = preprocess_and_save(uploaded)
    if path:
        st.subheader("📄 Preview Data")
        st.dataframe(df, use_container_width=True)

        ignore = ["name", "email", "id", "timestamp"]
        text_cols = [c for c in df.select_dtypes(include='object').columns if c.lower() not in ignore]

        # 🎯 Select Questions
        st.markdown("---")
        st.markdown("## ✅ Select Questions to Analyze")
        selected = st.multiselect("Choose Questions", text_cols, default=text_cols[:2])

        summary_data = []

        for i, col in enumerate(selected):
            responses = df[col].astype(str).dropna().tolist()
            sentiments = classify_sentiments(responses)
            kws = extract_keywords_tfidf(responses)

            st.markdown(f"---")
            with st.expander(f"🔍 Analysis: {col}", expanded=True):
                col1, col2 = st.columns(2) # Two columns for side-by-side display

                with col1:
                    st.markdown("### 📊 Sentiment Breakdown")
                    pie_df = pd.DataFrame(Counter(sentiments).items(), columns=["Sentiment", "Count"])
                    fig = px.pie(pie_df, names="Sentiment", values="Count", title="Sentiment Breakdown")
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("### 📋 Frequent Responses")
                    freq_df = pd.Series(responses).value_counts().reset_index()
                    freq_df.columns = ["Response", "Count"]
                    freq_df = freq_df[freq_df["Response"].str.len() > 10]
                    st.dataframe(freq_df.head(10), use_container_width=True)

                with col2:
                    st.markdown("### 🔤 Top Keywords")
                    kw_df = pd.DataFrame(kws, columns=["Keyword", "Score"])
                    fig = px.bar(kw_df, x="Keyword", y="Score", title="TF-IDF Keywords")
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("### 🧠 Gemini Summary")
                    try:
                        sample = "\n".join(pd.Series(responses).dropna().sample(min(15, len(responses)), random_state=42))
                        prompt = f"""You're a feedback analyst. Summarize the key themes, sentiment (positive, negative, neutral), and actionable improvements from the following responses to the question: "{col}". Make the summary detailed but concise, highlighting the most important aspects for a business user.

                        Feedbacks:
                        {sample}"""
                        reply = gemini.generate_content(prompt)
                        st.info(reply.text.strip()) # Removed character limit for fuller summary
                    except Exception as e:
                        st.error(f"Gemini Error: {e}")

            # Collect for overall summary
            summary_data.append({
                "Question": col,
                "Total Responses": len(responses),
                "👍 Positive": sentiments.count("Positive"),
                "👎 Negative": sentiments.count("Negative"),
                "😐 Neutral": sentiments.count("Neutral"),
                "Top Keywords": ", ".join([kw for kw, _ in kws if kw != "No keywords"])
            })

        # ---
        # 📋 Overall Summary Table
        st.markdown("---")
        st.markdown("## 🧾 Overall Feedback Summary")
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)

        # ---
        # 💬 Ask Gemini
        st.markdown("---")
        st.markdown("## 💬 Ask Gemini About All Feedback")
        userq = st.text_input("Ask your question about the feedback insights")
        if st.button("Ask Gemini"):
            try:
                tabular = summary_df.to_markdown(index=False)
                prompt = f"""You're a feedback report analyst. Given this summary table:\n\n{tabular}\n\nAnswer this question:\n{userq}\n\nProvide a concise and direct answer, focusing on actionable insights derived from the data."""
                final = gemini.generate_content(prompt)
                st.markdown("### 🧠 Gemini Answer")
                st.info(final.text.strip()) # Removed character limit
            except Exception as e:
                st.error(f"Gemini Error: {e}")

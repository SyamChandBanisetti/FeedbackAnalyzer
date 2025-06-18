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

# 🔐 Gemini Setup
def init_gemini(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")

# 📁 File Loader
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
        st.error(f"Processing error: {e}")
        return None, None, None

# 🧠 Efficient Sentiment Classifier (Vectorized)
@st.cache_data(show_spinner=False)
def classify_sentiments(texts):
    sentiments = []
    for text in texts:
        t = text.lower()
        if any(w in t for w in ["good", "great", "excellent", "love", "awesome", "satisfied", "happy", "helpful"]):
            sentiments.append("Positive")
        elif any(w in t for w in ["bad", "poor", "terrible", "hate", "worst", "unsatisfied", "boring", "rude"]):
            sentiments.append("Negative")
        else:
            sentiments.append("Neutral")
    return sentiments

# 🧠 Better Keyword Extraction
@st.cache_data(show_spinner=False)
def extract_keywords_tfidf(texts, top_n=10):
    clean_texts = [re.sub(rf"[{string.punctuation}]", "", t.lower()) for t in texts]
    vectorizer = TfidfVectorizer(stop_words='english', max_features=top_n)
    X = vectorizer.fit_transform(clean_texts)
    keywords = vectorizer.get_feature_names_out()
    scores = X.sum(axis=0).A1
    return sorted(zip(keywords, scores), key=lambda x: x[1], reverse=True)

# 🚀 Streamlit UI
st.set_page_config("Feedback Analyzer", layout="wide")
st.title("📋 Feedback Analyzer using Gemini Flash 2.0")

with st.sidebar:
    st.header("🔐 Gemini API Key")
    gemini_key = st.text_input("Enter Gemini API key", type="password")
    if gemini_key:
        gemini = init_gemini(gemini_key)
        st.success("Gemini ready!")
    else:
        st.warning("API key required.")

uploaded_file = st.file_uploader("📤 Upload CSV/Excel Feedback", type=["csv", "xlsx"])

if uploaded_file and gemini_key:
    tmp_path, cols, df = preprocess_and_save(uploaded_file)

    if tmp_path:
        st.subheader("📄 Feedback Preview")
        st.dataframe(df, use_container_width=True)

        # 🎯 Target Feedback Columns
        ignore = ['name', 'email', 'id', 'timestamp']
        feedback_cols = [c for c in df.select_dtypes(include='object').columns if c.lower() not in ignore]

        # 📈 Feedback Summary Table
        st.markdown("## 📊 Feedback Summary")
        summary_rows = []
        for col in feedback_cols:
            responses = df[col].astype(str).tolist()
            sentiments = classify_sentiments(responses)
            keywords = [kw for kw, _ in extract_keywords_tfidf(responses)]
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

        # 🔘 Multi-select Questions
        st.markdown("## ✅ Select Questions to Analyze")
        selected_qs = st.multiselect("Choose Questions:", feedback_cols, default=feedback_cols[:2])

        for idx, col in enumerate(selected_qs):
            st.markdown(f"### 🔍 **{col}**")
            responses = df[col].astype(str).tolist()
            sentiments = classify_sentiments(responses)

            # 🔘 Checkboxes with better performance
            with st.expander("⚙️ Analysis Options", expanded=True):
                show_pie = st.checkbox("🟢 Sentiment Pie", value=True, key=f"pie_{idx}")
                show_kw = st.checkbox("🔤 Top Keywords (TF-IDF)", value=True, key=f"kw_{idx}")
                show_freq = st.checkbox("📉 Frequent Responses", key=f"freq_{idx}")
                show_sum = st.checkbox("🧠 Gemini Summary", value=True, key=f"sum_{idx}")

            if show_pie:
                pie_df = pd.DataFrame(Counter(sentiments).items(), columns=["Sentiment", "Count"])
                fig = px.pie(pie_df, names="Sentiment", values="Count", title="Sentiment Breakdown")
                st.plotly_chart(fig, use_container_width=True)

            if show_kw:
                keywords = extract_keywords_tfidf(responses)
                kw_df = pd.DataFrame(keywords, columns=["Keyword", "Score"])
                fig = px.bar(kw_df, x="Keyword", y="Score", title="Top Keywords by TF-IDF")
                st.plotly_chart(fig, use_container_width=True)

            if show_freq:
                freq_df = pd.Series(responses).value_counts().reset_index()
                freq_df.columns = ["Response", "Count"]
                freq_df = freq_df[freq_df["Response"].str.len() > 10]  # Skip tiny comments
                st.write("### 📊 Most Frequent Responses")
                st.dataframe(freq_df.head(10), use_container_width=True)

            if show_sum:
                try:
                    sample = "\n".join(pd.Series(responses).dropna().sample(min(15, len(responses)), random_state=42))
                    prompt = f"""You're a professional feedback analyzer. Analyze the following responses for the question: "{col}". Provide main points, positive/negative sentiment themes, and improvement suggestions.\n\nFeedbacks:\n{sample}"""
                    summary = gemini.generate_content(prompt)
                    st.markdown("#### 🤖 Gemini Summary")
                    st.markdown(summary.text)
                except Exception as e:
                    st.error(f"Gemini Error: {e}")

        # 🤔 Ask Gemini a Question
        st.markdown("## 💬 Ask Gemini about Overall Feedback")
        query = st.text_input("Ask your question...")
        if st.button("Submit"):
            try:
                tabular = summary_df.to_markdown(index=False)
                q_prompt = f"""You are a feedback analyst. Below is the summary of multiple questions:\n\n{tabular}\n\nNow answer this:\n{query}"""
                response = gemini.generate_content(q_prompt)
                st.markdown("### 🧠 Gemini Says:")
                st.markdown(response.text)
            except Exception as e:
                st.error(f"Gemini Error: {e}")

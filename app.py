# Keep all imports as before
import streamlit as st
import pandas as pd
import tempfile
import csv
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
import google.generativeai as genai
from collections import Counter

# 🧠 Gemini API setup
def init_gemini(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")

# 📁 File preprocessor
def preprocess_and_save(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            st.error("Unsupported file format.")
            return None, None, None
        
        for col in df.select_dtypes(include='object'):
            df[col] = df[col].astype(str).replace({r'"': '""'}, regex=True)

        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp_path = tmp.name
            df.to_csv(tmp_path, index=False, quoting=csv.QUOTE_ALL)

        return tmp_path, df.columns.tolist(), df
    except Exception as e:
        st.error(f"Processing error: {e}")
        return None, None, None

# 💬 Sentiment
def classify_sentiment(text):
    text = text.lower()
    if any(w in text for w in ["good", "great", "excellent", "love", "awesome", "satisfied", "happy"]):
        return "Positive"
    elif any(w in text for w in ["bad", "poor", "terrible", "hate", "worst", "unsatisfied", "boring"]):
        return "Negative"
    elif text.strip() == "":
        return "Neutral"
    else:
        return "Neutral"

# 🔑 Keywords
@st.cache_data
def extract_keywords(texts, top_n=10):
    vectorizer = CountVectorizer(stop_words='english', max_features=top_n)
    X = vectorizer.fit_transform(texts)
    return vectorizer.get_feature_names_out().tolist()

# 🚀 Streamlit App
st.set_page_config(page_title="🧠 Feedback Analyzer", layout="wide")
st.title("📋 Feedback Analyzer using Gemini Flash 2.0")

with st.sidebar:
    st.header("🔐 Gemini API Key")
    gemini_key = st.text_input("Enter your Gemini API key", type="password")
    if gemini_key:
        gemini = init_gemini(gemini_key)
        st.success("Gemini API initialized!")
    else:
        st.warning("API key required to continue.")

uploaded_file = st.file_uploader("📤 Upload Feedback File (.csv or .xlsx)", type=["csv", "xlsx"])

if uploaded_file and gemini_key:
    tmp_path, all_columns, df = preprocess_and_save(uploaded_file)

    if tmp_path:
        st.subheader("📄 Feedback Preview")
        st.dataframe(df, use_container_width=True)

        # Detect feedback columns
        text_cols = df.select_dtypes(include='object').columns.tolist()
        ignore_cols = ["name", "email", "id", "timestamp"]
        feedback_cols = [col for col in text_cols if col.lower() not in ignore_cols]

        # Create summary
        st.markdown("## 📊 Feedback Summary")
        summary_data = []
        for col in feedback_cols:
            responses = df[col].dropna().astype(str)
            sentiments = [classify_sentiment(r) for r in responses]
            summary_data.append({
                "Question": col,
                "Total": len(responses),
                "👍 Positive": sentiments.count("Positive"),
                "👎 Negative": sentiments.count("Negative"),
                "😐 Neutral": sentiments.count("Neutral"),
                "Top Keywords": ", ".join(extract_keywords(responses))
            })
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)

        # Select questions
        st.markdown("## ✅ Select Questions to Analyze")
        selected_questions = st.multiselect("Pick questions:", options=feedback_cols, default=feedback_cols[:2])

        for idx, q in enumerate(selected_questions):
            st.markdown(f"### 🔍 Analysis: **{q}**")
            responses = df[q].dropna().astype(str)
            sentiments = [classify_sentiment(r) for r in responses]

            with st.expander("📌 Choose visualizations", expanded=True):
                show_pie = st.checkbox("🟢 Show sentiment pie chart", key=f"pie_{idx}", value=True)
                show_keywords = st.checkbox("🔤 Show top keywords", key=f"kw_{idx}", value=True)
                show_freq = st.checkbox("📉 Show frequent responses", key=f"freq_{idx}")
                show_summary = st.checkbox("🧠 Show Gemini summary", key=f"sum_{idx}", value=True)

            if show_pie:
                pie_df = pd.DataFrame(Counter(sentiments).items(), columns=["Sentiment", "Count"])
                fig = px.pie(pie_df, names="Sentiment", values="Count", title="Sentiment Breakdown")
                st.plotly_chart(fig, use_container_width=True)

            if show_keywords:
                keywords = extract_keywords(responses)
                counts = Counter(" ".join(responses).lower().split())
                keyword_freq = {k: counts[k] for k in keywords}
                kw_df = pd.DataFrame(keyword_freq.items(), columns=["Keyword", "Count"])
                fig = px.bar(kw_df, x="Keyword", y="Count", title="Top Keywords")
                st.plotly_chart(fig, use_container_width=True)

            if show_freq:
                freq_df = responses.value_counts().reset_index()
                freq_df.columns = ["Response", "Count"]
                st.write("### 📊 Most Frequent Responses")
                st.dataframe(freq_df.head(10))

            if show_summary:
                prompt = f"Summarize key insights from these feedbacks for the question:\n\"{q}\"\n\n"
                sample = "\n".join(responses.sample(min(10, len(responses))).tolist())
                try:
                    reply = gemini.generate_content(prompt + sample)
                    st.markdown("#### 🤖 Gemini Summary")
                    st.markdown(reply.text)
                except Exception as e:
                    st.error(f"Gemini Error: {e}")

        # Gemini Q&A
        st.markdown("## 💬 Ask Gemini a Question")
        user_query = st.text_input("Example: Which question had most negative feedback?")
        if st.button("Submit"):
            try:
                summary_txt = summary_df.to_markdown(index=False)
                final_prompt = f"This is a summary of feedback questions:\n\n{summary_txt}\n\nAnswer this: {user_query}"
                result = gemini.generate_content(final_prompt)
                st.markdown("### 🤖 Gemini Says")
                st.markdown(result.text)
            except Exception as e:
                st.error(f"Gemini Q&A Error: {e}")

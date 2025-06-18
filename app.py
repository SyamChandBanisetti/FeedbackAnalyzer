import streamlit as st
import pandas as pd
import tempfile
import csv
import re
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
import google.generativeai as genai

# 🔐 Gemini API Setup
def init_gemini(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")

# 🧹 Preprocessing function
def preprocess_and_save(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, encoding='utf-8')
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            st.error("Unsupported file format. Upload CSV or Excel.")
            return None, None, None
        
        for col in df.select_dtypes(include=['object']):
            df[col] = df[col].astype(str).replace({r'"': '""'}, regex=True)

        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')
            elif df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col])
                except:
                    pass

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp_path = tmp.name
            df.to_csv(tmp_path, index=False, quoting=csv.QUOTE_ALL)

        return tmp_path, df.columns.tolist(), df
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, None, None

# 🔠 Keyword extractor
@st.cache_data
def extract_keywords(texts, top_n=10):
    vectorizer = CountVectorizer(stop_words='english', max_features=top_n)
    X = vectorizer.fit_transform(texts)
    return vectorizer.get_feature_names_out().tolist()

# 😊 Simple sentiment classifier
def classify_sentiment(text):
    text = text.lower()
    if any(w in text for w in ["good", "great", "excellent", "love", "awesome"]):
        return "Positive"
    elif any(w in text for w in ["bad", "poor", "terrible", "hate", "worst"]):
        return "Negative"
    elif text.strip() == "":
        return "Neutral"
    else:
        return "Neutral"

# 🚀 Streamlit App
st.set_page_config(page_title="🧠 Feedback Analyzer (Gemini Flash)", layout="wide")
st.title("📋 Feedback Analyzer with Gemini Flash 2.0")

# 🔐 Sidebar - API Key
with st.sidebar:
    st.header("🔐 Gemini API Key")
    gemini_key = st.text_input("Enter your Gemini API key", type="password")
    if gemini_key:
        gemini = init_gemini(gemini_key)
        st.success("Gemini API key loaded!")
    else:
        st.warning("Please enter your Gemini API key to continue.")

# 📁 Upload file
uploaded_file = st.file_uploader("📤 Upload Feedback CSV or Excel", type=["csv", "xlsx"])

if uploaded_file and gemini_key:
    temp_path, columns, df = preprocess_and_save(uploaded_file)

    if temp_path:
        st.subheader("📄 Uploaded Feedback Table")
        st.dataframe(df)

        # Detect open-ended feedback columns
        text_cols = df.select_dtypes(include='object').columns.tolist()
        ignore_cols = ["name", "email", "id", "timestamp"]
        feedback_cols = []

        for col in text_cols:
            if col.lower() in ignore_cols:
                continue
            unique_vals = df[col].nunique()
            avg_len = df[col].astype(str).apply(len).mean()
            if unique_vals > 10 or avg_len > 15:
                feedback_cols.append(col)

        if not feedback_cols:
            st.warning("No descriptive feedback questions detected.")
        else:
            # 📊 Summary Table
            st.markdown("### 📊 Summary Table")
            summary_rows = []
            for col in feedback_cols:
                responses = df[col].dropna().astype(str).tolist()
                sentiments = [classify_sentiment(r) for r in responses]
                pos, neg, neu = sentiments.count("Positive"), sentiments.count("Negative"), sentiments.count("Neutral")
                total = len(responses)
                keywords = extract_keywords(responses)
                summary_rows.append({
                    "Question": col,
                    "Total": total,
                    "👍 Positive": pos,
                    "👎 Negative": neg,
                    "😐 Neutral": neu,
                    "Top Keywords": ", ".join(keywords)
                })

            summary_df = pd.DataFrame(summary_rows)
            st.dataframe(summary_df)

            # ✅ Visualization controls
            st.markdown("### 📈 Visualization Preferences")
            show_sentiment = st.checkbox("📊 Show Sentiment Pie Chart", value=True)
            show_keywords = st.checkbox("🔠 Show Keyword Bar Chart", value=True)
            show_freq_bar = st.checkbox("📉 Show Answer Frequency Bar Chart", value=True)
            show_trend = st.checkbox("⏳ Show Time Trend (if available)", value=True)

            timestamp_col = next((col for col in df.columns if "timestamp" in col.lower()), None)

            st.markdown("## 📌 Question-wise Charts")
            for col in feedback_cols:
                st.subheader(f"📝 {col}")
                responses = df[col].dropna().astype(str).tolist()

                # Sentiment Pie Chart
                if show_sentiment:
                    sentiments = [classify_sentiment(r) for r in responses]
                    sentiment_df = pd.DataFrame(sentiments, columns=["Sentiment"])
                    st.markdown("**Sentiment Distribution**")
                    fig1 = px.pie(sentiment_df, names='Sentiment', title='Sentiment Breakdown')
                    st.plotly_chart(fig1, use_container_width=True)

                # Keyword Bar Chart
                if show_keywords:
                    st.markdown("**Top Keywords**")
                    keywords = extract_keywords(responses)
                    if keywords:
                        keyword_freq = pd.Series(keywords).value_counts()
                        st.bar_chart(keyword_freq)

                # Frequency Bar Chart
                if show_freq_bar:
                    counts = df[col].value_counts().head(10)
                    if counts.count() > 1:
                        st.markdown("**Answer Frequency**")
                        st.bar_chart(counts)

                # Time Trend
                if show_trend and timestamp_col:
                    temp_df = df[[timestamp_col, col]].dropna()
                    temp_df[timestamp_col] = pd.to_datetime(temp_df[timestamp_col])
                    temp_df = temp_df.set_index(timestamp_col).resample("D").count()
                    temp_df.columns = ["Responses"]
                    st.markdown("**📆 Response Trend Over Time**")
                    st.line_chart(temp_df)

            # 🔍 Gemini Q&A
            st.markdown("## 💬 Ask Gemini About the Feedback")
            user_query = st.text_area("💡 E.g. 'Which question had the most negative responses?'")

            if st.button("Submit Query"):
                with st.spinner("Thinking..."):
                    try:
                        context = f"""You are analyzing a feedback summary:

{summary_df.to_markdown(index=False)}

Now answer this user query:
{user_query}
"""
                        response = gemini.generate_content(context)
                        st.markdown("### 🧠 Gemini's Answer")
                        st.markdown(response.text)
                    except Exception as e:
                        st.error(f"Gemini API error: {e}")

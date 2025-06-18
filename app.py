import streamlit as st
import pandas as pd
import tempfile
import csv
import json
import re
from sklearn.feature_extraction.text import CountVectorizer
import google.generativeai as genai

# 🔑 Gemini API Setup
def init_gemini(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")

# 🔁 Preprocessing function
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
            if 'date' in col.lower():
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

# 🧠 Keyword extractor
@st.cache_data
def extract_keywords(texts, top_n=10):
    vectorizer = CountVectorizer(stop_words='english', max_features=top_n)
    X = vectorizer.fit_transform(texts)
    return vectorizer.get_feature_names_out().tolist()

# 😐 Sentiment Classifier
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
st.set_page_config(page_title="🧠 Feedback Form Analyzer (Gemini Flash)", layout="wide")
st.title("📋 Feedback Analyzer with Gemini Flash 2.0")

# 🛡️ API Key Input
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
        st.subheader("📄 Feedback Table")
        st.dataframe(df)

        text_cols = df.select_dtypes(include='object').columns.tolist()
        ignore_cols = ["name", "email", "id", "timestamp"]
        feedback_cols = [col for col in text_cols if col.lower() not in ignore_cols]

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

        # 🤖 Gemini Q&A over DataFrame
        st.markdown("## 💬 Ask Gemini Anything About the Feedback")

        user_query = st.text_area("💡 Example: 'Which question had the most negative responses?'")

        if st.button("Submit Query"):
            with st.spinner("Thinking..."):
                try:
                    context = f"""This is a feedback summary table from a form:
{summary_df.to_markdown(index=False)}

Now answer this query from the user:
{user_query}
"""
                    response = gemini.generate_content(context)
                    st.markdown("### 🧠 Gemini's Answer")
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"Gemini API error: {e}")

import streamlit as st
import pandas as pd
import tempfile
import csv
import json
import re
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai

# --- Gemini API ---
def init_gemini(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")

# --- Preprocessing ---
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

# --- Sentiment + Keywords ---
@st.cache_data
def extract_keywords(texts, top_n=10):
    vectorizer = CountVectorizer(stop_words='english', max_features=top_n)
    X = vectorizer.fit_transform(texts)
    return vectorizer.get_feature_names_out().tolist()

def classify_sentiment(text):
    text = text.lower()
    if any(w in text for w in ["good", "great", "excellent", "love", "awesome", "nice", "amazing"]):
        return "Positive"
    elif any(w in text for w in ["bad", "poor", "terrible", "hate", "worst", "disappointing"]):
        return "Negative"
    elif text.strip() == "":
        return "Neutral"
    else:
        return "Neutral"

# --- Pie chart ---
def plot_sentiment_pie(sentiments, col):
    fig, ax = plt.subplots()
    sentiment_counts = pd.Series(sentiments).value_counts()
    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
    ax.set_title(f"Sentiment Distribution: {col}")
    st.pyplot(fig)

# --- Bar chart ---
def plot_keyword_bar(keywords, col):
    keyword_counts = pd.Series(keywords).value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=keyword_counts.values, y=keyword_counts.index, ax=ax, palette="Blues_r")
    ax.set_title(f"Top Keywords: {col}")
    st.pyplot(fig)

# --- Streamlit App ---
st.set_page_config(page_title="📋 Feedback Analyzer with Gemini", layout="wide")
st.title("🧠 Feedback Analyzer (Gemini Flash 2.0 + Charts)")

with st.sidebar:
    st.header("🔐 Gemini API Key")
    gemini_key = st.text_input("Enter your Gemini API key", type="password")
    if gemini_key:
        gemini = init_gemini(gemini_key)
        st.success("Gemini API key loaded!")
    else:
        st.warning("Please enter your Gemini API key.")

uploaded_file = st.file_uploader("📤 Upload Feedback CSV or Excel", type=["csv", "xlsx"])

if uploaded_file and gemini_key:
    temp_path, columns, df = preprocess_and_save(uploaded_file)

    if temp_path:
        st.subheader("📄 Raw Feedback Table")
        st.dataframe(df)

        # Identify feedback-relevant columns
        text_cols = df.select_dtypes(include='object').columns.tolist()
        ignore_cols = ["name", "email", "id", "timestamp"]
        feedback_cols = [col for col in text_cols if col.lower() not in ignore_cols]

        st.markdown("## 📊 Summary Table")
        summary_rows = []

        # For detailed analysis
        question_analysis = {}

        for col in feedback_cols:
            responses = df[col].dropna().astype(str).tolist()
            sentiments = [classify_sentiment(r) for r in responses]
            keywords = extract_keywords(responses, top_n=10)
            pos, neg, neu = sentiments.count("Positive"), sentiments.count("Negative"), sentiments.count("Neutral")
            summary_rows.append({
                "Question": col,
                "Total": len(responses),
                "👍 Positive": pos,
                "👎 Negative": neg,
                "😐 Neutral": neu,
                "Top Keywords": ", ".join(keywords)
            })

            question_analysis[col] = {
                "responses": responses,
                "sentiments": sentiments,
                "keywords": keywords
            }

        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df)

        st.markdown("## 📈 Per-Question Insights")

        for col in feedback_cols:
            st.markdown(f"### 🔹 {col}")
            qdata = question_analysis[col]

            if len(qdata["responses"]) < 5:
                st.info("Not enough data to generate insights.")
                continue

            col1, col2 = st.columns(2)
            with col1:
                plot_sentiment_pie(qdata["sentiments"], col)
            with col2:
                plot_keyword_bar(qdata["keywords"], col)

            st.markdown("#### 🔎 Sample Responses")
            for r in qdata["responses"][:5]:
                st.markdown(f"- {r}")

            st.divider()

        st.markdown("## 💬 Ask Gemini Anything About the Feedback")

        user_query = st.text_area("💡 Example: 'What are students most unhappy about?'")

        if st.button("Submit Query"):
            with st.spinner("Gemini is analyzing..."):
                try:
                    context = f"""This is a feedback summary table:
{summary_df.to_markdown(index=False)}

Now answer this question from the user:
{user_query}
"""
                    response = gemini.generate_content(context)
                    st.markdown("### 🧠 Gemini's Answer")
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"Gemini API error: {e}")

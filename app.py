import streamlit as st
import pandas as pd
import tempfile
import csv
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import google.generativeai as genai
import plotly.express as px

# --------------------- 🔑 Gemini API Setup ---------------------
def init_gemini(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")

# --------------------- 🧹 Preprocessing ---------------------
def preprocess_and_save(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, encoding='utf-8')
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            st.error("Unsupported file format.")
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

# --------------------- 🧠 TF-IDF Keyword Extraction ---------------------
@st.cache_data
def extract_keywords_tfidf(texts, top_n=10):
    cleaned = [re.sub(rf"[{string.punctuation}]", "", t.lower()) for t in texts if len(t.split()) > 2]
    cleaned = [t for t in cleaned if t.strip() and not t.isspace()]
    
    if not cleaned:
        return ["No significant keywords"]

    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=top_n)
        X = vectorizer.fit_transform(cleaned)
        keywords = vectorizer.get_feature_names_out()
        scores = X.sum(axis=0).A1
        result = sorted(zip(keywords, scores), key=lambda x: x[1], reverse=True)
        return [kw for kw, score in result]
    except ValueError:
        return ["No meaningful keywords"]

# --------------------- 😀 Sentiment Classifier ---------------------
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

# --------------------- 🚀 Streamlit UI ---------------------
st.set_page_config(page_title="🧠 Feedback Analyzer (Gemini)", layout="wide")
st.title("📋 Feedback Analyzer with Gemini Flash 2.0")

# 🔐 API Key Sidebar
with st.sidebar:
    st.header("🔐 Gemini API Key")
    gemini_key = st.text_input("Enter your Gemini API key", type="password")
    if gemini_key:
        gemini = init_gemini(gemini_key)
        st.success("Gemini API key loaded!")
    else:
        st.warning("Please enter your Gemini API key to continue.")

# 📤 File Upload
uploaded_file = st.file_uploader("📤 Upload Feedback CSV or Excel", type=["csv", "xlsx"])

if uploaded_file and gemini_key:
    temp_path, columns, df = preprocess_and_save(uploaded_file)

    if temp_path:
        st.subheader("📄 Uploaded Feedback Data")
        st.dataframe(df, use_container_width=True)

        text_cols = df.select_dtypes(include='object').columns.tolist()
        ignore_cols = ["name", "email", "id", "timestamp"]
        feedback_cols = [col for col in text_cols if col.lower() not in ignore_cols]

        st.markdown("## 🎯 Choose questions to analyze:")
        selected_questions = st.multiselect("Select feedback questions", feedback_cols, default=feedback_cols[:3])

        for col in selected_questions:
            st.markdown(f"### ❓ Question: `{col}`")
            responses = df[col].dropna().astype(str).tolist()
            sentiments = [classify_sentiment(r) for r in responses]
            total = len(responses)
            pos, neg, neu = sentiments.count("Positive"), sentiments.count("Negative"), sentiments.count("Neutral")
            keywords = extract_keywords_tfidf(responses)

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Total Responses", total)
                st.metric("👍 Positive", pos)
                st.metric("👎 Negative", neg)
                st.metric("😐 Neutral", neu)

            with col2:
                st.markdown("#### 🔤 Top Keywords")
                if "No" in keywords[0]:
                    st.warning("No significant keywords found.")
                else:
                    st.success(", ".join(keywords))

            # Sentiment Pie Chart
            fig = px.pie(
                names=["Positive", "Negative", "Neutral"],
                values=[pos, neg, neu],
                title="Sentiment Distribution",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig, use_container_width=True)

        # 🧠 Gemini Question Answering
        st.markdown("---")
        st.markdown("## 💬 Ask Gemini about the feedback summary")
        user_query = st.text_area("🧠 Ask a question (e.g., 'Which question has most negative feedback?')")

        if st.button("Submit Query to Gemini"):
            with st.spinner("Gemini is analyzing..."):
                summary_data = []
                for col in selected_questions:
                    responses = df[col].dropna().astype(str).tolist()
                    sentiments = [classify_sentiment(r) for r in responses]
                    pos, neg, neu = sentiments.count("Positive"), sentiments.count("Negative"), sentiments.count("Neutral")
                    keywords = extract_keywords_tfidf(responses)
                    summary_data.append({
                        "Question": col,
                        "Total": len(responses),
                        "Positive": pos,
                        "Negative": neg,
                        "Neutral": neu,
                        "Keywords": ", ".join(keywords)
                    })

                summary_df = pd.DataFrame(summary_data)

                context = f"""You are a feedback analysis assistant. Here is the summarized data:

{summary_df.to_markdown(index=False)}

Now answer the following user query clearly and concisely:
{user_query}
"""

                try:
                    response = gemini.generate_content(context)
                    st.markdown("### 🤖 Gemini's Answer")
                    st.info(response.text.strip()[:1000])  # limit output length
                except Exception as e:
                    st.error(f"Gemini error: {e}")


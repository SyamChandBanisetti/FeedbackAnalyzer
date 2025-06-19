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
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- Internal Gemini API Key ---
GEMINI_API_KEY = "your_actual_gemini_api_key_here"

# --- Basic Stopwords ---
STOP_WORDS = set([
    # Common stop words
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during", "before", "after", "above",
    "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
] + list(string.punctuation))

def simple_tokenize(text):
    cleaned = re.sub(r'\d+', '', text.lower())
    cleaned = re.sub(rf"[{re.escape(string.punctuation)}]", "", cleaned)
    return [word for word in cleaned.split() if word not in STOP_WORDS and len(word) > 1]

# --- Gemini Setup ---
def init_gemini():
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel("gemini-1.5-flash")

# --- Preprocessing ---
def preprocess_and_save(file):
    try:
        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        df.fillna("", inplace=True)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        df.to_csv(tmp.name, index=False, quoting=csv.QUOTE_ALL)
        return tmp.name, df.columns.tolist(), df
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, None, None

# --- Sentiment Classifier ---
@st.cache_data
def classify_sentiments(texts):
    sentiments = []
    for t in texts:
        t = t.lower()
        if any(w in t for w in ["good", "great", "excellent", "love", "awesome", "helpful", "satisfied", "positive", "happy", "nice", "recommend"]):
            sentiments.append("Positive")
        elif any(w in t for w in ["bad", "poor", "terrible", "hate", "worst", "problem", "issue", "slow", "unhappy", "frustrated"]):
            sentiments.append("Negative")
        else:
            sentiments.append("Neutral")
    return sentiments

# --- TF-IDF Keywords ---
@st.cache_data
def extract_keywords_tfidf(texts, top_n=10):
    processed = [" ".join(simple_tokenize(t)) for t in texts if len(t.strip()) > 5]
    try:
        vec = TfidfVectorizer(max_features=top_n)
        X = vec.fit_transform(processed)
        kw = vec.get_feature_names_out()
        scores = X.sum(axis=0).A1
        return sorted(zip(kw, scores), key=lambda x: x[1], reverse=True)
    except:
        return []

# --- Word Cloud ---
@st.cache_data
def generate_wordcloud(text):
    if not text.strip():
        return None
    wc = WordCloud(width=800, height=400, background_color='white', stopwords=STOP_WORDS).generate(text)
    return wc

# --- Gemini Summary ---
def get_gemini_summary(gemini, question, responses):
    sample = "\n".join(pd.Series(responses).dropna().sample(min(25, len(responses)), random_state=42))
    prompt = f"""
You are an expert feedback analyst. Read the following responses for the question: "{question}" and provide a structured summary:

1. Overview of major sentiments (positive, negative, neutral).
2. Key insights and recurring points.
3. Clear bullet-point suggestions for improvement.
Keep it professional and concise with bullet formatting.

Responses:
{sample}
"""
    reply = gemini.generate_content(prompt)
    return reply.text.strip()

# --- Streamlit App ---
st.set_page_config("Feedback Analyzer", layout="wide")
st.title("📋 Feedback Analyzer Tool")

gemini = init_gemini()

uploaded = st.file_uploader("📤 Upload Feedback CSV or Excel", type=["csv", "xlsx"])

if uploaded:
    path, cols, df = preprocess_and_save(uploaded)
    if path:
        st.subheader("📄 Data Preview")
        st.dataframe(df, use_container_width=True)

        text_cols = [c for c in df.select_dtypes('object').columns if c.lower() not in ["name", "email", "id", "timestamp"]]
        selected = st.multiselect("✅ Select Questions to Analyze", text_cols, default=text_cols[:2])

        summary_data = []

        for i, col in enumerate(selected):
            responses = df[col].astype(str).dropna().tolist()
            meaningful = [r for r in responses if len(r.strip()) > 5 and not r.isnumeric()]

            sentiments = classify_sentiments(meaningful) if meaningful else []
            keywords = extract_keywords_tfidf(meaningful) if meaningful else []

            st.markdown(f"---\n### 🔍 Analysis: **{col}**")
            with st.expander(f"Click to expand analysis for **{col}**", expanded=True):

                # Layout
                col1, col2 = st.columns(2)
                with col1:
                    if sentiments:
                        st.markdown("#### 📊 Sentiment Breakdown")
                        pie_df = pd.DataFrame(Counter(sentiments).items(), columns=["Sentiment", "Count"])
                        fig = px.pie(pie_df, names="Sentiment", values="Count")
                        st.plotly_chart(fig, use_container_width=True)

                    if meaningful:
                        st.markdown("#### ☁️ Word Cloud")
                        wc_img = generate_wordcloud(" ".join(meaningful))
                        if wc_img:
                            fig_wc, ax_wc = plt.subplots(figsize=(8, 4))
                            ax_wc.imshow(wc_img, interpolation='bilinear')
                            ax_wc.axis('off')
                            st.pyplot(fig_wc)

                with col2:
                    if keywords:
                        st.markdown("#### 🔤 Top Keywords")
                        for kw, score in keywords:
                            st.markdown(f"- **{kw}** (Score: {score:.2f})")
                    if meaningful:
                        st.markdown("#### 📋 Frequent Responses")
                        freq_df = pd.Series(meaningful).value_counts().reset_index()
                        freq_df.columns = ["Response", "Count"]
                        freq_df = freq_df[freq_df["Response"].apply(lambda x: len(x.split()) >= 2 and len(x) > 10)]
                        st.dataframe(freq_df.head(10), use_container_width=True)

                if meaningful:
                    st.markdown("#### 🧠 Gemini Summary")
                    try:
                        summary = get_gemini_summary(gemini, col, meaningful)
                        st.markdown(summary)
                    except Exception as e:
                        st.error(f"Gemini summary failed: {e}")

            # Summary table
            summary_data.append({
                "Question": col,
                "Total Responses": len(responses),
                "👍 Positive": sentiments.count("Positive"),
                "👎 Negative": sentiments.count("Negative"),
                "😐 Neutral": sentiments.count("Neutral"),
                "Top Keywords": ", ".join([kw for kw, _ in keywords]) if keywords else "N/A"
            })

        st.markdown("---\n## 🧾 Overall Summary")
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

        st.markdown("---\n## 💬 Ask Gemini About All Feedback")
        user_q = st.text_input("Type your question about overall feedback insights:")
        if st.button("Ask Gemini"):
            try:
                summary_df = pd.DataFrame(summary_data).to_markdown(index=False)
                prompt = f"You are an expert feedback consultant. Given this summary:\n\n{summary_df}\n\nAnswer this:\n{user_q}"
                response = gemini.generate_content(prompt)
                st.markdown("### 🧠 Gemini Answer")
                st.info(response.text.strip())
            except Exception as e:
                st.error(f"Gemini failed to answer: {e}")
else:
    st.info("📥 Upload a feedback file to begin.")

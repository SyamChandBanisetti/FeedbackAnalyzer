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

# --- For Word Cloud ---
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- NLTK Replacement: Simple Tokenization and Stop Words ---
STOP_WORDS = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
    "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now",
    "don't", "shouldn't", "can't", "won't", "isn't", "aren't", "wasn't", "weren't",
] + list(string.punctuation))

def simple_tokenize(text):
    cleaned_text = re.sub(r'\d+', '', text.lower())
    cleaned_text = re.sub(rf"[{re.escape(string.punctuation)}]", "", cleaned_text)
    tokens = cleaned_text.split()
    return [token for token in tokens if token.strip()]

# Gemini Setup

def init_gemini(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")

# File Preprocess

def preprocess_and_save(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None, None, None

        df.fillna("", inplace=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp_path = tmp.name
            df.to_csv(tmp_path, index=False, quoting=csv.QUOTE_ALL)

        return tmp_path, df.columns.tolist(), df
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, None, None

# Sentiment Classifier
@st.cache_data
def classify_sentiments(texts):
    sentiments = []
    for text in texts:
        t = text.lower()
        if any(w in t for w in ["good", "great", "excellent", "love", "awesome", "helpful", "satisfied", "nice", "positive", "happy", "smooth", "efficient", "recommend", "super", "perfect", "enjoy"]):
            sentiments.append("Positive")
        elif any(w in t for w in ["bad", "poor", "terrible", "hate", "worst", "boring", "rude", "unsatisfied", "negative", "unhappy", "difficult", "issue", "problem", "frustrating", "slow", "broken", "complaint"]):
            sentiments.append("Negative")
        else:
            sentiments.append("Neutral")
    return sentiments

# TF-IDF Keyword Extractor
@st.cache_data
def extract_keywords_tfidf(texts, top_n=10):
    processed_texts = [
        " ".join([token for token in simple_tokenize(t) if token not in STOP_WORDS and len(token) > 1])
        for t in texts if len(t.strip()) > 2
    ]
    processed_texts = [t for t in processed_texts if t and not t.isspace()]

    if not processed_texts:
        return []
    try:
        vec = TfidfVectorizer(max_features=top_n)
        X = vec.fit_transform(processed_texts)
        kw = vec.get_feature_names_out()
        scores = X.sum(axis=0).A1
        return sorted(zip(kw, scores), key=lambda x: x[1], reverse=True)
    except Exception as e:
        st.warning(f"Keyword extraction error: {e}")
        return []

# Word Cloud
@st.cache_data
def generate_wordcloud(text_corpus):
    if not text_corpus.strip():
        return None
    wordcloud = WordCloud(width=800, height=400, background_color='white',
                          stopwords=STOP_WORDS, min_font_size=10).generate(text_corpus)
    return wordcloud

# UI Setup
st.set_page_config("Feedback Analyzer", layout="wide")
st.markdown("""
    <style>
    .main {background-color: #fdfdfd;}
    .st-expander {border: 1px solid #ccc; border-radius: 6px; padding: 5px;}
    .stCheckbox > div {margin-bottom: 4px;}
    .element-container:has(> .stCheckbox) {margin-bottom: 1rem;}
    </style>
""", unsafe_allow_html=True)

st.title("📋 Feedback Analyzer Tool")

# Sidebar: Gemini API Key
with st.sidebar:
    st.header("🔐 Gemini API Key")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    if api_key:
        try:
            gemini = init_gemini(api_key)
            test_response = gemini.generate_content("hello", safety_settings={'HARASSMENT': 'block_none'})
            st.success("Gemini is ready and connected!")
        except Exception as e:
            st.error(f"Error connecting to Gemini: {e}")
            gemini = None
    else:
        st.warning("Please enter your Gemini API key.")
        gemini = None

# Upload File
uploaded = st.file_uploader("📤 Upload CSV or Excel Feedback File", type=["csv", "xlsx"])

if uploaded and api_key and gemini:
    path, cols, df = preprocess_and_save(uploaded)
    if path:
        st.subheader("📄 Preview Data")
        st.dataframe(df, use_container_width=True)

        ignore = ["name", "email", "id", "timestamp"]
        text_cols = [c for c in df.select_dtypes(include='object').columns if c.lower() not in ignore]

        st.markdown("---")
        st.markdown("## ✅ Select Questions to Analyze")
        selected = st.multiselect("Choose Questions", text_cols, default=text_cols[:min(len(text_cols), 2)])

        summary_data = []

        for i, col in enumerate(selected):
            responses = df[col].astype(str).dropna().tolist()
            meaningful_responses = [r for r in responses if len(r.strip()) > 5 and not r.isnumeric()]

            sentiments = classify_sentiments(meaningful_responses) if meaningful_responses else []
            kws = extract_keywords_tfidf(meaningful_responses) if meaningful_responses else []

            with st.expander(f"🔍 Analysis: {col}", expanded=True):
                pie_df = pd.DataFrame(Counter(sentiments).items(), columns=["Sentiment", "Count"])
                freq_df = pd.Series(meaningful_responses).value_counts().reset_index()
                freq_df.columns = ["Response", "Count"]
                freq_df = freq_df[freq_df["Response"].apply(lambda x: len(x.split()) >= 2 and len(x) > 10)]

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 📊 Sentiment Breakdown")
                    if not pie_df.empty:
                        fig = px.pie(pie_df, names="Sentiment", values="Count", title="Sentiment Breakdown")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Not enough responses for sentiment breakdown.")

                    st.markdown("### 🔤 Top Keywords")
                    if kws:
                        for kw, score in kws:
                            st.markdown(f"- **{kw}** (Score: {score:.2f})")
                    else:
                        st.info("No significant keywords found.")

                    st.markdown("### 📋 Frequent Responses")
                    if not freq_df.empty:
                        st.dataframe(freq_df.head(10), use_container_width=True)
                    else:
                        st.info("No frequent multi-word responses found.")

                with col2:
                    st.markdown("### ☁️ Word Cloud")
                    if meaningful_responses:
                        text_for_wc = " ".join(meaningful_responses)
                        wordcloud_img = generate_wordcloud(text_for_wc)
                        if wordcloud_img:
                            fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
                            ax_wc.imshow(wordcloud_img, interpolation='bilinear')
                            ax_wc.axis('off')
                            st.pyplot(fig_wc)
                        else:
                            st.info("Not enough text to generate a word cloud.")
                    else:
                        st.info("No responses for word cloud.")

                    st.markdown("### 🧠 Gemini Summary")
                    try:
                        if meaningful_responses:
                            sample_responses = "\n".join(pd.Series(meaningful_responses).dropna().sample(min(25, len(meaningful_responses)), random_state=42))
                            prompt = f"""Provide a concise, useful, and actionable summary of the key themes, overall sentiment (positive/negative/neutral breakdown), and specific suggestions for improvement from the following responses to the question: \"{col}\". Aim for a summary that is easy to read and provides valuable insights, without being too long or too short. Focus on a length that is 'considerable' and 'just useful'.\n\nFeedbacks:\n{sample_responses}"""
                            reply = gemini.generate_content(prompt)
                            st.info(reply.text.strip())
                        else:
                            st.info("Not enough responses for summary.")
                    except Exception as e:
                        st.error(f"Gemini Error for '{col}': {e}")

            summary_data.append({
                "Question": col,
                "Total Responses": len(responses),
                "👍 Positive": sentiments.count("Positive"),
                "👎 Negative": sentiments.count("Negative"),
                "😐 Neutral": sentiments.count("Neutral"),
                "Top Keywords": ", ".join([kw for kw, _ in kws]) if kws else "N/A",
            })

        st.markdown("---")
        st.markdown("## 🧾 Overall Feedback Summary")
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)

        st.markdown("---")
        st.markdown("## 💬 Ask Gemini About All Feedback")
        userq = st.text_input("Ask your question about the overall feedback insights")
        if st.button("Ask Gemini about overall feedback"):
            try:
                tabular = summary_df.to_markdown(index=False)
                prompt = f"""You're a feedback report analyst. Given this summary table:\n\n{tabular}\n\nAnswer this question:\n{userq}\n\nProvide a concise and direct answer, focusing on actionable insights derived from the data."""
                final = gemini.generate_content(prompt)
                st.markdown("### 🧠 Gemini Answer")
                st.info(final.text.strip())
            except Exception as e:
                st.error(f"Gemini Error: {e}")

elif uploaded and not api_key:
    st.warning("Please enter your Gemini API key in the sidebar to proceed.")
elif not uploaded:
    st.info("Upload a CSV or Excel file to begin feedback analysis.")

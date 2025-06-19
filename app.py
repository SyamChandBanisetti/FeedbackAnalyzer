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

# --- Basic Stop Words ---
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
    "don't", "shouldn't", "can't", "won't", "isn't", "aren't", "wasn't", "weren't"
] + list(string.punctuation))

def simple_tokenize(text):
    cleaned_text = re.sub(r'\d+', '', text.lower())
    cleaned_text = re.sub(rf"[{re.escape(string.punctuation)}]", "", cleaned_text)
    tokens = cleaned_text.split()
    return [token for token in tokens if token.strip()]

def init_gemini(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.0-flash")

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

@st.cache_data
def extract_keywords_tfidf(texts, top_n=10):
    processed_texts = [" ".join([token for token in simple_tokenize(t) if token not in STOP_WORDS and len(token) > 1]) for t in texts if len(t.strip()) > 2]
    processed_texts = [t for t in processed_texts if t and not t.isspace()]
    if not processed_texts:
        return []
    try:
        vec = TfidfVectorizer(max_features=top_n)
        X = vec.fit_transform(processed_texts)
        kw = vec.get_feature_names_out()
        scores = X.sum(axis=0).A1
        return sorted(zip(kw, scores), key=lambda x: x[1], reverse=True)
    except ValueError as e:
        st.warning(f"Could not extract keywords: {e}")
        return []
    except Exception as e:
        st.error(f"Keyword extraction error: {e}")
        return []

@st.cache_data
def generate_wordcloud(text_corpus):
    if not text_corpus.strip():
        return None
    wordcloud = WordCloud(width=800, height=400, background_color='white',
                          stopwords=STOP_WORDS, min_font_size=10).generate(text_corpus)
    return wordcloud

# --- Streamlit UI ---
st.set_page_config("Feedback Analyzer", layout="wide")
st.title("📋 Feedback Analyzer Tool")

# --- API Key Setup from secrets ---
with st.sidebar:
    st.header("🔐 Gemini API Key (Pre-configured)")
    try:
        api_key = st.secrets["gemini"]["api_key"]
        gemini = init_gemini(api_key)
        test_response = gemini.generate_content("hello", safety_settings={'HARASSMENT': 'block_none'})
        if test_response:
            st.success("Gemini is ready and connected!")
        else:
            st.error("Gemini connection failed.")
            gemini = None
    except Exception as e:
        st.error(f"Error connecting to Gemini: {e}")
        gemini = None

uploaded = st.file_uploader("📤 Upload CSV or Excel Feedback File", type=["csv", "xlsx"])

if uploaded and gemini:
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
            sentiments, kws = [], []

            if meaningful_responses:
                sentiments = classify_sentiments(meaningful_responses)
                kws = extract_keywords_tfidf(meaningful_responses)

            st.markdown(f"---")
            with st.expander(f"🔍 Analysis: **{col}**", expanded=True):
                st.markdown("### Choose Analysis Types for this Question:")
                col_c1, col_c2, col_c3 = st.columns(3)
                with col_c1:
                    pie_chart_on = st.checkbox("📊 Sentiment Breakdown", value=True, key=f"pie_{col}")
                    keywords_on = st.checkbox("🔤 Top Keywords (List)", value=True, key=f"kw_{col}")
                with col_c2:
                    frequent_on = st.checkbox("📋 Frequent Responses", value=True, key=f"freq_{col}")
                with col_c3:
                    gemini_on = st.checkbox("🧠 Gemini Summary", value=True, key=f"gemini_{col}")
                    word_cloud_on = st.checkbox("☁️ Word Cloud", value=True, key=f"wc_{col}")

                st.markdown("---")
                col1, col2 = st.columns(2)

                with col1:
                    if pie_chart_on:
                        st.markdown("### 📊 Sentiment Breakdown")
                        if sentiments:
                            pie_df = pd.DataFrame(Counter(sentiments).items(), columns=["Sentiment", "Count"])
                            fig = px.pie(pie_df, names="Sentiment", values="Count")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No meaningful responses for sentiment analysis.")

                    if keywords_on:
                        st.markdown("### 🔤 Top Keywords")
                        if kws:
                            st.markdown("\n".join([f"- **{kw}** (Score: {score:.2f})" for kw, score in kws]))
                        else:
                            st.info("No significant keywords found.")

                    if frequent_on:
                        st.markdown("### 📋 Frequent Responses")
                        freq_df = pd.Series(meaningful_responses).value_counts().reset_index()
                        freq_df.columns = ["Response", "Count"]
                        freq_df = freq_df[freq_df["Response"].apply(lambda x: len(x.split()) >= 2 and len(x) > 10)]
                        if not freq_df.empty:
                            st.dataframe(freq_df.head(10), use_container_width=True)
                        else:
                            st.info("No frequent multi-word responses.")

                with col2:
                    if word_cloud_on:
                        st.markdown("### ☁️ Word Cloud")
                        text_for_wordcloud = " ".join(meaningful_responses)
                        wc_img = generate_wordcloud(text_for_wordcloud)
                        if wc_img:
                            fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
                            ax_wc.imshow(wc_img, interpolation='bilinear')
                            ax_wc.axis('off')
                            st.pyplot(fig_wc)
                        else:
                            st.info("Not enough words for a word cloud.")

                    if gemini_on:
                        st.markdown("### 🧠 Gemini Summary")
                        try:
                            sample_text = "\n".join(pd.Series(meaningful_responses).sample(min(25, len(meaningful_responses)), random_state=42))
                            prompt = f"""Provide a concise, actionable summary of these feedback responses to the question "{col}". Include overall sentiment and key suggestions:\n\n{sample_text}"""
                            reply = gemini.generate_content(prompt)
                            st.info(reply.text.strip())
                        except Exception as e:
                            st.error(f"Gemini Error: {e}")

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
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

        st.markdown("---")
        st.markdown("## 💬 Ask Gemini About All Feedback")
        userq = st.text_input("Ask your question about the feedback:")
        if st.button("Ask Gemini"):
            try:
                tabular = pd.DataFrame(summary_data).to_markdown(index=False)
                prompt = f"""You're a feedback analyst. Based on the table below:\n\n{tabular}\n\nAnswer this question:\n{userq}"""
                final = gemini.generate_content(prompt)
                st.info(final.text.strip())
            except Exception as e:
                st.error(f"Gemini Error: {e}")
elif uploaded:
    st.warning("Gemini API key not configured or invalid.")
else:
    st.info("Upload a CSV or Excel file to begin.")

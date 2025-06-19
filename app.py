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
    # ... (same stop words list as before)
] + list(string.punctuation))

def simple_tokenize(text):
    cleaned_text = re.sub(r'\d+', '', text.lower())
    cleaned_text = re.sub(rf"[{re.escape(string.punctuation)}]", "", cleaned_text)
    tokens = cleaned_text.split()
    return [token for token in tokens if token.strip()]

def init_gemini(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")

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
    except Exception as e:
        st.error(f"Keyword extraction error: {e}")
        return []

@st.cache_data
def generate_wordcloud(text_corpus):
    if not text_corpus.strip():
        return None
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=STOP_WORDS, min_font_size=10).generate(text_corpus)
    return wordcloud

# 💅 Custom CSS Styling
st.set_page_config(page_title="📋 Feedback Analyzer", layout="wide", page_icon="📊")
st.markdown("""
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        .stDataFrame th, .stDataFrame td { text-align: left !important; }
        .stTabs [data-baseweb="tab-list"] { flex-wrap: wrap; }
        .stTabs [role="tab"] { font-size: 1rem; padding: 0.5rem 1rem; }
    </style>
""", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>📋 Feedback Analyzer Tool</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Sidebar setup
with st.sidebar:
    st.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", width=150)
    st.markdown("## 🔐 Gemini API Key (Pre-configured)")
    st.caption("This app uses a secure internal API key to connect to Google Gemini.")
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
        tab_list = st.tabs([f"🔍 {col}" for col in selected])

        for i, col in enumerate(selected):
            with tab_list[i]:
                responses = df[col].astype(str).dropna().tolist()
                meaningful_responses = [r for r in responses if len(r.strip()) > 5 and not r.isnumeric()]
                sentiments, kws = [], []

                if meaningful_responses:
                    sentiments = classify_sentiments(meaningful_responses)
                    kws = extract_keywords_tfidf(meaningful_responses)

                st.markdown("### Choose Analysis Types:")
                col1, col2, col3 = st.columns(3)
                with col1:
                    pie_chart_on = st.checkbox("📊 Sentiment Breakdown", value=True, key=f"pie_{col}")
                    keywords_on = st.checkbox("🔤 Top Keywords", value=True, key=f"kw_{col}")
                with col2:
                    frequent_on = st.checkbox("📋 Frequent Responses", value=True, key=f"freq_{col}")
                with col3:
                    gemini_on = st.checkbox("🧠 Gemini Summary", value=True, key=f"gem_{col}")
                    wordcloud_on = st.checkbox("☁️ Word Cloud", value=True, key=f"wc_{col}")

                st.markdown("---")
                lcol, rcol = st.columns(2)
                with lcol:
                    if pie_chart_on and sentiments:
                        pie_df = pd.DataFrame(Counter(sentiments).items(), columns=["Sentiment", "Count"])
                        fig = px.pie(pie_df, names="Sentiment", values="Count")
                        st.plotly_chart(fig, use_container_width=True)

                    if keywords_on:
                        if kws:
                            st.markdown("\n".join([f"- **{kw}** (Score: {score:.2f})" for kw, score in kws]))
                        else:
                            st.info("No significant keywords found.")

                    if frequent_on:
                        freq_df = pd.Series(meaningful_responses).value_counts().reset_index()
                        freq_df.columns = ["Response", "Count"]
                        freq_df = freq_df[freq_df["Response"].apply(lambda x: len(x.split()) >= 2 and len(x) > 10)]
                        if not freq_df.empty:
                            st.dataframe(freq_df.head(10), use_container_width=True)
                        else:
                            st.info("No frequent multi-word responses.")

                with rcol:
                    if wordcloud_on and meaningful_responses:
                        text_wc = " ".join(meaningful_responses)
                        wc_img = generate_wordcloud(text_wc)
                        if wc_img:
                            fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
                            ax_wc.imshow(wc_img, interpolation='bilinear')
                            ax_wc.axis('off')
                            st.pyplot(fig_wc)

                    if gemini_on and meaningful_responses:
                        try:
                            sample = "\n".join(pd.Series(meaningful_responses).sample(min(25, len(meaningful_responses)), random_state=42))
                            prompt = f"""Summarize the main points, tone, and suggestions for the question '{col}'.\n\nFeedback:\n{sample}"""
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

        st.markdown("<h2 style='color:#4CAF50;'>🧾 Overall Feedback Summary</h2>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(summary_data).style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)

        st.markdown("<h2 style='color:#FF9800;'>💬 Ask Gemini About All Feedback</h2>", unsafe_allow_html=True)
        st.caption("Use this to ask a specific question about the feedback (e.g., 'What needs improvement?')")
        userq = st.text_input("Ask a question:")
        if st.button("Ask Gemini"):
            try:
                tabular = pd.DataFrame(summary_data).to_markdown(index=False)
                prompt = f"You're a feedback analyst. Based on this table:\n\n{tabular}\n\nAnswer this question:\n{userq}"
                final = gemini.generate_content(prompt)
                st.info(final.text.strip())
            except Exception as e:
                st.error(f"Gemini Error: {e}")

st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>Built with ❤️ using Streamlit and Gemini AI</p>", unsafe_allow_html=True)

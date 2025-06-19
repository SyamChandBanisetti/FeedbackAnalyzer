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
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from fpdf import FPDF
from io import BytesIO
import os

# --- Load API Key from .env ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Gemini Setup ---
def init_gemini():
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel("gemini-1.5-flash")

# --- Basic Stop Words ---
STOP_WORDS = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all",
    "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don",
    "should", "now"
] + list(string.punctuation))

# --- Tokenizer ---
def simple_tokenize(text):
    text = re.sub(r'\d+', '', text.lower())
    text = re.sub(rf"[{re.escape(string.punctuation)}]", "", text)
    return [word for word in text.split() if word not in STOP_WORDS and len(word) > 1]

# --- Data Preprocessing ---
def preprocess_and_save(file):
    try:
        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        df.fillna("", inplace=True)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        df.to_csv(tmp.name, index=False, quoting=csv.QUOTE_ALL)
        return tmp.name, df.columns.tolist(), df
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None, None

# --- Sentiment Classifier ---
@st.cache_data
def classify_sentiments(texts):
    sentiments = []
    for t in texts:
        t = t.lower()
        if any(w in t for w in ["good", "great", "excellent", "love", "awesome", "helpful", "positive", "happy", "nice"]):
            sentiments.append("Positive")
        elif any(w in t for w in ["bad", "poor", "terrible", "hate", "worst", "issue", "problem", "slow", "unhappy"]):
            sentiments.append("Negative")
        else:
            sentiments.append("Neutral")
    return sentiments

# --- TF-IDF Keywords ---
@st.cache_data
def extract_keywords_tfidf(texts, top_n=10):
    cleaned = [" ".join(simple_tokenize(t)) for t in texts if len(t.strip()) > 5]
    try:
        vec = TfidfVectorizer(max_features=top_n)
        X = vec.fit_transform(cleaned)
        kws = vec.get_feature_names_out()
        scores = X.sum(axis=0).A1
        return sorted(zip(kws, scores), key=lambda x: x[1], reverse=True)
    except:
        return []

# --- Word Cloud ---
@st.cache_data
def generate_wordcloud(text):
    if not text.strip(): return None
    return WordCloud(width=800, height=400, background_color='white', stopwords=STOP_WORDS).generate(text)

# --- Gemini Summary ---
def get_gemini_summary(gemini, question, responses):
    sample = "\n".join(pd.Series(responses).sample(min(25, len(responses)), random_state=42))
    prompt = f"""
You are an expert feedback analyst. Analyze the following feedback responses for the question: "{question}".

- Summarize key sentiments (positive, negative, neutral).
- Highlight key themes and concerns.
- Offer 3-5 bullet-point suggestions to improve based on responses.

Responses:
{sample}
"""
    reply = gemini.generate_content(prompt)
    return reply.text.strip()

# --- PDF Report Generation ---
def generate_pdf(summary_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Feedback Analysis Report", ln=True, align="C")
    for row in summary_data:
        pdf.ln(5)
        pdf.set_font("Arial", style='B', size=11)
        pdf.cell(0, 8, txt=f"Question: {row['Question']}", ln=True)
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 6, txt=f"Total: {row['Total Responses']} | 👍 {row['👍 Positive']} | 👎 {row['👎 Negative']} | 😐 {row['😐 Neutral']}", ln=True)
        pdf.multi_cell(0, 6, txt=f"Top Keywords: {row['Top Keywords']}")
    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer

# --- Streamlit App ---
st.set_page_config("📋 Feedback Analyzer", layout="wide")
st.title("📋 Feedback Analyzer Tool")

# Gemini Init
try:
    gemini = init_gemini()
    st.sidebar.success("✅ Gemini Connected")
except Exception as e:
    gemini = None
    st.sidebar.error(f"❌ Gemini Error: {e}")

# File Upload
uploaded = st.file_uploader("📤 Upload CSV or Excel", type=["csv", "xlsx"])
if uploaded and gemini:
    path, cols, df = preprocess_and_save(uploaded)
    if path:
        st.subheader("📄 Preview")
        st.dataframe(df, use_container_width=True)

        ignore = ["name", "email", "id", "timestamp"]
        text_cols = [c for c in df.select_dtypes('object').columns if c.lower() not in ignore]
        selected = st.multiselect("✅ Select Feedback Questions", text_cols, default=text_cols[:2])

        summary_data = []

        for i, col in enumerate(selected):
            responses = df[col].astype(str).dropna().tolist()
            meaningful = [r for r in responses if len(r.strip()) > 5 and not r.isnumeric()]
            sentiments = classify_sentiments(meaningful)
            keywords = extract_keywords_tfidf(meaningful)

            st.markdown(f"---\n### 🔍 **{col}**")
            with st.expander(f"Expand analysis for **{col}**", expanded=True):
                c1, c2 = st.columns(2)

                with c1:
                    if sentiments:
                        st.markdown("#### 📊 Sentiment Breakdown")
                        pie_df = pd.DataFrame(Counter(sentiments).items(), columns=["Sentiment", "Count"])
                        fig = px.pie(pie_df, names="Sentiment", values="Count")
                        st.plotly_chart(fig, use_container_width=True)

                    if meaningful:
                        st.markdown("#### ☁️ Word Cloud")
                        wc = generate_wordcloud(" ".join(meaningful))
                        if wc:
                            fig_wc, ax_wc = plt.subplots(figsize=(8, 4))
                            ax_wc.imshow(wc, interpolation='bilinear')
                            ax_wc.axis('off')
                            st.pyplot(fig_wc)

                with c2:
                    if keywords:
                        st.markdown("#### 🔤 Top Keywords")
                        for kw, score in keywords:
                            st.markdown(f"- **{kw}** ({score:.2f})")

                    st.markdown("#### 📋 Frequent Responses")
                    freq_df = pd.Series(meaningful).value_counts().reset_index()
                    freq_df.columns = ["Response", "Count"]
                    freq_df = freq_df[freq_df["Response"].apply(lambda x: len(x.split()) >= 2 and len(x) > 10)]
                    st.dataframe(freq_df.head(10), use_container_width=True)

                st.markdown("#### 🧠 Gemini Summary")
                try:
                    summary = get_gemini_summary(gemini, col, meaningful)
                    st.markdown(summary)
                except Exception as e:
                    st.error(f"Gemini Summary Error: {e}")

            summary_data.append({
                "Question": col,
                "Total Responses": len(responses),
                "👍 Positive": sentiments.count("Positive"),
                "👎 Negative": sentiments.count("Negative"),
                "😐 Neutral": sentiments.count("Neutral"),
                "Top Keywords": ", ".join([kw for kw, _ in keywords]) if keywords else "N/A"
            })

        # Overall Summary
        st.markdown("---\n## 🧾 Overall Feedback Summary")
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)

        # Ask Gemini
        st.markdown("## 💬 Ask Gemini Anything About Feedback")
        user_q = st.text_input("Ask something about the feedback...")
        if st.button("Ask Gemini"):
            try:
                prompt = f"""You're an expert feedback consultant. Here's a summary:\n{summary_df.to_markdown(index=False)}\nNow answer this:\n{user_q}"""
                reply = gemini.generate_content(prompt)
                st.markdown("### 🧠 Gemini Answer")
                st.info(reply.text.strip())
            except Exception as e:
                st.error(f"Gemini Error: {e}")

        # Download PDF
        st.markdown("## 📥 Download Report")
        pdf = generate_pdf(summary_data)
        st.download_button("📄 Download Summary as PDF", data=pdf, file_name="feedback_report.pdf", mime="application/pdf")

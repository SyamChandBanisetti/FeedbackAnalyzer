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

STOP_WORDS = set(map(str, [
    # ... your stop words
    "i", "me", "my", "myself", "we", "our", "ours",  # etc.
] + list(string.punctuation)))
  # Omitted for brevity

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
    for t in texts:
        if not isinstance(t, str):
            t = str(t)  # Safely convert non-string input
        t = t.lower()
        if any(w in t for w in ["good", "great", "excellent", "love", "awesome", "helpful", "satisfied", "positive", "happy", "nice"]):
            sentiments.append("Positive")
        elif any(w in t for w in ["bad", "poor", "terrible", "hate", "worst", "issue", "problem", "slow", "unhappy"]):
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
        st.warning(f"Could not extract keywords: {e}.")
        return []
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return []

@st.cache_data
def generate_wordcloud(text_corpus):
    if not text_corpus.strip():
        return None
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=STOP_WORDS, min_font_size=10).generate(text_corpus)
    return wordcloud

st.set_page_config("Feedback Analyzer", layout="wide")
st.title("📋 Feedback Analyzer Tool")

with st.sidebar:
    st.header("🔐 Gemini API Key")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    if api_key:
        try:
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
    else:
        st.warning("Please enter your Gemini API key to enable AI features.")
        gemini = None

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
            sentiments = []
            kws = []
            if meaningful_responses:
                sentiments = classify_sentiments(meaningful_responses)
                kws = extract_keywords_tfidf(meaningful_responses)

            st.markdown(f"---")
            with st.expander(f"🔍 Analysis: **{col}**", expanded=True):
                pie_chart_on = st.checkbox("📊 Sentiment Breakdown", value=True, key=f"pie_toggle_{col}_{i}")
                keywords_on = st.checkbox("🔤 Top Keywords (List)", value=True, key=f"kw_toggle_{col}_{i}")
                frequent_on = st.checkbox("📋 Frequent Responses", value=True, key=f"freq_toggle_{col}_{i}")
                gemini_on = st.checkbox("🧠 Gemini Summary", value=True, key=f"gem_sum_toggle_{col}_{i}")
                word_cloud_on = st.checkbox("☁️ Word Cloud", value=True, key=f"wc_toggle_{col}_{i}")

                if pie_chart_on:
                    st.markdown("### 📊 Sentiment Breakdown")
                    if sentiments:
                        pie_df = pd.DataFrame(Counter(sentiments).items(), columns=["Sentiment", "Count"])
                        fig = px.pie(pie_df, names="Sentiment", values="Count", title="Sentiment Breakdown")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No meaningful responses for sentiment analysis.")

                if keywords_on:
                    st.markdown("### 🔤 Top Keywords")
                    if kws:
                        keyword_strings = [f"- **{kw}** (Score: {score:.2f})" for kw, score in kws]
                        st.markdown("\n".join(keyword_strings))
                    else:
                        st.info("No significant keywords found.")

                if frequent_on:
                    st.markdown("### 📋 Frequent Responses (2+ words)")
                    if meaningful_responses:
                        freq_df = pd.Series(meaningful_responses).value_counts().reset_index()
                        freq_df.columns = ["Response", "Count"]
                        freq_df = freq_df[freq_df["Response"].apply(lambda x: len(x.split()) >= 2 and len(x) > 10)]
                        if not freq_df.empty:
                            st.dataframe(freq_df.head(10), use_container_width=True)
                        else:
                            st.info("No frequent multi-word responses found.")
                    else:
                        st.info("No meaningful responses to analyze.")

                if word_cloud_on:
                    st.markdown("### ☁️ Word Cloud")
                    if meaningful_responses:
                        text_for_wordcloud = " ".join(meaningful_responses)
                        wordcloud_img = generate_wordcloud(text_for_wordcloud)
                        if wordcloud_img:
                            fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
                            ax_wc.imshow(wordcloud_img, interpolation='bilinear')
                            ax_wc.axis('off')
                            st.pyplot(fig_wc)
                        else:
                            st.info("Not enough content for word cloud.")
                    else:
                        st.info("No responses for word cloud generation.")

                if gemini_on:
                    st.markdown("### 🧠 Gemini Summary")
                    try:
                        if meaningful_responses:
                            sample_responses = "\n".join(pd.Series(meaningful_responses).dropna().sample(min(25, len(meaningful_responses)), random_state=42))
                            prompt = f"""You're an expert advisor analyzing survey feedback. Based on the following responses to the question: \"{col}\", generate a helpful, professional summary with key takeaways and 3–5 clear, bullet-pointed suggestions for improvement. Highlight sentiment and actionable advice.

Feedbacks:
{sample_responses}"""
                            reply = gemini.generate_content(prompt)
                            st.info(reply.text.strip())
                        else:
                            st.info("Not enough responses for Gemini summary.")
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
            if gemini:
                try:
                    tabular = summary_df.to_markdown(index=False)
                    prompt = f"""You're a feedback report analyst. Given this summary table:\n\n{tabular}\n\nAnswer this question:\n{userq}\n\nProvide a concise and direct answer with key suggestions based on the data."""
                    final = gemini.generate_content(prompt)
                    st.markdown("### 🧠 Gemini Answer")
                    st.info(final.text.strip())
                except Exception as e:
                    st.error(f"Gemini Error: {e}")
            else:
                st.warning("Gemini is not initialized.")

elif uploaded and not api_key:
    st.warning("Please enter your Gemini API key in the sidebar.")
elif not uploaded:
    st.info("Upload a CSV or Excel file to begin feedback analysis.")

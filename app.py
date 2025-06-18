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
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import nltk

# --- NLTK Data Download and Setup ---
# Use a placeholder to update download messages dynamically
download_status_placeholder = st.empty()

try:
    # Attempt to find the required NLTK data.
    # If not found, a DownloadError will be raised by find().
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    download_status_placeholder.success("NLTK data (punkt, stopwords) found and ready.")
except Exception as e: # Catch any exception, including DownloadError or others
    download_status_placeholder.warning(f"NLTK data not fully found or error during check: {e}. Attempting download...")
    try:
        # Download with quiet=True to prevent excessive console output in console
        # Use an empty st.empty() to clear the message once downloaded.
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        download_status_placeholder.success("NLTK data downloaded successfully and ready!")
    except Exception as download_error:
        download_status_placeholder.error(f"Failed to download NLTK data: {download_error}. Please check your internet connection or deployment environment.")
        st.stop() # Stop the app if crucial NLTK data can't be downloaded

# Initialize stop_words *after* ensuring NLTK stopwords are available
try:
    STOP_WORDS = set(nltk.corpus.stopwords.words('english') + list(string.punctuation))
except LookupError:
    # This block should ideally not be reached if the above download succeeded
    st.error("NLTK stopwords data not available even after download attempt. Please restart the app.")
    st.stop() # Crucial dependency, stop if it fails

# --- End NLTK Data Download and Setup ---


# 🔐 Gemini Setup
def init_gemini(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")

# 📁 Load & Clean File
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

# 🧠 Sentiment Classifier
@st.cache_data
def classify_sentiments(texts):
    sentiments = []
    for text in texts:
        t = text.lower()
        # Expanded keywords for better classification
        if any(w in t for w in ["good", "great", "excellent", "love", "awesome", "helpful", "satisfied", "nice", "positive", "happy", "smooth", "efficient", "recommend", "super", "perfect", "enjoy"]):
            sentiments.append("Positive")
        elif any(w in t for w in ["bad", "poor", "terrible", "hate", "worst", "boring", "rude", "unsatisfied", "negative", "unhappy", "difficult", "issue", "problem", "frustrating", "slow", "broken", "complaint"]):
            sentiments.append("Negative")
        else:
            sentiments.append("Neutral")
    return sentiments

# 🔤 TF-IDF Keywords
@st.cache_data
def extract_keywords_tfidf(texts, top_n=10):
    # Ensure texts are not empty or contain only whitespace after cleaning
    clean = [re.sub(rf"[{string.punctuation}]", "", t.lower()) for t in texts if len(t.strip()) > 2]
    clean = [t for t in clean if t and not t.isspace()]
    if not clean:
        return [] # Return empty list if no clean text
    try:
        vec = TfidfVectorizer(stop_words='english', max_features=top_n)
        X = vec.fit_transform(clean)
        kw = vec.get_feature_names_out()
        scores = X.sum(axis=0).A1
        return sorted(zip(kw, scores), key=lambda x: x[1], reverse=True)
    except ValueError as e: # Handle cases where vocabulary cannot be built (e.g., all texts are identical or too short)
        st.warning(f"Could not extract keywords: {e}. Not enough unique words in the provided texts.")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred during keyword extraction: {e}")
        return []

# 🔄 N-Gram Analysis
@st.cache_data
def extract_ngrams(texts, n=2, top_n=10):
    all_ngrams = []
    
    for text in texts:
        # Clean text: remove numbers, convert to lower, tokenize, filter stop words and short tokens
        cleaned_text = re.sub(r'\d+', '', text.lower()) # Remove numbers
        cleaned_text = re.sub(rf"[{string.punctuation}]", "", cleaned_text) # Remove punctuation
        tokens = word_tokenize(cleaned_text)
        # Use the globally initialized STOP_WORDS
        filtered_tokens = [word for word in tokens if word.isalpha() and word not in STOP_WORDS and len(word) > 1] # len > 1 to remove single letters
        
        if len(filtered_tokens) >= n: # Ensure enough tokens to form n-grams
            all_ngrams.extend(list(ngrams(filtered_tokens, n)))

    ngram_counts = Counter(all_ngrams)
    # Convert tuple n-grams to string for display
    return [(" ".join(ngram), count) for ngram, count in ngram_counts.most_common(top_n)]


# 📊 Feedback Analyzer App
st.set_page_config("Feedback Analyzer", layout="wide")
st.title("📋 Feedback Analyzer using Gemini Flash 2.0")

# Sidebar: Gemini API Key
with st.sidebar:
    st.header("🔐 Gemini API Key")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    if api_key:
        try:
            gemini = init_gemini(api_key)
            # Test Gemini connection
            test_response = gemini.generate_content("hello", safety_settings={'HARASSMENT': 'block_none'})
            if test_response:
                st.success("Gemini is ready and connected!")
            else:
                st.error("Gemini connection failed. Check API key or network.")
                gemini = None # Ensure gemini is None if connection fails
        except Exception as e:
            st.error(f"Error connecting to Gemini: {e}. Please check your API key.")
            gemini = None # Ensure gemini is None if connection fails
    else:
        st.warning("Please enter your Gemini API key in the sidebar to enable AI features.")
        gemini = None


# Upload File
uploaded = st.file_uploader("📤 Upload CSV or Excel Feedback File", type=["csv", "xlsx"])

if uploaded and api_key and gemini: # Proceed only if file uploaded and Gemini is ready
    path, cols, df = preprocess_and_save(uploaded)
    if path:
        st.subheader("📄 Preview Data")
        st.dataframe(df, use_container_width=True)

        ignore = ["name", "email", "id", "timestamp"]
        text_cols = [c for c in df.select_dtypes(include='object').columns if c.lower() not in ignore]

        # 🎯 Select Questions
        st.markdown("---")
        st.markdown("## ✅ Select Questions to Analyze")
        selected = st.multiselect("Choose Questions", text_cols, default=text_cols[:min(len(text_cols), 2)]) # Select up to 2 by default

        summary_data = []

        for i, col in enumerate(selected):
            responses = df[col].astype(str).dropna().tolist()
            # Filter out very short or non-meaningful responses for analysis
            meaningful_responses = [r for r in responses if len(r.strip()) > 5 and not r.isnumeric()]

            # Initialize with empty lists to avoid errors if no meaningful responses
            sentiments = []
            kws = []
            ngrams_list = []

            if meaningful_responses:
                sentiments = classify_sentiments(meaningful_responses)
                kws = extract_keywords_tfidf(meaningful_responses)
                ngrams_list = extract_ngrams(meaningful_responses, n=2) # Default to bigrams

            st.markdown(f"---")
            with st.expander(f"🔍 Analysis: **{col}**", expanded=True):
                # Checkboxes for analysis types
                st.markdown("### Choose Analysis Types for this Question:")
                col_c1, col_c2, col_c3 = st.columns(3)
                with col_c1:
                    pie_chart_on = st.checkbox("📊 Sentiment Breakdown", value=True, key=f"pie_toggle_{col}_{i}")
                    keywords_on = st.checkbox("🔤 Top Keywords (List)", value=True, key=f"kw_toggle_{col}_{i}")
                with col_c2:
                    frequent_on = st.checkbox("📋 Frequent Responses", value=True, key=f"freq_toggle_{col}_{i}")
                    ngrams_on = st.checkbox("🧩 Top N-Grams (Phrases)", value=True, key=f"ngram_toggle_{col}_{i}")
                with col_c3:
                    gemini_on = st.checkbox("🧠 Gemini Summary", value=True, key=f"gem_sum_toggle_{col}_{i}")
                    word_cloud_on = st.checkbox("☁️ Word Cloud (Upcoming)", value=False, key=f"wc_toggle_{col}_{i}")


                st.markdown("---") # Separator after toggles

                col1, col2 = st.columns(2) # Two columns for side-by-side display

                with col1:
                    if pie_chart_on:
                        st.markdown("### 📊 Sentiment Breakdown")
                        if sentiments:
                            pie_df = pd.DataFrame(Counter(sentiments).items(), columns=["Sentiment", "Count"])
                            fig = px.pie(pie_df, names="Sentiment", values="Count", title="Sentiment Breakdown")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No meaningful responses available for sentiment analysis for this question.")

                    if keywords_on:
                        st.markdown("### 🔤 Top Keywords")
                        if kws: # Check if kws is not empty
                            keyword_strings = [f"- **{kw}** (Score: {score:.2f})" for kw, score in kws]
                            st.markdown("\n".join(keyword_strings))
                        else:
                            st.info("No significant keywords found for this question.")

                    if frequent_on:
                        st.markdown("### 📋 Frequent Responses (2+ words)")
                        if meaningful_responses:
                            freq_df = pd.Series(meaningful_responses).value_counts().reset_index()
                            freq_df.columns = ["Response", "Count"]
                            # Filter for responses with at least 2 words and reasonable length
                            freq_df = freq_df[freq_df["Response"].apply(lambda x: len(x.split()) >= 2 and len(x) > 10)]
                            if not freq_df.empty:
                                st.dataframe(freq_df.head(10), use_container_width=True)
                            else:
                                st.info("No frequent multi-word responses (with more than 10 characters) found for this question.")
                        else:
                            st.info("No meaningful responses to find frequent patterns.")

                with col2:
                    if ngrams_on:
                        st.markdown("### 🧩 Top N-Grams (Common Phrases)")
                        if ngrams_list: # Check if ngrams_list is not empty
                            ngram_strings = [f"- **'{phrase}'** (Count: {count})" for phrase, count in ngrams_list]
                            st.markdown("\n".join(ngram_strings))
                        else:
                            st.info("No significant N-Grams (common phrases) found for this question.")

                    if word_cloud_on:
                        st.markdown("### ☁️ Word Cloud")
                        st.info("Word Cloud feature is planned. You can integrate libraries like `wordcloud` and `matplotlib` to display it here.")

                    if gemini_on:
                        st.markdown("### 🧠 Gemini Summary")
                        try:
                            if meaningful_responses:
                                # Sample more responses if available for better summary quality
                                sample_responses = "\n".join(pd.Series(meaningful_responses).dropna().sample(min(25, len(meaningful_responses)), random_state=42))
                                prompt = f"""Provide a concise, useful, and actionable summary of the key themes, overall sentiment (positive/negative/neutral breakdown), and specific suggestions for improvement from the following responses to the question: "{col}". Aim for a summary that is easy to read and provides valuable insights, without being too long or too short. Focus on a length that is 'considerable' and 'just useful'.

                                Feedbacks:
                                {sample_responses}"""
                                reply = gemini.generate_content(prompt)
                                st.info(reply.text.strip())
                            else:
                                st.info("Not enough meaningful responses to generate a Gemini summary for this question.")
                        except Exception as e:
                            st.error(f"Gemini Error for '{col}': {e}. Please ensure your API key is valid and there are sufficient responses.")

            # Collect for overall summary
            summary_data.append({
                "Question": col,
                "Total Responses": len(responses),
                "👍 Positive": sentiments.count("Positive"),
                "👎 Negative": sentiments.count("Negative"),
                "😐 Neutral": sentiments.count("Neutral"),
                "Top Keywords": ", ".join([kw for kw, _ in kws]) if kws else "N/A",
                "Top N-Grams": ", ".join([ng for ng, _ in ngrams_list]) if ngrams_list else "N/A"
            })

        # ---
        # 📋 Overall Summary Table
        st.markdown("---")
        st.markdown("## 🧾 Overall Feedback Summary")
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)

        # ---
        # 💬 Ask Gemini
        st.markdown("---")
        st.markdown("## 💬 Ask Gemini About All Feedback")
        userq = st.text_input("Ask your question about the overall feedback insights")
        if st.button("Ask Gemini about overall feedback"):
            if gemini:
                try:
                    tabular = summary_df.to_markdown(index=False)
                    prompt = f"""You're a feedback report analyst. Given this summary table:\n\n{tabular}\n\nAnswer this question:\n{userq}\n\nProvide a concise and direct answer, focusing on actionable insights derived from the data."""
                    final = gemini.generate_content(prompt)
                    st.markdown("### 🧠 Gemini Answer")
                    st.info(final.text.strip())
                except Exception as e:
                    st.error(f"Gemini Error: {e}. Please ensure your API key is valid.")
            else:
                st.warning("Gemini is not initialized. Please enter your API key.")

elif uploaded and not api_key:
    st.warning("Please enter your Gemini API key in the sidebar to proceed with analysis.")
elif not uploaded:
    st.info("Upload a CSV or Excel file to begin feedback analysis.")

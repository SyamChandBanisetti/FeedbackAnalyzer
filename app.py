import streamlit as st
import pandas as pd
import tempfile
import csv
import re
import string
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter, defaultdict
import google.generativeai as genai
import os
import networkx as nx
import matplotlib.pyplot as plt
import base64 # For embedding images in HTML
from io import BytesIO # For saving plot images to memory
from wordcloud import WordCloud # Add this line

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

# --- End NLTK Replacement ---


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
        if any(w in t for w in ["good", "great", "excellent", "love", "awesome", "helpful", "satisfied", "nice", "positive", "happy", "smooth", "efficient", "recommend", "super", "perfect", "enjoy", "pleased", "fantastic", "brilliant", "outstanding"]):
            sentiments.append("Positive")
        elif any(w in t for w in ["bad", "poor", "terrible", "hate", "worst", "boring", "rude", "unsatisfied", "negative", "unhappy", "difficult", "issue", "problem", "frustrating", "slow", "broken", "complaint", "disappointing", "awful", "unacceptable", "bug", "error"]):
            sentiments.append("Negative")
        else:
            sentiments.append("Neutral")
    return sentiments

# 🔤 TF-IDF Keywords
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
        # st.warning(f"Could not extract keywords: {e}. Not enough unique words in the provided texts.")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred during keyword extraction: {e}")
        return []

# 🔗 N-Gram Analysis
@st.cache_data
def extract_ngrams(texts, n_range=(2, 2), top_n=10):
    processed_texts = [" ".join([token for token in simple_tokenize(t) if token not in STOP_WORDS and len(token) > 1]) for t in texts if len(t.strip()) > 2]
    processed_texts = [t for t in processed_texts if t and not t.isspace()]

    if not processed_texts:
        return []

    try:
        vectorizer = TfidfVectorizer(ngram_range=n_range, max_features=top_n)
        X = vectorizer.fit_transform(processed_texts)
        ngrams = vectorizer.get_feature_names_out()
        scores = X.sum(axis=0).A1
        return sorted(zip(ngrams, scores), key=lambda x: x[1], reverse=True)
    except ValueError as e:
        # st.warning(f"Could not extract N-grams (range {n_range}): {e}. Not enough meaningful phrases in the provided texts.")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred during N-gram extraction: {e}")
        return []

# 📏 Response Length Analysis
@st.cache_data
def analyze_response_length(texts):
    lengths = [len(simple_tokenize(text)) for text in texts if len(text.strip()) > 2]
    if not lengths:
        return pd.DataFrame(), {} # Return empty dict for stats

    length_df = pd.DataFrame(lengths, columns=["Word Count"])

    stats = {
        "Average Word Count": length_df["Word Count"].mean(),
        "Median Word Count": length_df["Word Count"].median(),
        "Min Word Count": length_df["Word Count"].min(),
        "Max Word Count": length_df["Word Count"].max()
    }

    return length_df, stats

# ☁️ Word Cloud Generation
@st.cache_data
def generate_wordcloud(text_corpus):
    if not text_corpus.strip():
        return None
    wordcloud = WordCloud(width=800, height=400, background_color='white',
                            stopwords=STOP_WORDS, min_font_size=10).generate(text_corpus)
    return wordcloud

# ✨ New: Emotion Detection (Keyword-based Approximation)
@st.cache_data
def detect_emotions(texts):
    emotion_keywords = {
        "Joy": ["happy", "great", "love", "excellent", "pleased", "delighted", "fantastic", "awesome", "enjoy", "best"],
        "Sadness": ["sad", "unhappy", "disappointed", "frustrated", "bad", "poor", "terrible", "regret", "pity", "dreadful"],
        "Anger": ["angry", "frustrating", "issue", "problem", "bug", "slow", "broken", "hate", "worst", "annoyed", "ridiculous", "unacceptable"],
        "Surprise": ["surprise", "unexpected", "wow", "amazing", "shocked", "suddenly", "unbelievable"],
        "Fear": ["fear", "anxious", "worried", "concerned", "scared", "risk", "uncertainty"],
        "Trust": ["trust", "reliable", "dependable", "secure", "safe", "confident"],
        "Anticipation": ["looking forward", "excited", "eager", "hope", "soon", "future", "expecting"],
        "Disgust": ["disgust", "gross", "awful", "horrible", "nasty", "sickening"]
    }

    emotion_counts = defaultdict(int)
    total_meaningful_responses = 0

    for text in texts:
        if len(text.strip()) < 5:
            continue
        total_meaningful_responses += 1
        tokens = simple_tokenize(text)
        detected_in_response = set()

        for emotion, keywords in emotion_keywords.items():
            if any(keyword in tokens for keyword in keywords):
                detected_in_response.add(emotion)

        for emotion in detected_in_response:
            emotion_counts[emotion] += 1

    if not emotion_counts:
        return pd.DataFrame() # Return empty DataFrame

    emotion_df = pd.DataFrame(emotion_counts.items(), columns=["Emotion", "Count"])
    if total_meaningful_responses > 0:
        emotion_df["Percentage"] = (emotion_df["Count"] / total_meaningful_responses) * 100
    else:
        emotion_df["Percentage"] = 0

    emotion_df = emotion_df.sort_values(by="Percentage", ascending=False)

    return emotion_df


# ✨ New: Customer Effort Score (CES) Approximation
@st.cache_data
def calculate_ces_approximation(texts):
    high_effort_keywords = ["hard", "difficult", "struggle", "frustrating", "confusing", "complex", "took long", "complicated", "unclear", "trouble", "issue", "problem", "impossible", "couldn't"]
    low_effort_keywords = ["easy", "simple", "seamless", "quick", "fast", "efficient", "smooth", "effortless", "straightforward", "convenient", "no problem"]

    ces_categories = []

    for text in texts:
        if len(text.strip()) < 5:
            continue
        
        score = 0
        found_high = False
        found_low = False

        # Convert text to lower for keyword matching
        lower_text = text.lower()

        for keyword in high_effort_keywords:
            if keyword in lower_text:
                score -= 1
                found_high = True
        for keyword in low_effort_keywords:
            if keyword in lower_text:
                score += 1
                found_low = True

        if found_high and not found_low:
            ces_categories.append("High Effort")
        elif found_low and not found_high:
            ces_categories.append("Low Effort")
        elif found_high and found_low:
            if score > 0:
                ces_categories.append("Mixed (leaning Low)")
            elif score < 0:
                ces_categories.append("Mixed (leaning High)")
            else:
                ces_categories.append("Neutral Effort")
        else: # Neither high nor low effort keywords found
            ces_categories.append("Neutral Effort")

    if not ces_categories:
        return pd.DataFrame()

    ces_df = pd.DataFrame(ces_categories, columns=["Effort Level"])
    ces_counts = ces_df["Effort Level"].value_counts().reset_index()
    ces_counts.columns = ["Effort Level", "Count"]

    return ces_counts

# ✨ New: Word Co-occurrence Network
@st.cache_data
def generate_cooccurrence_graph(texts, top_n_nodes=20, edge_threshold=2):
    processed_texts = [" ".join([token for token in simple_tokenize(t) if token not in STOP_WORDS and len(token) > 1]) for t in texts if t and not t.isspace()]

    if not processed_texts:
        return None

    co_occurrences = defaultdict(lambda: defaultdict(int))
    word_counts = defaultdict(int)

    for text in processed_texts:
        words = list(set(text.split()))
        for i, word1 in enumerate(words):
            word_counts[word1] += 1
            for j, word2 in enumerate(words):
                if i < j:
                    co_occurrences[word1][word2] += 1
                    co_occurrences[word2][word1] += 1

    top_words_overall = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)[:top_n_nodes]
    top_word_set = {word for word, _ in top_words_overall}

    G = nx.Graph()

    for word, count in top_words_overall:
        G.add_node(word, size=count)

    for word1 in top_word_set:
        for word2 in top_word_set:
            if word1 != word2 and co_occurrences[word1][word2] >= edge_threshold:
                G.add_edge(word1, word2, weight=co_occurrences[word1][word2])

    if not G.edges():
        return None

    return G

# Helper function to convert Matplotlib figure to Base64 image
def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=150)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig) # Close the figure to free up memory
    return f"data:image/png;base64,{img_base64}"

# --- Report Generation Function ---
def generate_html_report(overall_summary_df, question_analyses, gemini_overall_summary=None):
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Feedback Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; margin: 20px; }}
            h1, h2, h3, h4 {{ color: #0056b3; }}
            .container {{ max-width: 1200px; margin: auto; background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .section {{ margin-bottom: 30px; padding: 15px; border: 1px solid #eee; border-radius: 5px; }}
            .section h3 {{ border-bottom: 2px solid #007bff; padding-bottom: 5px; margin-bottom: 15px; }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .plot-container {{ width: 100%; overflow-x: auto; margin-bottom: 20px; }}
            .dataframe {{ overflow-x: auto; margin-bottom: 20px; }}
            .info-box {{ background-color: #e0f7fa; border-left: 5px solid #00bcd4; padding: 10px; margin-top: 10px; border-radius: 4px; }}
            .warning-box {{ background-color: #fff3e0; border-left: 5px solid #ff9800; padding: 10px; margin-top: 10px; border-radius: 4px; }}
            .image-container {{ text-align: center; margin-top: 20px; }}
            .image-container img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Feedback Analysis Report</h1>
            <p>Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

            <div class="section">
                <h2>Overall Feedback Summary</h2>
                <div class="dataframe">
                    {overall_summary_df.to_html(index=False)}
                </div>
            </div>
    """

    if gemini_overall_summary:
        html_content += f"""
            <div class="section">
                <h2>Gemini's Overall Insights</h2>
                <div class="info-box">
                    <p>{gemini_overall_summary}</p>
                </div>
            </div>
        """

    for analysis in question_analyses:
        col = analysis['question_col']
        html_content += f"""
            <div class="section">
                <h2>Analysis for: {col}</h2>
        """
        # Sentiment Breakdown
        if analysis['sentiment_plot_html']:
            html_content += f"""
                <h3>📊 Sentiment Breakdown</h3>
                <div class="plot-container">{analysis['sentiment_plot_html']}</div>
            """
        else:
            html_content += f"<div class='info-box'>No meaningful responses for sentiment analysis.</div>"

        # Top Keywords
        if analysis['top_keywords_text']:
            html_content += f"""
                <h3>🔤 Top Keywords</h3>
                <ul>{''.join([f'<li>{kw}</li>' for kw in analysis['top_keywords_text'].split(', ') if kw != 'N/A'])}</ul>
            """
        else:
            html_content += f"<div class='info-box'>No significant keywords found.</div>"

        # Top N-grams
        if analysis['top_bigrams_text'] or analysis['top_trigrams_text']:
            html_content += f"""
                <h3>🔗 Top Phrases (Bigrams & Trigrams)</h3>
            """
            if analysis['top_bigrams_text'] and analysis['top_bigrams_text'] != 'N/A':
                html_content += f"""
                    <h4>Bigrams (2-word phrases):</h4>
                    <ul>{''.join([f'<li>{gram}</li>' for gram in analysis['top_bigrams_text'].split(', ')])}</ul>
                """
            if analysis['top_trigrams_text'] and analysis['top_trigrams_text'] != 'N/A':
                html_content += f"""
                    <h4>Trigrams (3-word phrases):</h4>
                    <ul>{''.join([f'<li>{gram}</li>' for gram in analysis['top_trigrams_text'].split(', ')])}</ul>
                """
        else:
            html_content += f"<div class='info-box'>No significant N-grams found.</div>"

        # Frequent Responses
        if analysis['frequent_responses_html']:
            html_content += f"""
                <h3>📋 Frequent Responses (2+ words)</h3>
                <div class="dataframe">{analysis['frequent_responses_html']}</div>
            """
        else:
            html_content += f"<div class='info-box'>No frequent multi-word responses found.</div>"

        # Emotion Detection
        if analysis['emotion_plot_html']:
            html_content += f"""
                <h3>🎭 Emotion Detection</h3>
                <div class="plot-container">{analysis['emotion_plot_html']}</div>
            """
        else:
            html_content += f"<div class='info-box'>No strong emotions detected or not enough data.</div>"

        # Customer Effort Score
        if analysis['ces_plot_html']:
            html_content += f"""
                <h3>💪 Customer Effort Score (CES) Approximation</h3>
                <div class="plot-container">{analysis['ces_plot_html']}</div>
            """
        else:
            html_content += f"<div class='info-box'>No meaningful responses for Customer Effort Score analysis.</div>"

        # Word Cloud
        if analysis['word_cloud_image_base64']:
            html_content += f"""
                <h3>☁️ Word Cloud</h3>
                <div class="image-container"><img src="{analysis['word_cloud_image_base64']}" alt="Word Cloud"></div>
            """
        else:
            html_content += f"<div class='info-box'>Not enough meaningful words to generate a word cloud.</div>"

        # Word Co-occurrence
        if analysis['cooccurrence_graph_image_base64']:
            html_content += f"""
                <h3>🕸️ Word Co-occurrence Network</h3>
                <div class="image-container"><img src="{analysis['cooccurrence_graph_image_base64']}" alt="Word Co-occurrence Graph"></div>
            """
        else:
            html_content += f"<div class='info-box'>Not enough significant co-occurrences to build a graph.</div>"
            
        # Response Length
        if analysis['response_length_plot_html']:
            html_content += f"""
                <h3>📏 Response Length Analysis (Word Count)</h3>
                <p><strong>Average Word Count:</strong> {analysis['response_length_stats'].get('Average Word Count', 0):.2f}</p>
                <p><strong>Median Word Count:</strong> {analysis['response_length_stats'].get('Median Word Count', 0):.0f}</p>
                <p><strong>Min Word Count:</strong> {analysis['response_length_stats'].get('Min Word Count', 0):.0f}</p>
                <p><strong>Max Word Count:</strong> {analysis['response_length_stats'].get('Max Word Count', 0):.0f}</p>
                <div class="plot-container">{analysis['response_length_plot_html']}</div>
            """
        else:
            html_content += f"<div class='info-box'>No meaningful responses to analyze length.</div>"


        # Gemini Summary
        if analysis['gemini_summary_text']:
            html_content += f"""
                <h3>🧠 Gemini Summary</h3>
                <div class="info-box">
                    <p>{analysis['gemini_summary_text']}</p>
                </div>
            """
        elif analysis['gemini_summary_on'] and not gemini: # If Gemini was supposed to run but wasn't initialized
             html_content += f"<div class='warning-box'>Gemini summary could not be generated (API key not initialized).</div>"
        else:
            html_content += f"<div class='info-box'>Not enough meaningful responses or Gemini summary was not enabled.</div>"

        html_content += "</div>" # Close section
    html_content += """
        </div>
    </body>
    </html>
    """
    return html_content

# 📊 Feedback Analyzer App
st.set_page_config("Feedback Analyzer", layout="wide")
st.title("📋 Feedback Analyzer Tool")

# Initialize session state for storing Gemini's overall summary
if 'gemini_overall_answer' not in st.session_state:
    st.session_state['gemini_overall_answer'] = None

# Sidebar: Gemini API Key
with st.sidebar:
    st.header("🔐 Gemini API Key")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        api_key = st.secrets.get("GEMINI_API_KEY")

    gemini_key_source = "None"
    if api_key:
        st.success("Gemini API Key loaded successfully.")
        gemini_key_source = "secrets.toml or Environment Variable"
    else:
        api_key = st.text_input("Enter Gemini API Key", type="password")
        if api_key:
            st.info("API Key entered manually for this session.")
            gemini_key_source = "Manual Input"
        else:
            st.warning("Please enter your Gemini API key in the sidebar or save it in secrets.toml.")

    gemini = None
    if api_key:
        try:
            with st.spinner("Connecting to Gemini..."):
                genai.configure(api_key=api_key)
                gemini = genai.GenerativeModel("gemini-1.5-flash")
                test_response = gemini.generate_content("hello", safety_settings={'HARASSMENT': 'block_none', 'HATE_SPEECH': 'block_none', 'SEXUALLY_EXPLICIT': 'block_none', 'DANGEROUS_CONTENT': 'block_none'})
                if test_response and test_response.text:
                    st.success(f"Gemini is ready and connected! (Source: {gemini_key_source})")
                else:
                    st.error("Gemini connection failed. Check API key or network. Response was empty or problematic.")
                    gemini = None
        except Exception as e:
            st.error(f"Error connecting to Gemini: {e}. Please check your API key.")
            gemini = None


# Upload File
uploaded = st.file_uploader("📤 Upload CSV or Excel Feedback File", type=["csv", "xlsx"])

if uploaded:
    path, cols, df = preprocess_and_save(uploaded)
    if path:
        st.subheader("📄 Preview Data")
        st.dataframe(df, use_container_width=True)

        ignore = ["name", "email", "id", "timestamp"]
        text_cols = [c for c in df.select_dtypes(include='object').columns if c.lower() not in ignore]

        st.markdown("---")
        st.markdown("## ✅ Select Questions to Analyze")
        selected = st.multiselect("Choose Questions", text_cols, default=text_cols[:min(len(text_cols), 2)])

        # Store all individual question analysis results for report generation
        all_question_analyses_for_report = []
        summary_data = []

        for i, col in enumerate(selected):
            responses = df[col].astype(str).dropna().tolist()
            meaningful_responses = [r for r in responses if len(r.strip()) > 5 and not r.isnumeric()]

            # Perform all analyses (cached functions will speed this up if inputs don't change)
            sentiments = classify_sentiments(meaningful_responses)
            kws_results = extract_keywords_tfidf(meaningful_responses)
            bigrams_results = extract_ngrams(meaningful_responses, n_range=(2, 2))
            trigrams_results = extract_ngrams(meaningful_responses, n_range=(3, 3))
            response_lengths_df, response_length_stats = analyze_response_length(meaningful_responses)
            emotion_df = detect_emotions(meaningful_responses)
            ces_counts_df = calculate_ces_approximation(meaningful_responses)
            cooccurrence_graph = generate_cooccurrence_graph(meaningful_responses)


            st.markdown(f"---")
            with st.expander(f"🔍 Analysis: **{col}**", expanded=True):
                st.markdown("### Choose Analysis Types for this Question:")
                col_c1, col_c2, col_c3, col_c4, col_c5 = st.columns(5)
                with col_c1:
                    pie_chart_on = st.checkbox("📊 Sentiment Breakdown", value=True, key=f"pie_toggle_{col}_{i}")
                    keywords_on = st.checkbox("🔤 Top Keywords (List)", value=True, key=f"kw_toggle_{col}_{i}")
                with col_c2:
                    ngrams_on = st.checkbox("🔗 Top Phrases (N-grams)", value=True, key=f"ngrams_toggle_{col}_{i}")
                    response_length_on = st.checkbox("📏 Response Length", value=True, key=f"len_toggle_{col}_{i}")
                with col_c3:
                    frequent_on = st.checkbox("📋 Frequent Responses", value=True, key=f"freq_toggle_{col}_{i}")
                    emotion_on = st.checkbox("🎭 Emotion Detection", value=True, key=f"emotion_toggle_{col}_{i}")
                with col_c4:
                    ces_on = st.checkbox("💪 Customer Effort Score", value=True, key=f"ces_toggle_{col}_{i}")
                    cooccurrence_on = st.checkbox("🕸️ Word Co-occurrence", value=True, key=f"cooccur_toggle_{col}_{i}")
                with col_c5:
                    word_cloud_on = st.checkbox("☁️ Word Cloud", value=True, key=f"wc_toggle_{col}_{i}")
                    gemini_on = st.checkbox("🧠 Gemini Summary", value=True, key=f"gem_sum_toggle_{col}_{i}")


                st.markdown("---")

                col1, col2 = st.columns(2)

                # Store analysis outputs for report generation
                current_question_analysis = {
                    "question_col": col,
                    "sentiment_plot_html": "",
                    "top_keywords_text": "",
                    "top_bigrams_text": "",
                    "top_trigrams_text": "",
                    "frequent_responses_html": "",
                    "emotion_plot_html": "",
                    "ces_plot_html": "",
                    "word_cloud_image_base64": "",
                    "cooccurrence_graph_image_base64": "",
                    "response_length_plot_html": "",
                    "response_length_stats": response_length_stats,
                    "gemini_summary_text": "",
                    "gemini_summary_on": gemini_on # To indicate if Gemini was requested
                }

                with col1:
                    if pie_chart_on:
                        st.markdown("### 📊 Sentiment Breakdown")
                        if sentiments:
                            pie_df = pd.DataFrame(Counter(sentiments).items(), columns=["Sentiment", "Count"])
                            fig = px.pie(pie_df, names="Sentiment", values="Count", title="Sentiment Breakdown")
                            st.plotly_chart(fig, use_container_width=True)
                            current_question_analysis["sentiment_plot_html"] = fig.to_html(full_html=False)
                        else:
                            st.info("No meaningful responses available for sentiment analysis for this question.")

                    if keywords_on:
                        st.markdown("### 🔤 Top Keywords")
                        if kws_results:
                            keyword_strings = [f"- **{kw}** (Score: {score:.2f})" for kw, score in kws_results]
                            st.markdown("\n".join(keyword_strings))
                            current_question_analysis["top_keywords_text"] = ", ".join([kw for kw, _ in kws_results])
                        else:
                            st.info("No significant keywords found for this question.")
                            current_question_analysis["top_keywords_text"] = "N/A"

                    if ngrams_on:
                        st.markdown("### 🔗 Top Phrases (Bigrams & Trigrams)")
                        if bigrams_results or trigrams_results:
                            if bigrams_results:
                                st.markdown("#### Bigrams (2-word phrases):")
                                bigram_strings = [f"- **{gram}** (Score: {score:.2f})" for gram, score in bigrams_results]
                                st.markdown("\n".join(bigram_strings))
                                current_question_analysis["top_bigrams_text"] = ", ".join([gram for gram, _ in bigrams_results])
                            else:
                                st.info("No significant bigrams found.")
                                current_question_analysis["top_bigrams_text"] = "N/A"

                            if trigrams_results:
                                st.markdown("#### Trigrams (3-word phrases):")
                                trigram_strings = [f"- **{gram}** (Score: {score:.2f})" for gram, score in trigrams_results]
                                st.markdown("\n".join(trigram_strings))
                                current_question_analysis["top_trigrams_text"] = ", ".join([gram for gram, _ in trigrams_results])
                            else:
                                st.info("No significant trigrams found.")
                                current_question_analysis["top_trigrams_text"] = "N/A"
                        else:
                            st.info("No significant N-grams found for this question.")
                            current_question_analysis["top_bigrams_text"] = "N/A"
                            current_question_analysis["top_trigrams_text"] = "N/A"

                    if frequent_on:
                        st.markdown("### 📋 Frequent Responses (2+ words)")
                        if meaningful_responses:
                            freq_df = pd.Series(meaningful_responses).value_counts().reset_index()
                            freq_df.columns = ["Response", "Count"]
                            freq_df = freq_df[freq_df["Response"].apply(lambda x: len(x.split()) >= 2 and len(x) > 10)]
                            if not freq_df.empty:
                                st.dataframe(freq_df.head(10), use_container_width=True)
                                current_question_analysis["frequent_responses_html"] = freq_df.head(10).to_html(index=False)
                            else:
                                st.info("No frequent multi-word responses (with more than 10 characters) found for this question.")
                        else:
                            st.info("No meaningful responses to find frequent patterns.")

                    if emotion_on:
                        st.markdown("### 🎭 Emotion Detection")
                        if not emotion_df.empty:
                            fig = px.bar(emotion_df, x="Emotion", y="Percentage",
                                         title="Detected Emotions in Feedback",
                                         labels={"Percentage": "Percentage of Responses Mentioning Emotion"},
                                         color="Emotion")
                            st.plotly_chart(fig, use_container_width=True)
                            current_question_analysis["emotion_plot_html"] = fig.to_html(full_html=False)
                        else:
                            st.info("No strong emotions detected or not enough data for emotion analysis.")

                with col2:
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
                                current_question_analysis["word_cloud_image_base64"] = fig_to_base64(fig_wc)
                            else:
                                st.info("Not enough meaningful words to generate a word cloud for this question.")
                        else:
                            st.info("No meaningful responses to generate a word cloud for this question.")

                    if response_length_on:
                        st.markdown("### 📏 Response Length Analysis (Word Count)")
                        if not response_lengths_df.empty:
                            st.write(f"**Average Word Count:** {response_length_stats['Average Word Count']:.2f}")
                            st.write(f"**Median Word Count:** {response_length_stats['Median Word Count']:.0f}")
                            st.write(f"**Min Word Count:** {response_length_stats['Min Word Count']:.0f}")
                            st.write(f"**Max Word Count:** {response_length_stats['Max Word Count']:.0f}")
                            fig_hist = px.histogram(response_lengths_df, x="Word Count", nbins=20,
                                                    title="Distribution of Response Word Counts")
                            st.plotly_chart(fig_hist, use_container_width=True)
                            current_question_analysis["response_length_plot_html"] = fig_hist.to_html(full_html=False)
                        else:
                            st.info("No meaningful responses to analyze length for this question.")

                    if ces_on:
                        st.markdown("### 💪 Customer Effort Score (CES) Approximation")
                        if not ces_counts_df.empty:
                            fig = px.bar(ces_counts_df, x="Effort Level", y="Count",
                                         title="Approximate Customer Effort Levels",
                                         color="Effort Level",
                                         category_orders={"Effort Level": ["Low Effort", "Neutral Effort", "Mixed (leaning Low)", "Mixed (leaning High)", "High Effort"]})
                            st.plotly_chart(fig, use_container_width=True)
                            current_question_analysis["ces_plot_html"] = fig.to_html(full_html=False)
                        else:
                            st.info("No meaningful responses for Customer Effort Score analysis.")

                    if cooccurrence_on:
                        st.markdown("### 🕸️ Word Co-occurrence Network")
                        if cooccurrence_graph:
                            fig_co, ax_co = plt.subplots(figsize=(10, 10))
                            pos = nx.spring_layout(cooccurrence_graph, k=0.5, iterations=50)
                            node_sizes = [cooccurrence_graph.nodes[node].get('size', 1) * 100 for node in cooccurrence_graph.nodes()]
                            nx.draw_networkx_nodes(cooccurrence_graph, pos, ax=ax_co, node_size=node_sizes, node_color='skyblue', alpha=0.9)
                            edge_weights = [cooccurrence_graph.edges[edge]['weight'] * 0.5 for edge in cooccurrence_graph.edges()]
                            nx.draw_networkx_edges(cooccurrence_graph, pos, ax=ax_co, width=edge_weights, alpha=0.6, edge_color='gray')
                            nx.draw_networkx_labels(cooccurrence_graph, pos, ax=ax_co, font_size=9, font_weight='bold')
                            ax_co.set_title("Top Word Co-occurrence Network")
                            ax_co.axis('off')
                            st.pyplot(fig_co)
                            current_question_analysis["cooccurrence_graph_image_base64"] = fig_to_base64(fig_co)
                        else:
                            st.info("Not enough significant co-occurrences to build a graph. Try more data or adjust parameters.")

                    if gemini_on:
                        st.markdown("### 🧠 Gemini Summary")
                        if not gemini:
                            st.warning("Gemini is not initialized. Please ensure your API key is correctly set in the sidebar or secrets.toml.")
                        else:
                            try:
                                if meaningful_responses:
                                    sample_responses = "\n".join(pd.Series(meaningful_responses).dropna().sample(min(25, len(meaningful_responses)), random_state=42))
                                    prompt = f"""Provide a concise, useful, and actionable summary of the key themes, overall sentiment (positive/negative/neutral breakdown), and specific suggestions for improvement from the following responses to the question: "{col}". Aim for a summary that is easy to read and provides valuable insights, without being too long or too short. Focus on a length that is 'considerable' and 'just useful'.

                                    Feedbacks:
                                    {sample_responses}"""
                                    with st.spinner("Generating Gemini summary..."):
                                        reply = gemini.generate_content(prompt)
                                        st.info(reply.text.strip())
                                        current_question_analysis["gemini_summary_text"] = reply.text.strip()
                                else:
                                    st.info("Not enough meaningful responses to generate a Gemini summary for this question.")
                            except Exception as e:
                                st.error(f"Gemini Error for '{col}': {e}. Please ensure your API key is valid and there are sufficient responses.")

            # Append current question's analysis data for report
            all_question_analyses_for_report.append(current_question_analysis)

            # Collect for overall summary (unchanged)
            summary_data.append({
                "Question": col,
                "Total Responses": len(responses),
                "Meaningful Responses": len(meaningful_responses),
                "👍 Positive": sentiments.count("Positive"),
                "👎 Negative": sentiments.count("Negative"),
                "😐 Neutral": sentiments.count("Neutral"),
                "Top Keywords": ", ".join([kw for kw, _ in kws_results]) if kws_results else "N/A",
                "Top Bigrams": ", ".join([gram for gram, _ in bigrams_results]) if bigrams_results else "N/A",
                "Top Trigrams": ", ".join([gram for gram, _ in trigrams_results]) if trigrams_results else "N/A",
                "Avg Response Length": f"{response_length_stats.get('Average Word Count', 0):.2f}" if response_length_stats else "N/A",
                "Top Emotions": ", ".join(emotion_df["Emotion"].tolist()[:3]) if not emotion_df.empty else "N/A",
                "CES Breakdown": ", ".join([f"{row['Effort Level']}: {row['Count']}" for idx, row in ces_counts_df.iterrows()]) if not ces_counts_df.empty else "N/A"
            })

        # ---
        # 📋 Overall Summary Table
        st.markdown("---")
        st.markdown("## 🧾 Overall Feedback Summary")
        overall_summary_df = pd.DataFrame(summary_data)
        st.dataframe(overall_summary_df, use_container_width=True)

        # ---
        # 💬 Ask Gemini
        st.markdown("---")
        st.markdown("## 💬 Ask Gemini About All Feedback")
        userq = st.text_input("Ask your question about the overall feedback insights")
        if st.button("Ask Gemini about overall feedback"):
            if gemini:
                try:
                    tabular = overall_summary_df.to_markdown(index=False)
                    prompt = f"""You're a feedback report analyst. Given this summary table:\n\n{tabular}\n\nAnswer this question:\n{userq}\n\nProvide a concise and direct answer, focusing on actionable insights derived from the data."""
                    with st.spinner("Generating Gemini answer..."):
                        final = gemini.generate_content(prompt)
                        st.markdown("### 🧠 Gemini Answer")
                        st.info(final.text.strip())
                        st.session_state['gemini_overall_answer'] = final.text.strip() # Store for report
                except Exception as e:
                    st.error(f"Gemini Error: {e}. Please ensure your API key is valid.")
            else:
                st.warning("Gemini is not initialized. Please enter your API key or ensure it's in secrets.toml.")

        # ---
        # ⬇️ Download Analysis Report
        st.markdown("---")
        st.markdown("## ⬇️ Download Analysis Report")
        st.info("Click the button below to generate and download a comprehensive HTML report of the analysis shown above.")

        report_html = generate_html_report(overall_summary_df, all_question_analyses_for_report, st.session_state['gemini_overall_answer'])

        st.download_button(
            label="Generate & Download HTML Report",
            data=report_html,
            file_name="feedback_analysis_report.html",
            mime="text/html"
        )


elif not uploaded:
    st.info("Upload a CSV or Excel file to begin feedback analysis.")

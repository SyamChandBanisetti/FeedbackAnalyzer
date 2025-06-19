import streamlit as st
import pandas as pd
import plotly.express as px # Keep this import if you want to retain the sentiment chart
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import os
import io
from datetime import datetime
import base64
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Feedback Analyzer Tool",
    page_icon="📋",
    layout="wide"
)

# --- Constants and Configuration ---
STOP_WORDS = set([
    "a", "an", "the", "and", "but", "or", "to", "of", "in", "on", "at", "for", "with",
    "as", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "not", "no", "don't", "can't", "won't", "shouldn't", "wouldn't",
    "couldn't", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "its", "our", "their", "this", "that", "these", "those",
    "some", "any", "all", "each", "every", "many", "much", "more", "most", "other",
    "such", "only", "own", "so", "than", "too", "very", "s", "t", "can", "will", "just",
    "don", "should", "now", "ve", "ll", "m", "re", "d", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "great", "good", "bad", "poor", "issue", "problem", "really", "very", "much",
    "like", "get", "got", "would", "could", "said", "say", "also", "well", "from", "into",
    "through", "during", "before", "after", "above", "below", "up", "down", "out", "off",
    "over", "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "don", "should", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren",
    "couldn", "didn", "doesn", "hadn", "hasn", "haven", "isn", "ma", "mightn", "mustn",
    "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn", "etc"
])

EMOTION_KEYWORDS = {
    "Joy": ["happy", "delighted", "pleased", "excited", "joyful", "satisfied", "great", "love"],
    "Sadness": ["sad", "unhappy", "down", "depressed", "disappointed", "frustrated", "grief"],
    "Anger": ["angry", "mad", "frustrated", "annoyed", "irritated", "furious", "rage"],
    "Surprise": ["surprised", "shocked", "amazed", "unexpected", "astonished"],
    "Fear": ["fear", "scared", "anxious", "worried", "nervous", "dread"],
    "Disgust": ["disgusted", "revolted", "sick", "nauseated", "repulsed"],
    "Trust": ["trust", "reliable", "confident", "secure", "faithful"],
    "Anticipation": ["eager", "expecting", "hopeful", "waiting", "looking forward"]
}

CES_KEYWORDS = {
    "Low Effort": ["easy", "simple", "smooth", "seamless", "effortless", "quick", "no hassle", "straightforward"],
    "High Effort": ["difficult", "hard", "complicated", "frustrating", "time-consuming", "struggle", "challenging", "complex"]
}

# --- Gemini API Configuration ---
gemini_api_key = None
try:
    # Attempt to load from Streamlit secrets (for Streamlit Cloud)
    # Corrected: Expects GEMINI_API_KEY directly, not under [gemini]
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
except KeyError:
    # If not in secrets, try from environment variables (for local deployment)
    gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    st.warning("🔐 Gemini API Key not found. Please ensure it's set in Streamlit Cloud secrets or as an environment variable (e.g., in a `.env` file for local testing).")
    st.session_state['gemini_model'] = None # Ensure model is not initialized if key is missing
else:
    # Initialize Gemini model if API key is available
    try:
        genai.configure(api_key=gemini_api_key)
        st.session_state['gemini_model'] = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        # Simple test prompt to check connectivity and initial blocking
        try:
            test_response = st.session_state['gemini_model'].generate_content("Hello, how are you today?")
            if test_response.candidates:
                st.sidebar.success("✅ Connected to Gemini API.")
            else:
                st.sidebar.error("Gemini API connected but no candidates returned for test prompt.")
                if test_response.prompt_feedback and test_response.prompt_feedback.block_reason:
                    st.sidebar.error(f"Test prompt blocked for reason: {test_response.prompt_feedback.block_reason.name}")
        except Exception as e:
            st.sidebar.error(f"Error during Gemini test prompt: {e}")
            st.session_state['gemini_model'] = None # Invalidate model if test fails
    except Exception as e:
        st.sidebar.error(f"Error initializing Gemini: '{e}'. Check your API key or network.")
        st.session_state['gemini_model'] = None # Invalidate model if initialization fails

# --- Helper Functions ---

@st.cache_data
def load_data(uploaded_file):
    """Loads data from CSV or Excel."""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload a CSV or Excel file.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

@st.cache_data
def analyze_sentiment(text_series):
    """Simple keyword-based sentiment analysis."""
    positive_keywords = ["good", "great", "excellent", "positive", "happy", "love", "satisfied", "recommend", "best", "smooth", "easy"]
    negative_keywords = ["bad", "poor", "terrible", "negative", "unhappy", "hate", "dissatisfied", "frustrating", "problem", "issue", "difficult"]

    sentiments = []
    for text in text_series.astype(str).fillna(""):
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in positive_keywords) and not any(keyword in text_lower for keyword in negative_keywords):
            sentiments.append("Positive")
        elif any(keyword in text_lower for keyword in negative_keywords) and not any(keyword in text_lower for keyword in positive_keywords):
            sentiments.append("Negative")
        elif any(keyword in text_lower for keyword in positive_keywords) and any(keyword in text_lower for keyword in negative_keywords):
            sentiments.append("Mixed") # Both positive and negative keywords
        else:
            sentiments.append("Neutral")
    return sentiments

@st.cache_data
def get_top_n_grams(text_series, n=1, top_n=10):
    """Generates top N-grams (single words, bigrams, trigrams)."""
    vectorizer = TfidfVectorizer(
        ngram_range=(n, n),
        stop_words=list(STOP_WORDS),
        token_pattern=r'\b[a-zA-Z]{2,}\b' # Only words, min 2 chars
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(text_series.astype(str).fillna(""))
        feature_names = vectorizer.get_feature_names_out()
        sums = tfidf_matrix.sum(axis=0)
        data = []
        for col, term in enumerate(feature_names):
            data.append((term, sums[0, col]))
        ranking = pd.DataFrame(data, columns=['term', 'tfidf']).sort_values(by='tfidf', ascending=False)
        return ranking.head(top_n)
    except ValueError:
        return pd.DataFrame(columns=['term', 'tfidf'])

@st.cache_data
def analyze_response_length(text_series):
    """Calculates word count for each response."""
    return text_series.astype(str).fillna("").apply(lambda x: len(x.split()))

@st.cache_data
def detect_emotions(text_series):
    """Detects emotions based on keywords."""
    emotion_counts = {emotion: 0 for emotion in EMOTION_KEYWORDS}
    total_responses = len(text_series)
    if total_responses == 0:
        return {emotion: 0 for emotion in EMOTION_KEYWORDS} # Return zeros if no responses

    for text in text_series.astype(str).fillna(""):
        text_lower = text.lower()
        found_emotion_in_response = False
        for emotion, keywords in EMOTION_KEYWORDS.items():
            if any(keyword in text_lower for keyword in keywords):
                emotion_counts[emotion] += 1
                found_emotion_in_response = True
        # If no specific emotion keyword found, consider it neutral for emotion analysis
        if not found_emotion_in_response:
            # We don't increment a "neutral" counter here, as it's about detecting *specific* emotions.
            pass
    
    # Convert counts to percentages
    emotion_percentages = {
        emotion: (count / total_responses) * 100 if total_responses > 0 else 0
        for emotion, count in emotion_counts.items()
    }
    return emotion_percentages


@st.cache_data
def calculate_ces(text_series):
    """Approximates Customer Effort Score based on keywords."""
    effort_scores = []
    for text in text_series.astype(str).fillna(""):
        text_lower = text.lower()
        low_effort_matches = sum(1 for keyword in CES_KEYWORDS["Low Effort"] if keyword in text_lower)
        high_effort_matches = sum(1 for keyword in CES_KEYWORDS["High Effort"] if keyword in text_lower)

        if low_effort_matches > high_effort_matches:
            effort_scores.append("Low Effort")
        elif high_effort_matches > low_effort_matches:
            effort_scores.append("High Effort")
        else:
            effort_scores.append("Neutral/Mixed Effort")
    return effort_scores

@st.cache_data
def generate_wordcloud(text_data):
    """Generates a word cloud image."""
    if not text_data:
        return None
    # Ensure text is string and handle potential non-string elements
    text_data_str = " ".join([str(item) for item in text_data if pd.notna(item)])
    if not text_data_str.strip():
        return None

    wordcloud = WordCloud(width=800, height=400, background_color='white',
                          stopwords=STOP_WORDS, min_font_size=10).generate(text_data_str)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    return fig

@st.cache_data
def generate_co_occurrence_graph(text_series, min_cooccurrence=2):
    """Generates a word co-occurrence graph."""
    sentences = text_series.astype(str).fillna("").apply(lambda x: x.lower().split())
    filtered_sentences = [[word for word in sentence if word not in STOP_WORDS and len(word) > 1] for sentence in sentences]

    co_occurrence_matrix = {}
    for sentence in filtered_sentences:
        for i, word1 in enumerate(sentence):
            if word1 not in co_occurrence_matrix:
                co_occurrence_matrix[word1] = {}
            for j, word2 in enumerate(sentence):
                if i != j:
                    if word2 not in co_occurrence_matrix[word1]:
                        co_occurrence_matrix[word1][word2] = 0
                    co_occurrence_matrix[word1][word2] += 1

    G = nx.Graph()
    for word1, connections in co_occurrence_matrix.items():
        for word2, count in connections.items():
            if count >= min_cooccurrence:
                G.add_edge(word1, word2, weight=count)

    if not G.edges():
        return None, "No strong co-occurrences found with current settings."

    pos = nx.spring_layout(G, k=0.15, iterations=20, seed=42)
    fig, ax = plt.subplots(figsize=(12, 10))
    
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    node_sizes = [G.degree(node) * 100 for node in G.nodes()] # Size nodes by their degree
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, width=[w * 0.5 for w in edge_weights], alpha=0.5, edge_color='gray', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
    
    ax.set_title(f'Word Co-occurrence Network (Min Co-occurrence: {min_cooccurrence})')
    ax.axis("off")
    return fig, None

# --- Gemini AI Summarization Function ---
@st.cache_data(show_spinner="Generating AI Summary with Gemini...")
def get_gemini_summary(text_to_summarize, model):
    if not model:
        return "Gemini AI is not available. Please provide a valid API key."

    try:
        # Gemini often prefers list of strings for content, even for single requests
        response = model.generate_content([f"Summarize the following feedback comments concisely, highlight key themes, and identify actionable insights:\n\n{text_to_summarize}"])
        if response.candidates:
            # st.write("Gemini Raw Response:", response) # Debugging line
            # st.write("Gemini Prompt Feedback:", response.prompt_feedback) # Debugging line
            return response.text
        else:
            block_reason = "Unknown reason"
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                block_reason = response.prompt_feedback.block_reason.name
            return f"Gemini did not return a summary. Content may have been blocked due to safety concerns. Block reason: '{block_reason}'."
    except Exception as e:
        return f"Error connecting to Gemini for summary: '{e}'. This might be due to safety filters or an API issue."

# --- HTML Report Generation ---
def create_html_report(df, selected_text_columns, analyses_results):
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Feedback Analysis Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; margin: 20px; }}
            h1, h2, h3 {{ color: #005f73; border-bottom: 2px solid #e0e0e0; padding-bottom: 5px; margin-top: 30px; }}
            .container {{ max-width: 1200px; margin: auto; padding: 20px; background: #fff; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            .section {{ margin-bottom: 40px; }}
            .chart-img {{ max-width: 100%; height: auto; display: block; margin: 15px auto; border: 1px solid #ddd; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .st-dataframe {{ overflow-x: auto; }} /* Allow horizontal scroll for dataframes */
            .info-box {{ background-color: #e6f7ff; border-left: 5px solid #2196f3; padding: 10px; margin-bottom: 15px; }}
            .warning-box {{ background-color: #fff3e0; border-left: 5px solid #ff9800; padding: 10px; margin-bottom: 15px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Feedback Analysis Report</h1>
            <p><strong>Report Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>This report provides an in-depth analysis of customer feedback data.</p>

            <div class="section">
                <h2>1. Data Overview</h2>
                <h3>Original Data Sample ({df.shape[0]} rows)</h3>
                <div class="st-dataframe">
                    {df.head().to_html()}
                </div>
                <p>Selected text columns for analysis: {', '.join(selected_text_columns)}</p>
            </div>

            <div class="section">
                <h2>2. Sentiment Analysis</h2>
                {analyses_results.get('sentiment_analysis', '<p>No sentiment analysis data.</p>')}
            </div>

            <div class="section">
                <h2>3. Key Phrase Analysis</h2>
                <h3>Top Keywords (Unigrams)</h3>
                {analyses_results.get('top_keywords_html', '<p>No top keywords data.</p>')}
                <h3>Top Phrases (Bigrams)</h3>
                {analyses_results.get('top_bigrams_html', '<p>No top bigrams data.</p>')}
                <h3>Top Phrases (Trigrams)</h3>
                {analyses_results.get('top_trigrams_html', '<p>No top trigrams data.</p>')}
            </div>

            <div class="section">
                <h2>4. Response Length Analysis</h2>
                {analyses_results.get('response_length_summary', '<p>No response length data.</p>')}
            </div>

            <div class="section">
                <h2>5. Emotion Detection</h2>
                {analyses_results.get('emotion_analysis_summary', '<p>No emotion analysis data.</p>')}
            </div>

            <div class="section">
                <h2>6. Customer Effort Score (CES) Approximation</h2>
                {analyses_results.get('ces_analysis_summary', '<p>No CES analysis data.</p>')}
            </div>
            
            <div class="section">
                <h2>7. Word Cloud</h2>
                {analyses_results.get('wordcloud_image', '<p>No word cloud generated.</p>')}
            </div>

            <div class="section">
                <h2>8. Word Co-occurrence Network</h2>
                {analyses_results.get('co_occurrence_graph', '<p>No co-occurrence graph generated.</p>')}
            </div>

            <div class="section">
                <h2>9. Gemini AI Summaries</h2>
                {analyses_results.get('gemini_summary_overall', '<p>No overall Gemini summary.</p>')}
                {analyses_results.get('gemini_summary_questions', '<p>No per-question Gemini summaries.</p>')}
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

def fig_to_base64(fig):
    """Converts a matplotlib figure to a base64 encoded image."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig) # Close the figure to free up memory
    return f'<img src="data:image/png;base64,{img_base64}" class="chart-img"/>'


# --- Main Streamlit Application ---

st.title("📋 Feedback Analyzer Tool")
st.markdown("Upload your feedback data (CSV or Excel) to get insights into sentiments, key phrases, emotions, and more, powered by Gemini AI.")

uploaded_file = st.sidebar.file_uploader("Upload your CSV or Excel file", type=["csv", "xls", "xlsx"], key="file_uploader")

if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None:
        st.sidebar.success("File uploaded successfully!")
        st.sidebar.dataframe(df.head(), use_container_width=True, key="sidebar_df_head")

        text_columns = [col for col in df.columns if df[col].dtype == 'object']
        if not text_columns:
            st.error("No text (object) columns found in your data. Please ensure your feedback is in a text format.")
            st.stop()

        selected_text_columns = st.sidebar.multiselect(
            "Select text columns for analysis:",
            options=text_columns,
            default=text_columns[0] if text_columns else [],
            key="column_multiselect"
        )

        if not selected_text_columns:
            st.warning("Please select at least one text column for analysis.")
            st.stop()

        # Combine selected text columns into a single series for overall analysis
        combined_feedback_series = df[selected_text_columns].astype(str).fillna('').agg(' '.join, axis=1)
        full_text_for_analysis = " ".join(combined_feedback_series.dropna().tolist())

        st.header("Overall Feedback Analysis")
        st.info("Results below are based on the combined text from selected columns.")

        # --- Display Raw Data (Optional) ---
        if st.checkbox("Show Raw Data Sample", key="show_raw_data_checkbox"):
            st.subheader("Raw Data Sample")
            st.dataframe(df.head(10), use_container_width=True, key="raw_data_df")

        # --- Sentiment Analysis ---
        st.subheader("Sentiment Analysis")
        with st.spinner("Analyzing sentiments..."):
            sentiment_df = pd.DataFrame()
            for col in selected_text_columns:
                sentiment_df[f'{col}_Sentiment'] = analyze_sentiment(df[col])
            
            # Combine all sentiment columns into a single Series for overall count
            overall_sentiments = pd.concat([sentiment_df[col] for col in sentiment_df.columns])
            sentiment_counts = overall_sentiments.value_counts(normalize=True) * 100
            
            if not sentiment_counts.empty:
                # Retained Plotly chart for Sentiment as it was not explicitly requested to be removed and is a core visualization
                fig_sentiment = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="Overall Sentiment Distribution",
                    hole=0.4
                )
                fig_sentiment.update_traces(textposition='inside', textinfo='percent+label')
                # ADDED UNIQUE KEY HERE
                st.plotly_chart(fig_sentiment, use_container_width=True, key="overall_sentiment_pie_chart") 
                
                st.write("Sentiment Breakdown:")
                st.dataframe(sentiment_counts.reset_index().rename(columns={'index': 'Sentiment', 0: 'Percentage'}), use_container_width=True, key="overall_sentiment_df")
            else:
                st.info("Not enough data to perform sentiment analysis.")

        # --- Key Phrase Analysis ---
        st.subheader("Key Phrase Analysis")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### Top Keywords (Unigrams)")
            top_keywords = get_top_n_grams(combined_feedback_series, n=1, top_n=15)
            st.dataframe(top_keywords, use_container_width=True, key="top_keywords_df")

        with col2:
            st.markdown("### Top Phrases (Bigrams)")
            top_bigrams = get_top_n_grams(combined_feedback_series, n=2, top_n=10)
            st.dataframe(top_bigrams, use_container_width=True, key="top_bigrams_df")

        with col3:
            st.markdown("### Top Phrases (Trigrams)")
            top_trigrams = get_top_n_grams(combined_feedback_series, n=3, top_n=5)
            st.dataframe(top_trigrams, use_container_width=True, key="top_trigrams_df")

        # --- Response Length Analysis (Direct Representation) ---
        st.subheader("Response Length Analysis")
        response_lengths = analyze_response_length(combined_feedback_series)
        if not response_lengths.empty:
            st.write(f"**Average response length:** {response_lengths.mean():.2f} words")
            st.write(f"**Median response length:** {response_lengths.median():.0f} words")
            st.write(f"**Minimum response length:** {response_lengths.min():.0f} words")
            st.write(f"**Maximum response length:** {response_lengths.max():.0f} words")
            st.markdown("---")
            st.markdown("##### Response Length Summary Statistics")
            st.dataframe(response_lengths.describe().to_frame().T, use_container_width=True, key="response_length_summary_df")
        else:
            st.info("Not enough data to analyze response lengths.")


        # --- Emotion Detection (Direct Representation) ---
        st.subheader("Emotion Detection (Keyword-Based)")
        with st.spinner("Detecting emotions..."):
            emotion_percentages = detect_emotions(combined_feedback_series)
            emotion_df = pd.DataFrame(emotion_percentages.items(), columns=['Emotion', 'Percentage']).sort_values(by='Percentage', ascending=False)
            
            if not emotion_df.empty and emotion_df['Percentage'].sum() > 0:
                st.write("Percentage of responses expressing each emotion:")
                st.dataframe(emotion_df, use_container_width=True, key="emotion_df")
                st.info("Note: A response can express multiple emotions if it contains keywords from different categories.")
            else:
                st.info("No specific emotion keywords detected in the feedback.")

        # --- Customer Effort Score (CES) Approximation (Direct Representation) ---
        st.subheader("Customer Effort Score (CES) Approximation")
        with st.spinner("Approximating CES..."):
            ces_scores = calculate_ces(combined_feedback_series)
            ces_counts = pd.Series(ces_scores).value_counts(normalize=True) * 100
            
            if not ces_counts.empty and ces_counts.sum() > 0:
                st.write("Distribution of approximated Customer Effort Levels:")
                st.dataframe(ces_counts.reset_index().rename(columns={'index': 'Effort Level', 0: 'Percentage'}), use_container_width=True, key="ces_df")
                st.info("Note: CES is approximated based on the presence of predefined keywords related to effort.")
            else:
                st.info("Not enough data to approximate Customer Effort Score.")


        # --- Word Cloud ---
        st.subheader("Word Cloud")
        with st.spinner("Generating word cloud..."):
            text_for_wordcloud = " ".join(combined_feedback_series.dropna().tolist())
            wordcloud_fig = generate_wordcloud(text_for_wordcloud)
            if wordcloud_fig:
                st.pyplot(wordcloud_fig, key="wordcloud_plot")
            else:
                st.info("Not enough relevant text to generate a word cloud.")

        # --- Word Co-occurrence Network ---
        st.subheader("Word Co-occurrence Network")
        min_cooccurrence = st.slider("Minimum co-occurrence for graph:", 1, 10, 2, key="min_cooccurrence_slider")
        with st.spinner("Generating co-occurrence graph..."):
            co_occurrence_fig, co_occurrence_message = generate_co_occurrence_graph(combined_feedback_series, min_cooccurrence)
            if co_occurrence_fig:
                st.pyplot(co_occurrence_fig, key="cooccurrence_plot")
            else:
                st.info(co_occurrence_message if co_occurrence_message else "No co-occurrences found to display a graph.")

        # --- Gemini AI Summarization ---
        st.subheader("Gemini AI Summarization")
        if st.session_state.get('gemini_model'): # Check if model was successfully initialized
            if st.button("Generate Overall AI Summary", key="generate_overall_ai_summary_btn"):
                with st.spinner("Asking Gemini to summarize overall feedback..."):
                    overall_summary = get_gemini_summary(full_text_for_analysis, st.session_state['gemini_model'])
                    st.markdown("#### Overall Summary by Gemini AI")
                    st.write(overall_summary)
                    # Store for report
                    st.session_state['overall_gemini_summary_generated'] = overall_summary 

            st.markdown("#### Summaries for Individual Feedback Questions (AI)")
            for col in selected_text_columns:
                question_text = df[col].astype(str).fillna("").tolist()
                question_text_filtered = [text for text in question_text if text.strip()]
                if question_text_filtered:
                    question_feedback_combined = "\n".join(question_text_filtered)
                    with st.expander(f"Generate Summary for: {col}", key=f"expander_{col}"):
                        if st.button(f"Summarize '{col}'", key=f"summarize_btn_{col}"):
                            with st.spinner(f"Asking Gemini to summarize '{col}'..."):
                                question_summary = get_gemini_summary(question_feedback_combined, st.session_state['gemini_model'])
                                st.write(question_summary)
                                st.session_state[f'gemini_summary_{col}'] = question_summary # Store for report
                else:
                    st.info(f"No valid text responses in column '{col}' to summarize.")
        else:
            st.warning("Gemini AI features are disabled because the API key was not found or the connection failed. Please ensure your API key is correctly configured in Streamlit Cloud secrets or environment variables.")

        # --- Generate and Download HTML Report ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("Download Report")

        if st.sidebar.button("Generate HTML Report", key="generate_html_report_btn"):
            with st.spinner("Generating comprehensive report..."):
                analyses_for_report = {}

                # 1. Sentiment Analysis
                sentiment_counts_html = ""
                if not sentiment_counts.empty:
                    sentiment_counts_html = sentiment_counts.reset_index().rename(columns={'index': 'Sentiment', 0: 'Percentage'}).to_html()
                    # Keep sentiment chart in report as it's a useful visualization
                    fig_sentiment_report = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, title="Overall Sentiment Distribution", hole=0.4)
                    fig_sentiment_report.update_traces(textposition='inside', textinfo='percent+label')
                    analyses_for_report['sentiment_analysis'] = f"<h3>Overall Sentiment Distribution</h3>{fig_to_base64(fig_sentiment_report)}<p>Sentiment Breakdown:</p>{sentiment_counts_html}"
                else:
                    analyses_for_report['sentiment_analysis'] = '<p>No sentiment analysis data.</p>'

                # 2. Key Phrase Analysis
                analyses_for_report['top_keywords_html'] = top_keywords.to_html() if not top_keywords.empty else '<p>No top keywords data.</p>'
                analyses_for_report['top_bigrams_html'] = top_bigrams.to_html() if not top_bigrams.empty else '<p>No top bigrams data.</p>'
                analyses_for_report['top_trigrams_html'] = top_trigrams.to_html() if not top_trigrams.empty else '<p>No top trigrams data.</p>'

                # 3. Response Length (Report version)
                if not response_lengths.empty:
                    length_stats_html = response_lengths.describe().to_frame().T.to_html()
                    analyses_for_report['response_length_summary'] = f"<h3>Response Length Analysis</h3>" \
                                                                       f"<p><strong>Average:</strong> {response_lengths.mean():.2f} words</p>" \
                                                                       f"<p><strong>Median:</strong> {response_lengths.median():.0f} words</p>" \
                                                                       f"<p><strong>Minimum:</strong> {response_lengths.min():.0f} words</p>" \
                                                                       f"<p><strong>Maximum:</strong> {response_lengths.max():.0f} words</p>" \
                                                                       f"<h4>Summary Statistics</h4>{length_stats_html}"
                else:
                    analyses_for_report['response_length_summary'] = '<p>No response length data.</p>'

                # 4. Emotion Detection (Report version)
                if not emotion_df.empty and emotion_df['Percentage'].sum() > 0:
                    analyses_for_report['emotion_analysis_summary'] = f"<h3>Emotion Detection (Keyword-Based)</h3>" \
                                                                       f"<h4>Percentage of responses expressing each emotion:</h4>" \
                                                                       f"{emotion_df.to_html()}" \
                                                                       f"<p><i>Note: A response can express multiple emotions if it contains keywords from different categories.</i></p>"
                else:
                    analyses_for_report['emotion_analysis_summary'] = '<p>No emotion analysis data.</p>'

                # 5. CES Approximation (Report version)
                if not ces_counts.empty and ces_counts.sum() > 0:
                    analyses_for_report['ces_analysis_summary'] = f"<h3>Customer Effort Score (CES) Approximation</h3>" \
                                                                   f"<h4>Distribution of approximated Customer Effort Levels:</h4>" \
                                                                   f"{ces_counts.reset_index().rename(columns={'index': 'Effort Level', 0: 'Percentage'}).to_html()}" \
                                                                   f"<p><i>Note: CES is approximated based on the presence of predefined keywords related to effort.</i></p>"
                else:
                    analyses_for_report['ces_analysis_summary'] = '<p>No CES analysis data.</p>'

                # 6. Word Cloud
                if wordcloud_fig:
                    analyses_for_report['wordcloud_image'] = fig_to_base64(wordcloud_fig)
                else:
                    analyses_for_report['wordcloud_image'] = '<p>No word cloud generated.</p>'

                # 7. Co-occurrence Graph
                if co_occurrence_fig:
                    analyses_for_report['co_occurrence_graph'] = fig_to_base64(co_occurrence_fig)
                else:
                    analyses_for_report['co_occurrence_graph'] = f'<p>{co_occurrence_message if co_occurrence_message else "No co-occurrence graph generated."}</p>'

                # 8. Gemini AI Summaries for Report
                overall_summary_report = ""
                # Check if overall summary was actually generated and stored in session state
                if st.session_state.get('gemini_model') and 'overall_gemini_summary_generated' in st.session_state and st.session_state['overall_gemini_summary_generated']:
                     overall_summary_report = f"<h4>Overall Feedback Summary</h4><p>{st.session_state['overall_gemini_summary_generated']}</p>"

                question_summaries_report = ""
                if st.session_state.get('gemini_model'):
                    for col in selected_text_columns:
                        # Check if per-question summary was actually generated and stored in session state
                        if f'gemini_summary_{col}' in st.session_state and st.session_state[f'gemini_summary_{col}']:
                            question_summaries_report += f"<h4>Summary for '{col}'</h4><p>{st.session_state[f'gemini_summary_{col}']}</p>"
                
                analyses_for_report['gemini_summary_overall'] = overall_summary_report if overall_summary_report else '<p>No overall Gemini summary generated for report.</p>'
                analyses_for_report['gemini_summary_questions'] = question_summaries_report if question_summaries_report else '<p>No per-question Gemini summaries generated for report.</p>'


                html_report_content = create_html_report(df, selected_text_columns, analyses_for_report)
                
                b64_html = base64.b64encode(html_report_content.encode()).decode()
                download_filename = f"feedback_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                
                st.sidebar.markdown(
                    f'<a href="data:text/html;base64,{b64_html}" download="{download_filename}">Click to Download HTML Report</a>',
                    unsafe_allow_html=True
                )
                st.sidebar.success("Report generated!")
            
else:
    st.info("Please upload a CSV or Excel file to begin the analysis.")

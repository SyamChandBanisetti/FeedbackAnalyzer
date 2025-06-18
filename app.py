import json
import tempfile
import csv
import streamlit as st
import pandas as pd
import google.generativeai as genai
import numpy as np
import re
import os

# Agno and Phi imports
from agno.models.openai import OpenAIChat # Agno agent's reasoning model (requires OpenAI Key)
from agno.agent import Agent # Generic Agno agent
from agno.tools import Tool # Base tool class for custom tools

# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="📝 AI Feedback Analyst Agent")

# --- Session State Initialization ---
# This ensures variables persist across reruns
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = {
        "sampled_analysis_df": pd.DataFrame(),
        "summary_themes": "",
        "recommendations": ""
    }
if "agno_feedback_agent" not in st.session_state:
    st.session_state.agno_feedback_agent = None
if "agent_chat_history" not in st.session_state:
    st.session_state.agent_chat_history = []
if "gemini_model" not in st.session_state:
    st.session_state.gemini_model = None
if "gemini_api_key_set" not in st.session_state:
    st.session_state.gemini_api_key_set = False
if "openai_api_key_set" not in st.session_state:
    st.session_state.openai_api_key_set = False

# --- Main App Title and Introduction ---
st.title("📝 AI Feedback Analyst Agent")
st.markdown("Upload your customer feedback data, and an AI agent will analyze it using Gemini 2.0 Flash and answer your questions.")

# --- API Key Input Section (Mandatory First Step) ---
st.header("🔑 Enter Your API Keys to Start")
st.info("Both Gemini and OpenAI API keys are required for full functionality. Your keys are used only for the current session and are not stored.")

gemini_api_key_input = st.text_input(
    "**Gemini API Key:** (for analysis functions)", 
    type="password", 
    key="gemini_api_key_form",
    help="Get your Gemini API key from Google AI Studio. It powers sentiment analysis, summarization, and recommendations."
)
openai_api_key_input = st.text_input(
    "**OpenAI API Key:** (for the AI Agent)", 
    type="password", 
    key="openai_api_key_form",
    help="Get your OpenAI API key from OpenAI Platform. It powers the conversational AI agent's reasoning and tool use."
)

# Check for API keys from Streamlit secrets (for deployment) or environment variables
if not gemini_api_key_input:
    try:
        gemini_api_key_input = st.secrets["GOOGLE_API_KEY"]
    except KeyError:
        pass # Will remain empty if not in secrets

if not openai_api_key_input:
    openai_api_key_input = os.getenv("OPENAI_API_KEY")


# Logic to set API keys and proceed
if gemini_api_key_input:
    try:
        genai.configure(api_key=gemini_api_key_input)
        st.session_state.gemini_model = genai.GenerativeModel("gemini-2.0-flash")
        st.session_state.gemini_api_key_set = True
    except Exception as e:
        st.error(f"Invalid Gemini API Key: {e}. Please check your key.")
        st.session_state.gemini_api_key_set = False
        st.session_state.gemini_model = None
else:
    st.session_state.gemini_api_key_set = False

if openai_api_key_input:
    try:
        # We don't directly "configure" agno.models.openai like genai.configure
        # The key is passed directly to the model instance.
        st.session_state.openai_api_key = openai_api_key_input
        st.session_state.openai_api_key_set = True
    except Exception as e:
        st.error(f"Issue with OpenAI API Key configuration: {e}. Please check your key.")
        st.session_state.openai_api_key_set = False
else:
    st.session_state.openai_api_key_set = False


# Proceed only if both keys are set
if not (st.session_state.gemini_api_key_set and st.session_state.openai_api_key_set):
    st.warning("Please enter both valid API keys above to unlock the application features.")
    # Show example data structure even if keys are not set
    st.markdown("---")
    st.markdown("### Example Data Structure:")
    sample_data = pd.DataFrame({
        "Timestamp": ["2024-06-01", "2024-06-02", "2024-06-03"],
        "Customer Feedback": [
            "The new app update is fantastic! So much faster and intuitive.",
            "Customer support was very slow, took ages to get a reply. Disappointed.",
            "Product features are good, but the pricing is too high."
        ],
        "Rating (1-5)": [5, 2, 4]
    })
    st.dataframe(sample_data, use_container_width=True)
    st.markdown("*(Your file can have many columns, but at least one should contain open-ended text feedback)*")
    st.stop() # Stop execution here if keys are missing


# --- Helper Functions (Gemini powered - unchanged) ---
@st.cache_data(show_spinner=False)
def get_sentiment_and_score(text_series_input, gemini_model_instance):
    """
    Analyzes sentiment of text using Gemini.
    """
    if not gemini_model_instance:
        st.error("Gemini model not initialized. Please provide API key.")
        return pd.Series(["Error"] * len(text_series_input), index=text_series_input.index), pd.Series([np.nan] * len(text_series_input), index=text_series_input.index)

    sentiments, scores = [], []
    texts_to_process = text_series_input.dropna().astype(str)

    progress_bar = st.progress(0, text="Analyzing sentiments...")
    total_texts = len(texts_to_process)

    for i, text in enumerate(texts_to_process):
        if not text.strip():
            sentiments.append("Neutral"); scores.append(0.0)
            continue

        prompt = f"""Analyze the sentiment of the following feedback and classify it as Positive, Negative, or Neutral.
        Then, provide a sentiment score: -1 for Negative, 0 for Neutral, and 1 for Positive.
        Return the output in the exact format: Sentiment: [Sentiment], Score: [Score]
        Feedback: "{text}"\n"""
        try:
            response = gemini_model_instance.generate_content(prompt)
            result = response.text.strip()

            sentiment = "Unknown"; score = np.nan
            sentiment_match = re.search(r"Sentiment:\s*([A-Za-z]+)", result)
            score_match = re.search(r"Score:\s*([-+]?\d*\.?\d+)", result)

            if sentiment_match: sentiment = sentiment_match.group(1).strip()
            if score_match:
                try: score = float(score_match.group(1).strip())
                except ValueError: pass

            sentiments.append(sentiment); scores.append(score)
        except Exception as e:
            st.warning(f"Error processing text (first 50 chars: '{text[:50]}...'): {e}")
            sentiments.append("Error"); scores.append(np.nan)

        progress_bar.progress((i + 1) / total_texts, text=f"Analyzing sentiments... ({i+1}/{total_texts})")
    progress_bar.empty()
    return pd.Series(sentiments, index=texts_to_process.index), pd.Series(scores, index=texts_to_process.index)


@st.cache_data(show_spinner=False)
def summarize_feedback_themes_llm(feedback_text_list, creativity, gemini_model_instance):
    """Summarizes feedback themes using Gemini."""
    if not gemini_model_instance:
        return "Gemini model not initialized. Cannot summarize themes."
    joined_feedback = "\n".join(feedback_text_list).strip()
    if not joined_feedback:
        return "No sufficient feedback to summarize themes."
    prompt = f"""Review the following customer feedback entries. Identify and list the top 3-5 most important recurring themes or topics discussed.
    Present them as a bullet-point list.
    Feedback:
    {joined_feedback}

    Themes:"""
    try:
        response = gemini_model_instance.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=creativity)
        )
        return response.text.strip()
    except Exception as e:
        st.error(f"Error generating summary themes: {e}")
        return "Could not generate key themes."

@st.cache_data(show_spinner=False)
def generate_recommendations_llm(negative_feedback_summary, creativity, gemini_model_instance):
    """Generates recommendations using Gemini."""
    if not gemini_model_instance:
        return "Gemini model not initialized. Cannot generate recommendations."
    if not negative_feedback_summary.strip() or "No sufficient feedback" in negative_feedback_summary:
        return "No specific negative feedback identified to generate recommendations."
    prompt = f"""Based on the following summary of negative feedback, provide 3-5 clear, actionable recommendations for improvement.
    Summary of negative feedback:
    {negative_feedback_summary}

    Actionable Recommendations:"""
    try:
        response = gemini_model_instance.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=creativity)
        )
        return response.text.strip()
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
        return "Could not generate recommendations."

def plot_wordcloud(text_data):
    """Generates and displays a word cloud."""
    if not text_data or not isinstance(text_data, list):
        st.info("No text data available for word cloud.")
        return
    text = " ".join([str(item) for item in text_data if pd.notna(item) and str(item).strip() != ""])
    if not text.strip():
        st.info("Not enough valid text to generate a word cloud.")
        return
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)


# --- Agno Tools (to query pre-computed results) ---
# These tools wrap direct access to st.session_state.analysis_results
class FeedbackAnalysisTools(Tool):
    def __init__(self, **kwargs):
        super().__init__(name="feedback_analysis_tools", **kwargs)
        self.register(self.get_overall_sentiment_data)
        self.register(self.get_key_themes_summary)
        self.register(self.get_actionable_recommendations)

    def _get_analysis_status(self):
        # Check if sampled_analysis_df is populated and not empty
        if st.session_state.analysis_results["sampled_analysis_df"].empty or st.session_state.analysis_results["sampled_analysis_df"].iloc[0].isnull().all():
            return "No feedback data has been analyzed yet. Please upload a file and click 'Run AI Analysis' first."
        return None # All data is available

    @Tool
    def get_overall_sentiment_data(self) -> str:
        """
        Provides the overall sentiment distribution (Positive, Negative, Neutral) and average sentiment score
        for the currently analyzed feedback data.
        Use this tool when the user asks about general sentiment, overall feeling, or sentiment statistics.
        """
        status = self._get_analysis_status()
        if status: return status

        df_analyzed = st.session_state.analysis_results["sampled_analysis_df"]
        sentiment_counts = df_analyzed['sentiment'].value_counts().to_dict()
        total_count = sum(sentiment_counts.values())
        sentiment_summary = ", ".join([f"{s}: {count} ({count/total_count:.1%})" for s, count in sentiment_counts.items()])

        avg_score = df_analyzed['score'].mean()
        score_str = f"Average sentiment score: {avg_score:.2f} (from -1 to 1)." if not np.isnan(avg_score) else "Average sentiment score is not available."

        return f"Overall sentiment distribution: {sentiment_summary}. {score_str}"

    @Tool
    def get_key_themes_summary(self) -> str:
        """
        Provides the AI-generated summary of the key recurring themes from the feedback.
        Use this tool when the user asks about main topics, recurring themes, or a summary of the feedback.
        """
        status = self._get_analysis_status()
        if status: return status
        summary = st.session_state.analysis_results["summary_themes"]
        if not summary.strip():
            return "Key themes summary has not been generated yet or is empty."
        return summary

    @Tool
    def get_actionable_recommendations(self) -> str:
        """
        Provides AI-generated actionable recommendations based on the negative feedback.
        Use this tool when the user asks for suggestions, improvements, or how to address problems.
        """
        status = self._get_analysis_status()
        if status: return status
        recommendations = st.session_state.analysis_results["recommendations"]
        if not recommendations.strip() or "No negative feedback" in recommendations:
            return "Actionable recommendations have not been generated yet, or there was no significant negative feedback to base them on."
        return recommendations


# --- Preprocessing Function (unchanged) ---
@st.cache_data(show_spinner=False)
def preprocess_and_save(file_content, file_name):
    """
    Preprocesses the uploaded file content and saves it to a temporary CSV.
    Returns the temporary file path, column names, and the DataFrame.
    Uses st.cache_data to avoid re-processing on every rerun.
    """
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(file_content, encoding='utf-8', na_values=['NA', 'N/A', 'missing'])
        elif file_name.endswith('.xlsx'):
            df = pd.read_excel(file_content, na_values=['NA', 'N/A', 'missing'])
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None, None, None
        
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).replace({r'"': '""'}, regex=True).str.strip()
            df[col] = df[col].replace({'': pd.NA})
        
        for col in df.columns:
            if 'date' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')
            elif df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except (ValueError, TypeError):
                    pass
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".csv", encoding='utf-8') as temp_file:
            temp_path = temp_file.name
            df.to_csv(temp_path, index=False, quoting=csv.QUOTE_ALL)
        
        return temp_path, df.columns.tolist(), df
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, None, None


# --- Main Application Logic (after API keys are set) ---
st.markdown("---")
st.header("📤 Upload Your Feedback Data")
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    file_bytes_data = uploaded_file.getvalue()

    if "last_uploaded_file_id" not in st.session_state or st.session_state.last_uploaded_file_id != uploaded_file.file_id:
        st.session_state.last_uploaded_file_id = uploaded_file.file_id
        st.session_state.df_loaded = False
        st.session_state.analysis_results = { # Reset results for new file
            "sampled_analysis_df": pd.DataFrame(),
            "summary_themes": "",
            "recommendations": ""
        }
        st.session_state.agent_chat_history = [] # Clear agent chat history
        st.session_state.agno_feedback_agent = None # Reset agent instance

    if not st.session_state.get("df_loaded", False):
        with st.spinner("Preprocessing your file... This might take a moment for large files."):
            temp_file_content_io = io.BytesIO(file_bytes_data)
            temp_path, columns, df = preprocess_and_save(temp_file_content_io, uploaded_file.name)
        
        if temp_path and columns and df is not None:
            st.session_state.temp_path = temp_path
            st.session_state.df_columns = columns
            st.session_state.original_df = df
            st.session_state.df_loaded = True
            st.success("File preprocessed successfully! Now configure analysis and run it.")
        else:
            st.session_state.df_loaded = False
            st.error("Failed to preprocess file. Please check its content and format.")

    if st.session_state.get("df_loaded", False):
        df = st.session_state.original_df

        with st.expander("View Uploaded Data and Columns"):
            st.subheader("Uploaded Data Preview:")
            st.dataframe(df)
            st.subheader("Detected Columns:")
            st.write(st.session_state.df_columns)

        st.markdown("---")
        st.header("⚙️ Analysis Configuration")
        col1_config, col2_config, col3_config = st.columns(3)

        with col1_config:
            selected_column = st.selectbox(
                "Select text column for analysis:",
                st.session_state.df_columns,
                key="selected_text_column",
                help="Choose the column containing text feedback."
            )
        with col2_config:
            max_rows_for_llm = st.slider(
                "Limit rows for AI processing:",
                min_value=10, max_value=500, value=100, step=10,
                key="max_rows_slider",
                help="Adjust to balance analysis depth and speed. Larger numbers mean more data processed."
            )
        with col3_config:
            llm_creativity = st.slider(
                "AI Creativity (for Summaries/Recs):",
                min_value=0.0, max_value=1.0, value=0.4, step=0.1,
                key="creativity_slider",
                help="Lower for more factual, higher for more diverse outputs."
            )

        if st.button("🚀 Run AI Analysis", use_container_width=True):
            if not st.session_state.get("gemini_api_key_set"):
                st.error("Gemini API key is not set. Please provide it in the API Key section above.")
            elif selected_column:
                st.markdown("---")
                st.header("📈 Analysis Results")
                
                responses_to_analyze = df[selected_column].dropna().sample(
                    n=min(len(df[selected_column].dropna()), max_rows_for_llm), random_state=42
                )
                if responses_to_analyze.empty:
                    st.info("No valid responses found in the selected sample for analysis.")
                    st.session_state.analysis_results = {**st.session_state.analysis_results, "sampled_analysis_df": pd.DataFrame()}
                    st.stop()

                # --- 1. Sentiment Analysis ---
                st.subheader("1. Sentiment Breakdown")
                with st.spinner(f"Running sentiment analysis on {len(responses_to_analyze)} responses..."):
                    sentiments, scores = get_sentiment_and_score(responses_to_analyze, st.session_state.gemini_model)

                sampled_analysis_df = pd.DataFrame({
                    'text': responses_to_analyze,
                    'sentiment': sentiments,
                    'score': scores
                })
                st.session_state.analysis_results["sampled_analysis_df"] = sampled_analysis_df

                if not sampled_analysis_df.empty:
                    sentiment_counts = sampled_analysis_df['sentiment'].value_counts().reset_index()
                    sentiment_counts.columns = ["Sentiment", "Count"]
                    fig_sentiment = px.bar(sentiment_counts, x="Sentiment", y="Count", color="Sentiment",
                                           title="Overall Sentiment Distribution",
                                           color_discrete_map={"Positive":"#28a745", "Negative":"#dc3545", "Neutral":"#ffc107", "Unknown":"#6c757d", "Error":"#6c757d"})
                    st.plotly_chart(fig_sentiment, use_container_width=True)

                    avg_score = sampled_analysis_df['score'].mean()
                    if not np.isnan(avg_score):
                        st.info(f"**Average Sentiment Score:** {avg_score:.2f} (from -1 to 1; higher is more positive)")
                else:
                    st.info("No sentiment data generated for visualization.")

                # --- 2. Word Cloud ---
                st.subheader("2. Key Themes Word Cloud")
                with st.spinner("Generating word cloud..."):
                    plot_wordcloud(responses_to_analyze.tolist())

                # --- 3. Summary of Themes ---
                st.subheader("3. AI-Generated Key Themes Summary")
                with st.spinner("Summarizing key themes..."):
                    summary_themes = summarize_feedback_themes_llm(responses_to_analyze.tolist(), llm_creativity, st.session_state.gemini_model)
                    st.success(summary_themes)
                    st.session_state.analysis_results["summary_themes"] = summary_themes

                # --- 4. Actionable Recommendations from Negative Feedback ---
                st.subheader("4. Actionable Recommendations (from Negative Feedback)")
                negative_responses_sampled = sampled_analysis_df[sampled_analysis_df['sentiment'] == 'Negative']['text'].dropna().tolist()

                if negative_responses_sampled:
                    with st.spinner(f"Generating recommendations from {len(negative_responses_sampled)} negative responses..."):
                        negative_summary_for_recs = summarize_feedback_themes_llm(negative_responses_sampled, llm_creativity, st.session_state.gemini_model)
                        recommendations = generate_recommendations_llm(negative_summary_for_recs, llm_creativity, st.session_state.gemini_model)
                        st.warning(recommendations)
                        st.session_state.analysis_results["recommendations"] = recommendations
                else:
                    st.info("No negative feedback identified in the sample to generate recommendations.")
                    st.session_state.analysis_results["recommendations"] = "No negative feedback to generate recommendations."

                st.success("Analysis complete! You can now chat with the AI Agent below.")
            else:
                st.warning("Please select a text column to run the analysis.")

# --- Agno Agent Chat Interface ---
st.markdown("---")
st.header("💬 Chat with AI Feedback Agent")
st.markdown("Ask the agent questions about the analyzed feedback (e.g., 'What is the overall sentiment?', 'Can you summarize the main themes?', 'What are the recommendations for improvement?').")

if st.session_state.get("openai_api_key_set"):
    # Initialize Agno agent if not already initialized or if OpenAI key changed
    if st.session_state.agno_feedback_agent is None or st.session_state.agno_feedback_agent.model.api_key != st.session_state.openai_api_key:
        st.session_state.agno_feedback_agent = Agent(
            model=OpenAIChat(model="gpt-4o", api_key=st.session_state.openai_api_key),
            tools=[FeedbackAnalysisTools()],
            markdown=True,
            add_history_to_messages=True, # Enable chat history for better agent context
            system_prompt="""You are an expert feedback analyst. Your goal is to answer user questions about the analyzed feedback data using the available tools.
            Always confirm that the analysis has been run before attempting to answer a question.
            If the analysis is not available, politely instruct the user to upload data and click 'Run AI Analysis'.
            Present the information retrieved from the tools concisely and clearly.
            Do not make up information. If a specific detail is not available through your tools, state that you cannot provide it.
            """
        )
    
    # Display chat messages from history
    for msg in st.session_state.agent_chat_history:
        if msg["role"] == "user":
            st.chat_message("user").markdown(msg["content"])
        else:
            st.chat_message("assistant").markdown(msg["content"])

    # User input for chat
    user_query_agent = st.chat_input("Ask the agent about the feedback data...")
    if user_query_agent:
        st.session_state.agent_chat_history.append({"role": "user", "content": user_query_agent})
        with st.chat_message("user"):
            st.markdown(user_query_agent)

        with st.spinner("Agent is thinking..."):
            try:
                agent_response = st.session_state.agno_feedback_agent.run(user_query_agent)
                
                if hasattr(agent_response, 'content'):
                    response_content = agent_response.content
                else:
                    response_content = str(agent_response)

                st.session_state.agent_chat_history.append({"role": "assistant", "content": response_content})
                with st.chat_message("assistant"):
                    st.markdown(response_content)
            except Exception as e:
                error_message = f"Agent encountered an error: {e}. Please ensure analysis has run or try rephrasing."
                st.error(error_message)
                st.session_state.agent_chat_history.append({"role": "assistant", "content": "Sorry, I couldn't process that request. " + str(e)})
else:
    st.info("Provide your OpenAI API key in the 'Enter Your API Keys to Start' section above to enable the AI Agent chat.")
    st.session_state.agent_chat_history = [] # Clear history if keys are removed/missing

# --- Cleanup temporary file on app close or rerun (best effort) ---
# This attempts to clean up the temp file. Streamlit's lifecycle can make perfect cleanup tricky.
# For production, consider persistent storage if temporary files are an issue.
if 'temp_path' in st.session_state and os.path.exists(st.session_state.temp_path):
    if uploaded_file is None or not st.session_state.get('df_loaded', False):
        try:
            os.remove(st.session_state.temp_path)
            del st.session_state.temp_path
        except OSError as e:
            st.warning(f"Could not remove temporary file {st.session_state.temp_path}: {e}")

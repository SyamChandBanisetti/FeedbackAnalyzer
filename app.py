Guide: Building and Deploying Your Dash Feedback Analyzer App
This guide will walk you through setting up your project files for a Dash application and deploying it online using GitHub and Render.com.

Step 1: Set Up Your Project Files
You'll need two main files in your project directory (which will become your GitHub repository).

1. app.py (Your Dash Application Code)
This is the core of your Dash application. It handles the UI layout, file uploads, triggers the Gemini API analysis, and displays interactive charts and summaries.

Key Differences from Streamlit:

No st. prefix: You'll use dash, dash.dcc (Dash Core Components), and dash.html (Dash HTML Components) instead.

Layout Structure: Dash apps define their UI using a tree of components, often nested.

Callbacks: Interactivity (like button clicks, file uploads) is handled by @app.callback decorators, which explicitly define inputs and outputs.

API Key: Your Gemini API key will be accessed via os.getenv (standard environment variables), which you'll configure on Render.com.

import os
import base64
import io
import pandas as pd
import google.generativeai as genai
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import re

# Dash Imports
from dash import Dash, html, dcc, Input, Output, State, no_update

# --- Setup ---
# Your Gemini API key must be set as an environment variable in your deployment environment (e.g., Render.com).
# On Render.com, add an environment variable: Key=GOOGLE_API_KEY, Value=YOUR_GEMINI_API_KEY_HERE
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GEMINI_API_KEY:
    # For local development, you might set it directly here for testing purposes
    # or load from a .env file (but DO NOT commit .env to GitHub)
    # GEMINI_API_KEY = "YOUR_LOCAL_GEMINI_API_KEY"
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please configure it in your deployment environment.")

genai.configure(api_key=GEMINI_API_KEY)
llm_model = genai.GenerativeModel("gemini-2.0-flash") # Renamed to avoid conflict with agent if integrated later

# Initialize Dash App
app = Dash(__name__)
server = app.server # This is needed for deployment on platforms like Render


# --- Helper Functions for AI Analysis ---

# Note: In Dash, we don't have st.cache_data directly.
# The `max_rows_for_llm` parameter and explicit button click
# manage the processing load.
def get_sentiment_and_score(text_series_input):
    """
    Analyzes the sentiment of each text in a Pandas Series and assigns a score.
    Score: -1 (Negative), 0 (Neutral), 1 (Positive)
    """
    sentiments, scores = [], []
    texts_to_process = text_series_input.dropna().astype(str)

    # In a real-time Streamlit app, we had a progress bar.
    # For Dash, progress updates within a single callback are more complex
    # and often done with loading states or background callbacks.
    # For simplicity here, we'll just process.
    for text in texts_to_process:
        if not text.strip():
            sentiments.append("Neutral")
            scores.append(0.0)
            continue

        prompt = f"""Analyze the sentiment of the following feedback and classify it as Positive, Negative, or Neutral.
        Then, provide a sentiment score: -1 for Negative, 0 for Neutral, and 1 for Positive.
        Return the output in the exact format: Sentiment: [Sentiment], Score: [Score]
        Feedback: "{text}"\n"""
        try:
            response = llm_model.generate_content(prompt)
            result = response.text.strip()

            sentiment = "Unknown"
            score = np.nan

            sentiment_match = re.search(r"Sentiment:\s*([A-Za-z]+)", result)
            score_match = re.search(r"Score:\s*([-+]?\d*\.?\d+)", result)

            if sentiment_match:
                sentiment = sentiment_match.group(1).strip()
            if score_match:
                try:
                    score = float(score_match.group(1).strip())
                except ValueError:
                    pass

            sentiments.append(sentiment)
            scores.append(score)
        except Exception as e:
            print(f"Error processing text: {e}") # Log to console for debugging
            sentiments.append("Error")
            scores.append(np.nan)

    return pd.Series(sentiments, index=text_series_input.index), pd.Series(scores, index=text_series_input.index)


def summarize_feedback_themes(feedback_text_list, creativity=0.4):
    """Summarizes a list of feedback texts into key themes using the LLM."""
    joined_feedback = "\n".join(feedback_text_list).strip()
    if not joined_feedback:
        return "No sufficient feedback to summarize themes."
    prompt = f"""Review the following customer feedback entries. Identify and list the top 3-5 most important recurring themes or topics discussed.
    Present them as a bullet-point list.
    Feedback:
    {joined_feedback}

    Themes:"""
    try:
        response = llm_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=creativity)
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error generating summary themes: {e}")
        return "Could not generate key themes."

def generate_recommendations(negative_feedback_summary, creativity=0.6):
    """Generates actionable recommendations based on a summary of negative feedback."""
    if not negative_feedback_summary.strip() or "No sufficient feedback" in negative_feedback_summary:
        return "No specific negative feedback identified to generate recommendations."
    prompt = f"""Based on the following summary of negative feedback, provide 3-5 clear, actionable recommendations for improvement.
    Summary of negative feedback:
    {negative_feedback_summary}

    Actionable Recommendations:"""
    try:
        response = llm_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=creativity)
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error generating recommendations: {e}")
        return "Could not generate recommendations."

def plot_wordcloud(text_data):
    """Generates a matplotlib word cloud image and returns it as a Plotly Figure."""
    if not text_data or not isinstance(text_data, list):
        return px.scatter() # Return an empty plot if no data

    text = " ".join([str(item) for item in text_data if pd.notna(item) and str(item).strip() != ""])
    if not text.strip():
        return px.scatter() # Return an empty plot if no valid text

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig_mpl, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')

    # Convert matplotlib figure to a Plotly figure (necessary for Dash)
    # This is a bit of a workaround as WordCloud directly outputs matplotlib.
    # Plotly can display image data, but it's simpler to show a placeholder or text.
    # For now, we'll return a blank plotly figure, as embedding matplotlib directly is complex.
    # A real solution might involve saving the matplotlib image to a buffer and serving it.
    # For this simplified Dash app, we'll return a placeholder.
    return px.scatter(title="Word Cloud (Requires direct image embedding, not directly supported in simple Plotly Fig)")


# --- Dash App Layout ---
app.layout = html.Div(style={'fontFamily': 'Inter, sans-serif', 'padding': '20px'}, children=[
    html.H1("✨ AI-Powered Feedback Analyzer", style={'textAlign': 'center', 'color': '#2C3E50'}),
    html.P("Upload your customer feedback data (CSV or Excel) to gain instant insights into sentiment, key themes, and actionable recommendations using Gemini 2.0 Flash.", style={'textAlign': 'center', 'color': '#7F8C8D', 'marginBottom': '30px'}),

    html.Div(style={'maxWidth': '800px', 'margin': '0 auto', 'padding': '20px', 'backgroundColor': '#ECF0F1', 'borderRadius': '10px', 'boxShadow': '0 4px 8px rgba(0,0,0,0.1)'}, children=[
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '100%', 'height': '60px', 'lineHeight': '60px',
                'borderWidth': '2px', 'borderStyle': 'dashed',
                'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px 0',
                'cursor': 'pointer', 'backgroundColor': '#FFFFFF', 'borderColor': '#BDC3C7'
            },
            multiple=False # Allow only one file
        ),
        html.Div(id='output-data-upload'), # To show raw data preview

        html.Div(id='analysis-config', children=[
            html.Hr(),
            html.H3("⚙️ Analysis Configuration", style={'color': '#2C3E50'}),
            dcc.Dropdown(
                id='text-column-selector',
                options=[], # Options will be loaded via callback
                placeholder="Select the text column for AI analysis",
                style={'marginBottom': '15px'}
            ),
            html.Label("Limit rows for AI processing (for speed)", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
            dcc.Slider(
                id='max-rows-slider',
                min=10, max=500, step=10, value=100,
                marks={i: str(i) for i in range(10, 501, 50)},
                tooltip={"placement": "bottom", "always_visible": True},
                className='mb-3'
            ),
            html.Label("AI Creativity (Temperature for Summaries/Recommendations)", style={'color': '#2C3E50', 'fontWeight': 'bold', 'marginTop': '20px'}),
            dcc.Slider(
                id='creativity-slider',
                min=0.0, max=1.0, step=0.1, value=0.4,
                marks={0.0: '0.0', 0.2: '0.2', 0.4: '0.4', 0.6: '0.6', 0.8: '0.8', 1.0: '1.0'},
                tooltip={"placement": "bottom", "always_visible": True},
                className='mb-3'
            ),
            html.Button('🚀 Run AI Analysis', id='run-analysis-button', n_clicks=0,
                        style={
                            'backgroundColor': '#2ECC71', 'color': 'white', 'padding': '12px 25px',
                            'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer',
                            'fontSize': '16px', 'fontWeight': 'bold', 'marginTop': '20px',
                            'width': '100%', 'transition': 'background-color 0.3s ease'
                        }),
            html.Div(id='analysis-status', style={'marginTop': '15px', 'textAlign': 'center'})
        ], style={'display': 'none'}), # Hidden until file is uploaded
    ]),

    html.Div(id='analysis-results', style={'marginTop': '40px', 'display': 'none'}, children=[
        html.H3("📈 Analysis Results", style={'textAlign': 'center', 'color': '#2C3E50', 'marginBottom': '20px'}),

        html.Div(className='row', children=[
            html.Div(className='col-md-6', children=[
                html.H4("1. Sentiment Breakdown", style={'color': '#34495E'}),
                dcc.Graph(id='sentiment-graph', config={'displayModeBar': False})
            ]),
            html.Div(className='col-md-6', children=[
                html.H4("Average Sentiment Score", style={'color': '#34495E'}),
                html.Div(id='average-score-display', style={'padding': '15px', 'backgroundColor': '#D6EAF8', 'borderRadius': '8px', 'fontSize': '1.1em', 'color': '#2874A6'})
            ])
        ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '30px'}), # Simple grid layout

        html.H4("2. Key Themes Word Cloud (Feature Placeholder)", style={'color': '#34495E'}),
        html.Div("Word cloud visualization is more complex to embed directly in Plotly/Dash and would typically require a separate image serving endpoint or custom component.", style={'padding': '15px', 'backgroundColor': '#FDEBD0', 'borderRadius': '8px', 'color': '#B8860B', 'marginBottom': '30px'}),
        # dcc.Graph(id='wordcloud-graph', config={'displayModeBar': False}), # Placeholder for word cloud

        html.H4("3. AI-Generated Key Themes Summary", style={'color': '#34495E'}),
        html.Div(id='summary-themes-output', style={'padding': '15px', 'backgroundColor': '#D4EFDF', 'borderRadius': '8px', 'fontSize': '1.1em', 'color': '#229954', 'marginBottom': '30px'}),

        html.H4("4. Actionable Recommendations (from Negative Feedback)", style={'color': '#34495E'}),
        html.Div(id='recommendations-output', style={'padding': '15px', 'backgroundColor': '#FADBD8', 'borderRadius': '8px', 'fontSize': '1.1em', 'color': '#C0392B', 'marginBottom': '50px'}),
    ]),

    # Hidden Divs to store processed data for callbacks
    dcc.Store(id='stored-data-df', data={}),
    dcc.Store(id='stored-analysis-df', data={})
])


# --- Callbacks ---

# Callback to parse uploaded data and populate column selector
@app.callback(
    [Output('output-data-upload', 'children'),
     Output('analysis-config', 'style'),
     Output('text-column-selector', 'options'),
     Output('text-column-selector', 'value')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def upload_file(contents, filename):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            if 'csv' in filename:
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            elif 'xls' in filename:
                df = pd.read_excel(io.BytesIO(decoded))
            else:
                return html.Div(['Unsupported file type. Please upload a CSV or XLSX file.']), {'display': 'none'}, [], None
        except Exception as e:
            print(e)
            return html.Div([f'There was an error processing your file: {e}.']), {'display': 'none'}, [], None

        # Store dataframe in dcc.Store
        df_json = df.to_json(date_format='iso', orient='split')

        # Identify text columns
        text_columns = [col for col in df.columns if df[col].dtype == 'object' and df[col].astype(str).apply(len).mean() > 10]
        column_options = [{'label': col, 'value': col} for col in text_columns]

        preview_table = html.Div([
            html.H4(f"File: {filename} (First 5 Rows)", style={'marginTop': '20px'}),
            html.Table([
                html.Thead(html.Tr([html.Th(col) for col in df.columns])),
                html.Tbody([
                    html.Tr([html.Td(df.iloc[i][col]) for col in df.columns]) for i in range(min(len(df), 5))
                ])
            ], style={'width': '100%', 'borderCollapse': 'collapse', 'marginTop': '10px'}),
            dcc.Store(id='stored-data-df', data=df_json) # Update stored data
        ])
        return preview_table, {'display': 'block'}, column_options, text_columns[0] if text_columns else None
    return html.Div("Upload a CSV or Excel file to begin analyzing your feedback!"), {'display': 'none'}, [], None


# Callback to run AI analysis
@app.callback(
    [Output('analysis-status', 'children'),
     Output('analysis-results', 'style'),
     Output('sentiment-graph', 'figure'),
     Output('average-score-display', 'children'),
     Output('summary-themes-output', 'children'),
     Output('recommendations-output', 'children'),
     Output('stored-analysis-df', 'data')], # Store processed df
    Input('run-analysis-button', 'n_clicks'),
    State('stored-data-df', 'data'),
    State('text-column-selector', 'value'),
    State('max-rows-slider', 'value'),
    State('creativity-slider', 'value'),
    prevent_initial_call=True
)
def run_analysis(n_clicks, df_json, selected_column, max_rows, creativity):
    if n_clicks > 0 and df_json is not None and selected_column is not None:
        df = pd.read_json(df_json, orient='split')
        
        # Sample responses for LLM
        responses_to_analyze = df[selected_column].dropna().sample(n=min(len(df[selected_column].dropna()), max_rows), random_state=42)
        if responses_to_analyze.empty:
            return "No valid responses found in the selected sample for analysis.", {'display': 'none'}, px.scatter(), "N/A", "N/A", "N/A", {}

        # Sentiment Analysis
        sentiments, scores = get_sentiment_and_score(responses_to_analyze)
        sampled_analysis_df = pd.DataFrame({
            'text': responses_to_analyze,
            'sentiment': sentiments,
            'score': scores
        })

        # Sentiment Distribution Plot
        if not sampled_analysis_df.empty:
            sentiment_counts = sampled_analysis_df['sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ["Sentiment", "Count"]
            fig_sentiment = px.bar(sentiment_counts, x="Sentiment", y="Count", color="Sentiment",
                                   title="Overall Sentiment Distribution",
                                   color_discrete_map={"Positive":"#28a745", "Negative":"#dc3545", "Neutral":"#ffc107", "Unknown":"#6c757d", "Error":"#6c757d"})
        else:
            fig_sentiment = px.scatter(title="No sentiment data to display")

        # Average Score Display
        avg_score = sampled_analysis_df['score'].mean()
        avg_score_text = f"**Average Score:** {avg_score:.2f} (closer to 1 is more positive, -1 more negative)" if not np.isnan(avg_score) else "Average score not available."

        # Key Themes Summary
        summary_themes = summarize_feedback_themes(responses_to_analyze.tolist(), creativity=creativity)

        # Recommendations from Negative Feedback
        negative_responses_sampled = sampled_analysis_df[sampled_analysis_df['sentiment'] == 'Negative']['text'].dropna().tolist()
        recommendations_text = ""
        if negative_responses_sampled:
            negative_summary_for_recs = summarize_feedback_themes(negative_responses_sampled, creativity=creativity) # Reuse summarize
            recommendations_text = generate_recommendations(negative_summary_for_recs, creativity=creativity)
        else:
            recommendations_text = "No negative feedback identified in the sample to generate recommendations."

        # Return all updated components
        return (
            html.Div("Analysis complete!", style={'color': 'green', 'fontWeight': 'bold'}),
            {'display': 'block'}, # Show analysis results section
            fig_sentiment,
            html.P(avg_score_text),
            html.Div(summary_themes),
            html.Div(recommendations_text),
            sampled_analysis_df.to_json(date_format='iso', orient='split') # Store processed data
        )
    return no_update # No update if conditions not met

if __name__ == '__main__':
    app.run_server(debug=True)

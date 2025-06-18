# (keep all previous imports)
import streamlit as st
import pandas as pd
import tempfile
import csv
import re
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
import google.generativeai as genai
from collections import Counter

# ✅ Gemini Init
def init_gemini(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")

# ✅ File Processing
def preprocess_and_save(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            st.error("Unsupported file format.")
            return None, None, None
        
        for col in df.select_dtypes(include='object'):
            df[col] = df[col].astype(str).replace({r'"': '""'}, regex=True)

        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')
            elif df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col])
                except:
                    pass

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp_path = tmp.name
            df.to_csv(tmp_path, index=False, quoting=csv.QUOTE_ALL)

        return tmp_path, df.columns.tolist(), df
    except Exception as e:
        st.error(f"Processing error: {e}")
        return None, None, None

# ✅ Sentiment
def classify_sentiment(text):
    text = text.lower()
    if any(w in text for w in ["good", "great", "excellent", "love", "awesome", "satisfied", "happy"]):
        return "Positive"
    elif any(w in text for w in ["bad", "poor", "terrible", "hate", "worst", "unsatisfied", "boring"]):
        return "Negative"
    elif text.strip() == "":
        return "Neutral"
    else:
        return "Neutral"

# ✅ Keywords
@st.cache_data
def extract_keywords(texts, top_n=10):
    vectorizer = CountVectorizer(stop_words='english', max_features=top_n)
    X = vectorizer.fit_transform(texts)
    return vectorizer.get_feature_names_out().tolist()

# ✅ Streamlit App UI
st.set_page_config(page_title="🧠 Feedback Analyzer", layout="wide")
st.title("📋 Feedback Analyzer using Gemini Flash 2.0")

with st.sidebar:
    st.header("🔐 Gemini API Key")
    gemini_key = st.text_input("Enter your Gemini API key", type="password")
    if gemini_key:
        gemini = init_gemini(gemini_key)
        st.success("Gemini API initialized!")
    else:
        st.warning("API key required to continue.")

uploaded_file = st.file_uploader("📤 Upload Feedback File (.csv or .xlsx)", type=["csv", "xlsx"])

if uploaded_file and gemini_key:
    tmp_path, all_columns, df = preprocess_and_save(uploaded_file)

    if tmp_path:
        st.subheader("📄 Feedback Preview")
        st.dataframe(df, use_container_width=True)

        # Identify feedback columns
        text_cols = df.select_dtypes(include='object').columns.tolist()
        ignore_cols = ["name", "email", "id", "timestamp"]
        feedback_cols = [col for col in text_cols if col.lower() not in ignore_cols]

        # Generate summary
        st.markdown("## 📊 Feedback Summary")
        summary_rows = []
        for col in feedback_cols:
            responses = df[col].dropna().astype(str).tolist()
            sentiments = [classify_sentiment(r) for r in responses]
            pos, neg, neu = sentiments.count("Positive"), sentiments.count("Negative"), sentiments.count("Neutral")
            total = len(responses)
            keywords = extract_keywords(responses)
            summary_rows.append({
                "Question": col,
                "Total": total,
                "👍 Positive": pos,
                "👎 Negative": neg,
                "😐 Neutral": neu,
                "Top Keywords": ", ".join(keywords)
            })

        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, use_container_width=True)

        # Question selection
        st.markdown("## ✅ Select Questions to Explore")
        selected_questions = st.multiselect("Choose questions to analyze in depth:",
                                            options=summary_df["Question"].tolist(),
                                            default=summary_df["Question"].tolist()[:3])

        for q in selected_questions:
            st.markdown(f"### ❓ {q}")
            q_df = df[q].dropna().astype(str)
            sentiments = [classify_sentiment(r) for r in q_df]

            # Chart selection
            with st.expander("📈 Select Visuals & Insights to Show"):
                show_pie = st.checkbox(f"🟢 Sentiment Pie Chart - {q}", value=True)
                show_keywords = st.checkbox(f"🔤 Top Keywords Bar Chart - {q}", value=True)
                show_freq = st.checkbox(f"📉 Response Frequency Chart - {q}", value=False)
                show_time = st.checkbox(f"📆 Time Trend (if available) - {q}", value=False)
                show_summary = st.checkbox(f"🧠 Gemini Summary - {q}", value=True)

            # 🟢 Pie Chart
            if show_pie:
                pie_data = pd.DataFrame(Counter(sentiments).items(), columns=['Sentiment', 'Count'])
                fig = px.pie(pie_data, names='Sentiment', values='Count', title="Sentiment Pie Chart")
                st.plotly_chart(fig, use_container_width=True)

            # 🔤 Keyword Bar
            if show_keywords:
                keyword_list = extract_keywords(q_df)
                keyword_counts = Counter(" ".join(q_df).lower().split())
                keyword_freq = {k: keyword_counts[k] for k in keyword_list}
                kw_df = pd.DataFrame(keyword_freq.items(), columns=["Keyword", "Count"])
                fig = px.bar(kw_df, x="Keyword", y="Count", title="Top Keywords")
                st.plotly_chart(fig, use_container_width=True)

            # 📉 Frequency Bar
            if show_freq:
                freq_df = q_df.value_counts().reset_index()
                freq_df.columns = ["Response", "Count"]
                fig = px.bar(freq_df.head(10), x="Response", y="Count", title="Top Responses")
                st.plotly_chart(fig, use_container_width=True)

            # 📆 Time Trend
            if show_time:
                date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
                if date_cols:
                    date_col = date_cols[0]
                    df_filtered = df[[date_col, q]].dropna()
                    df_filtered["Sentiment"] = df_filtered[q].apply(classify_sentiment)
                    trend = df_filtered.groupby([pd.Grouper(key=date_col, freq='D'), "Sentiment"]).size().reset_index(name="Count")
                    fig = px.line(trend, x=date_col, y="Count", color="Sentiment", title="Sentiment Over Time")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No timestamp column found for time trend.")

            # 🧠 Gemini Summary
            if show_summary:
                context = f"""Based on the following responses for the question "{q}":\n\n"""
                context += "\n".join(q_df.sample(min(10, len(q_df))).tolist())
                context += "\n\nGive a short insightful summary of what people are saying."

                try:
                    gemini_response = gemini.generate_content(context)
                    st.markdown("#### 🤖 Gemini Insights")
                    st.markdown(gemini_response.text)
                except Exception as e:
                    st.error(f"Gemini error: {e}")

        # Overall Q&A
        st.markdown("## 💬 Ask Gemini About the Whole Feedback")
        query = st.text_area("Example: Which question had the most negative responses?")
        if st.button("Submit Overall Query"):
            try:
                context = f"This is a feedback summary:\n\n{summary_df.to_markdown(index=False)}\n\nQuery: {query}"
                answer = gemini.generate_content(context)
                st.markdown("### 📢 Gemini's Answer")
                st.markdown(answer.text)
            except Exception as e:
                st.error(f"Error from Gemini: {e}")

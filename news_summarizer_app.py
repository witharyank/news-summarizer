import time
import base64
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect, LangDetectException
from newspaper import Article

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="News Summarizer",
    page_icon="📰",
    layout="centered"
)

# -------------------- CUSTOM STYLING --------------------
st.markdown("""
<style>
textarea {
    border-radius: 12px !important;
}
button {
    border-radius: 10px !important;
    height: 3em !important;
}
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# -------------------- DEVICE --------------------
DEVICE = "cpu"
# -------------------- MODEL LOADING --------------------
@st.cache_resource
def load_model():
    model_name = "sshleifer/distilbart-cnn-12-6"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# -------------------- HEADER --------------------
st.markdown("""
<h1 style='text-align: center;'>📰 AI News Summarizer</h1>
<p style='text-align: center; color: gray;'>
Paste your news article or provide a URL to generate a clean AI-powered summary.
</p>
""", unsafe_allow_html=True)

st.divider()

# -------------------- SUMMARY CONTROLS --------------------
col1, col2 = st.columns(2)

with col1:
    summary_length = st.selectbox(
        "📏 Summary Length",
        ["Short", "Medium", "Long"]
    )

with col2:
    st.metric("⚙️ Running On", DEVICE.upper())

length_map = {
    "Short": 60,
    "Medium": 120,
    "Long": 200
}

st.divider()

# -------------------- URL INPUT --------------------
url = st.text_input("🌐 Paste a news article URL :")

article = ""

if url:
    try:
        with st.spinner("Fetching article from URL..."):
            news_article = Article(url)
            news_article.download()
            news_article.parse()
            article = news_article.text
            st.success("Article fetched successfully!")
            st.caption(f"**Title:** {news_article.title}")
    except Exception:
        st.error("Failed to fetch article. Please check the URL.")

# -------------------- MANUAL ARTICLE INPUT --------------------
manual_article = st.text_area(
    "📄 Or paste your news article here:",
    height=350,
    placeholder="Paste full news article text here..."
)

# If manual text exists, override URL content
if manual_article.strip():
    article = manual_article

# -------------------- TEXT ANALYTICS --------------------
if article.strip():
    word_count = len(article.split())
    reading_time = max(1, round(word_count / 200))

    col1, col2 = st.columns(2)
    col1.metric("📝 Word Count", word_count)
    col2.metric("🕒 Reading Time (mins)", reading_time)

st.divider()

# -------------------- SESSION STATE --------------------
if "summary" not in st.session_state:
    st.session_state.summary = ""

# -------------------- GENERATE BUTTON --------------------
generate = st.button("🚀 Generate Summary", use_container_width=True)

# -------------------- GENERATION LOGIC --------------------
if generate:
    if not article.strip():
        st.warning("Please provide a news article or URL first.")
    else:
        try:
            detected_language = detect(article)
            if detected_language != "en":
                st.error("This summarizer currently works best with English text.")
            else:
                with st.spinner("Generating summary... ⏳"):
                    inputs = tokenizer(
                        article,
                        max_length=1024,
                        truncation=True,
                        return_tensors="pt"
                    ).to(DEVICE)

                    with torch.no_grad():
                        summary_ids = model.generate(
                            inputs["input_ids"],
                            max_length=length_map[summary_length],
                            min_length=40,
                            num_beams=4,
                            length_penalty=2.0,
                            early_stopping=True
                        )

                    st.session_state.summary = tokenizer.decode(
                        summary_ids[0],
                        skip_special_tokens=True
                    )

                time.sleep(0.3)

        except LangDetectException:
            st.error("Unable to detect language. Please check your input.")

# -------------------- OUTPUT SECTION --------------------
if st.session_state.summary:

    st.divider()
    st.subheader("🧠 Generated Summary")

    st.write(st.session_state.summary)

    # Download option as txt format
    b64 = base64.b64encode(st.session_state.summary.encode()).decode()
    st.markdown(
        f'<a href="data:text/plain;base64,{b64}" download="summary.txt">📥 Download Summary</a>',
        unsafe_allow_html=True
    )

    if st.button("🗑 Clear Summary"):
        st.session_state.summary = ""

import time
import base64
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect, LangDetectException
import requests
from bs4 import BeautifulSoup


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI News Summarizer",
    page_icon="📰",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

textarea {
    border-radius:12px !important;
}

button {
    border-radius:10px !important;
    height:3em !important;
}

.summary-box{
    background:#f5f7fb;
    padding:20px;
    border-radius:12px;
    border:1px solid #e6e6e6;
    font-size:16px;
    line-height:1.6;
}

.block-container{
    padding-top:2rem;
}

</style>
""", unsafe_allow_html=True)


# ---------------- DEVICE ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- MODEL LOADING ----------------
@st.cache_resource
def load_model():
    model_name = "sshleifer/distilbart-cnn-12-6"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    model.to(DEVICE)
    model.eval()

    return tokenizer, model


tokenizer, model = load_model()


# ---------------- HEADER ----------------
st.markdown("""
<h1 style='text-align:center;'>📰 AI News Summarizer</h1>
<p style='text-align:center;color:gray;font-size:18px'>
Summarize long news articles instantly using AI
</p>
""", unsafe_allow_html=True)

st.divider()


# ---------------- CONTROLS ----------------
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


# ---------------- INPUT TABS ----------------
tab1, tab2 = st.tabs(["🌐 From URL", "📄 Paste Article"])

article = ""

# -------- URL INPUT --------
with tab1:

    url = st.text_input("Paste news article URL")

    if url:
        try:

            with st.spinner("Fetching article..."):

                response = requests.get(url, timeout=10)
                soup = BeautifulSoup(response.text, "html.parser")

                paragraphs = soup.find_all("p")
                article = " ".join([p.get_text() for p in paragraphs])

            st.success("Article fetched successfully!")

        except Exception:
            st.error("Failed to fetch article. Check URL.")


# -------- MANUAL INPUT --------
with tab2:

    manual_article = st.text_area(
        "Paste full news article",
        height=350,
        placeholder="Paste article text..."
    )

    if manual_article.strip():
        article = manual_article


# -------- EXAMPLE ARTICLE --------
if st.button("📰 Load Example Article"):

    article = """
Artificial Intelligence is rapidly transforming industries worldwide.
Companies are investing heavily in machine learning technologies to automate
processes and gain deeper insights from data.

Experts believe AI will revolutionize healthcare, finance, transportation,
and many other sectors over the next decade.

However, concerns remain about job displacement, ethical AI usage,
and the need for strong governance frameworks.
"""


# ---------------- ARTICLE ANALYTICS ----------------
if article.strip():

    word_count = len(article.split())
    reading_time = max(1, round(word_count / 200))

    col1, col2 = st.columns(2)

    col1.metric("📝 Word Count", word_count)
    col2.metric("🕒 Reading Time (mins)", reading_time)

st.divider()


# ---------------- SESSION STATE ----------------
if "summary" not in st.session_state:
    st.session_state.summary = ""


# ---------------- GENERATE BUTTON ----------------
generate = st.button("🚀 Generate Summary", use_container_width=True)


# ---------------- GENERATION ----------------
if generate:

    if not article.strip():
        st.warning("Please provide a news article first.")

    else:

        try:

            detected_language = detect(article)

            if detected_language != "en":
                st.error("This summarizer currently works best with English text.")

            else:

                progress = st.progress(0)

                with st.spinner("Generating summary..."):

                    start_time = time.time()

                    for i in range(100):
                        time.sleep(0.005)
                        progress.progress(i + 1)

                    inputs = tokenizer(
                        article,
                        max_length=1054,
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

                    end_time = time.time()

                    st.session_state.generation_time = round(end_time - start_time, 2)

        except LangDetectException:
            st.error("Language detection failed.")


# ---------------- OUTPUT ----------------
if st.session_state.summary:

    st.divider()
    st.subheader("🧠 Generated Summary")

    st.markdown(
        f"<div class='summary-box'>{st.session_state.summary}</div>",
        unsafe_allow_html=True
    )

    summary_word_count = len(st.session_state.summary.split())

    col1, col2, col3 = st.columns(3)

    if "generation_time" in st.session_state:
        col1.metric("⏱ Time", f"{st.session_state.generation_time}s")

    col2.metric("📄 Summary Words", summary_word_count)

    if article:
        compression = round((1 - summary_word_count / word_count) * 100)
        col3.metric("📉 Compression", f"{compression}%")

    # -------- COPY FRIENDLY OUTPUT --------
    st.code(st.session_state.summary, language="text")

    # -------- DOWNLOAD --------
    b64 = base64.b64encode(st.session_state.summary.encode()).decode()

    st.markdown(
        f'<a href="data:text/plain;base64,{b64}" download="summary.txt">📥 Download Summary</a>',
        unsafe_allow_html=True
    )

    if st.button("🗑 Clear Summary"):
        st.session_state.summary = ""


# ---------------- FOOTER ----------------_--
st.divider()

st.caption("Model: DistilBART CNN | HuggingFace Transformers")
st.caption("Contact For Suggestion: kraryan2028@gmail.com")
st.markdown("[🔗 View Source Code](https://github.com/witharyank/news-summarizer)")
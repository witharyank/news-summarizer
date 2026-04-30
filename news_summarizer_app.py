import streamlit as st
import torch
import time
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------- CONFIG --------------------
st.set_page_config(
    page_title="AI News Summarizer",
    page_icon="📰",
    layout="centered"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- MODEL LOADING --------------------
@st.cache_resource
def load_model():
    model_name = "sshleifer/distilbart-cnn-12-6"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ---------------- FETCH ARTICLE --------------------
@st.cache_data
def fetch_article(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException:
        return ""

    soup = BeautifulSoup(response.text, "html.parser")

    paragraphs = [
        p.get_text().strip()
        for p in soup.find_all("p")
        if len(p.get_text().strip()) > 50
    ]

    return " ".join(paragraphs)

# ---------------- TEXT PROCESSING --------------------
def split_text(text: str, max_words: int = 600):
    words = text.split()
    return [
        " ".join(words[i:i + max_words])
        for i in range(0, len(words), max_words)
    ]

# ---------------- SUMMARIZATION --------------------
def summarize_chunk(text: str, max_len: int):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(DEVICE)

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=max_len,
                min_length=30,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def summarize_article(article: str, max_len: int):
    chunks = split_text(article)
    summaries = []

    progress = st.progress(0)

    for i, chunk in enumerate(chunks):
        summaries.append(summarize_chunk(chunk, max_len))
        progress.progress((i + 1) / len(chunks))

    progress.empty()
    return " ".join(summaries)

# ---------------- METRICS --------------------
def calculate_metrics(article: str, summary: str):
    word_count = len(article.split())
    summary_words = len(summary.split())
    reading_time = max(1, round(word_count / 200))
    compression = round((1 - summary_words / word_count) * 100)

    return word_count, summary_words, reading_time, compression

# ---------------- UI --------------------
st.markdown("<h1 style='text-align:center;'>📰 AI News Summarizer</h1>", unsafe_allow_html=True)
st.caption("Summarize long news articles instantly using AI")

st.divider()

# ---------------- INPUT --------------------
input_method = st.radio("Choose Input Method", ["🌐 URL", "📄 Paste Text"])

article = ""

if input_method == "🌐 URL":
    url = st.text_input("Enter Article URL")

    if url:
        with st.spinner("Fetching article..."):
            article = fetch_article(url)

        if article:
            st.success("Article fetched successfully!")
        else:
            st.error("Failed to fetch clean article content.")

else:
    article = st.text_area("Paste article text", height=300)

# ---------------- SETTINGS --------------------
length_option = st.selectbox("Summary Length", ["Short", "Medium", "Long"])

length_map = {
    "Short": 60,
    "Medium": 120,
    "Long": 200
}

st.divider()

# ---------------- ANALYTICS --------------------
if article.strip():
    word_count = len(article.split())
    reading_time = max(1, round(word_count / 200))

    col1, col2 = st.columns(2)
    col1.metric("📝 Words", word_count)
    col2.metric("⏱ Reading Time", f"{reading_time} min")

# ---------------- SUMMARIZE --------------------
if st.button("🚀 Generate Summary", use_container_width=True, disabled=not article.strip()):

    with st.spinner("Generating summary..."):
        start = time.time()
        summary = summarize_article(article, length_map[length_option])
        end = time.time()

    st.session_state.summary = summary
    st.session_state.time = round(end - start, 2)

# ---------------- OUTPUT --------------------
if "summary" in st.session_state:

    st.divider()
    st.subheader("🧠 Summary")

    st.text_area("📄 Summary Output", st.session_state.summary, height=250)

    wc, sw, rt, comp = calculate_metrics(article, st.session_state.summary)

    col1, col2, col3 = st.columns(3)
    col1.metric("⏱ Time", f"{st.session_state.time}s")
    col2.metric("📄 Summary Words", sw)
    col3.metric("📉 Compression", f"{comp}%")

    st.download_button(
        "📥 Download Summary",
        st.session_state.summary,
        file_name="summary.txt"
    )

    if st.button("🗑 Clear"):
        st.session_state.clear()
        st.rerun()

# ---------------- FOOTER --------------------
st.divider()

st.markdown("""
<div style='text-align:center; padding:20px; border-radius:12px; 
background:linear-gradient(135deg, #1e1e2f, #2b2b45); 
color:white; box-shadow:0 4px 12px rgba(0,0,0,0.2); margin-top:20px;'>

    <h3>📰 AI News Summarizer</h3>

    <p style='color:#dcdcdc;'>
        Powered by <b>DistilBART</b> | HuggingFace Transformers
    </p>

    <p style='color:#dcdcdc;'>
        Developed by <b>Kumar Aryan</b>
    </p>

    <p>
        📧 <a href="mailto:kraryan2028@gmail.com" 
        style='color:#4da6ff; text-decoration:none;'>
        kraryan2028@gmail.com
        </a>
    </p>

    <hr style='border:0.5px solid #444;'>

    <p style='font-size:13px; color:gray;'>
        © 2026 All Rights Reserved
    </p>

</div>
""", unsafe_allow_html=True)
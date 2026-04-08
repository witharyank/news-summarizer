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

# ---------------- UTIL FUNCTIONS ----------------

@st.cache_data
def fetch_article(url: str) -> str:
    """Fetch and clean article text from URL"""
    response = requests.get(url, timeout=8)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")

    text = " ".join(p.get_text() for p in paragraphs)
    return text.strip()


def split_text(text: str, max_words: int = 800):
    """Split long text into chunks"""
    words = text.split()
    return [
        " ".join(words[i:i + max_words])
        for i in range(0, len(words), max_words)
    ]


def summarize_chunk(text: str, max_len: int):
    """Summarize a single chunk"""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(DEVICE)

    with torch.no_grad():
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
    """Handle long articles via chunking"""
    chunks = split_text(article)

    summaries = []
    for chunk in chunks:
        summaries.append(summarize_chunk(chunk, max_len))

    return " ".join(summaries)


def calculate_metrics(article: str, summary: str):
    word_count = len(article.split())
    summary_words = len(summary.split())
    reading_time = max(1, round(word_count / 200))
    compression = round((1 - summary_words / word_count) * 100)

    return word_count, summary_words, reading_time, compression


# ---------------- UI ----------------

st.markdown("<h1 style='text-align:center;'>📰 AI News Summarizer</h1>", unsafe_allow_html=True)
st.caption("Summarize long news articles instantly using AI")

st.divider()

# ---------------- INPUT ----------------
input_method = st.radio("Choose Input Method", ["🌐 URL", "📄 Paste Text"])

article = ""

if input_method == "🌐 URL":
    url = st.text_input("Enter Article URL")

    if url:
        try:
            with st.spinner("Fetching article..."):
                article = fetch_article(url)
            st.success("Article fetched successfully!")
        except requests.exceptions.RequestException:
            st.error("Failed to fetch article. Please check the URL.")

else:
    article = st.text_area("Paste article text", height=300)


# ---------------- SETTINGS ----------------
length_option = st.selectbox("Summary Length", ["Short", "Medium", "Long"])

length_map = {
    "Short": 60,
    "Medium": 120,
    "Long": 200
}

st.divider()

# ---------------- ANALYTICS ----------------
if article.strip():
    word_count = len(article.split())
    reading_time = max(1, round(word_count / 200))

    col1, col2 = st.columns(2)
    col1.metric("📝 Words", word_count)
    col2.metric("⏱ Reading Time", f"{reading_time} min")

# ---------------- SUMMARIZE ----------------
if st.button("🚀 Generate Summary", use_container_width=True):

    if not article.strip():
        st.warning("Please provide an article first.")
    else:
        with st.spinner("Generating summary..."):
            start = time.time()
            summary = summarize_article(article, length_map[length_option])
            end = time.time()

        st.session_state.summary = summary
        st.session_state.time = round(end - start, 2)

# ---------------- OUTPUT ----------------
if "summary" in st.session_state:

    st.divider()
    st.subheader("🧠 Summary")

    st.markdown(
        f"<div style='background:#f5f7fb;padding:20px;border-radius:10px'>{st.session_state.summary}</div>",
        unsafe_allow_html=True
    )

    wc, sw, rt, comp = calculate_metrics(article, st.session_state.summary)

    col1, col2, col3 = st.columns(3)

    col1.metric("⏱ Time", f"{st.session_state.time}s")
    col2.metric("📄 Summary Words", sw)
    col3.metric("📉 Compression", f"{comp}%")

    st.code(st.session_state.summary)

    # Download
    st.download_button(
        "📥 Download Summary",
        st.session_state.summary,
        file_name="summary.txt"
    )

    if st.button("🗑 Clear"):
        st.session_state.clear()

# ---------------- FOOTER ----------------
st.divider()
st.caption("Model: DistilBART (HuggingFace)")
st.caption("Built with ❤️ using Streamlit")
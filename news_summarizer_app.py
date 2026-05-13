import streamlit as st
import torch
import time
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI News Summarizer",
    page_icon="📰",
    layout="wide"
)
# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.main {
    background-color: #0f172a;
    color: white;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1100px;
}

h1, h2, h3 {
    color: white;
}

.stTextInput input,
.stTextArea textarea {
    border-radius: 12px !important;
    border: 1px solid #334155 !important;
    background-color: #111827 !important;
    color: white !important;
}

.stSelectbox div[data-baseweb="select"] {
    border-radius: 12px !important;
}

.stButton button {
    width: 100%;
    border-radius: 12px;
    height: 3rem;
    font-size: 16px;
    font-weight: 600;
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    color: white;
    border: none;
}

.stDownloadButton button {
    width: 100%;
    border-radius: 12px;
    height: 3rem;
    font-size: 15px;
    font-weight: 600;
}

.metric-card {
    background: #111827;
    padding: 18px;
    border-radius: 16px;
    border: 1px solid #1e293b;
    text-align: center;
}

.summary-box {
    background: #111827;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #1e293b;
    margin-top: 10px;
}

.footer {
    text-align: center;
    padding: 30px;
    margin-top: 40px;
    border-top: 1px solid #1e293b;
    color: #94a3b8;
}

</style>
""", unsafe_allow_html=True)

# ---------------- DEVICE ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    model_name = "sshleifer/distilbart-cnn-12-6"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)

    model.eval()

    return tokenizer, model


tokenizer, model = load_model()

# ---------------- FETCH ARTICLE ----------------
@st.cache_data
def fetch_article(url: str) -> str:

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=10
        )

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

# ---------------- TEXT SPLIT ----------------
def split_text(text: str, max_words: int = 600):

    words = text.split()

    return [
        " ".join(words[i:i + max_words])
        for i in range(0, len(words), max_words)
    ]

# ---------------- SUMMARIZATION ----------------
def summarize_chunk(text: str, max_len: int):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(DEVICE)

    with torch.no_grad():

        with torch.cuda.amp.autocast(
            enabled=(DEVICE == "cuda")
        ):

            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=max_len,
                min_length=30,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )

    return tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

def summarize_article(article: str, max_len: int):

    chunks = split_text(article)

    summaries = []

    progress = st.progress(0)

    for i, chunk in enumerate(chunks):

        summaries.append(
            summarize_chunk(chunk, max_len)
        )

        progress.progress(
            (i + 1) / len(chunks)
        )

    progress.empty()

    return " ".join(summaries)

# ---------------- METRICS ----------------
def calculate_metrics(article: str, summary: str):

    word_count = len(article.split())

    summary_words = len(summary.split())

    reading_time = max(
        1,
        round(word_count / 200)
    )

    compression = round(
        (1 - summary_words / word_count) * 100
    )

    return (
        word_count,
        summary_words,
        reading_time,
        compression
    )

# ---------------- HERO SECTION ----------------
st.markdown("""
<h1 style='text-align:center; font-size:48px;'>
📰 AI News Summarizer
</h1>

<p style='text-align:center; color:#94a3b8; font-size:18px;'>
Summarize lengthy articles instantly using AI-powered NLP
</p>
""", unsafe_allow_html=True)

st.write("")

# ---------------- MAIN LAYOUT ----------------
left, right = st.columns([2, 1])

article = ""

# ---------------- LEFT PANEL ----------------
with left:

    st.subheader("📥 Input")

    input_method = st.radio(
        "Choose Input Method",
        ["🌐 URL", "📄 Paste Text"],
        horizontal=True
    )

    if input_method == "🌐 URL":

        url = st.text_input(
            "Article URL"
        )

        if url:

            with st.spinner(
                "Fetching article..."
            ):

                article = fetch_article(url)

            if article:
                st.success(
                    "Article fetched successfully!"
                )
            else:
                st.error(
                    "Unable to fetch article."
                )

    else:

        article = st.text_area(
            "Paste Article",
            height=350,
            placeholder="Paste your article text here..."
        )

# ---------------- RIGHT PANEL ----------------
with right:

    st.subheader("⚙ Settings")

    length_option = st.selectbox(
        "Summary Length",
        ["Short", "Medium", "Long"]
    )

    length_map = {
        "Short": 60,
        "Medium": 120,
        "Long": 200
    }

    if article.strip():

        wc = len(article.split())

        rt = max(
            1,
            round(wc / 200)
        )

        st.markdown(f"""
        <div class="metric-card">
            <h2>{wc}</h2>
            <p>Total Words</p>
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        st.markdown(f"""
        <div class="metric-card">
            <h2>{rt} min</h2>
            <p>Reading Time</p>
        </div>
        """, unsafe_allow_html=True)

# ---------------- GENERATE BUTTON ----------------
st.write("")

if st.button(
    "🚀 Generate Summary",
    disabled=not article.strip()
):

    with st.spinner(
        "Generating AI Summary..."
    ):

        start = time.time()

        summary = summarize_article(
            article,
            length_map[length_option]
        )

        end = time.time()

    st.session_state.summary = summary
    st.session_state.time = round(
        end - start,
        2
    )

# ---------------- OUTPUT ----------------
if "summary" in st.session_state:

    st.write("")
    st.subheader("🧠 Generated Summary")

    st.markdown(f"""
    <div class="summary-box">
        {st.session_state.summary}
    </div>
    """, unsafe_allow_html=True)

    wc, sw, rt, comp = calculate_metrics(
        article,
        st.session_state.summary
    )

    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "⏱ Time",
            f"{st.session_state.time}s"
        )

    with col2:
        st.metric(
            "📄 Summary Words",
            sw
        )

    with col3:
        st.metric(
            "📉 Compression",
            f"{comp}%"
        )

    st.write("")

    d1, d2 = st.columns(2)

    with d1:

        st.download_button(
            "📥 Download Summary",
            st.session_state.summary,
            file_name="summary.txt",
            use_container_width=True
        )

    with d2:

        if st.button(
            "🗑 Clear Session"
        ):

            st.session_state.clear()
            st.rerun()

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">

    <h3>📰 AI News Summarizer</h3>

    <p>
        Powered by DistilBART + HuggingFace Transformers
    </p>

    <p>
        Developed by Kumar Aryan
    </p>

    <p>
        📧 kraryan2028@gmail.com
    </p>

    <p style="font-size:13px;">
        © 2026 All Rights Reserved
    </p>

</div>
""", unsafe_allow_html=True)
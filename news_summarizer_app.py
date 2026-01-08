import time
import base64
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect, LangDetectException


# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="News Summarizer",
    layout="centered"
)

# -------------------- DEVICE DETECTION --------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

# -------------------- UI --------------------
st.title("📰 News Article Summarizer")
st.write("Paste a news article below and get an instant summary.")
st.caption(f"⚙️ Running on **{DEVICE.upper()}**")

# -------------------- SUMMARY LENGTH --------------------
summary_length = st.radio(
    "Select summary length:",
    ["Short", "Medium", "Long"],
    horizontal=True
)

length_map = {
    "Short": 60,
    "Medium": 120,
    "Long": 200
}

st.divider()

# -------------------- TEXT INPUT --------------------
article = st.text_area(
    "Paste your news article here:",
    height=400,
    placeholder="Paste full news article text here..."
)

# -------------------- LANGUAGE DETECTION --------------------
detected_language = None
is_english = True

if article.strip():
    try:
        detected_language = detect(article)
        if detected_language != "en":
            is_english = False
            st.warning(
                f"⚠️ Detected language: **{detected_language.upper()}**. "
                "This summarizer works best with **English** text."
            )
        else:
            st.success("✅ Detected language: **English**")
    except LangDetectException:
        st.warning("⚠️ Unable to detect language. Please check the input text.")

# -------------------- READING TIME --------------------
if article.strip():
    word_count = len(article.split())
    reading_time = round(word_count / 200, 2)
    st.info(f"🕒 Estimated reading time: **{reading_time} minutes**")

st.divider()

# -------------------- SESSION STATE --------------------
if "summary" not in st.session_state:
    st.session_state.summary = ""

# -------------------- GENERATE SUMMARY --------------------
if st.button("🚀 Generate Summary"):
    if not article.strip():
        st.warning("Please paste some text.")
    elif not is_english:
        st.error(
            "❌ Summarization aborted. Please provide an **English** article "
            "for best results."
        )
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

        time.sleep(0.5)

# -------------------- OUTPUT --------------------
if st.session_state.summary:
    st.subheader("🧠 Generated Summary")
    st.success(st.session_state.summary)

    st.code(st.session_state.summary, language="text")

    # Download summary
    b64 = base64.b64encode(st.session_state.summary.encode()).decode()
    st.markdown(
        f'<a href="data:text/plain;base64,{b64}" download="summary.txt">📥 Download Summary</a>',
        unsafe_allow_html=True
    )

    if st.button("🗑 Clear Summary"):
        st.session_state.summary = ""

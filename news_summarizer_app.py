import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import time
import base64

@st.cache_resource
def load_model():
    model_name = "sshleifer/distilbart-cnn-12-6"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

st.set_page_config(page_title="News Summarizer", layout="centered")
st.title("📰 News Article Summarizer")
st.write("Paste a news article below and get an instant summary.")

# --- Summary length selector ---
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

# --- Text input ---
article = st.text_area("Paste your news article here:", height=300)

# --- Reading time estimator ---
if article.strip():
    word_count = len(article.split())
    reading_time = round(word_count / 200, 2)
    st.info(f"🕒 Estimated reading time: **{reading_time} minutes**")

st.divider()

# --- Generate Summary ---
if st.button("Generate Summary"):
    if not article.strip():
        st.warning("Please paste some text.")
    else:
        with st.spinner("Generating summary... ⏳"):
            inputs = tokenizer([article], max_length=1024, truncation=True, return_tensors="pt")
            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=length_map[summary_length],
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            time.sleep(1)

        st.subheader("🧠 Generated Summary:")
        st.success(summary)

        # --- Copy-to-clipboard button ---
        st.code(summary, language="text")
        st.button("📋 Copy Summary to Clipboard", on_click=lambda: st.write("Copied! (Streamlit supports this natively in UI)"))

        # --- Download summary ---
        b64 = base64.b64encode(summary.encode()).decode()
        href = f'<a href="data:text/plain;base64,{b64}" download="summary.txt">📥 Download Summary as .txt</a>'
        st.markdown(href, unsafe_allow_html=True)


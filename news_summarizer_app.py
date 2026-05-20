import streamlit as st
import torch
import time
import requests
import urllib.parse
import re
from bs4 import BeautifulSoup
from collections import Counter
from langdetect import detect
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI News Summarizer Pro",
    page_icon="📰",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
/* Global background and text styling */
.stApp {
    background-color: #090d16 !important;
    background-image: radial-gradient(at 0% 0%, rgba(37, 99, 235, 0.1) 0px, transparent 50%),
                      radial-gradient(at 50% 0%, rgba(124, 58, 237, 0.08) 0px, transparent 50%) !important;
    color: #e2e8f0 !important;
}

/* Glassmorphism sidebar */
[data-testid="stSidebar"] {
    background-color: #0b0f19 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
}

/* Custom cards */
.metric-card {
    background: rgba(22, 28, 45, 0.4);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 22px;
    border-radius: 16px;
    text-align: center;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.4);
    transition: transform 0.3s ease, border-color 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-4px);
    border-color: rgba(59, 130, 246, 0.3);
}

.metric-card h2 {
    font-size: 2.2rem;
    margin: 0;
    background: linear-gradient(90deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
}

.metric-card p {
    font-size: 0.85rem;
    color: #94a3b8;
    margin: 8px 0 0 0;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
}

.summary-box {
    background: rgba(22, 28, 45, 0.3);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.06);
    padding: 24px;
    border-radius: 16px;
    line-height: 1.8;
    font-size: 1.1rem;
    color: #e2e8f0;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2);
}

/* Keywords badges */
.keyword-badge {
    display: inline-block;
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.12), rgba(139, 92, 246, 0.12));
    border: 1px solid rgba(139, 92, 246, 0.2);
    color: #c084fc;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    margin: 4px;
    transition: all 0.2s ease;
}

.keyword-badge:hover {
    transform: scale(1.05);
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.22), rgba(139, 92, 246, 0.22));
    border-color: rgba(139, 92, 246, 0.4);
    box-shadow: 0 4px 12px rgba(139, 92, 246, 0.15);
}

/* Footer styling */
.footer {
    text-align: center;
    padding: 40px 20px;
    margin-top: 60px;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
    background: linear-gradient(180deg, transparent, rgba(11, 15, 25, 0.9));
}

.footer h3 {
    background: linear-gradient(90deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    font-size: 1.5rem;
    margin-bottom: 15px;
}

.footer p {
    color: #64748b;
    margin: 6px 0;
    font-size: 0.95rem;
}

/* Gradient Buttons */
div.stButton > button {
    background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    height: 3rem !important;
    font-size: 16px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.2) !important;
}

div.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4) !important;
}

div.stButton > button:active {
    transform: translateY(0) !important;
}

/* Streamlit Native component styling overrides */
.stTextInput input, .stTextArea textarea, .stSelectbox select {
    background-color: rgba(22, 28, 45, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    color: white !important;
}

.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 1px #3b82f6 !important;
}

/* Tab styling overrides */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    background-color: rgba(22, 28, 45, 0.4) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 10px 10px 0 0 !important;
    padding: 12px 24px !important;
    color: #94a3b8 !important;
    transition: all 0.2s ease !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #f8fafc !important;
    background-color: rgba(22, 28, 45, 0.7) !important;
}

.stTabs [aria-selected="true"] {
    background-color: rgba(59, 130, 246, 0.15) !important;
    border-bottom: 2px solid #3b82f6 !important;
    color: #3b82f6 !important;
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
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ---------------- OPTIONAL SENTIMENT MODEL ----------------
@st.cache_resource
def load_sentiment_pipeline():
    from transformers import pipeline
    # Load lightweight, fast sentiment model
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=0 if DEVICE == "cuda" else -1)

# ---------------- SCRAPING & UTILITIES ----------------
@st.cache_data
def fetch_article(url: str) -> dict:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to connect or fetch data: {str(e)}"}

    soup = BeautifulSoup(response.text, "html.parser")
    
    # Extract Title
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    elif soup.find("h1"):
        title = soup.find("h1").get_text().strip()
    
    # Extract Domain
    domain = urllib.parse.urlparse(url).netloc
    if domain.startswith("www."):
        domain = domain[4:]

    # Remove script, style, nav, footer elements
    for element in soup(["script", "style", "nav", "footer", "header", "aside", "form", "iframe"]):
        element.decompose()

    paragraphs = []
    for p in soup.find_all("p"):
        p_text = p.get_text().strip()
        # Clean out boilerplates
        if len(p_text) > 40 and not any(term in p_text.lower() for term in ["cookie", "terms of use", "privacy policy", "copyright", "all rights reserved"]):
            paragraphs.append(p_text)

    article_text = " ".join(paragraphs)
    
    # Fallback to general divs if p yields empty text
    if not article_text.strip():
        visible_texts = []
        for tag in soup.find_all(['div', 'span', 'section', 'article']):
            text = tag.get_text(strip=True)
            if len(text) > 60 and tag.name != 'a':
                visible_texts.append(text)
        article_text = " ".join(visible_texts[:15])

    if not article_text.strip():
        return {"error": "No meaningful article content could be extracted from this page. Please try copy-pasting the text instead."}

    return {
        "title": title or "Untitled Article",
        "domain": domain,
        "text": article_text
    }

def split_text(text: str, max_tokens: int = 800) -> list:
    # Tokenizer-based splitting to prevent boundary cutoffs
    inputs = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(inputs), max_tokens):
        chunk_ids = inputs[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
    return chunks

def summarize_chunk(text: str, max_len: int):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(DEVICE)

    input_len = inputs["input_ids"].shape[1]
    # Set dynamic length constraints to prevent pipeline crashes on short parts
    adjusted_max_len = min(max_len, int(input_len * 0.7))
    adjusted_min_len = min(30, int(input_len * 0.2))
    
    if adjusted_max_len <= adjusted_min_len:
        adjusted_max_len = adjusted_min_len + 15

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=adjusted_max_len,
                min_length=adjusted_min_len,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def summarize_article(article: str, max_len: int):
    chunks = split_text(article)
    summaries = []
    
    progress = st.progress(0)
    progress_status = st.empty()
    
    for i, chunk in enumerate(chunks):
        progress_status.markdown(f"<p style='color: #94a3b8; font-size:14px;'>Processing part {i+1} of {len(chunks)}...</p>", unsafe_allow_html=True)
        summaries.append(summarize_chunk(chunk, max_len))
        progress.progress((i + 1) / len(chunks))
        
    progress.empty()
    progress_status.empty()
    
    return " ".join(summaries)

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "en"

def extract_keywords(text: str, top_n: int = 8) -> list:
    stopwords = {
        "the", "and", "a", "of", "to", "is", "in", "it", "that", "i", "you", "he", "she", "they", "we", "us", "him", "her", "them", 
        "on", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", 
        "below", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", 
        "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", 
        "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", 
        "now", "but", "this", "also", "has", "have", "had", "was", "were", "are", "been", "be", "an", "as", "their", "our", "its",
        "would", "could", "about", "one", "two", "three", "first", "new", "said", "also", "says", "mr", "mrs", "ms", "dr"
    }
    words = re.findall(r'\b[a-zA-Z]{3,15}\b', text.lower())
    filtered_words = [word for word in words if word not in stopwords]
    counter = Counter(filtered_words)
    return [word for word, count in counter.most_common(top_n)]

def analyze_sentiment(text: str) -> tuple:
    try:
        classifier = load_sentiment_pipeline()
        sample_text = " ".join(text.split()[:300]) # First 300 words
        result = classifier(sample_text)[0]
        label = result['label']
        score = result['score']
        if score < 0.65:
            return "NEUTRAL", score
        return label, score
    except Exception:
        return "UNKNOWN", 0.0

def format_as_bullets(summary_text: str) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', summary_text.strip())
    bullets = []
    for s in sentences:
        s = s.strip()
        if s:
            s = re.sub(r'^[-*•\s]+', '', s)
            bullets.append(f"• {s}")
    return "\n".join(bullets)

def calculate_metrics(article: str, summary: str):
    word_count = len(article.split())
    summary_words = len(summary.split())
    reading_time = max(1, round(word_count / 200))
    compression = round((1 - summary_words / word_count) * 100) if word_count > 0 else 0
    return word_count, summary_words, reading_time, compression

def add_to_history(title: str, url: str, summary: str, text: str, domain: str, source: str, time_taken: float):
    # Prevent duplicate history items based on title
    if any(item['title'] == title for item in st.session_state.history):
        return
    st.session_state.history.insert(0, {
        "title": title,
        "url": url,
        "summary": summary,
        "text": text,
        "domain": domain,
        "source": source,
        "time": time_taken
    })
    st.session_state.history = st.session_state.history[:5]

# ---------------- STATE INITIALIZATION ----------------
if "history" not in st.session_state:
    st.session_state.history = []

if "article_data" not in st.session_state:
    st.session_state.article_data = None

if "generated_summary" not in st.session_state:
    st.session_state.generated_summary = None

if "summary_time" not in st.session_state:
    st.session_state.summary_time = None

# ---------------- SIDEBAR / CONTROL PANEL ----------------
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0;'>
        <h2 style='font-size: 26px; font-weight: 700; margin-bottom: 0;'>⚙️ Settings Panel</h2>
        <p style='color: #64748b; font-size: 13px;'>Configure NLP Pipeline Parameters</p>
    </div>
    """, unsafe_allow_html=True)
    
    input_method = st.radio(
        "Choose Input Method",
        ["🌐 URL", "📄 Paste Text"],
        horizontal=True
    )
    
    st.write("---")
    
    st.markdown("#### Summary Options")
    length_option = st.selectbox(
        "Summary Length",
        ["Short", "Medium", "Long"],
        index=1
    )
    length_map = {
        "Short": 60,
        "Medium": 120,
        "Long": 200
    }
    
    format_option = st.selectbox(
        "Summary Format",
        ["Paragraph", "Bullet Points"],
        index=0
    )
    
    st.write("---")
    
    st.markdown("#### Advanced Models")
    enable_sentiment = st.toggle(
        "Enable Sentiment Analysis",
        value=False,
        help="Loads DistilBERT to classify the tone of the content."
    )
    
    st.write("---")
    
    # History section
    st.markdown("#### 📜 Recent Summaries")
    if st.session_state.history:
        for idx, item in enumerate(st.session_state.history):
            source_icon = "🌐" if item["source"] == "URL" else "📄"
            trunc_title = item["title"][:28] + "..." if len(item["title"]) > 28 else item["title"]
            if st.button(f"{source_icon} {trunc_title}", key=f"hist_{idx}", use_container_width=True):
                st.session_state.article_data = {
                    "title": item["title"],
                    "domain": item["domain"],
                    "text": item["text"],
                    "url": item["url"],
                    "source": item["source"]
                }
                st.session_state.generated_summary = item["summary"]
                st.session_state.summary_time = item["time"]
                st.rerun()
        
        st.write("")
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption("No summaries in session history.")

# ---------------- HERO BANNER ----------------
st.markdown("""
<div style='text-align:center; margin-top: 1rem; margin-bottom: 2.5rem;'>
    <h1 style='font-size: 50px; background: linear-gradient(90deg, #60a5fa, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; letter-spacing: -1.5px;'>
        📰 AI News Summarizer Pro
    </h1>
    <p style='color: #94a3b8; font-size: 18px;'>
        Instantly analyze, extract, and summarize any article or text with state-of-the-art NLP models.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- MAIN CONTENT AREA ----------------
if input_method == "🌐 URL":
    url = st.text_input(
        "Enter News Article URL",
        placeholder="https://www.nytimes.com/...",
        key="input_url"
    )
    if url:
        if "last_fetched_url" not in st.session_state or st.session_state.last_fetched_url != url:
            with st.spinner("🕷️ Fetching and scraping page content..."):
                res = fetch_article(url)
                if "error" in res:
                    st.error(res["error"])
                    st.session_state.article_data = None
                else:
                    st.session_state.article_data = {
                        "title": res["title"],
                        "domain": res["domain"],
                        "text": res["text"],
                        "url": url,
                        "source": "URL"
                    }
                    st.session_state.last_fetched_url = url
                    st.session_state.generated_summary = None
                    st.session_state.summary_time = None
                    st.rerun()
else:
    pasted_text = st.text_area(
        "Paste Article Content",
        height=280,
        placeholder="Paste your text here (minimum 10 words)...",
        key="input_text"
    )
    if pasted_text:
        words = pasted_text.split()
        if len(words) >= 10:
            if (st.session_state.article_data is None or 
                st.session_state.article_data.get("source") != "Pasted Text" or 
                st.session_state.article_data.get("text") != pasted_text):
                
                st.session_state.article_data = {
                    "title": "Pasted Content",
                    "domain": "Direct Text",
                    "text": pasted_text,
                    "url": "",
                    "source": "Pasted Text"
                }
                st.session_state.generated_summary = None
                st.session_state.summary_time = None
                st.rerun()
        else:
            st.caption("Please paste at least 10 words.")

# ---------------- ACTION AND SUMMARY DISPLAY ----------------
if st.session_state.article_data:
    data = st.session_state.article_data
    
    # Metadata Badge Display
    st.markdown(f"""
    <div style='background: rgba(30, 41, 59, 0.25); border: 1px solid rgba(255, 255, 255, 0.05); padding: 18px; border-radius: 12px; margin-bottom: 20px;'>
        <span style='background: linear-gradient(90deg, #3b82f6, #8b5cf6); color: white; padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 700; text-transform: uppercase;'>
            {data["domain"]}
        </span>
        <h3 style='margin: 8px 0 4px 0; font-size: 21px; color: #f8fafc; font-weight: 700;'>{data["title"]}</h3>
        <p style='color: #64748b; font-size: 13px; margin: 0;'>Word Count: {len(data["text"].split())} | Characters: {len(data["text"])}</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_btn, col_clear = st.columns([4, 1])
    with col_btn:
        generate_btn = st.button("🚀 Generate AI Summary", use_container_width=True)
    with col_clear:
        if st.button("🗑️ Reset", use_container_width=True):
            st.session_state.article_data = None
            st.session_state.generated_summary = None
            st.session_state.summary_time = None
            if "last_fetched_url" in st.session_state:
                del st.session_state.last_fetched_url
            st.rerun()

    if generate_btn:
        with st.spinner("🧠 Preparing DistilBART model & writing summary..."):
            start_time = time.time()
            raw_summary = summarize_article(data["text"], length_map[length_option])
            end_time = time.time()
            
            # Formatter logic
            if format_option == "Bullet Points":
                final_summary = format_as_bullets(raw_summary)
            else:
                final_summary = raw_summary
                
            st.session_state.generated_summary = final_summary
            st.session_state.summary_time = round(end_time - start_time, 2)
            
            add_to_history(
                title=data["title"],
                url=data["url"],
                summary=final_summary,
                text=data["text"],
                domain=data["domain"],
                source=data["source"],
                time_taken=st.session_state.summary_time
            )
            st.rerun()

    # Results Section
    if st.session_state.generated_summary:
        st.write("---")
        
        tab1, tab2, tab3 = st.tabs(["📄 Summary Output", "📊 Key Insights & Analytics", "📝 Original Text"])
        
        with tab1:
            st.write("")
            st.markdown("### 🧠 Generated Summary")
            
            # Format the output representation nicely
            summary_disp = st.session_state.generated_summary
            if format_option == "Bullet Points":
                lines = [f"<li style='margin-bottom: 8px;'>{line[2:]}</li>" for line in summary_disp.split("\n") if line.startswith("•")]
                summary_disp_html = f"<ul style='margin-left: 15px; padding-left: 10px; line-height: 1.8;'>{''.join(lines)}</ul>"
            else:
                summary_disp_html = f"<p style='line-height: 1.85; text-align: justify;'>{summary_disp}</p>"

            st.markdown(f"""
            <div class="summary-box">
                {summary_disp_html}
            </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            
            col_d, col_c = st.columns(2)
            with col_d:
                st.download_button(
                    label="📥 Download Summary (.txt)",
                    data=st.session_state.generated_summary,
                    file_name="ai_summary.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            with col_c:
                with st.expander("📋 Copy to Clipboard"):
                    st.code(st.session_state.generated_summary, language="text")

        with tab2:
            st.write("")
            st.markdown("### 📊 Metrics & AI Analytics")
            
            wc, sw, rt, comp = calculate_metrics(data["text"], st.session_state.generated_summary)
            srt = max(1, round(sw / 200))
            time_saved = max(0, rt - srt)
            
            # Visual Compression Bar
            st.markdown(f"""
            <div style='margin-bottom: 25px;'>
                <div style='display: flex; justify-content: space-between; margin-bottom: 6px;'>
                    <span style='font-size: 14px; font-weight: 700; color: #94a3b8;'>Text Size Compression</span>
                    <span style='font-size: 14px; font-weight: 700; color: #3b82f6;'>{comp}% Reduced</span>
                </div>
                <div style='background: #121826; height: 14px; border-radius: 7px; overflow: hidden; border: 1px solid rgba(255, 255, 255, 0.05);'>
                    <div style='background: linear-gradient(90deg, #3b82f6, #8b5cf6); width: {comp}%; height: 100%; border-radius: 7px;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Stats row
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.markdown(f"""
                <div class="metric-card">
                    <h2>{time_saved} min</h2>
                    <p>Time Saved</p>
                </div>
                """, unsafe_allow_html=True)
            with col_s2:
                st.markdown(f"""
                <div class="metric-card">
                    <h2>{wc - sw}</h2>
                    <p>Words Removed</p>
                </div>
                """, unsafe_allow_html=True)
            with col_s3:
                st.markdown(f"""
                <div class="metric-card">
                    <h2>{st.session_state.summary_time}s</h2>
                    <p>Processing Speed</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.write("---")
            
            # Language checking
            lang_code = detect_language(data["text"])
            lang_names = {
                "en": "English", "es": "Spanish", "fr": "French", "de": "German", "it": "Italian",
                "pt": "Portuguese", "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "ru": "Russian"
            }
            lang_name = lang_names.get(lang_code, lang_code.upper())
            if lang_code != "en":
                st.warning(f"⚠️ **Language Detected: {lang_name}**. The summarizer works best with **English** source material.")
            else:
                st.success(f"🌐 **Language Detected: English** (Optimal for BART Transformer)")

            st.write("")
            
            # Key Topics/Keywords Section
            keywords = extract_keywords(data["text"])
            badges_html = "".join([f"<span class='keyword-badge'>#{kw}</span>" for kw in keywords])
            st.markdown("#### 🏷️ Topic Keywords")
            st.markdown(f"<div style='margin-top: 10px; margin-bottom: 25px;'>{badges_html}</div>", unsafe_allow_html=True)
            
            st.write("---")
            
            # Sentiment details
            if enable_sentiment:
                sentiment_label, sentiment_score = analyze_sentiment(data["text"])
                color_map = {
                    "POSITIVE": "#10b981",
                    "NEGATIVE": "#ef4444",
                    "NEUTRAL": "#f59e0b",
                    "UNKNOWN": "#94a3b8"
                }
                emoji_map = {
                    "POSITIVE": "😊 Positive Tone",
                    "NEGATIVE": "😢 Negative Tone",
                    "NEUTRAL": "😐 Neutral Tone",
                    "UNKNOWN": "❓ Undetected Tone"
                }
                txt_color = color_map.get(sentiment_label, "#cbd5e1")
                emoji_label = emoji_map.get(sentiment_label, "Undetected")
                conf_score = int(sentiment_score * 100)
                
                st.markdown("#### 🎭 Sentiment Classification")
                st.markdown(f"""
                <div style='background: rgba(22, 28, 45, 0.4); border: 1px solid rgba(255, 255, 255, 0.06); padding: 20px; border-radius: 14px; display: flex; align-items: center; justify-content: space-between; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);'>
                    <div>
                        <span style='font-size: 13px; color: #64748b; text-transform: uppercase; font-weight: 700;'>Classifier Tone</span>
                        <p style='margin: 4px 0 0 0; font-size: 22px; font-weight: 800; color: {txt_color};'>{emoji_label}</p>
                    </div>
                    <div style='text-align: right;'>
                        <span style='font-size: 13px; color: #64748b; text-transform: uppercase; font-weight: 700;'>Confidence Level</span>
                        <p style='margin: 4px 0 0 0; font-size: 22px; font-weight: 800; color: #f8fafc;'>{conf_score}%</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("💡 Enable Sentiment Analysis in the Settings Panel to run classification.")

        with tab3:
            st.write("")
            st.markdown("### 📝 Full Article Text")
            st.markdown(f"""
            <div style='background: rgba(15, 23, 42, 0.5); border: 1px solid rgba(255, 255, 255, 0.05); padding: 24px; border-radius: 12px; height: 400px; overflow-y: scroll; font-size: 1rem; line-height: 1.75; color: #cbd5e1; text-align: justify;'>
                {data["text"]}
            </div>
            """, unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    <h3>📰 AI News Summarizer Pro</h3>
    <p>Powered by DistilBART & DistilBERT + HuggingFace Transformers</p>
    <p>Developed by Kumar Aryan</p>
    <p>📧 kraryan2028@gmail.com</p>
    <p style="font-size:12px; color: #475569; margin-top: 15px;">© 2026 All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)
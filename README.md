📰 News Summarizer App

A lightweight and efficient AI-powered web application built with Streamlit that automatically summarizes long news articles.
Powered by the DistilBART CNN model from Hugging Face, the app lets you paste any article, choose summary length, and instantly generate a concise summary — with automatic input language validation.



# 🚀 Features
🔹 AI Text Summarization

Uses the pretrained model sshleifer/distilbart-cnn-12-6, optimized for abstractive news summarization.

🔹 Automatic Language Detection (NEW ✅)

Detects the input language using langdetect

Warns users if the article is not in English

Prevents low-quality summaries from unsupported languages

⚠️ Best results are achieved with English articles.

🔹 Customizable Summary Length

Choose the summary style:

Short

Medium

Long

🔹 Reading Time Estimation

Estimates how long the original article would take to read based on word count.

🔹 Copy or Download Summary

View summary in a clean text block

Download summary as a .txt file

Easy copy via Streamlit UI

🔹 Optimized Performance

Automatic CPU/GPU detection

Model loaded once using Streamlit caching

Inference optimized with torch.no_grad()

# 🛠️ Tech Stack
Component	Technology
Framework	Streamlit
NLP Model	DistilBART (Hugging Face)
Language	Python
Backend	PyTorch
Utility	langdetect

# 📁 Project Structure
```
NewsSummarizerApp/
├── news_summarizer_app.py
├── requirements.txt
└── .gitignore
```

# 📦 Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/kraryan1946/news-summarizer.git
cd news-summarizer

2️⃣ Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # macOS/Linux

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the app
streamlit run news_summarizer_app.py

# 🧠 Model Information

DistilBART CNN-12-6
Distilled version of BART
Faster inference with minimal quality loss
Designed for news summarization tasks

# 🔗 Model page:
https://huggingface.co/sshleifer/distilbart-cnn-12-6

# 🖼️ How It Works
🔹Paste a news article or URL
🔹App detects input language
🔹Select summary length
🔹Click Generate Summary
🔹Copy or download the result

# ✨ Future Improvements
🔹URL and PDF-based article input
🔹Bullet-point summaries
🔹Dark mode support
🔹Auto-translation for non-English input
🔹Deployment to Streamlit Cloud / Hugging Face Spaces

# 👤 Author

Kumar Aryan
GitHub: https://github.com/witharyank

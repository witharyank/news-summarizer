# AI News Summarizer

A modern AI-powered web application that automatically summarizes long news articles using Hugging Face Transformers and Streamlit.

Live Demo:
https://news-summarizer-7g8fnozeszmvp49iscdm7w.streamlit.app/

GitHub Repository:
https://github.com/witharyank/news-summarizer

---

## Overview

AI News Summarizer is a lightweight and efficient NLP application designed to convert lengthy news articles into concise and meaningful summaries using the pretrained DistilBART transformer model.

The application supports:
- URL-based article extraction
- Direct text summarization
- Adjustable summary lengths
- Reading analytics
- Downloadable summaries
- GPU acceleration support
- Modern responsive UI

Built with performance optimization and simplicity in mind, the app uses Streamlit caching, efficient chunk processing, and optimized PyTorch inference for fast real-time summarization.

---

## Features

### AI-Powered Summarization

Uses the pretrained Hugging Face transformer model:

sshleifer/distilbart-cnn-12-6

Features:
- High-quality abstractive summarization
- Optimized for news articles
- Faster inference
- Lower memory usage

---

### URL-Based Article Extraction

Paste any news article URL and the app automatically:
- Fetches webpage content
- Extracts meaningful paragraphs
- Cleans unnecessary HTML
- Prepares text for summarization

Powered by:
- requests
- BeautifulSoup

---

### Direct Text Summarization

Users can directly paste article text into the application for instant summarization.

Useful for:
- News articles
- Blogs
- Research papers
- Documentation
- Reports

---

### Customizable Summary Length

Choose from multiple summary styles:

| Mode   | Description |
|--------|-------------|
| Short  | Quick concise overview |
| Medium | Balanced summary |
| Long   | Detailed summarized version |

---

### Reading Analytics

Automatically calculates:
- Total word count
- Estimated reading time
- Summary word count
- Compression percentage

---

### Optimized Performance

Performance-focused implementation includes:
- Streamlit caching
- Automatic GPU/CPU detection
- torch.no_grad() optimization
- Chunk-based processing for long articles
- Efficient transformer inference

---

### Download Summary

Generated summaries can be:
- Copied easily
- Downloaded as .txt files
- Shared instantly

---

### Modern UI

Features a clean and responsive interface with:
- Dark theme styling
- Modern cards and layout
- Responsive design
- Streamlit-based interactive components

---

## Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| Language | Python |
| NLP Framework | Hugging Face Transformers |
| Backend | PyTorch |
| Web Scraping | BeautifulSoup4 |
| HTTP Requests | Requests |
| Model | DistilBART CNN-12-6 |

---

## Model Information

### DistilBART CNN-12-6

A lightweight distilled version of Facebook BART optimized for summarization tasks.

Advantages:
- Faster inference
- Lower memory usage
- High summarization quality
- Optimized for news summarization

Model Page:
https://huggingface.co/sshleifer/distilbart-cnn-12-6

---

## Project Structure

```bash
NewsSummarizerApp/
│
├── news_summarizer_app.py
├── requirements.txt
├── .gitignore
└── README.md
```
# Installation & Setup
1. Clone the Repository
git clone https://github.com/witharyank/news-summarizer.git
cd news-summarizer
2. Create Virtual Environment

# Windows:
python -m venv .venv
.venv\Scripts\activate

macOS/Linux:
python3 -m venv .venv
source .venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
4. Run the Application
streamlit run news_summarizer_app.py
Application Workflow
User Input
   ↓
Article Extraction
   ↓
Text Cleaning
   ↓
Chunk Processing
   ↓
DistilBART Summarization
   ↓
Summary Generation
   ↓
Analytics + Download
Performance Optimizations

The application includes several optimizations:

Cached model loading
Automatic CUDA detection
Mixed precision inference
Chunk-based summarization
Efficient memory handling
Streamlit session state management
Key Functionalities
Feature	Status
URL Extraction	Yes
Text Summarization	Yes
Adjustable Summary Length	Yes
Reading Metrics	Yes
Download Summary	Yes
GPU Support	Yes
Streamlit Deployment	Yes
Future Improvements

Planned features:

Multi-language summarization
PDF support
Bullet-point summaries
Light/Dark theme toggle
Advanced transformer models
Docker deployment
Mobile optimization
Text-to-speech summaries
Summary quality scoring
Example Use Cases
News article summarization
Research paper overview
Blog summarization
Corporate news digest
Student study assistance
Quick content understanding
Author

Kumar Aryan

Computer Science Undergraduate interested in:

Artificial Intelligence
NLP Applications
Cloud Computing
Full Stack Development

GitHub:
https://github.com/witharyank

Support the Project

If you like this project:

Star the repository
Fork the project
Contribute improvements
Share with others
License

This project is open-source and available under the MIT License.

Built using Python, Streamlit, Hugging Face Transformers, and PyTorch.
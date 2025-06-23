Sentimo: Real-Time Sentiment Analysis for Korean & English Comments
Sentimo is a lightweight web app that classifies user-entered text into positive or negative sentiment in real-time. Built with pre-trained language models, it supports both Korean and English input.

💡 Project Highlights
Goal: Enable quick and intuitive emotion detection from live user comments (e.g. YouTube, Twitter)

Models Used:

Korean: KcELECTRA (via Hugging Face)

English: RoBERTa-base (with dynamic masking)

Why Pre-trained?: Reduced training cost and faster deployment. Optimized for practical performance in web environments.

Web Framework: Built with Streamlit for simple UI and real-time response

🧠 Methodology
Model selection based on use-case relevance and community performance

Fine-tuned pre-trained models applied without custom training due to GPU constraint

Simple frontend → user types a sentence, model returns sentiment score instantly

Example use case: Real-time monitoring of user feedback, automated comment screening

🔧 Tech Stack
Python, Streamlit, Hugging Face Transformers

Hosted on local machine / Streamlit Cloud (optional)

Korean & English dual-model routing logic

🚀 Try It
To run locally:

bash
Copy
Edit
git clone https://github.com/your-username/sentimo.git
cd sentimo
streamlit run sentimo_app.py

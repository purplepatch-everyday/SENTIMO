# 📌 Sentimo: Real-Time Sentiment Analysis for Korean & English Comments

**Sentimo** is a lightweight web app that classifies user-entered text into **positive** or **negative** sentiment in real-time. Built with pre-trained language models, it supports both **Korean** and **English** inputs.

---

## 💡 Project Highlights

- **Goal**: Enable intuitive emotion detection from live user comments (e.g., YouTube, Twitter)
- **Models Used**:
  - Korean: `KcELECTRA` (Hugging Face)
  - English: `RoBERTa-base` (with dynamic masking)
- **Why Pre-trained?**: Reduces training cost and enables fast, reliable deployment
- **Frontend**: Built using `Streamlit` for real-time interactivity and simplicity

---

## 🧠 Methodology

1. Selected models optimized for sentiment classification in each language
2. Applied pre-trained models without fine-tuning to reduce hardware dependency
3. Designed a dual-pipeline architecture to route Korean and English texts separately
4. Delivered a minimal UI that allows users to instantly test and see results

---

## 🔧 Tech Stack

- **Language**: Python
- **Models**: Hugging Face Transformers (KcELECTRA, RoBERTa)
- **Framework**: Streamlit
- **Utilities**: NumPy, Pandas, Regex

---

## 🚀 Run Locally

```bash
git clone https://github.com/your-username/sentimo.git
cd sentimo
streamlit run sentimo_app.py

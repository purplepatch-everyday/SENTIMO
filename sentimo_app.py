import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langdetect import detect_langs

# 언어 감지 함수 개선 (langdetect 확률 기반)
def detect_language(text):
    try:
        langs = detect_langs(text)
        if langs and langs[0].prob > 0.80:
            return langs[0].lang
        else:
            return "unknown"
    except:
        return "unknown"

# 입력 언어에 따라 모델 로딩 (메모리 절감 버전)
@st.cache_resource(show_spinner=False)
def load_model_for_lang(lang):
    if lang == "ko":
        tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base-finetuned-nsmc")
        model = AutoModelForSequenceClassification.from_pretrained("beomi/KcELECTRA-base-finetuned-nsmc")
    elif lang == "en":
        tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    else:
        return None, None
    return tokenizer, model

# 텍스트 전처리 함수 (강조 표현 정제)
def preprocess_text(text):
    return text.replace("너무너무", "너무 너무").strip()

# 감정 예측 함수
def predict_sentiment(text):
    lang = detect_language(text)
    cleaned_text = preprocess_text(text)

    tokenizer, model = load_model_for_lang(lang)
    if tokenizer is None or model is None:
        return "언어 감지 실패", 0.0, lang

    inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1).numpy()[0]

    if lang == 'ko':
        result = "긍정" if probs[1] > probs[0] else "부정"
        confidence = abs(probs[1] - probs[0])
    
    elif lang == 'en':
        label_map = {0: "부정", 1: "중립", 2: "긍정"}
        predicted_label = int(probs.argmax())
        result = label_map.get(predicted_label, "알 수 없음")
        confidence = probs[predicted_label]
    else:
        result = "언어 감지 실패"
        confidence = 0.0

    return result, confidence, lang

# Streamlit UI
st.set_page_config(page_title="Sentimo 감정 분석", layout="centered")
st.title("🧠 Sentimo: 실시간 감정 분석")
st.write("문장을 입력하고 실행 버튼을 누르면 감정 분석 결과를 보여줍니다.")

# 입력 인터페이스
with st.form(key="sentiment_form"):
    user_input = st.text_area("✍️ 감정을 알고 싶은 문장을 입력하세요:", height=100)
    submit_button = st.form_submit_button(label="🚀 실행하기")

if submit_button and user_input:
    result, confidence, lang = predict_sentiment(user_input)
    st.markdown(f"**🔍 감지된 언어:** `{lang}`")
    st.markdown(f"**📊 분석 결과:** `{result}`")
    st.markdown(f"**📈 확신도:** `{confidence:.2%}`")

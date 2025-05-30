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

# 모델 로드
@st.cache_resource(show_spinner=False)
def load_models():
    # 한국어 감정 분석 모델 교체 (공개 가능 모델)
    ko_tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base-v2022")
    ko_model = AutoModelForSequenceClassification.from_pretrained("beomi/KcELECTRA-base-v2022")

    # 영어 감정 분석 모델
    en_tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
    en_model = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")

    return (ko_tokenizer, ko_model), (en_tokenizer, en_model)

# 텍스트 전처리 함수 (강조 표현 정제)
def preprocess_text(text):
    return text.replace("너무너무", "너무 너무").strip()

# 감정 예측 함수
def predict_sentiment(text, ko_model_pack, en_model_pack):
    lang = detect_language(text)
    cleaned_text = preprocess_text(text)

    if lang == 'ko':
        tokenizer, model = ko_model_pack
        positive_label = 1
    elif lang == 'en':
        tokenizer, model = en_model_pack
        positive_emotions = ["joy", "love", "surprise"]
        negative_emotions = [
            "anger", "sadness", "fear", "disgust", "disappointment", "remorse",
            "grief", "disapproval", "embarrassment", "annoyance", "nervousness"
        ]
    else:
        return "언어 감지 실패", 0.0, lang

    inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1).numpy()[0]

    if lang == 'ko':
        result = "긍정" if probs[positive_label] > probs[1 - positive_label] else "부정"
        confidence = abs(probs[positive_label] - probs[1 - positive_label])  # 확신도 개선 아이디어 1 적용
    else:
        # 모델이 출력하는 클래스 수에 따라 labels 잘라서 사용
        labels = [
            "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
            "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
            "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
            "pride", "realization", "relief", "remorse", "sadness", "surprise"
        ]
        labels = labels[:len(probs)]  # 모델 출력 크기에 맞춰 자르기
        emotion_scores = {labels[i]: probs[i] for i in range(len(labels))}
        pos_score = sum([emotion_scores[e] for e in positive_emotions if e in emotion_scores])
        neg_score = sum([emotion_scores[e] for e in negative_emotions if e in emotion_scores])
        result = "긍정" if pos_score >= neg_score else "부정"
        confidence = abs(pos_score - neg_score)  # 확신도 개선 아이디어 1 적용

    return result, confidence, lang

# Streamlit UI
st.set_page_config(page_title="Sentimo 감정 분석", layout="centered")
st.title("🧠 Sentimo: 실시간 감정 분석")
st.write("문장을 입력하면 한국어/영어 감정 분석 결과를 보여줍니다.")

# 캐시 초기화 및 모델 로딩
st.cache_resource.clear()
ko_model_pack, en_model_pack = load_models()

# 입력 인터페이스
user_input = st.text_area("✍️ 감정을 알고 싶은 문장을 입력하세요:", height=100)

if user_input:
    result, confidence, lang = predict_sentiment(user_input, ko_model_pack, en_model_pack)

    st.markdown(f"**🔍 감지된 언어:** `{lang}`")
    st.markdown(f"**📊 분석 결과:** `{result}`")
    st.markdown(f"**📈 확신도:** `{confidence:.2%}`")

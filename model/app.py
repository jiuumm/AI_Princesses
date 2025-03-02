from fastapi import FastAPI
from fastapi.responses import FileResponse
import os
from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel
import torch
import joblib
app = FastAPI()

# 모델 로드
model_name = "BM-K/KoSimCSE-roberta-multitask"
tokenizer = AutoTokenizer.from_pretrained(model_name)
symptom_model = AutoModel.from_pretrained(model_name)
warning_model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

# 데이터 로드
individual_data = pd.read_excel("C:\\Users\\dbwld\\OneDrive\\바탕 화면\\Project\\individual_data_processed.xlsx")
nonindividual_data = pd.read_excel("C:\\Users\\dbwld\\OneDrive\\바탕 화면\\Project\\nonindividual_data_processed.xlsx")
symptom_data = pd.read_excel("C:\\Users\\dbwld\\OneDrive\\바탕 화면\\Project\\\\symptom_final_data.xlsx")

# 데이터 결합
individual_data['형태'] = '개별 인정형 품목'
nonindividual_data['형태'] = '고시형 품목'
combined_data = pd.concat([individual_data, nonindividual_data], ignore_index=True)

# 증상-건강 카테고리 매핑 추가
add_symptom_to_category = {
    "가슴이 답답하다": "심혈관 및 대사", "숨이 차다": "심혈관 및 대사","심장이 두근거린다": "심혈관 및 대사","혈압이 높다": "심혈관 및 대사",
 "손발이 차다": "심혈관 및 대사","당뇨가 있다": "심혈관 및 대사","살이 많이 찐다": "심혈관 및 대사", "비만이 걱정된다": "심혈관 및 대사",
"고혈당이 있다": "심혈관 및 대사", "속이 메스껍다": "간 및 소화","속이 쓰리다": "간 및 소화","가스가 자주 찬다": "간 및 소화","소화가 잘 안된다": "간 및 소화",
"소변 색이 짙다": "간 및 소화","변비가 있다": "간 및 소화","설사를 자주 한다": "간 및 소화","복부팽만감이 있다": "간 및 소화","감기에 자주 걸린다": "면역 및 체력","몸이 쉽게 피곤하다": "면역 및 체력", 
"콧물이 난다.": "면역 및 체력","기운이 없다": "면역 및 체력","너무 춥다.": "면역 및 체력","근육통이 있다": "면역 및 체력","면역력이 약하다": "면역 및 체력","잠을 잘 못 잔다": "수면 및 정신",
"스트레스를 많이 받는다": "수면 및 정신","우울감을 느낀다": "수면 및 정신","불안하다": "수면 및 정신","집중력이 떨어진다": "수면 및 정신", "무릎이 아프다": "뼈 및 구조",
"허리가 아프다": "뼈 및 구조","관절이 뻣뻣하다": "뼈 및 구조","골다공증이 걱정된다": "뼈 및 구조","등이 결린다": "뼈 및 구조"
}

# 증상 데이터에 추가
symptom_df = pd.DataFrame(list(add_symptom_to_category.items()), columns=['text', '건강 카테고리'])
symptom_data = pd.concat([symptom_data, symptom_df], ignore_index=True)

# 텍스트 인코딩 함수
def encode_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = symptom_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# 섭취 주의사항 확인 함수
def is_safe_for_condition(product_warning, user_condition, threshold=0.45):
    if isinstance(product_warning, float) and pd.isna(product_warning):
        return True
    bullets = product_warning.split('\n')
    for bullet in bullets:
        bullet = bullet.strip()
        if bullet == "":
            continue
        sim = util.cos_sim(warning_model.encode(bullet).reshape(1, -1), warning_model.encode(user_condition).reshape(1, -1))[0][0].item()
        if sim >= threshold:
            return False
    return True

@app.get("/favicon.ico")
def get_favicon():
    return FileResponse(os.path.join("static", "favicon.ico"))

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI app!"}

# 건강기능식품 추천 API
@app.post("/predict")
def recommend_health_products(user_input: str, user_condition: str, top_n: int = 5):
    input_vector = encode_text(user_input).reshape(1, -1)
    
    symptom_similarities = [
        (util.cos_sim(input_vector, encode_text(row['text']).reshape(1, -1))[0][0].item(), row['건강 카테고리']) 
        for _, row in symptom_data.iterrows()
    ]
    predicted_category = max(symptom_similarities, key=lambda x: x[0])[1]
    
    recommended_products = [
        (row['품목명'], row['형태'], row['건강 카테고리'])
        for _, row in combined_data.iterrows() 
        if row['건강 카테고리'] == predicted_category and is_safe_for_condition(row.get('섭취 주의사항', ''), user_condition)
    ]
    
    recommended_products = sorted(recommended_products, key=lambda x: x[1], reverse=True)[:top_n]
    
    if not recommended_products:
        raise HTTPException(status_code=404, detail="추천할 제품이 없습니다.")
    
    return {
        "예측된 건강 카테고리": predicted_category,
        "추천 제품": recommended_products
    }
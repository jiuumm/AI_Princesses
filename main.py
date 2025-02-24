import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

# 한글 폰트 설정
plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)

# 1. 데이터 불러오기
combined_data = pd.read_excel('combined_data.xlsx', sheet_name='Sheet1')
symptom_data = pd.read_excel('symptom_data.xlsx', sheet_name='Sheet1')

# 2. 데이터 전처리
texts = symptom_data['text']
labels = symptom_data['건강 카테고리']

# TF-IDF 벡터화 (n-gram (1, 2) 설정 및 max_features 1000으로 증가)
tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
text_vectors = tfidf_vectorizer.fit_transform(texts)

# 라벨 인코딩
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# 학습 데이터와 테스트 데이터 분리 (클래스 비율 유지)
X_train, X_test, y_train, y_test = train_test_split(
    text_vectors, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
)

# 3. 학습 데이터의 클래스 분포 시각화 (오버샘플링 전)
label_counts = Counter(y_train)
plt.bar(label_counts.keys(), label_counts.values())
plt.xticks(ticks=range(len(label_encoder.classes_)), labels=label_encoder.classes_, rotation=45)
plt.xlabel('건강 카테고리')
plt.ylabel('샘플 수')
plt.title('훈련 데이터의 건강 카테고리 분포 (오버샘플링 전)')
plt.savefig('train_data_distribution_before_oversampling.png')  # 이미지 저장
plt.close()

# 4. RandomOverSampler를 이용한 데이터 증강 (데이터 불균형 해결)
ros = RandomOverSampler(random_state=42)
X_train, y_train = ros.fit_resample(X_train, y_train)

# 데이터 증강 후 클래스 분포 확인
print("RandomOverSampler 후 훈련 데이터의 클래스 분포:")
print(Counter(y_train))

# 5. 학습 데이터의 클래스 분포 시각화 (오버샘플링 후)
label_counts = Counter(y_train)
plt.bar(label_counts.keys(), label_counts.values())
plt.xticks(ticks=range(len(label_encoder.classes_)), labels=label_encoder.classes_, rotation=45)
plt.xlabel('건강 카테고리')
plt.ylabel('샘플 수')
plt.title('훈련 데이터의 건강 카테고리 분포 (오버샘플링 후)')
plt.savefig('train_data_distribution_after_oversampling.png')  # 이미지 저장
plt.close()

# 6. 모델 학습
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 테스트 데이터 예측
y_pred = model.predict(X_test)

# 7. 모델 평가
print('📈 Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 8. 혼동 행렬 시각화
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, cmap='Blues', 
            xticklabels=label_encoder.classes_, 
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('혼동 행렬 (Confusion Matrix)')
plt.savefig('confusion_matrix.png')  # 이미지 저장
plt.close()

# 9. 모델 저장
joblib.dump(model, 'health_recommendation_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

# 10. 건강기능식품 추천 기능 구현 (유사도 기반 추천)
def recommend_health_products(user_input, top_n=5):
    # 사용자 입력 벡터화
    input_vector = tfidf_vectorizer.transform([user_input])
    
    # 건강 카테고리 예측
    predicted_category = model.predict(input_vector)
    category_name = label_encoder.inverse_transform(predicted_category)[0]
    
    # 예측된 건강 카테고리에 맞는 건강기능식품 필터링
    recommended_products = combined_data[combined_data['건강 카테고리'] == category_name]
    
    # 주요 기능 텍스트 벡터화 (추천 제품의 '주요 기능' 열 사용)
    product_descriptions = recommended_products['주요 기능'].fillna('')
    product_vectors = tfidf_vectorizer.transform(product_descriptions)
    
    # 사용자 입력과 제품 주요 기능 간의 코사인 유사도 계산
    similarities = cosine_similarity(input_vector, product_vectors).flatten()
    
    # 유사도에 따라 제품을 정렬하고 상위 N개 선택
    top_indices = similarities.argsort()[::-1][:top_n]
    top_products = recommended_products.iloc[top_indices]
    
    # 결과 출력
    print(f"\n🧠 예측된 건강 카테고리: {category_name}")
    print(f"💊 추천 건강기능식품 목록 (상위 {top_n}개):")
    for idx, row in enumerate(top_products.iterrows()):
        row_data = row[1]
        print(f"{idx+1}. {row_data['품목명']} ({row_data['주요 기능']}) [유사도: {similarities[top_indices[idx]]:.2f}]")

# 예시 입력 테스트
user_input = "허리가 아파요"
recommend_health_products(user_input, top_n=5)

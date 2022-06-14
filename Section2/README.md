## 주제: ML 모델을 사용해 데이터를 토대로 소득이 $50000 넘는지 예측해보기

#### 프로젝트 배경(선정 이유)
- 기술적 이유(기술 스택, 분석 스킬): colab 환경, python을 기반으로 분류문제에 대해 여러 머신러닝 모델을 사용해 학습과 성능 비교 가능하기 때문에 진행함
- 개인적 이유(개인 목표, 커리어 관련): section 2에서 배운 다양한 머신러닝 모델을 구현할 수 있음. 또한 기업의 입장에서 신용카드 발급이나 금융 상품 판매 시 고객의 소득이 $50000를 넘는지를 파악해 그에 맞는 상품 추천을 할 수 있을 것이라 생각됨

#### 데이터셋: 
- 컬럼
  - `age` : 나이
  - `workclass` : 고용 형태
  - `fnlwgt` : 사람 대표성을 나타내는 가중치 (final weight의 약자)
  - `education` : 교육 수준
  - `education_num` : 교육 수준 수치
  - `marital_status`: 결혼 상태
  - `occupation` : 업종
  - `relationship` : 가족 관계
  - `race` : 인종
  - `sex` : 성별
  - `capital_gain` : 자본이익
  - `capital_loss` : 자본손실
  - `hours_per_week` : 주당 근무 시간
  - `native_country` : 국적
  - `income` : 수익 

#### 프로젝트 방법
- 전처리: 


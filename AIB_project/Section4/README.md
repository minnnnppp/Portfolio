## 주제: 신경망 모델을 사용 financial sentiment analysis 


#### 🔍 프로젝트 배경
- 프로젝트 선정 이유(개인 목표, 커리어 관련): 
   - section 4 내용을 토대로 신경망을 활용한 모델과 transformer 모델을 직접 구현할 수 있음. 
   - 더불어 금융 시장은 사람들의 행동에 민감하게 변화하는데, 이러한 감성 분석이 가능하다면 시장의 흐름을 예측하는 데 도움이 될 것이라 예상됨.

- 기술 스택: `colab`, `python`, `tensorflow`, `pytorch`, `matplotlib`, `seaborn`, `sklearn`, `transformers`, `nltk`, `LabelEncoder`, `Tokenizer`

#### 데이터셋: https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis/code
- 컬럼
  - `Sentence`: 금융 관련 문장
  - `Sentiment`: 해당 문장이 어떤 감성을 나타내고 있는지 (Neutral/Positive/Negative)
  
#### 프로젝트 방법
- 전처리:
  - 중복값은 제거하여 처리
  - 텍스트 전처리 과정에서 @나 http와 같은 불용어 처리
  - 토큰 수를 줄이기 위해 정규 표현식을 사용하여 텍스트를 정제함
  - 토큰 길이를 맞추기 위해 padding 진행
  - 타겟 변수가 될 `Sentiment` 컬럼의 값을 `LabelEncoder`로 인코딩함

- 가설: 
  - LSTM 모델보다 BERT 모델의 성능이 감성 분석(sentiment analysis)에 더 뛰어날 것이다.

- 문제: 신경망 모델을 사용해 금융 관련 문장에 대해 감성 분류하기

- 모델: `LSTM` 모델과 `BERT` 모델을 사용


#### 프로젝트 결과 - 모델링 Logic에 따른 파이프라인 또는 그림 표현
- 모델 학습 결과
  - `LSTM` 모델:
    - 과적합 방지를 위해 `dropout`과 `학습률 감소` 적용 + `kernel reularzier`와 `bias regularizer` 설정함
    - 평가지표로 `loss`, `정확도`, `f1 score` 사용
    - `epoch` = 20, `batch_size` = 32, optimizer = `Adam`로 설정
  
  - `BERT` 모델:
    - `tensorflow`로 진행할 수 없어 `Pytorch`로 진행
    - 과적합 방지를 위해 `학습률 감소` 적용
    - `epoch` = 20, `batch_size` = 32, optimizer = `AdamW`로 설정
    
  - 모델 성능 비교 
    1. `LSTM` 모델 성능
        - 모델 평가 결과: epoch=20에서  `loss`: 1.1033, `accuracy`: 0.5708, `f1 score`: 2.0136
        
    2. `BERT` 모델 성능
        - epoch=20에서 `accuracy`: 0.7545
    3. 결과
        - 정확도 측면에서 LSTM보다 BERT가 더 높음 = financial sentiment analysis에 더 적합하다는 것을 보여줌

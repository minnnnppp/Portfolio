## 주제: ML 모델을 사용해 개인 소득이 $50000 넘는지 예측


#### 🔍 프로젝트 배경
- 프로젝트 선정 이유(개인 목표, 커리어 관련): 
  - section 2에서 배운 다양한 머신러닝 모델을 구현할 수 있음
  - 또한 기업의 입장에서 신용카드 발급이나 금융 상품 판매 시 고객의 소득이 $50000를 넘는지를 파악해 그에 맞는 상품 추천을 할 수 있을 것이라 생각됨

- 기술 스택: `colab`, `python`, `matplotlib`, `seaborn`, `sklearn`, `OrdinalEncoder`, `PermutationImportance`, `pdpbox`, `shap`


#### 데이터셋: https://www.kaggle.com/vardhansiramdasu/income
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
  - 결측치와 중복값은 제거하여 처리
  - 자본이익 특성과 자본손실 특성을 합해 자본 총합계라는 `total_capital`이라는 컬럼 추가로 생성함
  - `marital_status`를 결혼 여부를 나타내도록 컬럼의 값을 조정함: 결혼 x = 0, 결혼 o = 1
  - `education`, `educational-num`이 서로 연관되어 있으므로 `education`을 제거해줌
- 가설: 
  1. 주당 근무 시간이 길수록 소득 > $50000 일 것이다.
  2. 교육 수준이 높은 사람이 > $50000 일 가능성이 더 높을 것이다.
  3. 고용 형태에 따라, 즉 고용되어 있지 않거나 소득이 없는 경우 소득 > $50000 일 가능성이 낮을 것이다.

- 문제: 어떤 환경을 가질수록 소득이 $50000보다 많을 가능성이 있는지 예측하기
- 모델: `logistic regression`, `random forest`, `catboost` 모델을 사용해 학습 진행 + 하이퍼파라미터 조정은 `randomizedsearch CV`를 사용


#### 프로젝트 결과 - 모델링 Logic에 따른 파이프라인 또는 그림 표현
- 모델 학습 결과
  - 모델 성능 비교 
    1. 하이퍼파라미터를 조정하지 않은 1차 학습 성능 비교
        1. `baseline model`: 타겟변수의 최빈값으로 설정함 / 정확도: 0.752077
        2. `logistic regression model`: 훈련 정확도: 0.7967664409001529 / 검증 정확도: 0.7913225848735089 / f1 score: 0.3919022154316272
        3. `random forest model`: 훈련 정확도: 1.0 / 검증 정확도: 0.8547647135928693 / f1 score: 0.679212507237985
        4. `catboost model`: 훈련 정확도: 0.8667249289927901 / 검증 정확도: 0.8400838904181414 / f1 score: 0.7233560090702948
    2. 하이퍼파라미터를 조정한 2차 학습
       1. `catboost model` f1 score: 0.7790021426385063 
    3. 결과
        - 테스트 데이터의 f1 score는 훈련/검증 데이터의 f1 score와 큰 차이를 보이지 않았기 때문에 모델 학습에서 과적합이 발생하지 않았고, 학습이 상당 부분 일반화되었음을 확인

   <details>
      <summary>시각화 자료</summary>
        <img width="229" alt="스크린샷 2022-06-14 17 42 55" src="https://user-images.githubusercontent.com/93141881/173534330-b53bb744-fba7-4077-a2a7-9e1edb5c2705.png">
   <img width="229" alt="스크린샷 2022-06-14 17 42 55" src="https://user-images.githubusercontent.com/93141881/173534589-922b84e9-8b1e-45e8-9a3a-59085f296744.png">
   <img width="229" alt="스크린샷 2022-06-14 17 42 55" src="https://user-images.githubusercontent.com/93141881/173534521-0d28a50e-1ada-4cf3-8e40-10b6e124f201.png">
   <img width="229" alt="스크린샷 2022-06-14 17 42 55" src="https://user-images.githubusercontent.com/93141881/173534652-ec851054-8ed9-4d98-ab8f-19b12b2ceb3d.png">
   <img width="229" alt="스크린샷 2022-06-14 17 42 55" src="https://user-images.githubusercontent.com/93141881/173534696-5fb70fee-702e-4356-be0b-9458c947593c.png">
   <img width="229" alt="스크린샷 2022-06-14 17 42 55" src="https://user-images.githubusercontent.com/93141881/173534958-49ab4343-ae38-4594-8261-780489b67172.png">
   <img width="229" alt="스크린샷 2022-06-14 17 42 55" src="https://user-images.githubusercontent.com/93141881/173535013-f6bbac6e-09c3-405e-9787-50650eb0f093.png">

    </details>


   - 자료에 대한 해석: 
      - 순열중요도: 결혼 여부, 교육 수준, 나이, 자본 이익 및 총합계 순으로 개인 소득이 $50000를 넘는지에 대해 크게 작용하고 있음을 파악함
      - pdp plot: 
          - 자본이 많을수록, 교육수준이 높을수록, 결혼한 사람일수록 소득이 $50000를 넘을 확률이 높음
          - 20~50대에서는 나이가 많을수록, 반대로 50대 이상에서는 나이가 적을수록 소득이 $50000를 넘을 확률이 커짐
      - shap plot: 
          - 결혼 관계(marital status), 나이(age), 교육수준(eduacational-num), 자본총합계(total capital), 자본이익(capital-gain) 모두 특성값이 작을수록 `소득 > $50000` 에 negative한 영향을 주고, 특성값이 클수록 positive한 영향을 줌




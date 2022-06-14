## 주제: 국가별 기대수명에 대한 분석 및 예측 모델링

#### 프로젝트 배경(선정 이유)
- 기술적 이유(기술 스택, 분석 스킬): colab 환경, python을 사용해 데이터 분석을 진행할 수 있음
- 개인적 이유(개인 목표, 커리어 관련): section 1에서 배운 통계 검정을 적용해볼 수 있으며, 기업의 관점에서 어떤 게임을 출시하는 것이 좋을지 분석함으로써 데이터 분석 역량을 좀 더 키울 수 있기 때문에 선정

#### 데이터셋: https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who?datasetId=12603&sortBy=voteCount
- 컬럼
    - `Country`: 국가
    - `Year`: 연도
    - `Status`: 선진국/개발도상국 여부
    - `Life expectancy`: 기대수명
    - `Adult Mortality`: 성인 사망률
    - `infant deaths`: 유아 사망 건수
    - `Alcohol`: 알코올 소비량
    - `percentage expenditure`: GDP 대비 건강 관련 지출 비중
    - `Hepatitis B`: 1세 인구 대비 b형 간염 예방 접종자 수
    - `Measles`: 홍역 환자 수
    - `BMI`: 평균 체지방 지수
    - `under-five deaths`: 5세 미만 아동 사망률
    - `Polio`: 1세 인구 대비 소아마비 예방 접종자 수
    - `Total expenditure`: 정부 지출 대비 보건 관련 지출 비중
    - `Diphtheria`: 1세 인구 대비 급성전염병 예방 접종자 수
    - `HIV/AIDS`: 5세 미만 아동 hiv/aids 사망건수
    - `GDP`: 국내총생산
    - `Population`: 인구
    - `thinness 1-19 years`: 5~9세 아동인구 대비 저체중 아동의 비율
    - `thinness 5-9 years`: 10~19세 아동인구 대비 저체중 아동의 비율
    - `Income composition of resources`: HDI(각국의 인간 발전 정도와 선진화 정도를 평가한 지수)
    - `Schooling`: 교육수준

#### 프로젝트 방법
- 전처리: 
    - 결측치는 각 년도별로 보간하여 처리함
    - 이상치는 `winsorization`으로 처리
    - 올바르지 않은 컬럼명 수정
    - 건수를 나타낸 특성들을 percentage를 나타내도록 단위 통일

- 가설: 
      1. 아동 관련 보건 지표가 향상될수록 기대수명은 증가할 것이다.
      2. 국가의 선진화 정도가 높아질수록 기대수명이 증가할 것이다.
      3. 건강을 위한 지출이 증가할수록 기대수명은 늘어날 것이다.

- 문제: 나라별 기대수명 예측하기
- 모델: Linear Regression, Ridge Regression, Random Forest, XGBoost 사용 
    - 평가지표로 R2 score(결정계수) 사용

#### 프로젝트 결과 - Action Plan에 관련된 중요한 인사이트 위주 시각화 
- 분석 결과

    - 시각화 자료
    <img width="861" alt="image" src="https://user-images.githubusercontent.com/93141881/173583551-de8e5c22-967e-4770-86f8-b5315c60021a.png">
    
    <img width="570" alt="image" src="https://user-images.githubusercontent.com/93141881/173583299-ac154c36-df34-41b6-954e-2df4b45c8999.png">

    <img width="369" alt="image" src="https://user-images.githubusercontent.com/93141881/173583750-b6e5a16b-bc71-4c8c-bb29-95787c24ffd2.png">
    
    <img width="393" alt="image" src="https://user-images.githubusercontent.com/93141881/173583610-32570224-cbaf-42e8-8fe3-66a8367278fc.png">
    
    <img width="365" alt="image" src="https://user-images.githubusercontent.com/93141881/173583638-e126e4ee-90b8-43b0-b9c0-ed1098672f3e.png">
    
    ![image](https://user-images.githubusercontent.com/93141881/173590439-0a1bb5a0-2256-4746-a564-89149d95cf5b.png)


    - 자료에 대한 해석: 
            - 성능이 가장 뛰어났던 randomforest 모델에서 특성 중요도를 도출
                - hiv/aids 사망률과 성인 사망률, HDI 순으로 가장 중요도가 높음 
            
            - 특성 중요도와 시각화 자료를 토대로 기대 수명에는 아동의 건강과 의료기술의 발전이 지대한 영향을 미치고 있음을 파악함

    - Action Plan: 
            - 다양한 질병으로부터 선제적 보호 조치를 받고 있지 못하는 아동들에게 범국가적 차원의 조치가 필요
            - 선진화를 통해 건강한 생활습관 및 위생 관념 정립을 교육을 통해 이루어져야 할 것
            
- 예측 모델링 결과
    - Linear Regression: 0.8529984574582457
    - Ridge Regression: 0.8529984574582457
    - **Random Forest Regressor: 0.9642667362864851**
    - XGBoost Regressor:  0.9610947119687804

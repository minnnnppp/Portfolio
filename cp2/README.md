## 주제: Ecommerce 데이터 분석 및 추천 모델링(팀 프로젝트)

#### 🔍 프로젝트 배경(선정 이유)
- 기술적 이유(기술 스택, 분석 스킬):
    - 기술 스택: `jupyter`, `python`, `pyarrow`, `matplotlib`, `seaborn`, `plotly`, `sklearn`, `scipy`, `gensim`, `collections`
    - 대용량 데이터를 `pyarrow` 를 통해 `parquet` 파일 형식으로 변환하여 용량에 대한 이슈를 해결함: 
        - `sqlite`와 `MySQL`에 적재 시도: `sqlite`의 경우 데이터를 적재하여 불러오기까지 많은 시간이 소요되었고, `MySQL`의 경우 데이터 적재에 실패함
        - 최종적으로 **`parquet`으로 변환하는 방법**을 선택함
    
- 개인적 이유(개인 목표, 커리어 관련): 
    - Ecommerce 로그 데이터를 가지고 유저에 대한 분석과 추천 모델에 대한 프로젝트 경험이 없음
    - 데이터 분석가로서 로그 데이터 분석과 추천 시스템을 모델링을 통해 직접 구현하여 성과를 도출하기 위해 해당 프로젝트 진행

#### 🔍 팀 내 수행한 역할
- 박민경
    - 데이터 분석 관련 도메인 지식 학습, EDA 및 가설검정, Action plan 도출
    - `surprise` 패키지 이용한 모델과 `LightFM` 모델 구현 시도
    - `word2vec` 사용한 `content-based` 모델 구현
- 정호영
    - 데이터 분석 관련 도메인 지식 학습, EDA 및 `Funnel/Cohort` 분석, Action plan 도출
    - `Baseline model` 및 `TF-IDF` 사용한 `Content-Based Model` 구현


#### 🔍 데이터셋과 환경: https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store?select=2019-Oct.csv
- 데이터: 2019년 10월 데이터 
    - 2019.11~2020.04 데이터를 추가로 사용하여 코호트 분석 실시

- 컬럼
    - `event_time`: 이벤트(=유저의 행동)가 일어난 시간
    - `event_type`: 이벤트 타입
    - `product_id`: 상품 id
    - `category_code`: 상품의 카테고리 코드
    - `brand`: 상품의 브랜드
    - `price`: 상품 가격
    - `user_id`: 유저 id
    - `user_session`: 유저의 세션 id

- 환경
    - 카자흐스탄의 자국 Ecommerce 1위 기업인 Kaspi.kz를 가정함
    - 카자흐스탄 Ecommerce 시장에서 의류/전자/가전 제품의 규모가 가장 크고 성장률도 높음
    - 주로 도심과 젊은 층 위주로 Ecommerce의 소비가 발생 + Ecommerce 관련 인프라가 상대적으로 부족함
    - 모바일 구매 비율이 약 65%로 상당히 높음

#### 🔍 프로젝트 방법
- 전처리: 
    - `event_time`과 관련해 추가 feature engineering을 진행 
    - 결측치를 `unknown`으로 처리
    - 가격이 0이하인 데이터 제거

- 가설: 
 
      1. 주말/주중에 따라 구매 전환율은 차이가 날 것이다.
      2. 사이트에 오래 머물수록 구매 전환율이 더 높을 것이다.
      3. brand가 있는 상품일 경우의 평균 구매 전환율이 없는 상품일 경우보다 더 높을 것이다.
      4. categorized가 잘 되어 있지 않은 상품의 View 수는 잘 되어 있는 상품보다 더 적을 것이다.

- 문제: 데이터 분석 결과를 토대로 Action Plan 도출 및 추천시스템 구현

- 모델: 
    - `Baseline model`: 이벤트가 가장 많은 상위 20개의 상품을 일괄적으로 추천하는 통계 기반 모델을 설정 
    - 성능 개선 모델: `TF-IDF`와 `Word2Vec`을 사용한 `Content-Based model` 사용 
        1. `TF-IDF`을 사용한 `Content-Based model`: 각 상품의 메타정보(category_code, brand 등)를 활용하여 `cosine` 유사도를 계산해 유사도가 높은 순으로 유저별 20개의 상품 추천
        2. `Word2Vec`을 사용한 `Content-Based model`: `doc2vec`를 사용해 각 상품의 product_id와 category_code 벡터화하고, 상품별 유사도를 기준으로 유저가 본 상품과 연관성이 높은 20개의 상품 추천
    - 평가지표로 랭킹 기반에 사용되는 `MAP@K`, `NDCG@K` 사용

#### 🔍 프로젝트 결과
- 분석 결과

    - 시각화 자료
    <img width="556" alt="image" src="https://user-images.githubusercontent.com/93141881/174443601-bb06a71d-d3f4-4e9d-8383-2eb1e838ec66.png">

    
    <img width="737" alt="image" src="https://user-images.githubusercontent.com/93141881/173604551-f89eb9f8-09d1-49a6-8b43-4c91c7c741f5.png">
    
    <img width="673" alt="image" src="https://user-images.githubusercontent.com/93141881/173604618-ef8506f5-2729-4023-93cd-0c46e2506805.png">
    
   <img width="369" alt="image" src="https://user-images.githubusercontent.com/93141881/174443614-0245efe7-5943-40f2-8c76-28aa4520490a.png">
   
   <img width="369" alt="image" src="https://user-images.githubusercontent.com/93141881/174443620-5d1b273a-977d-4cb6-a795-83a8f201d68a.png">

    
    <img width="443" alt="image" src="https://user-images.githubusercontent.com/93141881/173605161-214c0343-5bc4-40ea-b240-552586330b5f.png">
    
    <img width="659" alt="image" src="https://user-images.githubusercontent.com/93141881/173605403-bd88e840-bbef-418e-807a-db06aa26e68b.png">
    
    <img width="648" alt="image" src="https://user-images.githubusercontent.com/93141881/173605433-9dd612bd-dd5f-4b4d-b711-ae9a17f647cc.png">

    - 자료에 대한 해석: 
        - 재구매율이 낮고, view에서 장바구니로의 전환율이 낮음
        - 기존 고객의 총 구매액과 1인당 평균 구매액이 신규 고객보다 더 큼
        - 가설 1과 2는 기각할 수 있고, 가설 3과 4는 기각할 수 없음
            - 즉 주말/주중의 구매 전환율에는 큰 차이가 없고, 접속 시간이 짧을수록 구매 전환율이 더 높은 경향이 있음
            - 브랜드가 있는 상품의 경우 구매 전환율이 더 높으며, categorized된 상품일수록 view 수가 더 높음

    - Action Plan: 
        - 신규 고객 확보를 위한 plan
            - 카테고리 등록 활성화
            - 옴니채널 서비스 활용
            - 신규 고객을 위한 모바일 어플 프로모션 및 할인행사 진행
       - 기존 고객 구매 장려를 위한 plan
            - 구매한 상품과 연관된 상품 추천하는 서비스 제공
            - 기존 고객을 위한 등급별 할인쿠폰 및 마일리지 제공
            - 저관여제품을 중심으로 상품의 다양성 확보
                - **생필품 & 소비 주기가 짧은 제품**(개인위생용품, 휴지, 생수 등)에 대한 반복적 광고 노출을 통해 기존 고객의 구매를 유도
            
- 예측 모델링 결과
    - `Baseline model`: MAP@20: 0.03470936329312459 / NDCG@20: 0.08250075125861263
    - `TF-IDF`을 사용한 `Content-Based model`: MAP@20: 0.017148624592693736 / NDCG@20: 0.04917596785249336
    - `Word2Vec`을 사용한 `Content-Based model`: MAP@20: 0.001588630171801049 / NDCG@20: 0.005573220462657566


    👉🏼 `TF-IDF`을 사용한 ``Content-Based model` 가장 좋은 성능을 보임

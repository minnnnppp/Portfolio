## 주제: Ecommerce 데이터 분석 및 추천 모델링

#### 프로젝트 배경(선정 이유)
- 기술적 이유(기술 스택, 분석 스킬): colab 환경, python을 사용해 데이터 분석을 진행할 수 있음
    - 기술 스택: `colab`, `python`, `pyarrow`, `matplotlib`, `seaborn`, `plotly`, `sklearn`, `scipy`, `gensim`, `collections`
- 개인적 이유(개인 목표, 커리어 관련): section 1에서 배운 통계 검정을 적용해볼 수 있으며, 기업의 관점에서 어떤 게임을 출시하는 것이 좋을지 분석함으로써 데이터 분석 역량을 좀 더 키울 수 있기 때문에 선정

#### 데이터셋과 환경: 
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

#### 프로젝트 방법
- 전처리: 
    - 대용량의 데이터를 parquet 형식으로 변환해 불러옴
        - 데이터 용량 줄이기 위해 컬럼의 타입을 변환함
    - `event_time`과 관련해 추가 feature engineering을 진행 
    - 결측치를 unknown으로 처리하거나 직접 유추하여 처리함
    - 가격이 0이하인 데이터 제거
    - 건수를 나타낸 특성들을 percentage를 나타내도록 단위 통일

- 가설: 
      1. 주말/주중에 따라 구매 전환율은 차이가 날 것이다.
      2. 사이트에 오래 머물수록 구매 전환율이 더 높을 것이다.
      3. brand가 있는 상품일 경우의 평균 구매 전환율이 없는 상품일 경우보다 더 높을 것이다.
      4. categorized가 잘 되어 있지 않은 상품의 View 수는 잘 되어 있는 상품보다 더 적을 것이다.

- 문제: 추천시스템 구현
- 모델: TF-IDF와 Word2Vec을 사용한 Content-Based model 사용 
    - Baseline Model로는 이벤트가 가장 많은 상위 20개의 상품을 일괄적으로 추천하는 모델을 설정
    - 평가지표로 MAP@K, NDCG@K 사용

#### 프로젝트 결과 - Action Plan에 관련된 중요한 인사이트 위주 시각화 / 모델링 Logic에 따른 파이프라인 또는 그림 표현
- 분석 결과

    - 시각화 자료
    <img width="737" alt="image" src="https://user-images.githubusercontent.com/93141881/173604551-f89eb9f8-09d1-49a6-8b43-4c91c7c741f5.png">
    
    <img width="673" alt="image" src="https://user-images.githubusercontent.com/93141881/173604618-ef8506f5-2729-4023-93cd-0c46e2506805.png">
    
    <img width="810" alt="image" src="https://user-images.githubusercontent.com/93141881/173604864-b23219e3-00d1-490e-b85f-40baeccaa87e.png">
    
    <img width="443" alt="image" src="https://user-images.githubusercontent.com/93141881/173605161-214c0343-5bc4-40ea-b240-552586330b5f.png">
    
    <img width="659" alt="image" src="https://user-images.githubusercontent.com/93141881/173605403-bd88e840-bbef-418e-807a-db06aa26e68b.png">
    
    <img width="648" alt="image" src="https://user-images.githubusercontent.com/93141881/173605433-9dd612bd-dd5f-4b4d-b711-ae9a17f647cc.png">


    - 자료에 대한 해석: 
        - 재구매율이 낮고, view에서 장바구니로의 전환율이 낮음
        - 기존 고객의 구매액 > 신규 고객의 구매액
        - 가설 1/2는 기각할 수 있고, 가설 3/4는 기각할 수 없음
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
                    - 생필품 & 소비 주기가 짧은 제품에 대한 구매 장려
            
- 예측 모델링 결과
    - Baseline model: MAP@20: 0.03470936329312459 / NDCG@20: 0.08250075125861263
    - TF-IDF을 사용한 Content-Based model: MAP@20: 0.017148624592693736 / NDCG@20: 0.04917596785249336
    - Word2Vec을 사용한 Content-Based model: MAP@20: 0.001588630171801049 / NDCG@20: 0.005573220462657566


    👉🏼 TF-IDF을 사용한 Content-Based model 가장 좋은 성능을 보임

## 주제: 비디오 게임 데이터와 다음 분기에 출시해야 할 비디오 게임에 대한 분석

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


#### 프로젝트 결과 - Action Plan에 관련된 중요한 인사이트 위주 시각화 
- 분석 결과

    - 시각화 자료
    


    - 자료에 대한 해석: 최근 5년간 일본 지역을 제외한 지역의 장르 선호도는 상당히 유사함 + 가장 많은 대중이 선호하는 플랫폼은 `PS4`

    - Action Plan: 북미/유럽/기타 지역이 선호하는 `Shooter/Sports/Action` 장르의 `PS5` 플랫폼인 게임을 출시해애 함

- 예측 모델링 결과

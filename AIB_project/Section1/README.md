## 주제: 비디오 게임 데이터와 다음 분기에 출시해야 할 비디오 게임에 대한 분석

#### 🔍 프로젝트 배경
- 프로젝트 선정 이유(개인 목표, 커리어 관련): 
    - Section 1에서 배운 통계 검정을 적용해볼 수 있음
    - 기업의 관점에서 어떤 게임을 출시하는 것이 좋을지 분석함으로써 데이터 분석 역량을 좀 더 키울 수 있을 것이라 생각함

#### 기술 스택:
- `colab`, `python`, `matplotlib`, `seaborn`, `sklearn`, `scipy`

#### 데이터셋: https://drive.google.com/file/d/1XnswYGySzrdUzuBst0L63XuWuMRJbLvH/view?usp=sharing
- 컬럼
    - `Name`: 게임명
    - `Platform`: 게임 플랫폼
    - `Year`: 출시 연도
    - `Genre`: 게임 장르
    - `Publisher`: 게임 회사
    - `NA_Sales`: 북미 지역 출고량
    - `EU_Sales`: 유럽 지역 출고량
    - `JP_Sales`: 일본 출고량
    - `Other_Sales`: 기타 지역 출고량

#### 프로젝트 방법
- 전처리: 
    - 결측치는 제거하여 처리
    - 출시 연도 표기 통일
    - 매출액 단위 통일
    - 모든 지역 매출액 합해 `총 출고량(Total_Sales)` 컬럼 생성

- 접근법: 총 출고량이 높은 top 50에 대한 분석

- 가설: 
    1. 지역별로 게임 장르 선호도에 차이가 있다.
    2. 연도별 게임 트렌드가 있다. 

- 문제: 다음 분기에 출시해야 할 게임은 어떤 게임이어야 하는가


#### 프로젝트 결과 - Action Plan에 관련된 중요한 인사이트 위주 시각화 
- 분석 결과

    - 시각화 자료

    ![다운로드](https://user-images.githubusercontent.com/93141881/173243595-121e1e3a-25fb-4f09-afcd-bfa1f7b8fa75.png)
    ![다운로드 (1)](https://user-images.githubusercontent.com/93141881/173243601-6b13ea31-f2cb-4c13-9a71-92c47c05eca8.png)
    ![다운로드 (2)](https://user-images.githubusercontent.com/93141881/173243610-116f1652-f69d-4cc6-843e-e61c7c7bd458.png)
    ![플랫폼](https://user-images.githubusercontent.com/93141881/173243615-35900875-81c9-4a10-a88e-1b6094b12cb7.png)


    - 자료에 대한 해석: 최근 5년간 일본 지역을 제외한 지역의 장르 선호도는 상당히 유사함 + 가장 많은 대중이 선호하는 플랫폼은 `PS4`

    - Action Plan: 북미/유럽/기타 지역이 선호하는 `Shooter/Sports/Action` 장르의 `PS5` 플랫폼인 게임을 출시해애 함

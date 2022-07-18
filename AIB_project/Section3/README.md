## 주제: 폐암 사전 자가 진단 웹서비스 개발

#### 🔍 프로젝트 배경
- 프로젝트 선정 이유 
  - 기술적 이유:
    - `Flask`를 사용한 이유: 구현하고자 하는 웹 어플리케이션이 매우 간단한 어플리케이션이기 때문
      - `Flask`: `Django`에 비해 매우 가벼운 web framework이며 기본 기능만 제공

  - 개인적 이유(개인 목표, 커리어 관련): 
    - 발병하기 어려운 폐암에 대해 경각심을 일깨울뿐만 아니라 폐암 조기 발병으로 인해 완치율을 높일 수 있는 웹 어플리케이션 서비스를 제공할 수 있음
 - 기술 스택
    - `colab`, `python`, `sqlite`, `matplotlib`, `seaborn`, `sklearn`, `OrdinalEncoder`, `Metabase`, `Flask`, `Heroku`

#### 데이터셋: https://www.kaggle.com/h13380436001/h-lung-cancer
- 컬럼
  - `Gender`: 성별
  - `Age`: 나이
  - `Smoking`: 흡연여부
  - `Yellow fingers`: 황달 증상 유무
  - `Anxiety`: 불안 증세
  - `Peer_pressure`: 사회적 압박감
  - `Chronic Disease`: 만성 질환 유무
  - `Fatigue`: 만성 피로감 유무
  - `Allergy`: 알러지 유무
  - `Wheezing`: 색색거림 증상 유무
  - `Alcohol`: 음주 여부
  - `Coughing`: 기침 증상 유무
  - `Shortness of Breath`: 숨가쁨 증상 유무
  - `Swallowing Difficulty`: 삼킴 장애 유무
  - `Chest pain`: 흉통 증상 유무
  - `Lung Cancer`: 폐암 여부

#### 프로젝트 방법
- 전처리: 
    - 결측치는 존재하지 x
    - 성별을 남성은 1, 여성은 0으로 표기하도록 전처리 진행

- 데이터 적재 방법: `sqlite`을 사용하여 진행

- 문제: 웹 어플리케이션 서비스 사용자가 자신의 생활습관 및 증상에 대해 자가 체크하여 해당 사용자가 폐암의 위험으로부터 안전한 지를 예측

- 모델: 분류 문제이기 때문에 이에 적합한 `catboost model`을 사용
  - 웹 어플리케이션에 모델을 구현하기 위해 (모델 크기 이슈로 인해) 해당 모델에 대해 하이퍼파라미터를 조정하지 않음

- 대시보드 구현: `metabase`를 사용
- 웹 어플리케이션 구현: `Flask` 사용
- 배포: `heroku` 사용


#### 프로젝트 결과 - 웹 어플리케이션 구현

- 구현 영상 링크: https://drive.google.com/file/d/1XBSWS8cvC2TMiZj3BMGwTPougRX9O-40/view?usp=sharing

  **(영상에는 배포가 나와있지 않습니다. 영상 촬영 후 이후에 heroku를 사용한 배포를 시도하여 성공했습니다.)**





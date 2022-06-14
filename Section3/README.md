## 주제: 폐암 사전 자가 진단 웹서비스 개발

#### 프로젝트 배경(선정 이유)
- 기술적 이유(기술 스택, 분석 스킬): colab 환경, python을 사용해 데이터 분석을 진행할 수 있음
- 개인적 이유(개인 목표, 커리어 관련): section 1에서 배운 통계 검정을 적용해볼 수 있으며, 기업의 관점에서 어떤 게임을 출시하는 것이 좋을지 분석함으로써 데이터 분석 역량을 좀 더 키울 수 있기 때문에 선정

#### 데이터셋: 
- 컬럼
  - Gender: 성별
  - Age: 나이
  - Smoking: 흡연여부
  - Yellow fingers: 황달 증상
  - Anxiety: 불안 증세
  - Peer_pressure: 사회적 압박감
  - Chronic Disease: 만성 질환 유무
  - Fatigue: 만성 피로감 유무
  - Allergy: 알러지 유무
  - Wheezing: 색색거림 증상 유무
  - Alcohol: 음주 여부
  - Coughing: 기침 증상 유무
  - Shortness of Breath: 숨가쁨 증상 유무
  - Swallowing Difficulty: 삼킴 장애 유무
  - Chest pain: 흉통 증상 유무
  - Lung Cancer: 폐암 여부

#### 프로젝트 방법
- 전처리: 
    - 결측치는 존재하지 x
    - 성별을 남성은 1, 여성은 0으로 표기하도록 전처리 진행

- 데이터 적재 방법: sqlite을 사용하여 진행

- 문제: 웹 어플리케이션 서비스 사용자가 자신의 생활습관 및 증상에 대해 자가 체크하여 해당 사용자가 폐암의 위험으로부터 안전한 지를 예측

- 모델: 분류 문제이기 때문에 이에 적합한 catboost model을 사용
  - 웹 어플리케이션에 모델을 구현하기 위해 (모델 크기 이슈로 인해) 해당 모델에 대해 하이퍼파라미터를 조정하지 않음

- 대시보드 구현: metabase를 사용
- 웹 어플리케이션 구현: Flask 사용
- 배포: heroku 사용


#### 프로젝트 결과 - 웹 어플리케이션 구현
- 구현 영상 링크: https://drive.google.com/file/d/1XBSWS8cvC2TMiZj3BMGwTPougRX9O-40/view?usp=sharing
  (영상에는 배포가 나와있지 않습니다. 영상 촬영 후 이후에 heroku를 사용한 배포를 시도하여 성공했습니다.)





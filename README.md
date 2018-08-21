### 네이버 리뷰 데이터를 활용한 한글 데이터 감정 분석


---



#### Overview

네이버 영화 리뷰데이터(Naver Sentiment Movie Corpus,NSMC)를 활용해서 감정분석 문제를 해결합니다.

아래와 같은 순서로 진행했습니다.

* 데이터 전처리 과정에서의 데이터 분석
* 단어 임베딩 및 모델링
---

#### Data

데이터는 해당 [github](https://github.com/e9t/nsmc)에서 받으실 수 있습니다. 데이터는 다음과 같습니다.

* `rating_train.txt`: 학습 데이터 총 15만개
* `rating_test.txt`: 테스트 데이터 총 5만개

class는 긍정, 부정을 2개이며 각각 분포는 정확히 50:50으로 분포되어 있습니다.

---
#### Package
필요한 패키지는 다음과 같다.

* Konlpy
* Tensorflow
* Gensim
* Pandas
* Numpy
* Re

---

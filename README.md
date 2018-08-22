### 네이버 리뷰 데이터를 활용한 한글 데이터 감정 분석


---



#### Overview

네이버 영화 리뷰데이터(Naver Sentiment Movie Corpus,NSMC)를 활용해서 감정분석 문제를 해결합니다.

아래와 같은 순서로 진행했습니다.

* 데이터 전처리
* CNN을 사용한 Classification
---

#### Data

데이터는 해당 [github](https://github.com/e9t/nsmc)에서 받으실 수 있습니다. 데이터는 다음과 같습니다.

* `rating_train.txt`: 학습 데이터 총 15만개
* `rating_test.txt`: 테스트 데이터 총 5만개

class는 긍정, 부정을 2개이며 각각 분포는 정확히 50:50으로 분포되어 있습니다.

---
#### Package
필요한 패키지는 다음과 같습니.

* Konlpy
* Tensorflow
* Pandas
* Numpy
* Re

---

#### Preprocessing

word2vec 혹은 GloVe 등을 사용해 임베딩은 하지않고, 데이터 전처리와 `tensorflow.python.keras.prerprocessing` 모듈을 이용해 index로 벡터화만 진행했습니다.

전처리 과정에서 전체 데이터인 `ratings.txt`를 사용해서 Tokenizer에 fit했고 해당 객체를 이용해서 `ratings_train.txt`와 `ratings_test.txt`를 각각 index sequence 넘파이 배열로 만들었습ㄴ다.

---

#### Training

모델은 간단한 CNN 모델을 사용했습니다. input값을 random하게 embedding 해준뒤 tensorflow의 conv1d를 사용했고 임베딩과 convolution에 모두 dropout을 0.2의 확률로 주었습니다.

코드는 tf.data와 tf.estimator를 사용해서 작성되었습니다.

---


#### Result

모델에는 10에폭으로 학습한다고 나와있는데, 전체 학습하지는 않고 60k step까지만 학습 후 evaluation 과 test를 진행했습니다.
![loss](https://i.imgur.com/PLRmbMo.jpg)

evaluation 결과는 다음과 같습니다.

```python
acc = 0.8341333, global_step = 60700, loss = 0.45636088
```

test 데이터 결과는 다음과 같습니다.

accuracy = 83.5168 %

(다시 학습한 후 결과 update 예정)



---

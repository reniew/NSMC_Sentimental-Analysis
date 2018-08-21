import os
import tensorflow as tf
import numpy as np
import json
import re
import pandas as pd
from konlpy.tag import Okt
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing import sequence

DEFAULT_PATH='./data/nsmc/'

train = pd.read_csv(DEFAULT_PATH+'ratings_train.txt', header=0, delimiter='\t' ,quoting=3 )
test = pd.read_csv(DEFAULT_PATH+'ratings_test.txt', header=0, delimiter='\t' ,quoting=3)

def preprocessing(review, okt, remove_stopwords = False, stop_words = []):
    # 함수의 인자는 다음과 같다.
    # review : 전처리할 텍스트
    # okt : okt 객체를 반복적으로 생성하지 않고 미리 생성후 인자로 받는다.
    # remove_stopword : 불용어를 제거할지 선택 기본값은 False
    # stop_word : 불용어 사전은 사용자가 직접 입력해야함 기본값은 비어있는 리스트

    # 1. 한글 및 공백을 제외한 문자 모두 제거.
    review_text = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", "", review)

    # 2. okt 객체를 활용해서 형태소 단위로 나눈다.
    word_review = okt.morphs(review_text, stem=True)

    if remove_stopwords:

        # 불용어 제거(선택적)
        word_review = [token for token in word_review if not token in stop_words]


    return word_review

stop_words = [ '은', '는', '이', '가', '하', '아', '것', '들','의', '있', '되', '수', '보', '주', '등', '한']
okt = Okt()

clean_review = []
clean_review_test = []

for review in train['document']:
    # 비어있는 데이터에서 멈추지 않도록 string인 경우만 진행
    if type(review) == str:
        clean_review.append(preprocessing(review, okt, remove_stopwords = True, stop_words=stop_words))
    else:
        clean_review.append([])

for review in test['document']:
    # 비어있는 데이터에서 멈추지 않도록 string인 경우만 진행
    if type(review) == str:
        clean_review_test.append(preprocessing(review, okt, remove_stopwords = True, stop_words=stop_words))
    else:
        clean_review_test.append([])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(clean_review)
text_sequences = tokenizer.texts_to_sequences(clean_review)

word_vocab = tokenizer.word_index # 딕셔너리 형태
print("전체 단어 개수: ", len(word_vocab)) # 전체 단어 개수 확인

MAX_SEQUENCE_LENGTH = 10 # 문장 최대 길이

inputs = pad_sequences(text_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post') # 문장의 길이가 50 단어가 넘어가면 자르고, 모자르면 0으로 채워 넣는다.
labels = np.array(train['label']) # 각 리뷰의 감정을 넘파이 배열로 만든다.

print('Shape of input data tensor:', inputs.shape) # 리뷰 데이터의 형태 확인
print('Shape of label tensor:', labels.shape) # 감정 데이터 형태 확인

tokenizer_test = Tokenizer()
tokenizer_test.fit_on_texts(clean_review_test)
text_sequences_test = tokenizer_test.texts_to_sequences(clean_review_test)

word_vocab_test = tokenizer_test.word_index # 딕셔너리 형태
print("전체 단어 개수: ", len(word_vocab_test)) # 전체 단어 개수 확인

MAX_SEQUENCE_LENGTH = 10 # 문장 최대 길이

inputs_test = pad_sequences(text_sequences_test, maxlen=MAX_SEQUENCE_LENGTH, padding='post') # 문장의 길이가 50 단어가 넘어가면 자르고, 모자르면 0으로 채워 넣는다.
labels_test = np.array(test['label']) # 각 리뷰의 감정을 넘파이 배열로 만든다.

print('Shape of input data tensor:', inputs_test.shape) # 리뷰 데이터의 형태 확인
print('Shape of label tensor:', labels_test.shape) # 감정 데이터 형태 확인

# 경로 및 파일 이름 지정
FILE_DIR_PATH = './data/'
INPUT_TRAIN_DATA_FILE_NAME = 'nsmc_train_input.npy'
LABEL_TRAIN_DATA_FILE_NAME = 'nsmc_train_label.npy'
DATA_CONFIGS_FILE_NAME = 'data_configs.json'
INPUT_TEST_DATA_FILE_NAME = 'nsmc_test_input.npy'
LABEL_TEST_DATA_FILE_NAME = 'nsmc_test_label.npy'


data_configs = word_vocab
data_configs['vocab_size'] = len(word_vocab) # vocab size 추가


# 저장하는 디렉토리가 존재하지 않으면 생성
if not os.path.exists(FILE_DIR_PATH):
    os.makedirs(FILE_DIR_PATH)

# 전처리 된 데이터를 넘파이 형태로 저장
np.save(open(FILE_DIR_PATH + INPUT_TRAIN_DATA_FILE_NAME, 'wb'), inputs)
np.save(open(FILE_DIR_PATH + LABEL_TRAIN_DATA_FILE_NAME, 'wb'), labels)

np.save(open(FILE_DIR_PATH + INPUT_TEST_DATA_FILE_NAME, 'wb'), inputs_test)
np.save(open(FILE_DIR_PATH + LABEL_TEST_DATA_FILE_NAME, 'wb'), labels_test)

# 데이터 사전을 json 형태로 저장
json.dump(word_vocab, open(FILE_DIR_PATH + DATA_CONFIGS_FILE_NAME, 'w'), ensure_ascii=False)

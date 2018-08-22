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
ALL_REVIEW = 'ratings.txt'
TRAIN_REVEIW = 'ratings_train.txt'
TEST_REVEIW = 'ratings_test.txt'


all_review = pd.read_csv(FILE_PATH + ALL_REVIEW, quoting=3, header= 0, delimiter='\t')
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

def review_to_clean(clean_list, reviews):

    print('total {0} reviews, processing start'.format(len(reviews)))
    count = 0clean_review
    for review in reviews['documremove_stopwords=       if type(review) == str:
            clean_list.append(preprocessing(review, okt, remove_stopwords = True, stop_words=stop_words))
        else:
            clean_list.append([])
        count = count + 1

        if count % 1000 == 0:
            print('{0} reviews done'.format(count))


stop_words = [ '은', '는', '이', '가', '하', '아', '것', '들','의', '있', '되', '수', '보', '주', '등', '한']
okt = Okt()

clean_review = []
clean_train = []
clean_test = []

review_to_clean(clean_list = clean_review, reviews = all_review)
review_to_clean(clean_list = clean_train, reviews = train)
review_to_clean(clean_list = clean_test, reviews = test)


tokenizer = Tokenizer()
tokenizer.fit_on_texts(clean_review)

train_sequences = tokenizer.texts_to_sequences(clean_train)
test_sequences = tokenizer.texts_to_sequences(clean_test)

word_vocab = tokenizer.word_index # 딕셔너리 형태
print("전체 단어 개수: ", len(word_vocab)) # 전체 단어 개수 확인

MAX_SEQUENCE_LENGTH = 12 # 문장 최대 길이

train_inputs = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
train_labels = np.array(train['label'])

test_inputs = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
test_labels = np.array(test['label'])


############################################


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
json.dump(data_configs, open(FILE_DIR_PATH + DATA_CONFIGS_FILE_NAME, 'w'), ensure_ascii=False)

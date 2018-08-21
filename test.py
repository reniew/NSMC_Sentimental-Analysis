import tensorflow
import model
import numpy as np
import json

import model

FILE_DIR_PATH = './data/'
INPUT_TEST_DATA_FILE_NAME = 'nsmc_test_input.npy' # 전처리한 데이터
LABEL_TEST_DATA_FILE_NAME = 'nsmc_test_label.npy' # 전처리한 데이터
DATA_CONFIGS_FILE_NAME = 'data_configs.json' # vocab size, vocab dictionary

input_test_data = np.load(open(FILE_DIR_PATH + INPUT_TEST_DATA_FILE_NAME, 'rb'))
label_test_data = np.load(open(FILE_DIR_PATH + LABEL_TEST_DATA_FILE_NAME, 'rb'))
prepro_test_configs = json.load(open(FILE_DIR_PATH + DATA_CONFIGS_FILE_NAME, 'r'))



def mapping_fn(X, Y):
    input, label = {'text': X}, Y
    return input, label

def test_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((input_test, label_test))
    dataset = dataset.shuffle(buffer_size=len(input_eval))
    dataset = dataset.batch(16)
    dataset = dataset.map(mapping_fn)
    iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()

prediction = model.est.predict(test_input_fn)

num_score = []


#prediction을 진행하여, 마지막에 있는 값을 추출 (label 제외)
for i, p in enumerate(prediction):
    num_score.append(p['prob'][0])

def test_acc():

    value = 0
    count = 0
    result_label = []

    for score in num_score:
        if score >= 0.5:
            result_label.append(1)
        else:
            result_label.append(0)

    for label_predict in result_label:
        if label_predict == test_label_data[count]:
            value = value + 1
        count = count + 1

    accr = (value / count) * 100

    print(accr,"%")

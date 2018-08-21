import numpy as np
import json
from sklearn.model_selection import train_test_split

FILE_DIR_PATH = './data/'
INPUT_TRAIN_DATA_FILE_NAME = 'nsmc_train_input.npy' # 전처리한 데이터
LABEL_TRAIN_DATA_FILE_NAME = 'nsmc_train_label.npy' # 전처리한 데이터
DATA_CONFIGS_FILE_NAME = 'data_configs.json' # vocab size, vocab dictionary

input_data = np.load(open(FILE_DIR_PATH + INPUT_TRAIN_DATA_FILE_NAME, 'rb'))
label_data = np.load(open(FILE_DIR_PATH + LABEL_TRAIN_DATA_FILE_NAME, 'rb'))
prepro_configs = json.load(open(FILE_DIR_PATH + DATA_CONFIGS_FILE_NAME, 'r'))

TEST_SPLIT = 0.1
RNG_SEED = 13371447

# 학습과 검증 데이터 구분
input_train, input_eval, label_train, label_eval = train_test_split(input_data, label_data, test_size=TEST_SPLIT, random_state=RNG_SEED)

def mapping_fn(X, Y):
    input, label = {'text': X}, Y
    return input, label

def train_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((input_train, label_train))
    dataset = dataset.shuffle(buffer_size=len(input_train))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(mapping_fn)
    dataset = dataset.repeat(count=NUM_EPOCHS)
    iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()

# eval 시 사용
def eval_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((input_eval, label_eval))
    dataset = dataset.shuffle(buffer_size=len(input_eval))
    dataset = dataset.batch(16)
    dataset = dataset.map(mapping_fn)
    iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()

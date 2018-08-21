import os
from datetime import datetime
import tensorflow as tf

import data_tf

BATCH_SIZE = 16
NUM_EPOCHS = 10
vocab_size = data_tf.prepro_configs['vocab_size']
embedding_size = 128

def model_fn(features, labels, mode, params):
    TRAIN = mode == tf.estimator.ModeKeys.TRAIN
    EVAL = mode == tf.estimator.ModeKeys.EVAL
    PREDICT = mode == tf.estimator.ModeKeys.PREDICT

   #embedding layer를 선언합니다.
    input_layer = tf.contrib.layers.embed_sequence(
                    features['text'],
                    vocab_size = vocab_size,
                    embedding_size=embedding_size,
                    initializer=params['embedding_initializer']
                    )
    # 현재 모델이 학습모드인지 여부를 확인하는 변수입니다.
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    # embedding layer에 대한 output에 대해 dropout을 취합니다.
    dropout_emb = tf.layers.dropout(inputs=input_layer,
                                  rate=0.2,
                                  training=training)

    #### CNN 구현체 부분 ####
    conv = tf.layers.conv1d(
           inputs=dropout_emb,
           filters=32,
           kernel_size=3,
           padding='same',
           activation=tf.nn.relu)

    pool = tf.reduce_max(input_tensor=conv, axis=1) #max-pooling layer


    #Fully-connected layer
    hidden = tf.layers.dense(inputs=pool, units=250, activation=tf.nn.relu)

    #####################
    dropout_hidden = tf.layers.dropout(inputs=hidden, rate=0.2, training=training)
    logits = tf.layers.dense(inputs=dropout_hidden, units=1)

    #prediction 진행 시, None
    if labels is not None:
        labels = tf.reshape(labels, [-1, 1])
    #최종적으로 학습, 평가, 테스트의 단계로 나누어 활용
    if TRAIN:
        global_step = tf.train.get_global_step()
        loss = tf.losses.sigmoid_cross_entropy(labels, logits)
        train_op = tf.train.AdamOptimizer(0.001).minimize(loss, global_step)

        return tf.estimator.EstimatorSpec(mode=mode, train_op=train_op, loss = loss)

    elif EVAL:
        loss = tf.losses.sigmoid_cross_entropy(labels, logits)
        pred = tf.nn.sigmoid(logits)
        accuracy = tf.metrics.accuracy(labels, tf.round(pred))
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops={'acc': accuracy})

    elif PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={
                'prob': tf.nn.sigmoid(logits),
            }
        )

params = {'embedding_initializer': tf.random_uniform_initializer(-1.0, 1.0)}

model_dir = os.path.join(os.getcwd(), "checkpoint/cnn_model")
os.makedirs(model_dir, exist_ok=True)

config_tf = tf.estimator.RunConfig()
config_tf._save_checkpoints_steps = 100
config_tf._save_checkpoints_secs = None
config_tf._keep_checkpoint_max =  2
config_tf._log_step_count_steps = 100

est = tf.estimator.Estimator(model_fn, model_dir=model_dir, config=config_tf, params=params)

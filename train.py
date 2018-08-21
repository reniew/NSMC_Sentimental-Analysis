import tensorflow as tf
from datetime import datetime

import model
import data_tf

tf.logging.set_verbosity(tf.logging.INFO)
print(tf.__version__)
time_start = datetime.utcnow()
print("Experiment started at {}".format(time_start.strftime("%H:%M:%S")))
print(".......................................")

model.est.train(data_tf.train_input_fn)

time_end = datetime.utcnow()
print(".......................................")
print("Experiment finished at {}".format(time_end.strftime("%H:%M:%S")))
print("")
time_elapsed = time_end - time_start
print("Experiment elapsed time: {} seconds".format(time_elapsed.total_seconds()))


print("#######################################")
print("############# evaluation ##############")
print("#######################################")

valid = model.est.evaluate(data_tf.eval_input_fn)

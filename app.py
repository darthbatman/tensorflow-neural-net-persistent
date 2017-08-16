from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import random
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

tf.logging.set_verbosity(tf.logging.ERROR)

training_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename="iris_training.csv", target_dtype=np.int, features_dtype=np.float32)

test_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename="iris_test.csv", target_dtype=np.int, features_dtype=np.float32)

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[10, 20, 10],n_classes=3,model_dir=str(os.path.dirname(os.path.abspath(__file__)))+"/tmp/iris_model")

classifier.fit(x=training_set.data, y=training_set.target, steps=2000)

accuracy_score = classifier.evaluate(x=test_set.data, y=test_set.target)["accuracy"]

print('Accuracy: {0:f}'.format(accuracy_score))
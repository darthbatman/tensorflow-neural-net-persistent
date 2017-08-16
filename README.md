# tensorflow-neural-net-persistent
Tensorflow DNN implemented to classify data read from CSV that can save and reload a trained model.

The code for this model can be found in app.py

#### Code

```python
# dependencies

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import random
import os

# hiding warning messages

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

tf.logging.set_verbosity(tf.logging.ERROR)

# reading training set from file

training_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename="iris_training.csv", target_dtype=np.int, features_dtype=np.float32)

# reading testing set from file

test_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename="iris_test.csv", target_dtype=np.int, features_dtype=np.float32)

# getting feature names of attributes

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

# creating classifier

classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[10, 20, 10],n_classes=3,model_dir=str(os.path.dirname(os.path.abspath(__file__)))+"/tmp/iris_model")

# training classifier

classifier.fit(x=training_set.data, y=training_set.target, steps=2000)

# calculating accuracy of classifier's evaluation of test set

accuracy_score = classifier.evaluate(x=test_set.data, y=test_set.target)["accuracy"]

# showing classifier accuracy

print('Accuracy: {0:f}'.format(accuracy_score))
```

The code for the implementation of a reloaded previously created model can be found in pretrained.py

#### Code

```python
# dependencies

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import random
import os

# hiding warning messages

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

tf.logging.set_verbosity(tf.logging.ERROR)

# reading testing set from file

test_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename="iris_test.csv", target_dtype=np.int, features_dtype=np.float32)

# getting feature names of attributes

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

# recreating classifier

classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[10, 20, 10],n_classes=3,model_dir=str(os.path.dirname(os.path.abspath(__file__)))+"/tmp/iris_model")

# calculating accuracy of classifier's evaluation of test set

accuracy_score = classifier.evaluate(x=test_set.data, y=test_set.target)["accuracy"]

# showing accuracy of classifier

print('Accuracy: {0:f}'.format(accuracy_score))
```
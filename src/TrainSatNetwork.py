#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""DNNRegressor with custom input_fn for Housing dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

COLUMNS = ["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31","32","33","34","35","36","37","38","39","40","41","42","43","44","45","46","47","48","49"]
FEATURES = ["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31","32","33","34","35","36","37","38","39","40","41","42","43","44","45","46","47","48"]
LABEL = ["49"]


def input_fn(data_set):
  feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
  labels = tf.constant(data_set[LABEL].values)
  return feature_cols, labels

sess = tf.InteractiveSession()

training_set = pd.read_csv("/home/willie/workspace/SAT_Solvers_GPU/data/tfData/PI4000Train.csv", skipinitialspace=True,
                             skiprows=1, names=COLUMNS)
test_set = pd.read_csv("/home/willie/workspace/SAT_Solvers_GPU/data/tfData/PI4000Val.csv", skipinitialspace=True,
                         skiprows=1, names=COLUMNS)

# Set of 6 examples for which to predict median house values
prediction_set = pd.read_csv("/home/willie/workspace/TensorFlow/data/boston_predict.csv", skipinitialspace=True,
                               skiprows=1, names=COLUMNS)

# Feature cols
feature_cols = [tf.contrib.layers.real_valued_column(k)
                  for k in FEATURES]

# Build 2 layer fully connected DNN with 10, 10 units respectively.
regressor = tf.contrib.learn.DNNClassifier(
    feature_columns=feature_cols, hidden_units=[20], n_classes=2, enable_centered_bias=False)
regressor.fit(input_fn=lambda: input_fn(training_set), steps=100)

for variable in regressor.get_variable_names():
	print(variable)
	print(regressor.get_variable_value(variable))

ev = regressor.evaluate(input_fn=lambda: input_fn(test_set), steps=1)
loss_score = ev["loss"]
print("Loss: {0:f}".format(loss_score))

#def printCNet(classifier):



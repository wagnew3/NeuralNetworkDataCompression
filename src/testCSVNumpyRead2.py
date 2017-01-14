import tensorflow as tf
import numpy as np
from numpy import genfromtxt
import sklearn

# Convert to one hot
def convertOneHot(data):
    y=np.array([int(i[0]) for i in data])
    y_onehot=[0]*len(y)
    for i,j in enumerate(y):
        y_onehot[i]=[0]*(y.max() + 1)
        y_onehot[i][j]=1
    return (y,y_onehot)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

data = genfromtxt('/home/willie/workspace/SAT_Solvers_GPU/data/tfData/PI4000TrainInNoCC.csv',delimiter=',')  # Training data
y_train_onehot = genfromtxt('/home/willie/workspace/SAT_Solvers_GPU/data/tfData/PI4000TrainOutNoCC.csv',delimiter=',')
y_test_onehot = genfromtxt('/home/willie/workspace/SAT_Solvers_GPU/data/tfData/PI4000ValOutNoCC.csv',delimiter=',')  # Testing output

test_data = genfromtxt('/home/willie/workspace/SAT_Solvers_GPU/data/tfData/PI4000ValInNoCC.csv',delimiter=',')  # Test data

x_train=np.array([ i[1::] for i in data])
#y_train,y_train_onehot = convertOneHot(data)

x_test=np.array([ i[1::] for i in test_data])
#y_test,y_test_onehot = convertOneHot(test_data)

A=data.shape[1]-1 # Number of features, Note first is y
B=len(y_train_onehot[0])
hiddenSize=10

tf_in = tf.placeholder("float", [None, A]) # Features
tf_weight = tf.Variable(tf.zeros([A,hiddenSize]))
tf_bias = bias_variable([hiddenSize])
#tf_softmax = tf.nn.relu(tf.matmul(tf_in,tf_weight) + tf_bias)

y_in=tf.nn.relu(tf.matmul(tf_in,tf_weight) + tf_bias)
tf_weight_out = tf.Variable(tf.zeros([hiddenSize,B]))
tf_bias_out = tf.Variable(tf.zeros([B]))
tf_softmax = tf.nn.softmax(tf.matmul(y_in,tf_weight_out) + tf_bias_out)

# Training via backpropagation
tf_softmax_correct = tf.placeholder("float", [None,B])
tf_cross_entropy = -tf.reduce_sum(tf_softmax_correct*tf.log(tf_softmax))

# Train using tf.train.GradientDescentOptimizer
tf_train_step = tf.train.AdamOptimizer(1e-4).minimize(tf_cross_entropy)

# Add accuracy checking nodes
tf_correct_prediction = tf.equal(tf.argmax(tf_softmax,1), tf.argmax(tf_softmax_correct,1))
tf_accuracy = tf.reduce_mean(tf.cast(tf_correct_prediction, "float"))

saver = tf.train.Saver([tf_weight,tf_bias])

# Initialize and run
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

print("...")
# Run the training
for i in range(10000):
    sess.run(tf_train_step, feed_dict={tf_in: x_train, tf_softmax_correct: y_train_onehot})
    result = sess.run(tf_cross_entropy, feed_dict={tf_in: x_test, tf_softmax_correct: y_test_onehot})
    print result

ans= sess.run(tf_softmax, feed_dict={tf_in: x_train, tf_softmax_correct: y_train_onehot})

for i in range(0, 30):
	print(ans[i])
	print(y_train_onehot[i])

print('in weight---------------------------------------------')
print(sess.run(tf_weight))

print('in bias---------------------------------------------')
print(sess.run(tf_bias))

print('out weight---------------------------------------------')
print(sess.run(tf_weight_out))

print('out bias---------------------------------------------')
print(sess.run(tf_bias_out))


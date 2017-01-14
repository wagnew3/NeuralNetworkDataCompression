import numpy as np
import tensorflow as tf
# Global config variables
num_steps = 5 # number of truncated backprop steps ('n' in the discussion above)
batch_size = 200
num_classes = 8
state_size = 8
learning_rate = 0.1

def readToLines(file):
    csvFile=open(file)
    lines=csvFile.read().splitlines()
    return lines

def gen_data(size=1000000):

    xList=[]
    yList=[]

    tLines=readToLines('/home/willie/workspace/TensorFlow/data/LRBPI4096.csv')
    state=0
    tempXs=[]
    tempYs=[]
    amt=0
    for tLine in tLines:
	if tLine=='n':
		xList+=[tempXs]
		yList+=[tempYs]
		#print(tempYs)
		tempXs=[]
		tempYs=[]
		state=0
		amt+=1
		if amt>100:
			break;
	else:
        	stringFloats=tLine.split(',')
        	floats=[]
        	for ind in range(0, len(stringFloats)):
		    floats+=[float(stringFloats[ind])]
		if state%2==0:
			tempXs+=[np.asarray(floats)]
		else:
			tempYs+=[np.asarray(floats)]
		state+=1
    rawX=[]
    rawY=[]

    
   
    for ind in range(0, len(xList)):
        for time in range(num_steps, len(xList[ind]), 1):
		#yield(xList[ind][time-num_steps:time], yList[ind][time])
		rawX+=[xList[ind][time-num_steps:time]]
		rawY+=[yList[ind][time]]

    X=np.asarray(rawX)
    print(X.shape)
    Y=np.asarray(rawY)

    #print(Y)

    return X, Y

# adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/reader.py
def gen_batch(raw_data, batch_size, num_steps):
    raw_x, raw_y = raw_data
    data_length = len(raw_x)

    # partition raw data into batches and stack them vertically in a data matrix
    batch_partition_length = data_length // batch_size
    data_x = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    data_y = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    for i in range(batch_size):
        data_x[i] = raw_x[batch_partition_length * i:batch_partition_length * (i + 1)]
        data_y[i] = raw_y[batch_partition_length * i:batch_partition_length * (i + 1)]
    # further divide batch partitions into num_steps for truncated backprop
    epoch_size = batch_partition_length // num_steps

    for i in range(epoch_size):
        x = data_x[:, i * num_steps:(i + 1) * num_steps]
        y = data_y[:, i * num_steps:(i + 1) * num_steps]
        yield (x, y)

def gen_epochs(n, num_steps):
    for i in range(n):
        yield gen_data()#gen_batch(gen_data(), batch_size, num_steps)

"""
Placeholders
"""

x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')
init_state = tf.zeros([batch_size, state_size])

"""
RNN Inputs
"""

# Turn our x placeholder into a list of one-hot tensors:
# rnn_inputs is a list of num_steps tensors with shape [batch_size, num_classes]
x_one_hot = tf.one_hot(x, num_classes)
rnn_inputs = tf.unpack(x_one_hot, axis=1)
"""
Definition of rnn_cell

This is very similar to the __call__ method on Tensorflow's BasicRNNCell. See:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn_cell.py
"""
with tf.variable_scope('rnn_cell'):
    W = tf.get_variable('W', [num_classes + state_size, state_size])
    b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))

def rnn_cell(rnn_input, state):
    with tf.variable_scope('rnn_cell', reuse=True):
        W = tf.get_variable('W', [num_classes + state_size, state_size])
        b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))
    return tf.tanh(tf.matmul(tf.concat(1, [rnn_input, state]), W) + b)
"""
Adding rnn_cells to graph

This is a simplified version of the "rnn" function from Tensorflow's api. See:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn.py
"""
state = init_state
rnn_outputs = []
for rnn_input in rnn_inputs:
    state = rnn_cell(rnn_input, state)
    rnn_outputs.append(state)
final_state = rnn_outputs[-1]
"""
Predictions, loss, training step

Losses and total_loss are simlar to the "sequence_loss_by_example" and "sequence_loss"
functions, respectively, from Tensorflow's api. See:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/seq2seq.py
"""

#logits and predictions
with tf.variable_scope('softmax'):
    W = tf.get_variable('W', [state_size, num_classes])
    b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]
predictions = [tf.nn.softmax(logit) for logit in logits]

# Turn our y placeholder into a list labels
y_as_list = [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(1, num_steps, y)]

#losses and train_step
losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logit,label) for \
          logit, label in zip(logits, y_as_list)]
total_loss = tf.reduce_mean(losses)
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)
"""
Function to train the network
"""

def train_network(num_epochs, num_steps, state_size=4, verbose=True):
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        training_losses = []
        for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps)):
            training_loss = 0
            training_state = np.zeros((batch_size, state_size))
            if verbose:
                print("\nEPOCH", idx)
		print(epoch)
            for i in range(0, 1):
		X, Y=gen_data()
                tr_losses, training_loss_, training_state, _ = \
                    sess.run([losses,
                              total_loss,
                              final_state,
                              train_step],
                                  feed_dict={x:X, y:Y, init_state:training_state})
                training_loss += training_loss_
                #if step % 100 == 0 and step > 0:
                #    if verbose:
                #        print("Average loss at step", step,
                #              "for last 250 steps:", training_loss/100)
                #    training_losses.append(training_loss/100)
                #    training_loss = 0

    return training_losses
training_losses = train_network(1,num_steps, state_size=state_size)

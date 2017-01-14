import numpy as np
import tensorflow as tf
from random import shuffle

num_steps = 10
fracExamples=0.95

def readToLines(file):
    csvFile=open(file)
    lines=csvFile.read().splitlines()
    return lines 

def gen_data(size):

    xList=[]
    yList=[]

    tLines=readToLines('/home/willie/workspace/TensorFlow/data/LRBPI6065.csv')
    state=0
    tempXs=[]
    tempYs=[]

    maxX=[0]*13
    maxY=0
    
    amt=0
    totalY=0
    numY=0
    totalExpDist=0
    lastExpEst=0
    for tLine in tLines:
	amt+=1
	if amt>size and state==0:
		break;
	if tLine=='n':
		xList+=[tempXs]
		yList+=[tempYs]
		#print(tempYs)
		tempXs=[]
		tempYs=[]
		state=0
	else:
        	stringFloats=tLine.split(',')
        	floats=[]
        	for ind in range(0, len(stringFloats)):
                    val=float(stringFloats[ind])
		    floats+=[val]
                    if state%2==0 and val>maxX[ind]:
			maxX[ind]=val
		    elif state%2==1 and val>maxY:
			maxY=val;
		if state%2==0:
			tempXs+=[np.asarray(floats)]
			lastExpEst=floats[10]
		else:
			tempYs+=[np.asarray(floats)]
			totalY+=floats[0]*floats[0]
			totalExpDist+=(lastExpEst-floats[0])*(lastExpEst-floats[0])
			numY+=1
		state+=1

    xList+=[tempXs]
    yList+=[tempYs]
    print('avg y', totalY/numY)
    print('avg exp dist', totalExpDist/numY)
    #print(tempYs)
    tempXs=[]
    tempYs=[]
    state=0
    amt+=1

    for varSeq in range(0, len(xList)):
	for choiceInd in range(0, len(xList[varSeq])):
		yList[varSeq][choiceInd]/=maxY
		for inputInd in range(0, len(xList[varSeq][choiceInd])):
			xList[varSeq][choiceInd][inputInd]/=maxX[inputInd]

    #print(xList)

    rawX=[]
    rawY=[]

    for ind in range(0, len(xList)):
        for time in range(num_steps, len(xList[ind]), 1):
		#yield(xList[ind][time-num_steps:time], yList[ind][time])
		rawX+=[xList[ind][time-num_steps:time]]
		rawY+=[yList[ind][time]]

    X=np.asarray(rawX)
    Y=np.asarray(rawY)

    print('X', len(X))
    print('Y', len(Y))

    return X, Y

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

train_input, train_output=gen_data(50000000)

test_input = train_input[len(train_input)*fracExamples:]
test_output = train_output[len(train_input)*fracExamples:] 


 
train_input = train_input[:len(train_input)*fracExamples]
train_output = train_output[:len(train_output)*fracExamples] #till 10,000

print('train_input', len(train_input))
print('train_output', len(train_output))

data = tf.placeholder(tf.float32, [None, num_steps, len(train_input[0][0])])
target = tf.placeholder(tf.float32, [None, 1])

num_hidden = 4

print('num_hidden', num_hidden)

cell = tf.nn.rnn_cell.BasicRNNCell(num_hidden)

val, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)

val = tf.transpose(val, [1, 0, 2])
last = tf.gather(val, int(val.get_shape()[0]) - 1)

weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))

prediction = tf.nn.sigmoid(tf.matmul(last, weight) + bias)

cross_entropy = tf.reduce_sum((target-prediction)*(target-prediction))

optimizer = tf.train.AdamOptimizer() #tf.train.GradientDescentOptimizer(0.01)
minimize = optimizer.minimize(cross_entropy)

mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)

batch_size = 10000

print("train len", len(train_input))

no_of_batches = int(len(train_input)/batch_size)

print("no_of_batches", no_of_batches)

epoch = 1000
for i in range(epoch):
    ptr = 0
    shuffle_in_unison(train_input, train_output)
    for j in range(no_of_batches):
        inp, out = train_input[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]
        ptr+=batch_size

	#print('in', len(inp))
	#print('out', len(out))

        sess.run(minimize,{data: inp, target: out})

	#incorrect = sess.run(cross_entropy,{data: inp, target: out})
	#print('Error: ', incorrect)
    if i%5==0:
    	print "Epoch - ",str(i)
	incorrect = sess.run(cross_entropy,{data: test_input, target: test_output})
	print('Validation Error: ', incorrect/len(test_input))

print('rnn cell---------------------------------------------')

for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
	print("variable --------------------------------------------------")
	print(sess.run(variable))

sess.close()

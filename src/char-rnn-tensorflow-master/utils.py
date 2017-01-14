import codecs
import os
import collections
from six.moves import cPickle
import numpy as np

class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, encoding='utf-8'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding

        input_file = os.path.join(data_dir, "ptb.train.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        #if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
        print("reading text file")
        self.preprocess(input_file, vocab_file, tensor_file)
        #else:
        #print("loading preprocessed files")
        #self.load_preprocessed(vocab_file, tensor_file)
        self.create_batches()
        self.reset_batch_pointer()

    def preprocess(self, input_file, vocab_file, tensor_file):
        with codecs.open(input_file, "r") as f:
            data = f.read()
	print(len(data))
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
	numGreater=0
	numOccLess=0
	remove={}
	count=0
	self.vocab={}
        max=-1
	for key in data:
		self.vocab[key]=ord(key)
		if ord(key)>max:
			max=ord(key)
	print(numGreater, numOccLess)
        self.chars=self.vocab.keys()
        self.vocab_size = max+1
	print('vocab len', len(self.chars))
	with open(vocab_file, 'wb') as f:
            cPickle.dump(self.chars, f)
        self.tensor = np.array(list(map(self.vocab.get, data)))
	print('shape', self.tensor.shape)
        np.save(tensor_file, self.tensor)

    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.chars = cPickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.tensor = np.load(tensor_file)
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

    def create_batches(self):
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

        # When the data (tensor) is too small, let's give them a better error message
        if self.num_batches==0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.tensor
	print('xshape', xdata.shape)
        ydata = np.copy(self.tensor)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
	print('yshape', ydata.shape)

        #self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
	#self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1)

	self.x_batches = np.zeros((self.num_batches, self.batch_size, self.seq_length), dtype=np.int)
	self.y_batches = np.zeros((self.num_batches, self.batch_size, self.seq_length), dtype=np.int)

	for batchInd in range(self.num_batches):
		for exampleInd in range(self.batch_size):
			for charInd in range(self.seq_length):
				self.x_batches[batchInd][exampleInd][charInd]=xdata[charInd
					+exampleInd*self.seq_length
					+batchInd*self.batch_size*self.seq_length]
				self.y_batches[batchInd][exampleInd][charInd]=ydata[charInd
					+exampleInd*self.seq_length
					+batchInd*self.batch_size*self.seq_length]
	
	#xString=''
	#for batchInd in range(self.num_batches):
	#	for exampleInd in range(self.batch_size):
	#		for charInd in range(self.seq_length):
	#			xString+=chr(self.x_batches[batchInd][exampleInd][charInd])
	
	#for batchInd in range(len(self.x_batches)):
	#	for exampleInd in range(len(self.x_batches[batchInd])):
			#xString=''
			#yString=''
	#		for charInd in range(len(self.x_batches[batchInd][exampleInd])):
	#			xString+=chr(self.x_batches[batchInd][exampleInd][charInd])
				#yString+=chr(self.y_batches[batchInd][exampleInd][charInd])
			#print('xString', xString)	
			#print('yString', yString)	

	#text_file = open('/home/willie/workspace/TensorFlow/src/char-rnn-tensorflow-master/saveGRU3.128/saveNetRanks_nomod', "w")
	#text_file.write(xString)
	#text_file.close()

	xReshapedPart=xdata.reshape(self.batch_size, -1)
	xReshaped=np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
	for stringInd in range(len(xReshaped[0])):
		xString=''
		xReshapedString=''
		for ind in range(0, len(xReshaped[0][stringInd])):
		#	xString+=chr(xReshaped[0][stringInd][ind])
			xReshapedString+=chr(xReshapedPart[stringInd][ind])
		#print('xString', xString)
		print('xReshapedString', xReshapedString)

        


    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0

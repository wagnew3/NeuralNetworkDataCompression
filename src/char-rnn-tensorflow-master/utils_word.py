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

        input_file = os.path.join(data_dir, "input.txt")
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
        with codecs.open(input_file, "r", encoding=self.encoding) as f:
            data = f.read().split()
        counter = collections.Counter(data)
	#print(counter)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
	numGreater=0
	numOccLess=0
	remove={}
	count=0
	totalInc=0
	totalIgnoredChars=0
	plotStr='{'
	for key in count_pairs:
		if count<1024:
			numGreater+=1
			totalInc+=key[1]
			print(key[0], key[1])
			plotStr+=str(key[1])+', '	
		else:
			remove[key[0]]=key[0]
			totalIgnoredChars+=len(key[0])*key[1]
		count+=1
	removedData=[]
	#print(plotStr)
	for word in data:
		if word in remove:
			removedData+=[word]

	ignored_file = open('/home/willie/workspace/TensorFlow/src/char-rnn-tensorflow-master/saveGRU3.128_1024/ignoredWords_1024', "w")
	for item in removedData:
  		ignored_file.write(item.encode())
	ignored_file.close()

	print(numGreater, totalInc, numOccLess, 'totalIgnoredChars', totalIgnoredChars)
        self.chars, _ = zip(*count_pairs)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
	for key in self.vocab:
		if remove.has_key(key):
			self.vocab[key]=numGreater
        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.chars, f)
	self.vocab_size=numGreater+1
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
	print('yshape', ydata.shape)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1)


    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0

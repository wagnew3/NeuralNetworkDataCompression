from __future__ import print_function
import numpy as np
import tensorflow as tf

import argparse
import time
import os
import random
from six.moves import cPickle

import utils
import utils_word
from model import Model
from model2 import Model2

import array

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

rnntype='gru'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/willie/workspace/TensorFlow/data/enwiki8',
                       help='data directory containing input.txt')
    parser.add_argument('--save_dir', type=str, default='saveGRU7.128', #'saveGRU3.128.ReTrained',
                       help='directory to store checkpointed models')
    parser.add_argument('--rnn_size', type=int, default=128,
                       help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=7,
                       help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default=rnntype,
                       help='rnn, gru, or lstm')
    parser.add_argument('--batch_size', type=int, default=3000,
                       help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=100,
                       help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=200,
                       help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1000,
                       help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5.,
                       help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                       help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97,
                       help='decay rate for rmsprop')                       
    parser.add_argument('--init_from', type=str, default=None,#'/home/willie/workspace/TensorFlow/src/char-rnn-tensorflow-master/saveGRU3.256',
                       help="""continue training from saved model at this path. Path must contain files saved by previous training process: 
                            'config.pkl'        : configuration;
                            'chars_vocab.pkl'   : vocabulary definitions;
                            'checkpoint'        : paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)
                        """)
    args = parser.parse_args()


    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/willie/workspace/TensorFlow/data/enwiki8',
                       help='data directory containing input.txt')
    parser.add_argument('--save_dir', type=str, default='saveGRU3.128', #'saveGRU3.128.ReTrained',
                       help='directory to store checkpointed models')
    parser.add_argument('--rnn_size', type=int, default=128,
                       help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default=rnntype,
                       help='rnn, gru, or lstm')
    parser.add_argument('--batch_size', type=int, default=5000,
                       help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=100,
                       help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=1,
                       help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1000,
                       help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5.,
                       help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                       help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97,
                       help='decay rate for rmsprop')                       
    parser.add_argument('--init_from', type=str, default=None, #'/home/willie/workspace/TensorFlow/src/char-rnn-tensorflow-master/saveGRU3.512'
                       help="""continue training from saved model at this path. Path must contain files saved by previous training process: 
                            'config.pkl'        : configuration;
                            'chars_vocab.pkl'   : vocabulary definitions;
                            'checkpoint'        : paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)
                        """)
    args2 = parser.parse_args()

    train(args, args2)

word=False

def train(args, args2):
    if(word):
	data_loader = utils_word.TextLoader(args.data_dir, args.batch_size, args.seq_length)
    else:
    	data_loader = utils.TextLoader(args.data_dir, args.batch_size, args.seq_length)

    args.vocab_size = data_loader.vocab_size
    args2.vocab_size = data_loader.vocab_size
    
    # check compatibility if training is continued from previously saved model
    if args.init_from is not None:
        # check if all necessary files exist 
        assert os.path.isdir(args.init_from)," %s must be a a path" % args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"config.pkl")),"config.pkl file does not exist in path %s"%args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"chars_vocab.pkl")),"chars_vocab.pkl.pkl file does not exist in path %s" % args.init_from
        ckpt = tf.train.get_checkpoint_state(args.init_from)
        assert ckpt,"No checkpoint found"
        assert ckpt.model_checkpoint_path,"No model path found in checkpoint"

        # open old config and check if models are compatible
        with open(os.path.join(args.init_from, 'config.pkl')) as f:
            saved_model_args = cPickle.load(f)
        need_be_same=["model","rnn_size","num_layers","seq_length"]
        for checkme in need_be_same:
            assert vars(saved_model_args)[checkme]==vars(args)[checkme],"Command line argument and saved model disagree on '%s' "%checkme
        
        # open saved vocab/dict and check if vocabs/dicts are compatible
        with open(os.path.join(args.init_from, 'chars_vocab.pkl')) as f:
            saved_chars, saved_vocab = cPickle.load(f)
        assert saved_chars==data_loader.chars, "Data and loaded model disagree on character set!"
        assert saved_vocab==data_loader.vocab, "Data and loaded model disagree on dictionary mappings!"
        
    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(args, f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'wb') as f:
        cPickle.dump((data_loader.chars, data_loader.vocab), f)
    
    model = Model(args)
    
    #trainModel=Model2(args2)
    
    saver = tf.train.Saver(tf.all_variables())
    
    print("---------------------Models created")
    with tf.Session() as sess:
	#netOuts = open('/home/willie/workspace/TensorFlow/src/char-rnn-tensorflow-master/saveGRU3.512/netOuts.gz', 'w')
        tf.initialize_all_variables().run()
	
	loss=0        

	print(sess.run(model.initial_state))
        # restore model
	init=False
        if args.init_from is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)

	#print(sess.run(trainModel.initial_state))
        for e in range(args.num_epochs):
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
	    #sess.run(tf.assign(trainModel.lr, args.learning_rate * (args.decay_rate ** e)))
            data_loader.reset_batch_pointer()
            state = sess.run(model.initial_state)
	    #trainState=sess.run(trainModel.initial_state)
	    ranks=[]
            for b in range(data_loader.num_batches):
                start = time.time()
                x, y = data_loader.next_batch()
                feed = {model.input_data: x, model.targets: y}
		if rnntype=='gru':
			feed[model.initial_state]=state
		else:
                	for i, (c, h) in enumerate(model.initial_state):
                    		feed[c] = state[i].c
                    		feed[h] = state[i].h
		#probs, loss, state =sess.run([model.probs, model.cost, model.final_state], feed)
		probs, loss, state, _ =sess.run([model.probs, model.cost, model.final_state, model.train_op], feed)
		
		
		total=0
		totalInc=0
		numberInc=0
		numberWordsCorrect=0
		targets=np.reshape(y, [-1])
		i=0
		while i<probs.shape[0]:

			#count=0
			#numCorrect=-1
			#while count<1 and i+1+numCorrect<probs.shape[0]:
			#	prob=probs[i+1+numCorrect][targets[i+1+numCorrect]]
			#	count=np.where(probs[i+1+numCorrect]>prob)[0].shape[0]
			#	numCorrect+=1

			#count=0
			#if numCorrect>10:
			#	i+=numCorrect
			#	total+=numCorrect
			#	for j in range(numCorrect):
			#		ranks+=[count+249]
			#else:
			#	ranks+=[targets[i]]

			prob=probs[i][targets[i]]
			count=np.where(probs[i]>prob)[0].shape[0]
			if count<1000:
				ranks+=[count]
				totalInc+=count
				numberInc+=1
			else:
				ranks+=[targets[i]]		
			i+=1
		print(total, totalInc, numberInc, numberWordsCorrect)

		text=''
		#print(ranks)
#		for asciiInt in ranks:
#			text+=chr(asciiInt)
		#print(text)
		#print('probs.shape', probs.shape)

		#print(probs)
		#trainFeed = {trainModel.input_data: x, trainModel.targets: probs, trainModel.char_targets: y}
		#batchState=trainState
                #if rnntype=='gru':
		#	trainFeed[trainModel.initial_state]=trainState
		#else:
                #	for i, (c, h) in enumerate(trainModel.initial_state):
                #    		trainFeed[c] = trainState[i].c
                #    		trainFeed[h] = trainState[i].h

		#print('probs[0]', probs.shape)
		#train_loss=0		
		#trainProbs, train_loss=sess.run([trainModel.probs, trainModel.cost], trainFeed)
		#print('trainProbs[0]', trainProbs[0].shape, train_loss)
		#train_loss, chartrain_loss, trainState, _ = sess.run([trainModel.cost, trainModel.char_cost, trainModel.final_state, trainModel.train_op], trainFeed)

		#probs=sess.run([model.probs], feed)
		#print(probs[0].shape)
		#for prob in probs:
		#	prob[prob<1e-7]=0
		#	np.savetxt(netOuts, prob, delimiter=',', newline='\n', fmt='%.4e')

		end = time.time()

		#print(train_loss)
                
		#print("{}/{} (epoch {}), main train loss={:.3f}, train_loss = {:.3e}, time/batch = {:.3f}" \
                #    .format(e * data_loader.num_batches + b,
                #            args.num_epochs * data_loader.num_batches,
                #            e, loss, train_loss, end - start))
		print("{}/{} (epoch {}), main train loss={:.3f}, time/batch = {:.3f}" \
                    .format(e * data_loader.num_batches + b,
                            args.num_epochs * data_loader.num_batches,
                            e, loss, end - start))
		#print('chartrainloss', chartrain_loss)
                if (e * data_loader.num_batches + b) % args.save_every == 0\
                    or (e==args.num_epochs-1 and b == data_loader.num_batches-1): # save for the last result
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step = e * data_loader.num_batches + b)
                    print("model saved to {}".format(checkpoint_path))

	    text_file = open('/home/willie/workspace/TensorFlow/src/char-rnn-tensorflow-master/saveGRU3.256/saveNetRanks_comped'+str(e), "w")
	    text=''
	    for asciiInt in ranks:
		text+=chr(asciiInt)
	    text_file.write(text)
	    text_file.close()
	#np.savetxt(netOuts, saveArr, delimiter=',', newline='\n')
	#netOuts.close()
	    #totalMissed=0
	    #numberBatches=0
            #data_loader.reset_batch_pointer()
            #state2 = sess.run(model.initial_state)
            #for b2 in range(data_loader.num_batches):
	#	if random.uniform(0.0, 1.0)<200.0/data_loader.num_batches:
	#		numberBatches+=1
         #       	start = time.time()
          #      	x, y = data_loader.next_batch()
          #      	feed = {model.input_data: x, model.targets: y}
	#		if rnntype=='gru':

#				feed[model.initial_state]=state2
#			else:
 #               		for i, (c, h) in enumerate(model.initial_state):
  #                  			feed[c] = state2[i].c
   #                 			feed[h] = state2[i].h
	#		missed, state2 = sess.run([model.missedChars, model.final_state], feed)
	#		totalMissed+=missed
	#		if (b2)%10==0:
	#			print(totalMissed)
	#		
	 #   if numberBatches>0:
	#	print(totalMissed, numberBatches, totalMissed/numberBatches)

def word_correct(i, probs, targets):
	if targets[i]==9 or targets[i]==10 or targets[i]==11 or targets[i]==12 or targets[i]==13:
		ind=i+1
		nChars=0
		while ind<len(targets) and not (targets[ind]==9 or targets[ind]==10 or targets[ind]==11 or targets[ind]==12 or targets[ind]==13):
			nChars+=1
			if np.where(probs[ind]>probs[ind][targets[ind]])[0].shape[0]>0:
				return [False, -1]
			ind+=1
		if nChars>2:
			return [True, ind-1];
		else:
			return [False, -1];
	else:
		return [False, -1];
	

if __name__ == '__main__':
    main()

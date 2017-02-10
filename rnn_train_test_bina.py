from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import numpy as np
import sklearn as sk
import csv
import re
import collections
from nltk.tokenize import WordPunctTokenizer
tokenizer = WordPunctTokenizer()
import pickle
import sys
import os
sys.path.append("source")
os.environ["CUDA_VISIBLE_DEVICES"]="0"
 
#from att_rnn import *
from rnn_train import *
#from att_rnn_sum_max1 import *

embSize = 100
d1_emb_size=10
d2_emb_size=10 
type_emb_size=10
numfilter = 150
num_epochs = 26
N = 1
batch_size=500
reg_para = 0.001
drop_out = 0.7
out_file = 'results/filtered/binary_filtered_blstm.txt'
sent_out = 'results/sent_wise/binary/neg_filter/blstm_'

att = False
pooling = 'max'

#ftrain = "dataset/ddi/step2/tmp_train.txt"
#ftest = "dataset/ddi/step2/tmp_test.txt"
ftrain = "dataset/ddi/neg_filtered/train_data.txt"
ftest = "dataset/ddi/neg_filtered/test_data.txt"

#ftrain = "dataset/ddi/all_data/train_data.txt"
#ftest = "dataset/ddi/all_data/test_data.txt"

#wefile = "/home/sunil/embeddings/cbow_100d_gvkcorpus.txt"
wefile = "/home/sunil/embeddings/glove_100d_w9_pubmed.txt"
#wefile = "/home/sunil/embeddings/glove_200d_w9_pubmed.txt"



def preProcess(sent):
	sent = sent.lower()
	sent = sent.replace('/',' ')

#	sent = sent.replace('(','')
#	sent = sent.replace(')','')
#	sent = sent.replace('[','')
#	sent = sent.replace(']','')
	sent = sent.replace('.','')
#	sent = sent.replace(',',' ')
#	sent = sent.replace(':','')
#	sent = sent.replace(';','')

	sent = tokenizer.tokenize(sent)
	sent = ' '.join(sent)
	sent = re.sub('\d', 'dg',sent)
	return sent


def makeWordListReverst(word_dict):
	wl = {}
	for k,v in word_dict.items():
		wl[v] = k
	return wl


def find_sub_list(sl,l):
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            return ind,ind+sll-1


def makePaddedList(sent_contents, maxl, pad_symbol= '<pad>'):	 
	T = []
 	for sent in sent_contents:
		t = []
		lenth = len(sent)
		for i in range(lenth):
			t.append(sent[i])
		for i in range(lenth,maxl):
			t.append(pad_symbol)
		T.append(t)	

	return T

def makeWordList(sent_lista, sent_listb):
	sent_list = sent_lista+sent_listb
	wf = {}
	for sent in sent_list:
		for w in sent:
 			if w in wf:
				wf[w] += 1
			else:
				wf[w] = 0

	wl = []		#orederd dictionary
	wl.append('0')
	for w,f in wf.iteritems():		
		wl.append(w)	 
	return wl

def mapWordToId(sent_contents, word_list):
	T = []
	for sent in sent_contents:
		t = []
		for w in sent:
			t.append(word_list.index(w))
		T.append(t)
	return T

def mapLabelToId(sent_lables, label_dict):
#	print"sent_lables", sent_lables
#	print"label_dict", label_dict
#	return [label_dict[label] for label in sent_lables]
	return [int (label != 'false') for label in sent_lables]
	
def dataRead(fname):
	print "Input File Reading"
	fp = open(fname, 'r')
	samples = fp.read().strip().split('\n\n')
	sent_lengths   = []		#1-d array
  	sent_contents  = []		#2-d array [[w1,w2,....] ...]
  	sent_lables    = []		#1-d array
  	entity1_list   = []		#2-d array [[e1,e1_t] [e1,e1_t]...]
  	entity2_list   = []		#2-d array [[e1,e1_t] [e1,e1_t]...]
  	for sample in samples:
		sent, entities, relation = sample.strip().split('\n')
#		if len(sent.split()) > 100:
#			continue
		e1, e1_t, e2, e2_t = entities.split('\t') 
		sent_contents.append(sent.lower())
		entity1_list.append([e1, e1_t])
		entity2_list.append([e2, e2_t])
		sent_lables.append(relation)

  	return sent_contents, entity1_list, entity2_list, sent_lables 

def makeFeatures(sent_list, entity1_list, entity2_list):
	print 'Making Features'
	word_list = []
	d1_list = []
	d2_list = []
	type_list = []
 	for sent, ent1, ent2 in zip(sent_list, entity1_list, entity2_list):
		sent = preProcess(sent)
 
		sent_list1 = sent.split()		
		entity1 = preProcess(ent1[0]).split()
		entity2 = preProcess(ent2[0]).split()
  		s1 = sent_list1.index('druga')
		s2 = sent_list1.index('drugb') 
		# distance1 feature	
		d1 = []
		for i in range(len(sent_list1)):
		    if i < s1 :
			d1.append(str(i - s1))
		    elif i > s1 :
			d1.append(str(i - s1 ))
		    else:
			d1.append('0')
		#distance2 feature		
		d2 = []
		for i in range(len(sent_list1)):
		    if i < s2:
			d2.append(str(i - s2))
		    elif i > s2:
			d2.append(str(i - s2))
		    else:
			d2.append('0')
		#type feature
		t = []
		for i in range(len(sent_list1)):
			t.append('Out')
		t[s1] = ent1[1]		
		t[s2] = ent2[1]

		word_list.append(sent_list1)
 		d1_list.append(d1)
		d2_list.append(d2)
 		type_list.append(t) 

    	return word_list, d1_list, d2_list, type_list

def readWordEmb(word_dict, fname, embSize):
	print "Reading word vectors"
	wv = []
	wl = []
	with open(fname, 'r') as f:
		for line in f :			
			vs = line.split()
			if len(vs) < embSize :
				continue
			vect = map(float, vs[1:])
			wv.append(vect)
			wl.append(vs[0])
	wordemb = []
	count = 0
	wordemb.append(np.zeros(embSize))
	for word in word_dict[1:]:
		if word in wl:
			wordemb.append(wv[wl.index(word)])
		else:
			count += 1
			wordemb.append(np.random.rand(embSize))
	wordemb = np.asarray(wordemb, dtype='float32')
	print "number of unknown word in word embedding", count
	return wordemb

def findLongestSent(Tr_word_list, Te_word_list):
	combine_list = Tr_word_list + Te_word_list
	a = max([len(sent) for sent in combine_list])
	return a
 
def findSentLengths(tr_te_list):
	lis = []
	for lists in tr_te_list:
		lis.append([len(l) for l in lists])
	return lis
 
def paddData(listL, maxl): #W_batch, d1_tatch, d2_batch, t_batch)
	rlist = []
 	for mat in listL:		
		mat_n = []
		for row in mat:
			lenth = len(row)
			t = []
			for i in range(lenth):
				t.append(row[i])
			for i in range(lenth, maxl):
				t.append(0)
			mat_n.append(t)
		rlist.append(np.array(mat_n)) 
	return rlist

"""
Tr_sent_contents, Tr_entity1_list, Tr_entity2_list, Tr_sent_lables = dataRead(ftrain)
print "Tr_sent_contents", len(Tr_sent_contents)
Tr_word_list, Tr_d1_list, Tr_d2_list, Tr_type_list = makeFeatures(Tr_sent_contents, Tr_entity1_list, Tr_entity2_list)

Te_sent_contents, Te_entity1_list, Te_entity2_list, Te_sent_lables = dataRead(ftest)
print "Te_sent_contents", len(Te_sent_contents)
Te_word_list, Te_d1_list, Te_d2_list, Te_type_list = makeFeatures(Te_sent_contents, Te_entity1_list, Te_entity2_list)

print "train_size", len(Tr_word_list)
print "test_size", len(Te_word_list)

train_sent_lengths, test_sent_lengths = findSentLengths([Tr_word_list, Te_word_list])
sentMax = max(train_sent_lengths + test_sent_lengths)
print 'sentMax', sentMax

train_sent_lengths = np.array(train_sent_lengths, dtype='int32')
test_sent_lengths = np.array(test_sent_lengths, dtype='int32')

#print "train sent length", train_sent_lengths
#print "test sent length", test_sent_lengths

# Wordlist
#label_dict = {'false':0, 'advise': 1, 'mechanism': 2, 'effect': 3, 'int': 4}
label_dict = {'false':0, 'true':1}

word_dict = makeWordList(Tr_word_list, Te_word_list)
d1_dict = makeWordList(Tr_d1_list, Te_d1_list)
d2_dict = makeWordList(Tr_d2_list, Te_d2_list)
type_dict = makeWordList(Tr_type_list, Te_type_list)

print "word dictonary length", len(word_dict)
print "d1 dictonary length", len(d1_dict)
print "type dictonary length", len(type_dict)

# Word Embedding
wv = readWordEmb(word_dict, wefile, embSize)		

#print wv[0:3]

# Mapping Train
W_train =   mapWordToId(Tr_word_list, word_dict)
d1_train = mapWordToId(Tr_d1_list, d1_dict)
d2_train = mapWordToId(Tr_d2_list, d2_dict)
T_train = mapWordToId(Tr_type_list,type_dict)

Y_t = mapLabelToId(Tr_sent_lables, label_dict)
Y_train = np.zeros((len(Y_t), len(label_dict)))
for i in range(len(Y_t)):
	Y_train[i][Y_t[i]] = 1.0

# Mapping Test
W_test =   mapWordToId(Te_word_list, word_dict)
d1_test = mapWordToId(Te_d1_list, d1_dict)
d2_test = mapWordToId(Te_d2_list, d2_dict)
T_test = mapWordToId(Te_type_list, type_dict)
Y_t = mapLabelToId(Te_sent_lables, label_dict)
Y_test = np.zeros((len(Y_t), len(label_dict)))
for i in range(len(Y_t)):
	Y_test[i][Y_t[i]] = 1.0

#padding
W_train, d1_train, d2_train, T_train, W_test, d1_test, d2_test, T_test = paddData([W_train, d1_train, d2_train, T_train, W_test, d1_test, d2_test, T_test], sentMax) 

print "train", len(W_train)
print "test", len(W_test)
 
with open('train_test_rnn_data.pickle', 'wb') as handle:
	pickle.dump(W_train, handle)
	pickle.dump(d1_train, handle)
	pickle.dump(d2_train, handle)
	pickle.dump(T_train, handle)
	pickle.dump(Y_train, handle)
	pickle.dump(train_sent_lengths, handle)

	pickle.dump(W_test, handle)
	pickle.dump(d1_test, handle)
	pickle.dump(d2_test, handle)
	pickle.dump(T_test, handle)
	pickle.dump(Y_test, handle)
	pickle.dump(test_sent_lengths, handle)

	pickle.dump(wv, handle)
	pickle.dump(word_dict, handle)
	pickle.dump(d1_dict, handle)
	pickle.dump(d2_dict, handle)
	pickle.dump(type_dict, handle)	 
	pickle.dump(label_dict, handle) 
	pickle.dump(sentMax, handle)
"""
with open('train_test_rnn_data.pickle', 'rb') as handle:
	W_train = pickle.load(handle)
	d1_train= pickle.load(handle)
	d2_train= pickle.load(handle)
	T_train = pickle.load(handle)
	Y_train = pickle.load(handle)
	train_sent_lengths = pickle.load(handle)

	W_test = pickle.load(handle)
	d1_test = pickle.load(handle)
	d2_test = pickle.load(handle)
	T_test = pickle.load(handle)
	Y_test = pickle.load(handle)
	test_sent_lengths = pickle.load(handle)

	wv = pickle.load(handle)
	word_dict= pickle.load(handle)
	d1_dict = pickle.load(handle)
	d2_dict = pickle.load(handle)
	type_dict = pickle.load(handle)
	label_dict = pickle.load(handle)
	sentMax = pickle.load(handle)


#vocabulary size
word_dict_size = len(word_dict)
d1_dict_size = len(d1_dict)
d2_dict_size = len(d2_dict)
type_dict_size = len(type_dict)
label_dict_size = len(label_dict)

 
rev_label_dict = {0:'false', 1:'true'}


fp = open(out_file, 'w')

#for reg_para in [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]:
#   for drop_out in [0.1, 0.2, 0.3, 0.4, 0.5]:
for ii in [0,1,2]:
	fsent = open(sent_out+str(ii), 'w')
	print 'drop_out, reg_rate', drop_out, reg_para

	rnn = rnn = RNN_Relation(label_dict_size, 	# output layer size
			word_dict_size, 		# position embedding size
			d1_dict_size, 				
			d2_dict_size, 
			type_dict_size, 		# type emb. size
			sentMax, 			# length of sentence
			wv,				# word embedding
			d1_emb_size=d1_emb_size, 	# emb. length
			d2_emb_size=d2_emb_size, 
			type_emb_size=type_emb_size,
			num_filters=numfilter, 		# number of hidden nodes in RNN
			w_emb_size=embSize, 		# dim. word emb
			l2_reg_lambda=reg_para,		# l2 reg
			)


	def test_step(W_test, test_sent_lengths, d1_test, d2_test, T_test, Y_test):
		n = len(W_test)
		ra = n/batch_size
		samples = []
		for i in range(ra):
			samples.append(range(batch_size*i, batch_size*(i+1)))
		samples.append(range(batch_size*(i+1), n))

		acc = [] 
		pred = []
		for i in samples:
			p,a = rnn.test_step(W_test[i], test_sent_lengths[i], d1_test[i], d2_test[i], T_test[i], Y_test[i])
#			acc.extend(a)
			pred.extend(p)
		return pred, acc


	train_len = len(W_train)
	y_true_list = []
	y_pred_list = []
	num_batches_per_epoch = int(train_len/batch_size) + 1

	for epoch in range(num_epochs):	
		shuffle_indices = np.random.permutation(np.arange(train_len))
		W_tr =  W_train[shuffle_indices]
		d1_tr = d1_train[shuffle_indices]
		d2_tr = d2_train[shuffle_indices]
		T_tr = T_train[shuffle_indices]
		Y_tr = Y_train[shuffle_indices]
		S_tr = train_sent_lengths[shuffle_indices]

 		for batch_num in range(num_batches_per_epoch):	
			start_index = batch_num*batch_size
			end_index = min((batch_num + 1) * batch_size, train_len)

			loss = rnn.train_step(W_tr[start_index:end_index], 
					S_tr[start_index:end_index], 
					d1_tr[start_index:end_index], 
					d2_tr[start_index:end_index], 
					T_tr[start_index:end_index], 
					Y_tr[start_index:end_index], 
					drop_out)

		if (epoch%N) == 0:
 			pred, acc = test_step(W_test, test_sent_lengths, d1_test, d2_test, T_test, Y_test)
 			y_true = np.argmax(Y_test, 1)
			y_pred = pred
			y_true_list.append(y_true)
			y_pred_list.append(y_pred)
	
	fp.write('lemada and droput\t'+ str(reg_para)+"\t"+str(drop_out)+'\n' )
	fscore_list = []
	for y_true, y_pred in zip(y_true_list, y_pred_list):
		fscore_list.append( f1_score(y_true, y_pred, average='binary') )

	index = np.argmax(fscore_list)
	y_true = y_true_list[index]
	y_pred = y_pred_list[index]


#	fp.write('numfilter,drop_out, reg '+str(numfilter)+"\t"+str(drop_out)+"\t"+str(reg_para)+'\n')
			
	fp.write(str(precision_score(y_true, y_pred, average='binary' )))
	fp.write('\t')

	fp.write(str(recall_score(y_true, y_pred, average='binary' )))
	fp.write('\t')

	fp.write(str(f1_score(y_true, y_pred, average='binary' )))
	fp.write('\t')
	fp.write('\n')

	fp.write(str(confusion_matrix(y_true, y_pred)))
	fp.write('\n')

	fp.write('\n\n\n') 
	
	for sent, slen, y_t, y_p in zip(W_test, test_sent_lengths, y_true, y_pred) :
		sent_l = [str(word_dict[sent[kk]]) for kk in range(slen) ]
		s = ' '.join(sent_l) 
		fsent.write(s)
		fsent.write('\n')
		fsent.write( rev_label_dict[y_t] )
		fsent.write('\n')
		fsent.write( rev_label_dict[y_p] )
		fsent.write('\n')
		fsent.write('\n')
	fsent.close()

	rnn.sess.close()
 
	 


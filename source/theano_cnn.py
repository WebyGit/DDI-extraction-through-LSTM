import theano
import theano.tensor as T
import numpy as np 
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

srng = RandomStreams()
def rectify(X):
    return T.maximum(X, 0.)

def init_ones(size, name):
    return theano.shared(value=np.ones(size, dtype='float64'), name=name, borrow=True)

def init_zeros(size, name):
    return theano.shared(value=np.zeros(size, dtype='float64'), name=name, borrow=True)

def init_uniform(size, name):
    return theano.shared(value=np.asarray(np.random.uniform(low = -np.sqrt(6. / np.sum(size)), high = np.sqrt(6. / np.sum(size)), size=size), dtype='float64'), name=name, borrow=True)

def init_ortho(size, name):
    W = np.random.randn(max(size[0],size[1]),min(size[0],size[1]))
    u, s, v = np.linalg.svd(W)
    return theano.shared(value=u.astype('float64')[:,:size[1]], name=name, borrow=True)

def dropout(X, p=0.):
 #   if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    	return X

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

def Adam(cost, params, lr=0.0002, b1=0.1, b2=0.001, e=1e-8):
        updates = []
        grads = T.grad(cost, params)
        i = theano.shared(numpy.float64(0.))
        i_t = i + 1.
        fix1 = 1. - (1. - b1)**i_t
        fix2 = 1. - (1. - b2)**i_t
        lr_t = lr * (T.sqrt(fix2) / fix1)
        for p, g in zip(params, grads):
            m = theano.shared(p.get_value() * 0.)
            v = theano.shared(p.get_value() * 0.)
            m_t = (b1 * g) + ((1. - b1) * m)
            v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
            g_t = m_t / (T.sqrt(v_t) + e)
            p_t = p - (lr_t * g_t)
            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((p, p_t))
        updates.append((i, i_t))
        return updates

class CNN_Relation(object):

	def __init__(self, num_classes, seq_len, word_dict_size, pos_dict_size, chunk_dict_size, d1_dict_size, d2_dict_size, type_dict_size, wv, batch_size = 60, w_emb_size=50, d1_emb_size=10, d2_emb_size=10, pos_emb_size=5, chunk_emb_size=5, dep_emb_size=5, type_emb_size=5, filter_size=3, num_filters=80, l2_reg_lambda = 0.0):

		emb_size = w_emb_size + d1_emb_size + d2_emb_size 
		w  = T.lmatrix('x1')
		d1 = T.imatrix('x2')
		d2 = T.imatrix('x3')		 
		input_y = T.imatrix('y')
		dropout_keep_prob = T.iscalar('drop_param')

		W_wemb = theano.shared(value = np.asarray(wv, dtype='float64'), name='WE', borrow=True)	#(L,50)		 
		W_d1emb = init_uniform((d1_dict_size, d1_emb_size), 'W_d1emb')  			#(n,5)
		W_d2emb = init_uniform((d2_dict_size, d2_emb_size), 'W_d2emb')				#(m,5)
		#Embedding layer
		emb1 = W_wemb[w]									#(60,100,50)
		emb2 = W_d1emb[d1]									#(60,100,10)
		emb3 = W_d2emb[d2]									#(60,100,10)
		X = T.concatenate((emb1, emb2, emb3), axis=2)						#(60,100,70)
		X = T.reshape(X, (-1, 1, seq_len, emb_size) )						#(60,1,100,70)

		#Conv + Pooling layer
		filter_shape = [num_filters, 1, filter_size, emb_size]
		W_conv = init_uniform(filter_shape, 'W_conv')						#(80,1,3,70)
		l1a = rectify(conv2d(X, W_conv))							#(60,80,98,1)	
    		l1 = max_pool_2d(l1a, (seq_len-filter_size+1, 1))					#(60,80,1,1)
    		l1 = dropout(l1, dropout_keep_prob)							#(60,80,1,1)

		#Fully connected with sigmoid layer
		l1 = T.reshape(l1, (-1, num_filters) )							#(60,80)
		W_nn1 = init_uniform((num_filters, 3*num_classes), 'W_nn1')				#(80,15)
		b_nn1 = init_uniform((3*num_classes), 'b_nn1')						#(15)
		h1 = T.nnet.sigmoid(T.dot(l1, W_nn1)+b_nn1)						#(60,15)

		#Fully connected with softmax
		W_nn2 = init_uniform((3*num_classes, num_classes), 'W_nn2')				#(15,5)
		b_nn2 = init_uniform((num_classes), 'b_nn2')						#(5)
		h2 = T.nnet.softmax(T.dot(h1, W_nn2)+b_nn2)						#(60,5)

		#loss and prediction  
		losses = T.nnet.categorical_crossentropy(h2, input_y)
		loss = T.mean(losses)
		pred = T.argmax(h2, axis=1)
#		accuracy = T.mean(np.argmax(input_y, axis=1) == self.pred)
 		#Optimization staffs
		params = [W_wemb, W_d1emb, W_d2emb, W_conv, W_nn1, b_nn1, W_nn2, b_nn2]
		updates = RMSprop(loss, params, lr=0.001)

		self.train_step1 = theano.function(inputs=[w, d1, d2, input_y, dropout_keep_prob], outputs=loss, updates=updates, allow_input_downcast=True)
		self.test_step1 = theano.function(inputs=[w, d1, d2, dropout_keep_prob], outputs=pred, allow_input_downcast=True)


	def train_step(self, W_batch, pos_batch, chunk_batch, d1_batch, d2_batch, t_batch, y_batch):
		loss = self.train_step1(W_batch, d1_batch, d2_batch, y_batch, 0.5)
		return loss
		

	def test_step(self, W_batch, pos_batch, chunk_batch, d1_batch, d2_batch, t_batch, y_batch):
		predict  = self.test_step1(W_batch, d1_batch, d2_batch, 1.0)
		return predict 
			










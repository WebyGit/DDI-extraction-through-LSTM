import tensorflow as tf

class RNN_Relation(object):

	def __init__(self, num_classes, word_dict_size, d1_dict_size, d2_dict_size, type_dict_size, sentMax, wv, w_emb_size=50, d1_emb_size=5, d2_emb_size=5, dep_emb_size=5, type_emb_size=5, filter_sizes=4, num_filters=100, l2_reg_lambda = 0.0, pooling='max'):

		tf.reset_default_graph()
		emb_size = w_emb_size + d1_emb_size + d2_emb_size + type_emb_size  
#		emb_size = w_emb_size + type_emb_size  

		self.w  = tf.placeholder(tf.int32, [None, sentMax], name="x")
		self.d1 = tf.placeholder(tf.int32, [None, sentMax], name="x3")
		self.d2 = tf.placeholder(tf.int32, [None, sentMax], name='x4')
		self.type = tf.placeholder(tf.int32, [None, sentMax], name='x5')
		self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
		self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
		
		# Initialization
#		W_wemb =    tf.Variable(tf.random_uniform([word_dict_size, w_emb_size], -1.0, +1.0))
		W_wemb = tf.Variable(wv)
		W_d1emb =   tf.Variable(tf.random_uniform([d1_dict_size, d1_emb_size], -1.0, +1.0))
		W_d2emb =   tf.Variable(tf.random_uniform([d2_dict_size, d2_emb_size], -1.0, +1.0))
		W_typeemb = tf.Variable(tf.random_uniform([type_dict_size, type_emb_size], -1.0, +1.0))
 		
		# Embedding layer
		emb0 = tf.nn.embedding_lookup(W_wemb, self.w)				#word embedding
		emb3 = tf.nn.embedding_lookup(W_d1emb, self.d1)				#POS embedding
		emb4 = tf.nn.embedding_lookup(W_d2emb, self.d2)				#POS embedding
		emb5 = tf.nn.embedding_lookup(W_typeemb, self.type)			#POS embedding
 
		X = tf.concat(2, [emb0, emb3, emb4, emb5])
#		X = tf.concat(2, [emb0, emb5])
		X_expanded = tf.expand_dims(X, -1) 					#shape (?, 21, 80, 1)
		print"X_expanded", X_expanded.get_shape
		l2_loss = tf.constant(0.0)
		
		# CNN Layer
		pooled_outputs = []
		filter_shape = [filter_sizes, emb_size, 1, num_filters]
		W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")	#Convolution parameter
		b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")		#Convolution bias parameter
		conv = tf.nn.conv2d(X_expanded, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")

        	# Apply nonlinearity
		#h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu") 		#shape (?, 19, 1, 200)
		h = tf.nn.bias_add(conv, b) 					#shape (?, 19, 1, 200)
		sentMax = sentMax-filter_sizes+1				#new sent max
		h = tf.reshape(h, [-1, sentMax, num_filters])			#shape (?, 19, 200)
		print "h ", h.get_shape
		
		#Attention weight
		W_a1 = tf.get_variable("W_a1", shape=[num_filters, num_filters])	    	# 200x200
		tmp1 = tf.matmul(tf.reshape(h, shape=[-1, num_filters]), W_a1, name="Wy") 	# NMx200
		h = tf.reshape(tmp1, shape=[-1, sentMax, num_filters])		 		#NxMx200

 		M = tf.tanh(h)									# NxMx200
		W_a2 = tf.get_variable("W_a2", shape=[num_filters, 1]) 				# 200 x 1
		print "W_a2", W_a2.get_shape()
       		tmp3 = tf.matmul(tf.reshape(M, shape=[-1, num_filters]), W_a2)  		# NMx1
		print "tmp3", tmp3.get_shape()
		alpha = tf.nn.softmax(tf.reshape(tmp3, shape=[-1, sentMax], name="att"))	# NxM	
		self.ret_alpha = alpha

		#attention applying
		alpha = tf.expand_dims(alpha, 1) 						# Nx1xM
		print "alpha", alpha.get_shape()
		h2 =  tf.reshape(tf.batch_matmul(alpha, h, name="r"), shape=[-1, num_filters] )
		print 'h2', h2.get_shape()						
		# dropout layer	 
		h_drop = tf.nn.dropout(h2, self.dropout_keep_prob)			#(?, 100)

		# Fully connetected layer
		W = tf.Variable(tf.truncated_normal([num_filters, num_classes], stddev=0.1), name="W")
		b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
		l2_loss += tf.nn.l2_loss(W)
		l2_loss += tf.nn.l2_loss(b)
		scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
	
		# prediction and loss function
		self.predictions = tf.argmax(scores, 1, name="predictions")
		self.losses = tf.nn.softmax_cross_entropy_with_logits(scores, self.input_y)
        	self.loss = tf.reduce_mean(self.losses) + l2_reg_lambda * l2_loss

        	# Accuracy
        	self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
        	self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")	
 
		session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
		self.sess = tf.Session(config=session_conf)  

		self.optimizer = tf.train.AdamOptimizer(1e-2)

		self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
		self.global_step = tf.Variable(0, name="global_step", trainable=False)
		self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)

		self.sess.run(tf.initialize_all_variables())

	def train_step(self, W_batch, Sent_len, d1_batch, d2_batch, t_batch, y_batch, drop_out):
    		feed_dict = {
				self.w 		:W_batch,
				self.d1		:d1_batch,
				self.d2		:d2_batch,
				self.type	:t_batch,
				self.dropout_keep_prob: drop_out,
				self.input_y 	:y_batch
	    			}
   		_, step, loss, accuracy, predictions = self.sess.run([self.train_op, self.global_step, self.loss, self.accuracy, self.predictions], feed_dict)
    		print ("step "+str(step) + " loss "+str(loss) +" accuracy "+str(accuracy))
		return loss

	def test_step(self, W_batch, Sent_len, d1_batch, d2_batch, t_batch, y_batch):
    		feed_dict = {
				self.w 		:W_batch,
				self.d1		:d1_batch,
				self.d2 	:d2_batch,
				self.type	:t_batch,
				self.dropout_keep_prob:1.0,
				self.input_y 	:y_batch
	    		}
   		step, loss, accuracy, predictions = self.sess.run([self.global_step, self.loss, self.accuracy, self.predictions], feed_dict)
    		print "Accuracy in test data", accuracy
		return predictions, accuracy

	





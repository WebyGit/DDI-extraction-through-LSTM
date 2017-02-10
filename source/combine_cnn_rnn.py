import tensorflow as tf
import numpy as np


class RNN_Relation(object):
	def __init__(self, num_classes, word_dict_size, d1_dict_size, d2_dict_size, type_dict_size, sentMax, wv, w_emb_size=50, d1_emb_size=5, d2_emb_size=5, type_emb_size=5, num_filters=100, l2_reg_lambda = 0.02, pooling = 'max', filter_size=3):
		tf.reset_default_graph()
 		emb_size = w_emb_size + d1_emb_size + d2_emb_size + type_emb_size  

		self.sent_len = tf.placeholder(tf.int64, [None], name='sent_len')
		self.w  = tf.placeholder(tf.int32, [None, None], name="x")
 		self.d1 = tf.placeholder(tf.int32, [None, None], name="x3")
		self.d2 = tf.placeholder(tf.int32, [None, None], name='x4')
		self.type = tf.placeholder(tf.int32, [None, None], name='x5')
		self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
		self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

		# Initialization
		W_wemb =    tf.Variable(tf.random_uniform([word_dict_size, w_emb_size], -1.0, +1.0))
#		W_wemb = tf.Variable(wv)
 		W_d1emb =   tf.Variable(tf.random_uniform([d1_dict_size, d1_emb_size], -1.0, +1.0))
		W_d2emb =   tf.Variable(tf.random_uniform([d2_dict_size, d2_emb_size], -1.0, +1.0))
		W_typeemb = tf.Variable(tf.random_uniform([type_dict_size, type_emb_size], -1.0, +1.0))
		
		# embedding Layer
		emb0 = tf.nn.embedding_lookup(W_wemb, self.w)				# word embedding NxMx50
 		emb3 = tf.nn.embedding_lookup(W_d1emb, self.d1)				# POS embedding  NxMx5
		emb4 = tf.nn.embedding_lookup(W_d2emb, self.d2)				# POS embedding  NxMx5
		emb5 = tf.nn.embedding_lookup(W_typeemb, self.type)			# POS embedding  NxMX5

		X = tf.concat(2, [emb0, emb3, emb4, emb5])				# (N,M,130)
		print 'X', X.get_shape()
		
		# Bi-Recurrent Layer
		with tf.variable_scope('rnn'):
			cell_f = tf.nn.rnn_cell.LSTMCell(num_units=num_filters, state_is_tuple=True)
			cell_b = tf.nn.rnn_cell.LSTMCell(num_units=num_filters, state_is_tuple=True)
			outputs, states = tf.nn.bidirectional_dynamic_rnn(
										cell_fw	=cell_f, 
										cell_bw	=cell_b, 
										dtype	=tf.float32, 	
										sequence_length=self.sent_len, 
										inputs	=X
									)

			output_fw, output_bw = outputs						# NxMx100
			states_fw, states_bw = states
			print 'output_fw', output_fw.get_shape()

			h1_rnn = tf.concat(2, [output_fw, output_bw])				# NxMx200
			print 'h1_rnn', h1_rnn.get_shape()		 
	 		h1_rnn = tf.expand_dims(h1_rnn, -1)						# NxMx200x1
			pooled = tf.nn.max_pool(h1_rnn, 
						ksize=[1, sentMax, 1, 1], 
						strides=[1, 1, 1, 1], 
						padding='VALID', 
						name="pool"
						) # Nx1x200x1
			h2_rnn = tf.reshape(pooled, [-1, 2*num_filters])				# Nx200
			print 'h2_rnn', h2_rnn.get_shape()		

		with tf.variable_scope('cnn'):
			# Convolution Layer
			X_cnv = tf.expand_dims(X, -1) 							#(N, M, 130, 1)
			filter_shape = [filter_size, emb_size, 1, 2*num_filters]
			W_cnn = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")	  #Convolution parameter
			b_cnn = tf.Variable(tf.constant(0.1, shape=[2*num_filters]), name="b")		  #Convolution bias parameter
			conv = tf.nn.conv2d(X_cnv, 
						W_cnn, 
						strides=[1, 1, 1, 1], 
						padding="VALID", 
						name="conv") #(N, M, 1, 200)
			#h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu") 				#shape (N, M, 1, 200)
			h1_cnn = tf.nn.bias_add(conv, b_cnn) 						#shape (N, M-3-1, 1, 200)
			pool=tf.nn.max_pool(h1_cnn, 
						ksize=[1,sentMax-filter_size+1,1,1],
						strides=[1, 1, 1, 1],
						padding='VALID',
						name="pool")#N,1,1,200
			#h2_cnn = tf.reshape(pool, [-1, num_filters])					#shape (N,200)
			h2_cnn = tf.squeeze(pool)							#shape (N,200)
			print "h2_cnn ", h2_cnn.get_shape

		#concatenate RNN and CNN
		h2 = tf.concat(1,[h2_rnn, h2_cnn])		
		print "h2 ", h2.get_shape 	# (N,400)
		h2 = tf.tanh(h2)

		#fully connected layer
		W = tf.Variable(tf.truncated_normal([4*num_filters, num_classes], stddev=0.1), name="W")
		b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
		
		scores = tf.nn.xw_plus_b(h2, W, b, name="scores")				# 200x8
		print 'score', scores.get_shape()
		
		self.predictions = tf.argmax(scores, 1, name="predictions")
		losses = tf.nn.softmax_cross_entropy_with_logits(scores, self.input_y)
		self.loss = tf.reduce_mean(losses)  + l2_reg_lambda * (tf.nn.l2_loss(W) + tf.nn.l2_loss(b))
		
		self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")
		
		self.optimizer = tf.train.AdamOptimizer(1e-2)
		self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
		self.global_step = tf.Variable(0, name="global_step", trainable=False)
		self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)
		
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
		session_conf = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=False)

		self.sess = tf.Session(config=session_conf)  
		self.sess.run(tf.initialize_all_variables())

	def train_step(self, W_batch, Sent_len, d1_batch, d2_batch, t_batch, y_batch, drop_out):
		#Padding data 
    		feed_dict = {
				self.w 		:W_batch,
				self.d1		:d1_batch,
				self.d2		:d2_batch,
				self.type	:t_batch,
				self.sent_len 	:Sent_len,
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
				self.d2		:d2_batch,
				self.type	:t_batch,
				self.sent_len 	:Sent_len,
				self.dropout_keep_prob: 1.0,
				self.input_y 	:y_batch
	    			}
    		step, loss, accuracy, predictions = self.sess.run([self.global_step, self.loss, self.accuracy, self.predictions], feed_dict)

    		print "Accuracy in test data", accuracy
		return predictions, accuracy







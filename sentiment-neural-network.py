import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from create_sentiment_featuresets import create_feature_sets_and_labels
import numpy as np

train_x, train_y, test_x, test_y = create_feature_sets_and_labels('/Users/surthi/gitrepos/tensorflow/ml-data/pos.txt', '/Users/surthi/gitrepos/tensorflow/ml-data/neg.txt')
num_nodes_hl1 = 500
num_nodes_hl2 = 500
num_nodes_hl3 = 500

n_classes = 2
batch_size= 100

# x matrix size = height x width (None x 784). (784 = 28x28 image) 
x = tf.placeholder('float', [None, len(train_x[0])])
y = tf.placeholder('float')

def nn_model(data):
	hidden_layer_1 = {'weights': tf.Variable(tf.random_normal([len(train_x[0]), num_nodes_hl1])),
			'biases':tf.Variable(tf.random_normal([num_nodes_hl1]))}
	hidden_layer_2 = {'weights': tf.Variable(tf.random_normal([num_nodes_hl1, num_nodes_hl2])),
                        'biases':tf.Variable(tf.random_normal([num_nodes_hl2]))}
	hidden_layer_3 = {'weights': tf.Variable(tf.random_normal([num_nodes_hl2, num_nodes_hl3])),
                        'biases':tf.Variable(tf.random_normal([num_nodes_hl3]))}
	output_layer = {'weights': tf.Variable(tf.random_normal([num_nodes_hl3, n_classes])),
                        'biases':tf.Variable(tf.random_normal([n_classes]))}
	# (input_data * weights + biases)
	l1 = tf.add(tf.matmul(data, hidden_layer_1['weights']), hidden_layer_1['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']), hidden_layer_2['biases'])
        l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, hidden_layer_3['weights']), hidden_layer_3['biases'])
        l3 = tf.nn.relu(l3)

	output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']
	return output

def train_nn(x):
	prediction = nn_model(x)
	
	# define cost function. we are using cross entropy here
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
	
	# now we want to minimize the cost with the help of optimizer. We can use SGD, AdaGrad etc
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	n_epochs = 5

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

		# training the network
		for epoch in range(n_epochs):
			epoch_loss = 0

			i=0
			while (i<len(train_x)):
				start = i
				end = i + batch_size
				epoch_x = np.array(train_x[start, end])
				epoch_y = np.array(train_y[start, end])
				i += batch_size
				_, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
				epoch_loss += c
			print('Epoch', epoch, 'completed out of ', n_epochs, 'loss:', epoch_loss)
		
		# evaluate model accuracy
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy:', accuracy.eval({x:test_x, y: test_y}))

train_nn(x)

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle


#todo: script for data downloading from https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
print('Using  tf version=', tf.__version__, ' to implement logistic regression for iris dataset')

DEVICE = '/gpu:0'
IRIS_DATA_FILE = './data/iris.csv'
LEARNING_RATE = 0.005
BATCH_SIZE = 15
ITERATIONS = 3000

iris_data = pd.read_csv(IRIS_DATA_FILE)
print(iris_data.head())
#choosing iris-setosa and iris-versicolor for binary classification
iris_data = iris_data[:99]
iris_data = iris_data.replace('Iris-setosa',1)
iris_data = iris_data.replace('Iris-versicolor',2)
iris_data = shuffle(iris_data)
labels = iris_data[iris_data.columns[-1]]
data = iris_data.drop(iris_data.columns[-1], axis=1)
scaler = MinMaxScaler()
scaler.fit(data)
data = scaler.transform(data)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=101)
y_train = y_train.values.reshape(y_train.shape[0],1)
y_train = y_train.astype(np.float64)
y_test = y_test.values.reshape(y_test.shape[0],1)
y_test = y_test.astype(np.float64) 

#Variables and placeholders
W = tf.Variable(tf.random_normal(shape=[data.shape[1],1],  stddev=0.01, dtype=tf.float64), name='weights', dtype=tf.float64)
b = tf.Variable(tf.random_normal(shape=[1,1], stddev=0.01, dtype=tf.float64), name='bias', dtype=tf.float64)
x = tf.placeholder(tf.float64, shape=(None, X_train.shape[1]))
y = tf.placeholder(tf.float64, shape=(None, 1))


with tf.device(DEVICE):
	y_pred = tf.matmul(x, W) + b
	loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=y_pred)
	loss = tf.reduce_mean(loss)
	optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
	_predicted = tf.round(tf.sigmoid(y_pred))
	_correct = tf.cast(tf.equal(_predicted, y), dtype=tf.float64)
	accuracy = tf.reduce_mean(_correct)
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		n_batches = int(X_train.shape[0]/BATCH_SIZE)
		for step in range(ITERATIONS):
			indicies = np.random.randint(X_train.shape[0]-1, size=BATCH_SIZE)		
			X_train_batch = X_train[indicies, :]
			y_train_batch = y_train[indicies, :]
			sess.run(optimizer, feed_dict={x:X_train_batch, y:y_train_batch})
			if step%100==0:
				print('************************** On epoch : {} ***************************'.format(step))
				loss_result = sess.run(loss, feed_dict={x:X_train_batch, y:y_train_batch})
				train_accuracy = sess.run(accuracy, feed_dict={x:X_train, y:y_train})
				print(X_test.shape, y_test.shape)
				test_accuracy = sess.run(accuracy, feed_dict={x:X_test, y:y_test})
				print('Loss = {}, train accuracy = {}, test accuracy = {}'.format(loss_result,train_accuracy, test_accuracy))

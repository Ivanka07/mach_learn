import tensorflow as tf
import numpy as np
import pickle
import glob
import datetime

print('CIFAR-10 classification with tf version={}'.format(tf.__version__))

CIFAR_DIR = './data/cifar-10-batches-py/'
MODEL_DIR = './model/'
CLASSES_NUM = 10
LEARNING_RATE = 0.0001
BATCH_SIZE = 50
ITERATION_NUM = 1000000 
C1, C2, C3 = 32, 64, 96


X_train = []
y_train = []
X_test = []
y_test = []
image_counter = 0


def bias(shape):
	return tf.Variable(tf.random_normal(shape=shape,
				stddev=0.1, dtype=tf.float32), dtype=tf.float32, name='bias')

def weights(shape):
	return tf.Variable(tf.truncated_normal(shape=shape,
				stddev=0.1, dtype=tf.float32), dtype=tf.float32, name='weights')


def conv_layer(x, kernel_shape):
	output = x.shape[1] - kernel_shape[0] + 2*2 + 1
	output_shape = [None, output, output, kernel_shape[3]]
	print('Creating convolutional  layer with input shape=', x.shape, ' and output shape=', output_shape)
	W = weights(kernel_shape)
	b = bias([kernel_shape[3]])
	conv_op = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME') + b
	return tf.nn.relu(conv_op)


def max_pool(x, size=2):
	print('Pooling by choosing max value on input shape=', x.shape, ' and kernel size=', size)
	return tf.nn.max_pool(x,ksize=[1,size,size,1],strides=[1,size,size,1], padding='SAME')


def fully_connected(x, size):
	print('Creating fully connected layer with input shape=', x.shape, ' and output shape=', size , 'x1')
	in_size = int(x.shape[1])
	W = weights([in_size, size])
	b = bias([size])
	return tf.matmul(x,W) + b


def get_next_batch(batch_size, images, labels):
	global image_counter
	assert batch_size > 0
	assert len(images) and len(labels)
	start = image_counter
	end = image_counter + batch_size
	batch_images, batch_labels = images[start:end], labels[start:end]
	image_counter = (image_counter + batch_size) % len(images)
	return batch_images, batch_labels
	
	
def get_data(batch_name):
	files = glob.glob(CIFAR_DIR+batch_name)
	data = []
	for f in files:
		with open(f, 'rb') as file_object:
			data.append( pickle.load(file_object, encoding='bytes'))
	images = np.vstack([d[b'data'] for d in data])
	images = images.reshape(len(images),3,32,32)	
	images = images.transpose(0, 2, 3, 1).astype(np.float32)/255
	labels = np.hstack(d[b'labels']  for d in data) 	
	encoded_labels = np.zeros((len(labels), CLASSES_NUM))
	encoded_labels[range(len(labels)),labels] = 1
	return images, encoded_labels	

	
X_train, y_train = get_data('data_batch_*')
X_test, y_test = get_data('test_batch')
print('Loaded training images=', X_train.shape, '. Loaded training labels = ', y_train.shape)
print('Loaded test images=', X_test.shape, '. Loaded test labels = ', y_test.shape)
#i, l = get_next_batch(100, X_train, y_train)
x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y = tf.placeholder(tf.float32, shape=[None, 10])
hold_probability = tf.placeholder(tf.float32)
with tf.device('/gpu:0'):
	conv1_1 = conv_layer(x, kernel_shape=[3,3,3,C1])
	conv1_2 = conv_layer(conv1_1, kernel_shape=[3,3,C1,C1])
	conv1_3 = conv_layer(conv1_2, kernel_shape=[3,3,C1,C1])
	pool1 = max_pool(conv1_3)
	drop1 = tf.nn.dropout(pool1, keep_prob=hold_probability)

	conv2_1 = conv_layer(drop1, kernel_shape=[3,3,C1,C2])
	conv2_2 = conv_layer(conv2_1, kernel_shape=[3,3,C2,C2])
	conv2_3 = conv_layer(conv2_2, kernel_shape=[3,3,C2,C2])
	pool2 = max_pool(conv2_3)
	drop2 = tf.nn.dropout(pool2, keep_prob=hold_probability)

	conv3_1 = conv_layer(drop2, kernel_shape=[3,3,C2,C3])
	conv3_2 = conv_layer(conv3_1, kernel_shape=[3,3,C3,C3])
	conv3_3 = conv_layer(conv3_2, kernel_shape=[3,3,C3,C3])
	pool3 = tf.nn.max_pool(conv3_3,ksize=[1,8,8,1],strides=[1,8,8,1], padding='SAME')
	
	conv3_flat = tf.reshape(pool3, [-1, C3])

	drop3 = tf.nn.dropout(conv3_flat, keep_prob=hold_probability)
	

	fully_connected1 = tf.nn.relu(fully_connected(drop3, 1024))
	_fully_connected1 = tf.nn.dropout(fully_connected1, keep_prob=hold_probability)
	y_pred = fully_connected(_fully_connected1, 10)
	
	#define loss
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pred))
	optimize = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
	correct = tf.equal(tf.argmax(y_pred,1), tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
	
	saver = tf.train.Saver()
	
	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)) as sess:
		tf.summary.scalar('Cross entropy', cross_entropy)
		tf.summary.scalar('Accuracy', accuracy)
		merged = tf.summary.merge_all()
		logdir = 'tensorboard/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '/'
		writer = tf.summary.FileWriter(logdir, sess.graph)
		sess.run(tf.global_variables_initializer())
		
		for epoch in range(ITERATION_NUM):
			batch_x, batch_y = get_next_batch(BATCH_SIZE, X_train, y_train)
			summary, _ = sess.run([merged, optimize], feed_dict={x:batch_x, y:batch_y, hold_probability:0.5})
			writer.add_summary(summary, epoch)
			if epoch%10 == 0:
				print('On epoch={}: loss={}'.format(epoch, cross_entropy))
				X_test = X_test.reshape(10,1000,32,32,3)	
				y_test = y_test.reshape(10,1000,10)
				acc = np.mean([sess.run(accuracy, feed_dict={x:X_test[i], y:y_test[i], hold_probability:1.0}) for i in range(10)])	
				print('Accuracy={}'.format(acc))
	

			
		print('Saving a model')
		saver.save(sess, MODEL_DIR+'cifar_32_64_96.ckpt')

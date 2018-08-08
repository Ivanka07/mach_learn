import tensorflow as tf
import numpy as np
import pickle
import glob

print('CIFAR-10 classification with tf version={}', tf.__version__)

CIFAR_DIR = './data/cifar-10-batches-py/'
CLASSES_NUM = 10

def get_data():
	files = glob.glob(CIFAR_DIR+'data_batch_*')
	data = []
	for f in files:
		with open(f, 'rb') as file_object:
			data.append( pickle.load(file_object, encoding='bytes'))
	images = np.vstack([d[b'data'] for d in data])
	images = images.reshape(len(images),3,32,32)	
	images = images.transpose(0, 2, 3, 1).astype(float)/255
	labels = np.hstack(d[b'labels']  for d in data) 	
	encoded_labels = np.zeros((len(labels), CLASSES_NUM))
	encoded_labels[range(len(labels)),labels] = 1
	return images, encoded_labels	
get_data()

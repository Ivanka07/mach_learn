import tensorflow as tf
import numpy as np
import datetime
from training_data import *
from line_gan import *

Z_DIMENSION = 100
BATCH_SIZE = 30
G_LEARNING_RATE = 2e-4
D_LEARNING_RATE = .00005
EPOCHS = 500
NUM_HIDDEN = 3

if __name__== '__main__':
    training_data = generate_training_data(628)
    batches = get_batches(training_data, BATCH_SIZE)
    with tf.device('/gpu:0'):
        train_gan(BATCH_SIZE, EPOCHS, Z_DIMENSION, D_LEARNING_RATE, G_LEARNING_RATE, batches)


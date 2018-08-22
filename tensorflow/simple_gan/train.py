import tensorflow as tf
import numpy as np
import datetime
import sys
from training_data import *
from line_gan import *
'''
arg1 = [train | generate]
'''
argument = ['train', 'generate']
Z_DIMENSION = 100
BATCH_SIZE = 30
G_LEARNING_RATE = 2e-4
D_LEARNING_RATE = .00005
EPOCHS = 500
NUM_HIDDEN = 3

if __name__== '__main__':
    assert len(sys.argv) > 1
    arg = sys.argv[1]
    assert arg in argument, 'Please use anarguments: train to train a model and generate for model generation. You used = ' % arg
    if arg == 'train':
            training_data = generate_training_data(628)
            batches = get_batches(training_data, BATCH_SIZE)
            with tf.device('\gpu:0'):
                train_gan(BATCH_SIZE, EPOCHS, Z_DIMENSION, D_LEARNING_RATE, G_LEARNING_RATE, batches)
    elif arg=='generate':
        load_from_model('./models/line_gan_model.ckpt.meta')


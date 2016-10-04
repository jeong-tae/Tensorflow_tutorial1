# main
import tensorflow as tf
import numpy as np
import sys

from config import base_config as config
from models import ANN

from tensorflow.examples.tutorials.mnist import input_data

def main(_):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
    train_data, valid_data, test_data = mnist.train, mnist.validation, mnist.test

    nn_config = config(train_data)

    with tf.Session() as sess:
        model = ANN(nn_config, sess, train_data, valid_data, test_data)
        model.build_model()
        model.run()

if __name__ == '__main__':
    tf.app.run()

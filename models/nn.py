# practice for MNIST data

import tensorflow as tf
import numpy as np
import random
from tqdm import tqdm
from .ops import *

class ANN(object):
    def __init__(self, config, sess, train_data, valid_data, test_data):
        self.sess = sess
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.train_range = config.train_range
        
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.max_epoch = config.max_epoch

        self.x = tf.placeholder(tf.float32, [None, 784])
        self.y_ = tf.placeholder(tf.float32, [None, 10])

    def build_model(self):
        """
        Write down your code here for what you want to make.
        Fill in the missing opertion in this function.
        """
        
        self.loss = #...
        self.preds = #output of softmax
        self.opt = #optimizer

    def run_epoch(self, ops, data, is_train = True, desc = 'Train'):
        total_preds = []
        total_labels = []
        total_loss = []

        for step, (data_in, label_in) in tqdm(enumerate(self.data_iteration(data, is_train)), desc = desc):
            _, loss, preds = self.sess.run([ops, self.loss, self.preds],
                            feed_dict = {
                                    self.x: data_in,
                                    self.y_: label_in
                            })
            total_preds.append(preds)
            total_labels.append(label_in)
            total_loss.append(loss)

        total_loss = sum(total_loss)
        total_preds = np.concatenate(total_preds, axis = 0)
        total_labels = np.concatenate(total_labels, axis = 0)
        return total_loss, accuracy(total_preds, total_labels)

    def run(self):
        tf.initialize_all_variables().run()

        for i in range(self.max_epoch):
            train_loss, train_acc = self.run_epoch(self.opt, self.train_data)
            print(" [*] Epoch: %d, Train loss: %.3f, Train Acc: %.3f" % (i+1, train_loss, train_acc))
            valid_loss, valid_acc = self.run_epoch(tf.no_op(), self.valid_data, False)
            print(" [*] Epoch: %d, Validation loss: %.3f, Validation Acc: %.3f" % (i+1, valid_loss, valid_acc))
        test_loss, test_acc = self.run_epoch(tf.no_op(), self.test_data, False)
        print(" [*] Test loss: %.3f, Test Acc: %.3f" % (test_loss, test_acc))

    def data_iteration(self, data, is_train = True):
        data_range = None
        if is_train:
            data_range = self.train_range
            random.shuffle(data_range)
            
            batch_len = len(data_range) // self.batch_size

            for l in xrange(batch_len):
                b_idx = data_range[self.batch_size * l:self.batch_size * (l+1)]
                batch_xs, batch_ys = data.images[b_idx], data.labels[b_idx]

                yield batch_xs, batch_ys

        else:
            yield data.images, data.labels
            

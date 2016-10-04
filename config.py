



import numpy as np

class base_config(object):
    def __init__(self, train_data):
        self.train_range = np.array(range(len(train_data.labels)))
        self.batch_size = 32
        self.lr = 0.02
        self.max_epoch = 5

import numpy as np
class config(object):
    def __init__(self):
        self.train_data_path= 'data/train'
        self.valid_data_path= ''
        self.edg_path       = 'data/contuor'#edge of all the data
        self.label_path     = 'data/mask'#ground truth of all the data
        self.test_data_path = 'data/test'
        self.save_path      = 'output'
        self.model_path     = 'model/model-20.hdf5'
        self.train          = 1 #train:1 test:0
        self.cutoff         = 0.65 #the cutoff of the prediction map

        self.epoches        = 100
        self.batch_size     = 2
          

        self.optim_conf     = {
        'learning_rate':0.0001,
        'weight_decay':0.0001,
        'betas':(0.9, 0.999)
        }
        self.lr_scheduler   = {
        'gamma':0.96
        }

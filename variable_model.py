# a neural net class for evolution purposes

import random
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Activation, Dropout, Flatten, Reshape, UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D, Convolution2D, Deconvolution2D

class ModelInstance(object):
    """An instance of a Keras neural network for evolution"""
    def __init__(self, id):
        super(ModelInstance, self).__init__()
        self.id = id
        # dna is a list of options:
        # first element is always optimizer, loss, output activation
        # number of conv+pool layers
        # conv/pool kernels, kernel shapes, strides
        # number of dense layers
        # their width and activation function

    def create_random(self):
        # generate with random dna
        print("making a random network")
        optimizer = random.choice(['adam','rmsprop','adagrad'])
        loss = random.choice(['mse','mae','msle','mape','accuracy'])
        output_act = random.choice(['softmax','elu','selu','relu','sigmoid','exponential','linear'])
        hidden_act = random.choice(['softmax','elu','selu','relu','sigmoid','exponential','linear'])
        conv_layers = random.choice(list(range(1,7)))
        conv_filters = random.choice(list(range(5,40)))
        ker_x = random.choice(list(range(3,10)))
        ker_y = random.choice(list(range(3,50)))
        stride_x = random.choice(list(range(1,6)))
        stride_y = random.choice(list(range(1,26)))
        dense_layers = random.choice(list(range(1,8)))
        dense_sizes = random.choice(list(range(50,1500)))

        self.dna = [optimizer, loss, output_act, hidden_act, conv_layers,
                    conv_filters, ker_x, ker_y, stride_x, stride_y,
                    dense_layers, dense_sizes]
        self.build_model()


    def create_child(self, other_dna):
        # generate a child
        print("making babies")

    def build_model(self):
        # instantiate
        print("building model")
        print(self.dna)
        model = Sequential()
        for i in range(self.dna[3]):
            # add conv and pool

my_inst = ModelInstance(15)
my_inst.create_random()
# dna example:
# ['adam'|'rmsprop'|'adagrad',
#  'mse'|'mae'|'msle'|'mape'|'accuracy',
#  'softmax'|'elu'|'selu'\'relu'|'sigmoid'|'exponential'|'linear',
#  1-6, # number of conv+pool
#  5-40, # number of filters
#  3-10, # kernel x (history)
#  3-50, # kernel y (freq)
#  1-5, # stride x
#  1-25, # stride y
#  1-7, # dense layers
#  50-1500, # dense sizes
#  'softmax'|'elu'|'selu'\'relu'|'sigmoid'|'exponential'|'linear']

# class ModelGenerator(object):
#     """functions for generating new or child networks"""
#     def __init__(self):
#         super(ModelGenerator, self).__init__()
#         # self.arg = arg
#
#     def get_random(self):
#         # generate some random dna
#         optim

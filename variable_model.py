# a neural net class for evolution purposes

import random
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Activation, Dropout, Flatten, Reshape, UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D, Convolution2D, Deconvolution2D

class ModelInstance(object):
    """An instance of a Keras neural network for evolution"""
    def __init__(self, id, history_samples, n_freq_in, n_freq_out):
        super(ModelInstance, self).__init__()
        self.id = id
        self.history_samples = history_samples
        self.n_freq_in = n_freq_in
        self.n_freq_out = n_freq_out
        # dna is a list of options:
        # first element is always optimizer, loss, output activation
        # number of conv+pool layers
        # conv/pool kernels, kernel shapes, strides
        # number of dense layers
        # their width and activation function

    def set_dna(self, dna):
        # set dna, build model. should be run in a try loop.
        self.dna = dna
        self.build_model()

    def create_random(self):
        # generate with random dna
        optimizer = random.choice(['adam','rmsprop','adagrad'])
        loss = random.choice(['mse','mae','msle','mape','accuracy'])
        output_act = random.choice(['softmax','elu','selu','relu','sigmoid','exponential','linear'])
        hidden_act = random.choice(['softmax','elu','selu','relu','sigmoid','exponential','linear'])
        conv_act = random.choice(['softmax','elu','selu','relu','sigmoid','exponential','linear'])
        conv_layers = random.choice(list(range(1,5)))
        conv_filters = random.choice(list(range(10,32)))
        ker_x = random.choice([3,5,7])
        ker_y = random.choice([3,5,7,9,17,25])
        stride_x = int(ker_x/2)
        stride_y = int(ker_y/2)
        pool_size = random.choice([2,3])
        conv_dropout = random.uniform(0,1)
        dense_dropout = random.uniform(0,1)
        dense_layers = random.choice(list(range(0,4)))
        dense_sizes = random.choice(list(range(50,1500)))

        self.dna = {'optimizer': optimizer,
                    'loss': loss,
                    'output_act': output_act,
                    'hidden_act': hidden_act,
                    'conv_layers': conv_layers,
                    'conv_filters': conv_filters,
                    'conv_act': conv_act,
                    'ker_x': ker_x,
                    'ker_y': ker_y,
                    'stride_x': stride_x,
                    'stride_y': stride_y,
                    'pool_size': pool_size,
                    'conv_dropout': conv_dropout,
                    'dense_dropout': dense_dropout,
                    'dense_layers': dense_layers,
                    'dense_sizes': dense_sizes}
        self.build_model()


    def create_child(self, other_dna):
        # generate a child
        # returns its dna
        print("making babies")
        try:
            # this might fail, dna isn't perfect
            print("yolo XD")
        except Exception as ex:
            print(ex)

    def build_model(self):
        # instantiate
        model = Sequential()
        for i in range(self.dna['conv_layers']):
            # add conv and pool
            if i == 0:
                model.add(Convolution2D(self.dna['conv_filters'],
                                        input_shape=(1,self.history_samples,self.n_freq_in),
                                        kernel_size=(self.dna['ker_x'], self.dna['ker_y']),
                                        strides=(self.dna['stride_x'], self.dna['stride_y']),
                                        data_format='channels_first',
                                        activation=self.dna['conv_act']))
                model.add(MaxPooling2D(pool_size=self.dna['pool_size']))
                model.add(Dropout(self.dna['conv_dropout']))
            else:
                model.add(Convolution2D(self.dna['conv_filters'],
                                        kernel_size=(self.dna['ker_x'], self.dna['ker_y']),
                                        strides=(self.dna['stride_x'], self.dna['stride_y']),
                                        data_format='channels_first',
                                        activation=self.dna['conv_act']))
                model.add(MaxPooling2D(pool_size=self.dna['pool_size']))
                model.add(Dropout(self.dna['conv_dropout']))

        # now Flatten
        model.add(Flatten())

        # and add dense layers
        for i in range(self.dna['dense_layers']):
            model.add(Dense(self.dna['dense_sizes'],
                            activation=self.dna['hidden_act']))
            model.add(Dropout(self.dna['dense_dropout']))

        # add the output layer
        model.add(Dense(self.n_freq_out,
                        activation=self.dna['output_act'],
                        name='main_output'))
        model.compile(optimizer=self.dna['optimizer'],
                      loss=self.dna['loss'],
                      metrics=['mse', 'msle', 'accuracy', 'mae', 'mape'])
        print(model.summary())
        self.model = model


for j in range(100):
    thedna = {'pool_size': 3, 'dense_layers': 2, 'conv_layers': 2, 'dense_dropout': 0.745175810249754, 'conv_dropout': 0.9615585532255168, 'dense_sizes': 590, 'conv_act': 'exponential', 'output_act': 'exponential', 'stride_x': 1, 'ker_x': 3, 'conv_filters': 28, 'loss': 'mae', 'hidden_act': 'sigmoid', 'optimizer': 'rmsprop', 'ker_y': 9, 'stride_y': 4}
    try:
        my_inst = ModelInstance(15,30,2000,4000)
        my_inst.create_random()
        print("")
        print(my_inst.dna)
        print(my_inst.create_child(thedna))
        print("THAT'S A BEAUTIFUL BABY")
    except Exception as ex:
        # print("")
        # print(my_inst.dna)
        # print(ex)
        pass
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

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Dense

tf.keras.backend.set_floatx('float64')

class MLP(tf.keras.Model):

    def __init__(self, layers, use_bias_input=False):
        super(MLP, self).__init__()
        self.from_layer = 0
        self.seq = []
        input_size = layers[0]
        for layer, output_size in enumerate(layers[1:]):
            if layer == len(layers) - 2:
                self.seq.append(Dense(output_size,
                                      activation='softmax',
                                      input_shape=(input_size,),
                                      use_bias=use_bias_input))
            elif layer == 0:
                self.seq.append(Dense(output_size,
                                      activation='linear',
                                      input_shape=(input_size,),
                                      use_bias=use_bias_input))
                self.seq.append(tf.math.sigmoid)
            else:
                self.seq.append(Dense(output_size,
                                      activation='linear',
                                      input_shape=(input_size,)))
                self.seq.append(tf.math.sigmoid)
            input_size = output_size

    def set_from_layer(self, layer=0):
        self.from_layer = layer

    def call(self, x):
        for layer, fc in enumerate(self.seq):
            if layer >= self.from_layer:
                x = fc(x)
        return x

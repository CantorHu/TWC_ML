import tensorflow as tf
import numpy as np

class autoEncoderClass(object):
    def __init__(self, size, learning_rate=0.01, batch_size=256, Echo=30, num_ntr=256):
    self.LR = learning_rate
    self.batch = batch_size
    self.ECHO = Echo
    self.size = size
    self.num_set = []
    self.time_set = []
    self.num_ntr = num_ntr
    
    def autoEncoder(self):
        input_data = tf.placeholder("float", [None, self.size])

        self.num_set.append(self.num_ntr)

        encoder = tf.keras.layers.Dense(self.num_ntr, activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-5))(input_data)
        decoder = tf.keras.layers.Dense(self.size, activation='sigmoid')(encoder)

        pred = decoder
        true=input_data
        
        loss = tf.reduce_mean(tf.pow(pred - true, 2))
        optimizer = tf.train.RMSPropOptimizer(LR).minimize(loss)

        init = tf.global_variables_initializer()
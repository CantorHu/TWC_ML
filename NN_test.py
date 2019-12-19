import numpy as np
import tensorflow as tf
import pandas as pd
import os

def nnTest(input_shape,x_test):

    # This part needs to keep the same as nnTrain
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3),strides=(3, 3), activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128,activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(2,activation='softmax'))


    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])

    print(model.summary())

    model.load_weights('./saved_weights')

    predictions = model.predict(x_test)

    f = open('output.csv', 'w')
    f.write('Index,Label\n')
    for i, v in enumerate(predictions):
        f.write(i + ',' + str(np.argmax(predictions[i]))+"\n")
    print('predict all')
    f.close()
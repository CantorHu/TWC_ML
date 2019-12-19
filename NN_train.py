import numpy as np
import tensorflow as tf
import pandas as pd
import os

def nnTrain(input_shape,x_train,y_train,x_val,y_val):

    # This part need to be changed in practice
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

    model.summary()

    history = model.fit(x_train, y_train, epochs=20, batch_size=50, validation_data = (x_val,y_val))

    model.save('./saved_weights')

    print(history.history.keys())

    check = model.evaluate(x_val,y_val)
    print(model.metrics_names)

    print(check)
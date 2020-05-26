
from keras.layers import Dropout,Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
import tensorflow as tf
import tunings
import keras


print(tunings.filters)

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(filters=tunings.filters, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1))) 
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(filters=tunings.filters//2, kernel_size=2, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(tunings.filters*4, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))


model.compile(loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])





model.fit(x_train.reshape(60000, 28, 28, 1),
         y_train,
         batch_size=64,
         epochs=tunings.epochs)


# Evaluate the model on test set
score = model.evaluate(x_test.reshape(10000, 28, 28, 1), y_test, verbose=0)

acc = round(score[1] * 100, 3)

print(" Accuracy ", acc)

import os

if acc >= 98:
  os.system("echo 'true' > /accuracy.txt")
else:
  os.system("echo 'false' > /accuracy.txt")




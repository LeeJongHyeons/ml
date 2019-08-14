import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.datasets import mnist
from keras.preprocessing.image import img_to_array, array_to_img

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train=np.dstack([x_train] * 12)
x_test=np.dstack([x_test] * 12)

x_train = x_train.reshape(x_train.shape[0], 56,56,3)
x_test = x_test.reshape(x_test.shape[0], 56,56,3)

x_train = x_train.reshape(x_train.shape[0], 56, 56, 3).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 56, 56, 3).astype('float32') / 255

#y_train=np.dstack([y_train] * 12)
#y_test=np.dstack([y_test] * 12)

#y_train = y_train.reshape(y_train.shape[0], 56, 56, 3).astype('float32') / 255
#y_test = y_test.reshape(y_test.shape[0], 56, 56, 3).astype('float32') / 255

#print(x_train.shape)
#print(x_test.shape)

#print(y_train.shape)
#print(y_test.shape)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

#print(y_train.shape)
#print(y_test.shape)


from keras.applications import MobileNet
conv_base = MobileNet(weights='imagenet', include_top=False,input_shape=(56,56,3))

#conv_base.summary()

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras import layers, models

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(10, activation='sigmoid'))

# model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100, batch_size=50, shuffle=True, verbose=1, validation_data=(x_test, y_test))

loss, acc = model.evaluate(x_test, y_test)

print(loss, acc)





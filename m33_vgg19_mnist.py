# vgg19 훈련
'''
VGG16, VGG19차이: layer와 노드의 개수(16, 19), CNN, , MAXPOOIING 조합

'''
import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.preprocessing.image import img_to_array, array_to_img

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255

x_test = x_test.astype('float32') / 255

y_train = np_utils.to_categorical(y_train)

y_test = np_utils.to_categorical(y_test)

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1] * x_train.shape[2]))

x_test = x_test.reshape((x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

x_train = np.dstack([x_train] * 3)
x_test = np.dstack([x_test] * 3)

print(x_train.shape)
print(x_test.shape)

x_train  = x_train.reshape((x_train.shape[0], 28, 28, 3))
x_test  = x_test.reshape((x_test.shape[0], 28, 28, 3))

from keras.applications import VGG19
from keras.models import model
from keras.layers import Dense, Flatten, Dropout

conv_base = VGG19(weight='imagenet', include_top=False, input_shape=(48,48,3))

model = Sequential()
model.add(conv_base)
model.add(Dropout(0.3))
model.add(Dense(261), activation='relu')
model.add(Dense(156), activation='relu')
model.add(Dense(56), activation='relu')
model.add(Dropout(0.2))
model.add(Dense(45), activation='relu')
model.add(Dense(22), activation='relu')
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(10), activation='softmax')
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

model.fit(x_train ,y_train, epochs=50, batch_size=100, verbose=1)

print("score :", model.evaluate(x_test, y_test)[1])
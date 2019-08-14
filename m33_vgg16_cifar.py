from keras.datasets import cifar10
import numpy as np 
from keras.layers import Dense, Flatten, Dropout,Activation
from keras.applications import VGG16 ,VGG19, Xception, InceptionV3, ResNet50, MobileNet
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import model, Sequential
from keras.utils import np_utils

(cifar10_x_train, cifar10_y_train), (cifar10_x_test, cifar10_y_test) = cifar10.load_data()
#(cifar10_train_x, _), (cifar10_test_x, _) = cifar10.load_data()

x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

y_train = np_utils.to_categorical(cifar10_y_train)
y_test = np_utils.to_categorical(cifar10_y_test)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(323))
model.add(Dropout(0.5))
model.add(Dense(221))
model.add(Activation('softmax'))
model.summary()


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

model.fit(x_train, y_train, epochs=1, batch_size=256)

print("Score:", model.evaluate(x_test, y_test)[1])


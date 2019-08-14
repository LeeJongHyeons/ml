'''
Xception: keras 라이브러리의 창시자이자 최고 관리자인 Francois Chollet 자신 외에는 아무도 제한하지 않음

Xception은 표준 Inception 모듈을 분리 가능한 컨볼루션으로 대체하는 Inception 아키텍처의 확정

원본 간행물인 Xception: 깊이있는 분리 가능한 컨볼루션을 사용한 딥러닝(91MB)로 가장 작은 중량 직렬화를 자랑

'''
import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.datasets import mnist
from keras.preprocessing.image import img_to_array, array_to_img

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)
x_train=np.dstack([x_train] * 3)
x_test=np.dstack([x_test] * 3)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 3).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 3).astype('float32') / 255

# reshape 28 => 48 증폭
x_train = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in x_train])
x_test = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in x_test])

print(x_train.shape)
print(x_test.shape)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

from keras.applications import xception
conv_base = xception(weights='imagenet', include_top=False,
                  input_shape=(48,48,3))

# 모델 구성
from keras.models import Sequential
from keras.layers import Dense, Input, Flatten, Conv2D, MaxPooling2D

model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=200, verbose=1)

acc = model.predict(x_test)
print(acc)

from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping  
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import tensorflow as tf  
import numpy as np
import os

# CIFAR_10은 3채널로 구성된 32x32 이미지 60000장을 갖는다.
IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_COLS = 32

#상수정의
BATCH_SIZE = 128
NB_EPOCH = 20
NB_CLASSES = 10
VERBOSE = 1
VALIDATION_SPLIT = 0.2
OPTIM = RMSprop()

#데이터셋 불러오기
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
X_train, _, Y_train, _ = train_test_split(X_train, Y_train, train_size=300)

X_train = X_train[:300]
X_test = X_test[:300]
Y_train = Y_train[:300]
Y_test = Y_test[:300]

# X_train = X_train.reshape(X_train.shape[0], 32 * 32 * 3)
# X_test = X_test.reshape(X_test.shape[0], 32 * 32 * 3)

Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
print(Y_train.shape)  # (60000, 10)
print(Y_test.shape)   # (10000, 10)

print(X_train.shape) #(60000, 28, 28, 1) 
print(X_test.shape)  #(10000, 28, 28, 1)

#실수형으로 지정하고 정규화
X_train = X_train.astype('float32')
X_test = X_test.astype('float32') # 255로 나눴을때 0~1이 나옴
X_train /= 255  # 한개의 셀에 255값이 있음 
X_test /= 255

# X_train = X_train.reshape(X_train.shape[0], 32, 32, 3)
# X_test = X_test.reshape(X_test.shape[0], 32, 32, 3)

# ==from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Input
import numpy as np

def build_network(keep_prob=0.5, optimizer='adam'):
    inputs = Input(shape=(32,32,3),name='input')
    x = Conv2D(32, kernel_size=(3,3), activation='relu', name='hidden1')(inputs)
    x = Conv2D(32, kernel_size=(3,3), activation='relu', name='hidden2')(inputs)
    x = Dropout(keep_prob)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    x = Dropout(keep_prob)(x)
    prediction = Dense(10, activation='softmax', name = 'output')(x)
    model = Model(inputs=inputs, outputs = prediction)
    model.compile(optimizer=optimizer, loss= 'categorical_crossentropy',metrics=['accuracy'])
    return model

def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(10,50, 100, 500)
    return{"batch_size":batches, "optimizer":optimizers, "keep_prob":dropout}


def data_scaling(x_data):
    x_data = x_data.reshape(x_data.shape[0], 32*32*1)
    
#============================================================

#신경망 정의
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Conv2D 

# 모델 정의
model = Sequential()

model.add(Conv2D(32, kernel_size=(3,3), input_shape=(32,32,3), activation='relu'))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(74,activation='relu'))
model.add(Dense(51, activation='relu'))
model.add(Dense(46, activation='relu'))
model.add(Dense(95, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

#학습
model.compile(loss = 'categorical_crossentropy', optimizer = OPTIM, metrics = ['accuracy'])
print(X_train.shape,Y_train.shape)
history = model.fit(X_train, Y_train, batch_size=62, epochs=3, validation_data=(X_test, Y_test))

early_stopping = EarlyStopping(monitor='val_loss', patience=50)

data_generator = ImageDataGenerator(rotation_range=20, width_shift_range=0.02, height_shift_range=0.02, horizontal_flip=True)
                                    # 회전값(rotation), 넓이(width), 높이(height), 수평(horizontal_flip)


model.fit_generator(data_generator.flow(X_train, Y_train, batch_size=100), #generator
steps_per_epoch=len(X_train)//32,epochs=100,validation_data=(X_test, Y_test),shuffle=True, verbose=1) #, callbacks=callbacks

# X_train을 32로 나눔, 200번돌리고,X_test, Y_test에 검증, 실행하기전에 이미지를 만든뒤 실행

print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))

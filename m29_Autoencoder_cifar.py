from keras.datasets import cifar10
import numpy as np 

(x_train, _), (x_test, _) = cifar10.load_data()

x_train = x_train.astype("float32")/255
x_test = x_test.astype("float32")/255

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print(x_train.shape)  #(50000, 3072)
print(x_test.shape)   #(50000, 3072)

############################# 모델 구성 ###############################
from keras.models import Model
from keras.layers.core import Input, Dense, Dropout

encoding_dim = 32

input_img = Input(shape=(784,))

x = Dense(512, activation='relu')(input_img)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.05)
x = Dense(64, activation='relu')(x)
x = Dropout(0.05)
x = Dense(32, activation='relu')(x)


encode = Dense(encoding_dim, activation='relu')(input_img)

y = Dense(322, activation='relu')(encode)
y = Dense(221, activation='relu')(y)
y = Dense(110, activation='relu')(y)
y = Dense(58, activation='relu')(y)

Y = Dense(32, activation='relu')(y)


decode = Dense(784, activation='softmax')(y)

autoencoder = Model(input_img, decode)

encoder = Model(input_img, encode)

encode_input = Input(shape=(encoding_dim,))

decode_layer = autoencoder.layers[-1]

decoder = Model(encode_input, decode_layer(encode_input))

autoencoder.compile(optimizer = 'adadelta', loss='mse', metrics=['accuracy'])

history = autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, verbose=2, shuffle=True, validation_data=(x_test, x_test))

encode_img = encoder.predict(x_test)
decode_img = decoder.predict(encode_img)

print(encode_img)
print(decode_img)
print(encode_img.shape)
print(decode_img.shape)

#################### cifar10 이미지 출력 ##############
import matplotlib.pyplot as plt 

n = 0

plt.figure(figsize=(20,4))

for i in range(n):
    
    ax = plt.subplot(2, n, i +1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()


################ cifar10 그래프 그리기 #######################33
def plot_acc(history, title=None):
    if not isinstance(history, dict):
        history = history.history
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    if title is not None:
        plt.title(title)
    plt.ylabel('Accracy')
    plt.xlabel('Epoch')
    plt.legend(['Training data', 'Validation data'], loc=0)
    
def plot_loss(history, title=None):
    if not isinstance(history, dict):
        history = history.history
        
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    if title is not None:
         plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training data', 'Validation data'], loc=0)
    # plt.show()
    
plot_acc(history, '학습 경과에 따른 정확도 변화 추이')
plt.show()
plot_loss(history, '학습 경과에 따른 손실된 변화 추이')
plt.show()
    
loss, acc = autoencoder.evaluate(x_test, x_test)
print(loss, acc)
    

    

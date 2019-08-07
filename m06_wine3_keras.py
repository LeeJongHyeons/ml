# 랜포 95% => keras 96 이상
import pandas as pd 
import numpy as np
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split 
from keras.models import Sequential
from keras.utils import np_utils

# RandomForestClassifier: XG_Boost => keras 

# 데이터 읽기
wine = pd.read_csv(".\data\winequality-white.csv", sep=";", encoding="utf-8")

# 데이터를 레이블과 데이터로 분리
y = wine["quality"]
x = wine.drop("quality", axis=1)

# y 레이블 변경하기
newlist = []
for v in list(y):
    if v <= 4:
        newlist += [0]
    elif v <= 7:
        newlist += [1]

    else:
        newlist += [2]

y = newlist

y = np_utils.to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = Sequential()
model.add(Dense(256, input_dim=11, activation='relu'))
model.add(Dense(46))
model.add(Dense(87))
model.add(Dropout(0.7))
model.add(Dense(44))
model.add(Dropout(0.5))
model.add(Dense(84))
model.add(Dense(64))
model.add(Dense(53))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x, y, epochs=20000, batch_size=10)

loss, acc = model.evaluate(x_test, y_test)

y_pred = model.predict(x_test)
print("acc:", acc)


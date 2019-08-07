from sklearn.datasets import load_boston 
from sklearn.model_selection import train_test_split 
from keras.models import Sequential
from keras.layers import Dense, Dropout
import pandas as pd 
import numpy as np

boston = load_boston()
print(boston.data.shape)
print(boston.keys())
print(boston.target)
print(boston.target.shape)

x = boston.data 
y = boston.target 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = Sequential()
model.add(Dense(100, input_dim=11, activation='relu'))
model.add(Dense(46))
model.add(Dense(87))
model.add(Dropout(0,3))
model.add(Dense(15))
model.add(Dense(120))
model.add(Dense(36))
model.add(Dense(58))
model.add(Dense(62))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.summary()

from sklearn.linear_model import LinearRegression, Ridge, Lasso 

model = LinearRegression()
model.fit(x_train, y_train)
a = model.score(x_train, y_train)


model2 = Ridge()
model2.fit(x_train, y_train)
b = model2.score(x_train, y_train)

model3 = Lasso()
model3.fit(x_train, y_train)
c = model3.score(x_train, y_train)

print(a)
print(b)
print(c)




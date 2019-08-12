from keras.models import Sequential 
from keras.layers import Dense 
from sklearn.model_selection import train_test_split 
import numpy 
from keras.utils import np_utils
import pandas as pd 

wine = pd.read_csv(".\data\winequality-white.csv", sep=";", encoding="utf-8")

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

from sklearn.pipeline import Pipline 
from sklearn.preprocessing import MinMaxScaler 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


model = kerasClassfiter(build_fn=build_model, verbose=1)
hyperparameters = create_hyperparameters()

model = Sequential()
model.add(Dense(100, input_dim=11, activation='relu'))
model.add(Dense(46))
model.add(Dense(87))
model.add(Dense(120))
model.add(Dense(36))
model.add(Dense(58))
model.add(Dense(10, activation='softmax'))
model.summary()

def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = numpy.linspace(0.1, 0.5, 5)
    return{"batch_size":batches, "optimizer":optimizers, "keep_prob":dropout}

parameters = [
    {"n_estimators:": [200, 700, 800, 1100],
     "max_dept:": [600, 550, 460],
     "max_features": ['auto', 'sqrt', 'log2'],
     "min_samples_split": [0.1, 0.5, 0.6, 0.8]}
    
]

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=200, batch_size=10)

print("테스트 점수:", pipe.score(x_test, y_test))
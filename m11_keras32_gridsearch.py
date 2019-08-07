import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import classification_report 


iris_data = pd.read_csv("G:/study/data/iris2.csv", encoding="utf-8")

y = iris_data.loc[:, "Name"]
x = iris_data.loc[:,["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8, shuffle=True)


from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Input
import numpy as np
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

def _network(keep_prob=0.5, optimizer='adam'):
    inputs = Input(shape=(28,28,1),name='input')
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
    dropout = np.linspace(0.1, 0.5, 5)
    return{"batch_size":batches, "optimizer":optimizers, "keep_prob":dropout}

parameters = [
    {"n_estimators:": [200, 700, 800, 1100],
     "max_dept:": [600, 550, 460],
     "max_features": ['auto', 'sqrt', 'log2'],
     "min_samples_split": [0.1, 0.5, 0.6, 0.8]}
    
]

Kfold_cv = KFold(n_splits=5, shuffle=True)
clf =GridSearchCV(RandomForestClassifier(), parameters, cv=Kfold_cv)

clf.fit(x_train, y_train)
print("최적의 매개 변수=", clf.best_estimator_)

y_pred = clf.predict(x)
print("최종 정답률=", accuracy_score(y_test, y_pred))
last_score = clf.score(x_test, y_test)
print("정답률=4", last_score)
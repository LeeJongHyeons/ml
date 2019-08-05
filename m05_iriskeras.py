from keras.models import Sequential
from keras.layers import Dense
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score 


iris_data = pd.read_csv("G:/study/data/iris.csv", encoding='utf-8', names=['a','b','c','d','y'])
print(iris_data)
print(iris_data.shape)
print(type(iris_data))

y = iris_data.loc[:, "y"]

x = iris_data.loc[:,["a", "b","c","d"]]

#=============== OneHotEncoder, LabelEncoder ==================================
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
lenc = LabelEncoder()
lenc.fit_transform(y.values)
enc = OneHotEncoder()
y2 = enc.fit_transform(y.values.reshape(-1,1)).toarray()

print(y[0], " -- One Hot Encoding -->", y2[0])
print(y[50], " -- One Hot Encoding -->", y2[50])
print(y[100], " -- One Hot Encoding -->", y2[100])
# ==============================================================================

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8, shuffle=True)
# 테스트 전용 분리
clf = Sequential()
clf.add(Dense(5, input_shape=(4,), activation='relu'))
clf.add(Dense(8, activation='relu'))
clf.add(Dense(12, activation='relu'))
clf.add(Dense(6, activation='relu'))
clf.add(Dense(3, activation='softmax'))

clf,compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
clf.fit(x_test, y_test, epochs=300)

_, acc = clf.evaluate(x_test, y_test, batch_size=1)

y_pred = clf.predict(x_test)
print("acc = ", acc)
#print(y_pred)
print("결과:", np.round(y_pred))

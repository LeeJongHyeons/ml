from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score 
from keras.utils import np_utils
from sklearn.metrics import classification_report 
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
import pandas as pd 
import numpy as np

# 데이터 읽기
wine = pd.read_csv(".\data\winequality-white.csv", sep=";", encoding="utf-8")

y = wine["quality"]
x = wine.drop("quality", axis=1)


y = np_utils.to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# 2. 모델
#model = LinearSVC()
model = Sequential()
model.add(Dense(100, input_dim=11, activation='relu'))
model.add(Dense(46))
model.add(Dense(87))
model.add(Dropout(0,3))
model.add(Dense(15))
model.add(Dense(120))
model.add(Dense(36))
model.add(Dense(58))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.summary()

# 3. 실행
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=200, batch_size=10)
score = model.evaluate(x_test, y_test)


loss, acc = model.evaluate(x_test, y_test)

y_pred = model.predict(x_test)

print("\n정답률: ", accuracy_score(y_test, y_pred))
print("acc의 결과:", acc)
print("score의 결과:", score)

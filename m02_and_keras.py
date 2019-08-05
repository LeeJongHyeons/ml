from sklearn.svm import LinearSVC, SVC         #LinearSVC : 선형회귀
from sklearn.metrics import accuracy_score #metrics: 정확도
import numpy as np  
from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터
x_data = [[0,0],[1,0],[0,1],[1,1]]
y_data = [0,0,0,1]

# 2. 모델
#model = LinearSVC()
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(6,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(23,activation='relu'))
model.add(Dense(34,activation='relu'))
model.add(Dense(15,activation='relu'))
model.add(Dense(12,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(26,activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 실행
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_data, y_data, epochs=100)

# 4. 평가
x_test = [[0,0],[1,0],[0,1],[1,1]]
_, acc = model.evaluate(x_test, y_data, batch_size=1)
y_predict = model.predict(x_test)

print(x_test, "의 예측결과: ", y_predict)
print("acc = ", acc)

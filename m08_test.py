from sklearn.linear_model import LinearRegression 
import pandas as pd 
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 
from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np
# 기온 데이터 읽기
df = pd.read_csv("G:/study/data/tem10y.csv", encoding="utf-8")
# 데이터를 학습 전용과 테스트 전용으로 분리
train_year = (df["연"]<= 2015)
test_year = (df["연"]>= 2016)
interval = 6
# 과거 6일의 데이터를 기반으로 학습할 데이터 만들기
# ========================= split 작업 ============================
def make_data(data):
    x = [] # 학습 데이터
    y = [] # 결과
    temps = list(data["기온"])
    for i in range(len(temps)):
        if i < interval: continue
        y.append(temps[i])
        xa = []
        for p in range(interval):
            d = i + p - interval
            xa.append(temps[d])
        x.append(xa)
    return (np.array(x), np.array(y))
# =================================================================
train_x, train_y = make_data(df[train_year])
test_x, test_y = make_data(df[test_year])


train_x = train_x.reshape((train_x.shape[0],train_x.shape[1],1))
test_x = test_x.reshape((test_x.shape[0],test_x.shape[1],1))

#print(type(train_x))


#train_x = numpy.array(train_x)
#train_y = numpy.array(train_y)
#test_x = numpy.array(test_x)
#test_y = numpy.array(test_y)

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

# 직선 회귀 분석
model = Sequential()
model.add(LSTM(64, input_shape=(6,1), return_sequences=True))

#model.add(Dense(3, input_dim=6, activation='relu'))
model.add(LSTM(6))
model.add(Dense(16, activation='relu'))
model.add(Dense(36, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(train_x, train_y, epochs=300)   

_, mse = model.evaluate(test_x, test_y)

pre_y = model.predict(test_x) 

print("mse:", mse)

from sklearn.metrics import r2_score

r2_y_predict = r2_score(test_y, pre_y)
print("R2 :", r2_y_predict)
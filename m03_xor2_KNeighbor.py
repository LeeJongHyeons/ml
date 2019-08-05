from sklearn.svm import LinearSVC, SVC          #LinearSVC : 선형회귀
from sklearn.metrics import accuracy_score      #metrics: 정확도
from sklearn.neighbors import KNeighborsClassifier

# 1. 데이터
x_data = [[0,0],[1,0],[0,1],[1,1]]
y_data = [0,1,1,0]
# 2. 모델
model = KNeighborsClassifier(n_neighbors=1)
# 3. 실행
model.fit(x_data, y_data)
# 4. 평가
x_test = [[0,0],[1,0],[0,1],[1,1]]
y_predict = model.predict(x_test)

print(x_test, "의 예측결과: ", y_predict)
print("acc = ", accuracy_score([0,1,1,0],y_predict))

#[[0, 0], [1, 0], [0, 1], [1, 1]] 의 예측결과:  [0 1 1 0]
#acc =  1.0

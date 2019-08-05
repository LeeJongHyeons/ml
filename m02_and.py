from sklearn.svm import LinearSVC          #LinearSVC : 선형회귀
from sklearn.metrics import accuracy_score #metrics: 정확도

# 1. 데이터
x_data = [[0,0],[1,0],[0,1],[1,1]]
y_data = [0,0,0,1]
# 2. 모델
model = LinearSVC()
# 3. 실행
model.fit(x_data, y_data)
# 4. 평가
x_test = [[0,0],[1,0],[0,1],[1,1]]
y_predict = model.predict(x_test)

print(x_test, "의 예측결과: ", y_predict)
print("acc = ", accuracy_score([0,0,0,1],y_predict))

#  결과
#[[0, 0], [1, 0], [0, 1], [1, 1]] 의 예측결과:  [0 0 0 1]
#acc = 1.0

# 딥러닝과 차이점: svc모델의 최적화된 값을 함수화 시킴
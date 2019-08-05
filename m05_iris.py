# Linear Regression, Ridge(l2), Lasso(l1)
import pandas as pd  
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score 

iris_data = pd.read_csv("G:/study/data/iris.csv", encoding='utf-8', names=['a','b','c','d','y'])
print(iris_data)
print(iris_data.shape)
print(type(iris_data))

# 붓꽃 데이터를 레이블과 입력 데이터로 분리
y = iris_data.loc[:, "y"]
x = iris_data.loc[:,["a", "b","c","d"]]

#print(x.shape) #(150,4)
#print(y.shape) #(150,)

# 학습 전용과 테스트 전용 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8, shuffle=True)

#print(x_test.shape) #(105, 4)
#print(y_test.shape) #(30, 4)

# 학습
#clf = SVC # 0.9
#clf = KNeighborsClassifier(n_neighbors=1) # 0.966666
clf = LinearSVC()
clf.fit(x_train, y_train)

# 평가
y_pred = clf.predict(x_test)
print("정답률: ", accuracy_score(y_test, y_pred)) # 0.933 ~ 1.0



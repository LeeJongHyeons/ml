# 아래 gridsearchcv 코드를 randomsearchcv 코드로 변경
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC, KerasClassifier
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV 
import warnings

iris_data = pd.read_csv("G:/study/data/iris2.csv", encoding="utf-8")

y = iris_data.loc[:, "Name"]
x = iris_data.loc[:,["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]

warnings.filterwarnings('ignore')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8, shuffle=True, verbose=1)

# 그리드 서치에서 사용할 매개 변수 --- (*1)
parameters = [
   {"C": [1, 10, 100, 1000], "kernel":["linear"]},
   {"C": [1, 10, 100, 1000], "Kernel":["rbf"], "gamma":[0.001, 0.0001]},
   {"C": [1, 10, 100, 1000], "Kernel":["sigmoid"], "gamma": [0.001, 0.0001]}
]
# 그리드 서치 ---(*2)
Kfold_cv = KFold(n_splits=5, shuffle=True)
clf =GridSearchCV(KerasClassifier(), parameters, cv=Kfold_cv)

clf.fit(x_train, y_train)
print("최적의 매개 변수=", clf.best_estimator_)

# 최적의 매개 변수로 평가 --- (*3)
y_pred = clf.predict(x_test)
print("최종 정답률=", accuracy_score(y_test, y_pred))
last_score = clf.score(x_test, y_test)
print("최종 정답률=", last_score)
# last_score = clf.score(x_test, y_test)
# print("최종 정답률 =", last_score)

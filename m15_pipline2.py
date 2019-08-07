import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators # 측정평가
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC 
import numpy as np 

# 붓꽃 데이터 읽어 들이기
iris_data = pd.read_csv("G:/study/data/iris2.csv", encoding="utf-8")

# 붗꽃 데이터를 레이블과 입력 데이터로 분리하기
y = iris_data.loc[:, "Name"]
x = iris_data.loc[:,["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, train_size = 0.8, shuffle = True)

from sklearn.pipeline import Pipline 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

parameters = [{"SVM_C": [1, 10, 100, 1000], "kernel":["linear","rbf","sigmoid"], "gamma": [0.001, 0.0001]}]

kfold_cv = KFold(n_splits=5, shuffle=True)
clf = GridSearchCV(RandomizedSearchCV(estimator=pipe, param_distributions=param_dist, n_iter=n_iter_search, cv=5, iid=False))

pipe = Pipline([("scler", MinMaxScaler()), ('svm', SVC())]) # SVC모델앞에 fit 전부터 전처리 처리를 하고, 작업을 수행
#from sklearn.pipeline import make_pipeline
#pipe = make_pipeline(MinMaxScaler(), SVC(C=100))

pipe.fit(x_train, y_train)

print("테스트 점수:", pipe.score(x_test, y_test))

y_pred = clf.predict(x_test)
print("최종 정답률:", accuracy_score(y_test, y_pred))
last_score = clf.score(x_test, y_test)
print("최종 정답률: ", last_score)


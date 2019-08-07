import pandas as pd 
import warnings
from sklearn.model_selection import train_test_split 
from sklearn.svm import LinearSVC,SVC 
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV 
from sklearn.neighbors import KNeighborsClassifier

iris_data = pd.read_csv("G:/study/data/iris2.csv", encoding="utf-8")

y = iris_data.loc[:, "Name"]
x = iris_data.loc[:,["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]

warnings.filterwarnings('ignore')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8, shuffle=True)

parameters = [
    {"n_neighbors": [1, 100, 300, 500]}
 ]

Kfold_cv = KFold(n_splits=5, shuffle=True)
clf =GridSearchCV(KNeighborsClassifier(), parameters, cv=Kfold_cv)

clf.fit(x, y)
print("최적의 매개 변수=", clf.best_estimator_)

y_pred = clf.predict(x)
print("최종 정답률=", accuracy_score(y_test, y_pred))
last_score = clf.score(x, y)
print("최종 정답률=", last_score)
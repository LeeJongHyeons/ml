'''
결정 트리의 단점: 과대 적합
장점: 전처리가 필요

m06_wine3.py
-----------------------
m25_DT.PY
m25_RF.PY
m25_GB.PY
m25_XGB.PY
-----------------------
RandomSearch, k-fold, cv
'''
from sklearn.tree import DecisionTreeClassifier 
from sklearn.datasets import load_breast_cancer 
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score 


cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

parameters = [
    {"max_depth": [10, 200, 500, 1100]},
    {"min_samples_split": [60, 150, 300]}, 
    {"max_sampels_split": [10, 20, 30]},
    {"min_weight_fraction_leaf":[15,25]},
    {"min_impurity_split":[55]}
]

kfold_cv =KFold(n_splits=5, shuffle=True)

search = RandomizedSearchCV(DecisionTreeClassifier(), parameters, cv=kfold_cv, n_jobs=-1)
search.fit(x_train, y_train)
score = score.search(x_test, y_test)

y_pred = search.predict(x_test)
print("최대 정답률:", accuracy_score(y_test, y_pred))
last_score = search.score(x_test, y_test)
print("최종 정답률은:", last_score)

'''

for(name, algorithm) in allAlgorithms:
    clf =algorithm()

    if hasattr(clf, "score"):

        scores = cross_val_score(clf, x_test, y_test, cv=kfold_cv)
        print(name, "의 정답률")
        print(scores)

'''
from sklearn.linear_model import LinearRegression 
import pandas as pd 
import numpy as np 
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer 

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
    return (x, y)
# =================================================================
train_x, train_y = make_data(df[train_year])
test_x, test_y = make_data(df[test_year])

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

search = RandomizedSearchCV(RandomForestClassifier(), parameters, cv=kfold_cv, n_jobs=-1)
search.fit(x_train, y_train)
score = score.search(x_test, y_test)

y_pred = search.predict(x_test)
print("최대 정답률:", accuracy_score(y_test, y_pred))
last_score = search.score(x_test, y_test)
print("최종 정답률은:", last_score)

print("특성 중요도:\n", search.feature_importances_) 

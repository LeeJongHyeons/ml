import numpy as np   
import pandas as pd   
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split 
                                            
cancer = load_breast_cancer() # 분류 데이터 모델

x_train, x_test, y_train, y_test = train_test_split(cancer.data,cancer.target, stratify=cancer.target, random_state=66)

training_accuracy = []
test_accuracy = []

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
neighbors_setting = range(1, 11)

for n_neighbors in neighbors_setting:
    clf =KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(x_train, y_train)

    training_accuracy.append(clf.score(x_test, y_test))

clf.fit(x_train, y_train)

print("테스트 예측:".format(clf.predict(x_test)))
print("테스트 상태 정확도: {:.2f}".format(clf.score(x_test, y_test)))





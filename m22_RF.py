from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer 
from sklearn.model_selection import train_test_split 

cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

tree = RandomForestClassifier(n_estimators=10, n_jobs=-1, random_state=0, max_features=4)
tree.fit(x_train, y_train)
print("훈련 세트 정확도 : {:.3f}".format(tree.score(x_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(tree.score(x_test, y_test)))

# n_estimators: 클수록 좋다. 단점: 메모리 많이 차지, 기본값: 100
# n_jobs =-1: cpu 병렬처리
# max_features: 기본값

#tree = RandomForestClassifier(max_depth=5, random_state=0) #depth: 깊이
#tree.fit(x_train, y_train)
#print("훈련 세트 정확도 : {:.3f}".format(tree.score(x_train, y_train)))
#print("테스트 세트 정확도: {:.3f}".format(tree.score(x_test, y_test)))

print("특성 중요도:\n", tree.feature_importances_) 

import matplotlib.pyplot as plt 
import numpy as np  

def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("특성 중요도")
    plt.ylabel("특성")
    plt.ylim(-1, n_features)

plot_feature_importances_cancer(tree)
plt.show()
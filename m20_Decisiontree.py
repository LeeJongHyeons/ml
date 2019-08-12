# 훈련세트에 과적합하여 테스트 정확도가 낮음
# 특성이 한쪽으로 몰리게 할 경우엔 왼쪽으로 몰린 쪽 위주로 평가모델을 수정

from sklearn.tree import DecisionTreeClassifier 
from sklearn.datasets import load_breast_cancer 
from sklearn.model_selection import train_test_split 

cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

tree = DecisionTreeClassifier(random_state=0)
tree.fit(x_train, y_train)
print("훈련 세트 정확도 : {:.3f}".format(tree.score(x_train, y_train))) # 훈련 세트 정확도 : 1.000
print("테스트 세트 정확도: {:.3f}".format(tree.score(x_test, y_test)))  # 테스트 세트 정확도: 0.937

# tree = DecisonTreeClassifier(max_depth=4, random_state=0)
# tree.fit(x_train, y_train)
#print("훈련 세트 정확도 : {:.3f}".format(tree.score(x_train, y_train)))  
#print("테스트 세트 정확도: {:.3f}".format(tree.score(x_test, y_test)))   

#print("특성 중요도:\n", tree.feature_importances_) # 각 컬럼들 전부 더하면 1
#[0.         0.00752597 0.         0.         0.00903116 0.
#0.00752597 0.         0.         0.         0.00975731 0.04630969
#0.         0.00238745 0.00231135 0.         0.         0.
#0.         0.00668975 0.69546322 0.05383211 0.         0.01354675
#0.         0.         0.01740312 0.11684357 0.01137258 0.        ]


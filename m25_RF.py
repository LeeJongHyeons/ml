from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score

cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)


model = RandomForestClassifier()
model.fit(x_train, y_train)

parameters = [
         {"max_depth": [15, 65, 95]},
         {"n_estimators": [100, 160, 280, 450, 330]}
]

score = score.search(x_test, y_test)

y_pred = model.predict(x_test)
print("최대 정답률:", accuracy_score(y_test, y_pred))

last_score = model.score(x_test, y_test)
print("최종 정답률은:", last_score)

print("특성 중요도:\n", model.feature_importances_) 

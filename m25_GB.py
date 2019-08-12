from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer 
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

kfold_cv =KFold(n_splits=5, shuffle=True)


parameters = [
         {"max_depth": [15, 65, 95]},
         {"n_estimators": [100, 160, 280, 450, 330]}
]

search = RandomizedSearchCV(GradientBoostingClassifier(),parameters, n_jobs=1, cv=kfold_cv)
search.fit(x_train, y_train)

score = score.search(x_test, y_test)

y_pred = search.predict(x_test)
print("최대 정답률:", accuracy_score(y_test, y_pred))

last_score = search.score(x_test, y_test)
print("최종 정답률은:", last_score)

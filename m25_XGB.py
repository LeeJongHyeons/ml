from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer 
from sklearn.model_selection import train_test_split 
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

search = XGBClassifier()


kfold_cv =KFold(n_splits=5, shuffle=True)

parameters = [
    {"max_depth": [10, 200, 500, 1100]},
    {"min_samples_split": [60, 150, 300]} 
]

search = RandomizedSearchCV(XGBClassifier(),parameters,cv=kfold_cv, n_jobs=-1)

search.fit(x_train, y_train)

score = score.search(x_test, y_test)

y_pred = search.predict(x_test)
print("최대 정답률:", accuracy_score(y_test, y_pred))

last_score = search.score(x_test, y_test)
print("최종 정답률은:", last_score)

print("특성 중요도:\n", search.feature_importances_) 

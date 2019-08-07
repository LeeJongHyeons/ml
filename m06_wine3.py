from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
import pandas as pd 

# RandomForestClassifier: XG_Boost => keras 
# acc 66%를 70% 이상으로 올리기

# 데이터 읽기
wine = pd.read_csv(".\data\winequality-white.csv", sep=";", encoding="utf-8")

# 데이터를 레이블과 데이터로 분리
y = wine["quality"]
x = wine.drop("quality", axis=1)

# y 레이블 변경하기
newlist = []
for v in list(y):
    if v <= 4:
        newlist += [0]
    elif v <= 7:
        newlist += [1]

    else:
        newlist += [2]

y = newlist

#from sklearn.preprocessing import MinMaxScaler, LabelEncoder
#MNS = MinMaxScaler()
#MNS.fit(wine)
#datapoint = MNS.transform(wine)

#lec = LabelEncoder()
#lec.fit(wine)
#datalabel = lec.transform(lec)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# 학습
model = RandomForestClassifier(n_estimators=800, max_depth=100, max_features="auto", min_samples_split=2,min_samples_leaf=2) # n_neighbors=15, n_estimators=10, random_state=3
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
aaa =model.score(x_test, y_test)

# 출력
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print("정답률:", accuracy_score(y_test, y_pred))
print("score:", score)
print(aaa) # 0.7148979591836735

#정답률= 0.9428571428571428
#0.9428571428571428
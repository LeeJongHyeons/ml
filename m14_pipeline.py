import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators # 측정평가
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC 

# 붓꽃 데이터 읽어 들이기
iris_data = pd.read_csv("G:/study/data/iris2.csv", encoding="utf-8")

# 붗꽃 데이터를 레이블과 입력 데이터로 분리하기
y = iris_data.loc[:, "Name"]
x = iris_data.loc[:,["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, train_size = 0.8, shuffle = True)

from sklearn.pipeline import Pipline 
from sklearn.preprocessing import MinMaxScaler 

pipe = Pipline([("scler", MinMaxScaler()), ('svm', SVC())]) # SVC모델앞에 fit 전부터 전처리 처리를 하고, 작업을 수행
pipe.fit(x_train, y_train)

print("테스트 점수:", pipe.score(x_test, y_test))
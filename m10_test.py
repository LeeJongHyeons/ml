import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators # all_estimators: 측정평가
import warnings

warnings.filterwarnings('ignore')

# 붓꽃 데이터 읽어 들이기
iris_data = pd.read_csv("G:/study/data/iris2.csv", encoding="utf-8")

# 붗꽃 데이터를 레이블과 입력 데이터로 분리하기
y = iris_data.loc[:, "Name"]
x = iris_data.loc[:,["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]

# classifier 알고리즘 모두 추출
warnings.filterwarnings('ignore')
allAlgorithms = all_estimators(type_filter="classifier")

for i in range(3, 11):
    print("n_split_value:", 7) 
    kfold_cv = KFold(n_splits=2, shuffle=True)

    for(name, algorithm) in allAlgorithms:

        clf = algorithm()
    
        if hasattr(clf, "score"): 
    
            # 크로스 밸리데이션
            scores = cross_val_score(clf, x, y, cv=kfold_cv)
            print(name, "의 정답률=")
            print(scores)
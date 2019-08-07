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


# classifier 알고리즘 모두 추출
warnings.filterwarnings('ignore')
allAlgorithms = all_estimators(type_filter="classifier")

# k-분할 크로스 밸리데이션 전용 객체
kfold_cv = KFold(n_splits=5, shuffle=True) # 변차 데이터를 5조각으로 나눔, train_set: 80%

for(name, algorithm) in allAlgorithms:
    # 각 알고리즘 객체 생성
    clf = algorithm()

    # score 메서드를 가진 클래스를 대상으로 하기
    if hasattr(clf, "score"): # hasattr: 객체가 있는지 확인

        # 크로스 밸리데이션
        scores = cross_val_score(clf, x, y, cv=kfold_cv) # cross_val_score: 교차 검증
        print(name, "의 정답률=")
        print(scores)
# Linear Regression, Ridge(l2), Lasso(l1)
import pandas as pd  
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import RandomSearch, GridSearchCV
import warnings 

iris_data = pd.read_csv("G:/study/data/iris.csv", encoding='utf-8', names=['a','b','c','d','y'])
print(iris_data)
print(iris_data.shape)
print(type(iris_data))

y = iris_data.loc[:, "Name"]
x = iris_data.loc[:,["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]

warnings.filterwarnings('ignore')

# 붓꽃 데이터를 레이블과 입력 데이터로 분리
y = iris_data.loc[:, "y"]
x = iris_data.loc[:,["a", "b","c","d"]]

#print(x.shape) #(150,4)
#print(y.shape) #(150,)

# 학습 전용과 테스트 전용 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.55, shuffle=True)

from keras.models import Sequential, Model
import numpy as np  
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers import Dense, LSTM, Dropout, Input

def _network(keep_prob=0.5, optimizer='adam'):
    inputs = Input(shape=(28,28,1),name='input')
    x = Conv2D(32, kernel_size=(3,3), activation='relu', name='hidden1')(inputs)
    x = Conv2D(32, kernel_size=(3,3), activation='relu', name='hidden2')(inputs)
    x = Dropout(keep_prob)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    x = Dropout(keep_prob)(x)
    prediction = Dense(10, activation='softmax', name = 'output')(x)
    model = Model(inputs=inputs, outputs = prediction)
    model.compile(optimizer=optimizer, loss= 'categorical_crossentropy',metrics=['accuracy'])
    return model

def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.5, 5)
    return{"batch_size":batches, "optimizer":optimizers, "keep_prob":dropout}

parameters = [
    {"n_estimators:": [200, 700, 800, 1100],
     "max_dept:": [600, 550, 460],
     "max_features": ['auto', 'sqrt', 'log2'],
     "min_samples_split": [0.1, 0.5, 0.6, 0.8]}
    
]

model = kerasClassfiter(build_fn=build_model, verbose=1)
hyperparameters = create_hyperparameters()


#print(x_test.shape) #(105, 4)
#print(y_test.shape) #(30, 4)

# 학습
#clf = SVC # 0.9
#clf = KNeighborsClassifier(n_neighbors=1) # 0.966666667
clf = RandomSearch()
clf.fit(x_train, y_train)

# 평가
y_pred = clf.predict(x_test)
print("정답률: ", accuracy_score(y_test, y_pred))
print("최적의 매개 변수=", clf.best_estimator_)
print("최종 정답률=", last_score)





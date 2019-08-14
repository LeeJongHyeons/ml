
'''
### 모델 저장하기 ###
model.save('savetest01.h5')

### 모델 불러오기 ###
from keras.models import load_model
model = load_model("savetest01.h5")
from keras.layers import Dense
model.add(Dense(1))

import numpy as np 
a = np.arange(10)
print(a)
np.save("aaa.npy", a)  # 저장 
b = np.load("aaa.npy") # 불러오기
print(b)

## pandas를 numpy 불러오기 ##
#pandas.value()

### csv 불러오기 ###
dataset = numpy.loadtxt("./data/pima-indians-diabetes.csv", delimiter=",")
iris_data = pd.read_csv("./data/iris.csv", encoding='utf-8')
index_col = 0, encoding='c949', sep=',', header=None
names= ['x1','x2','x3','x4', y]
wine = pd.read_csv("./data/winequality-white.csv", sep=",", encoding="utf-8")


### csv 저장 ###
데이터를 저장할 때 많이 사용되는 확장자 파일이기 때문

예) writer()를 사용하여 csv 파일로 저장
import csv
data = [[1,2,3,4], [5,6,7,8]]

csvfile = open("저장할 csv파일 경로", "w", newline="")

csvwriter = csv.writer(csvfile)

for row in data:
    csvwriter.writer(row)

csvfile.close()

### 한글처리###
-*- coding: utf-8 -*-

1. 인코딩 기본
파이썬을 이용해 한글을 처리한다고 하면, 아래의 코드를 페이지 최상단에 넣어주는 것이 기본

-*- coding: utf-8 -*-

2. 파일 읽기
외부에서 파일을 불러들일 대 파일에 쓰어진 언어는 코드와는 상관이 없으므로 그냥 읽어들이면 문제가 발생

path = 'test.txt'                   # 현재 작업 경로의 test.txt 파일에는 한글이 들어 있다.
test = open(path).readline()     # 그 파일의 첫 줄을 읽어 변수에 저장한다.
print test                          # 그리고 그 변수를 출력한다.

file= 'test.txt'
test = open(file).readline().decode('cp949')      # 파일이 ANSI로 저장되었다면 'cp949'
print test

프로그래밍에서 한글 처리를 할 때는 cp949 등으로 인코딩을 하게 되면 여러 번거러운 일이 발생하여, 당장 현재의 페이지는 서로 
저장한다고 해놓고, 들어오는 데이터가 cp949라는 것만으로도 혼란함이 있고, 따라서 프로그램 내에서 흐르는 모든 코드와 데이터가 나의 인코딩으로 맞추는것이 좋음

한글이 들어있는 파일을 불러 읽어들이는 코드
file= 'test-utf8.txt'
test = open(file).readline().decode('utf-8')
print test

3. 문자열 비교
현재의 페이지도 utf-8로 작성되었고, 들어오는 데이터도 utf-8로 디코딩 되었으니 잘 궁합이 맞지않고 에러가 발생

print (test.find('한글'))    # test 변수에 '한글'이라는 문자열이 있는지를 검사한다.
따라서 코드에 삽입되는 한글을 이용해 문자열 처리를 하려고 한다면, 이것 역시 utf-8로 디코딩을 하여야 올바른 처리가 이루어진다.

print (test.find('한글'.decode('utf-8'))) 

먼저 변수에 한글을 집어 넣고, 나중에 비교하는 형태라고 하더라도 위와 같은 방식을 따라야 한다. 즉, 다음과 같이 해야 한다.

test = "한글".decode('utf-8')

print (test.find(test))

4. 파일에 저장
파일에 저장할 때도 또 utf-8로 인코딩을 해주어야 한다. ANSI로 저장하려고 cp949로 인코딩을 해주면 에러가 난다. 따라서 저장을 위해서는 다음과 같은 코드를 작성해 주어야 한다.

my_file = open('testWrite.txt', 'w')

my_file.write(test.encode('utf-8'))

파이썬에서 해주는 인코딩, 디코딩은 기본적으로 그 상태가 변하지 않는 구간에서만 유효한 것으로 보인다. 짐작컨대, 데이터가 파일에서 입력 또는 출력되는 과정을 거치면 ascii 로 바뀌는 것 같다.

### 각종 샘플 데이터 셋 ###
from keras.datasets import mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

from keras.datasets import cifar10
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

from keras.datasets import boston_housing
(X_train, Y_train), (X_test, Y_test) = boston_housing.load_data()

from sklearn.datasets import load_boston
boston = load_boston()
print(boston.keys())    # data, target
#boston.data: x값, 넘파이
#boston.target: y값, 넘파이

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

'''
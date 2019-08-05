from sklearn.svm import LinearSVC, SVC         #LinearSVC : 선형회귀
from sklearn.metrics import accuracy_score     #metrics: 정확도
from keras.models import Sequential
from keras.layers import Dense
from sklearn.neighbors import KNeighborsClassifier
import numpy
import tensorflow as tf 

seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

# 1. 데이터
dataset = numpy.loadtxt("G:/study/data/pima-indians-diabetes.csv", delimiter=",")
X = dataset[:, 0:8]
Y = dataset[:,8]

# 2. 모델

#model = SVC()
#model = Sequential()
model = LinearSVC()
model.fit(X, Y)

y_predict= model.predict
print(y_predict)
print("\n Accuracy: %.4f" % (accuracy_score(Y, y_predict)[1]))

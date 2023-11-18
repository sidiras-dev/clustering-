import pprint
from __future__ import division
from builtins import range
import pandas as pd 


from sklearn.svm import SVC
from tensorflow.keras.datasets import mnist
from datetime import datetime

#get the data from kagg;le
#we can lood from Minst folder

# (X_train, y_train), (X_test, y_test) = mnist.load_data()
train = pd.read_csv("my_svm/MINST/train.csv")
test = pd.read_csv("my_svm/MINST/test.csv")

X_train = train.drop(labels = ["label"], axis = 1)
y_train = train['label']

X_test = test

print(X_train.shape, X_test.shape)

model=SVC()

t0=datetime.now()
model.fit(X_train,y_train)
pprint("train duration",datetime.now()-t0)

t0=datetime.now()
pprint("train score",model.score(X_train,y_train),"duration",datetime.now()-t0)

t0=datetime.now()
pprint("test score",model.score(X_test,test),"duration",datetime.now()-t0)

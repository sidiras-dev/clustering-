import pprint,numpy as np
from __future__ import division
from builtins import range
import pandas as pd 


from sklearn.svm import SVC
from tensorflow.keras.datasets import mnist
from datetime import datetime

#get the data from kagg;le
#we can lood from Minst folder

# (X_train, y_train), (X_test, y_test) = mnist.load_data()
train = pd.read_csv("my_svm/MINST/train.csv")#.values.astype(np.float32)
test = pd.read_csv("my_svm/MINST/test.csv").values.astype(np.float32)

X_train = train.drop(labels = ["label"], axis = 1)
#X_train=np.delete(train, column_to_drop='label', axis=1)
y_train = train['label']

X_test = test[-1000:1]
Y_test=test[-1000:0]
_

print(X_train.shape, X_test.shape)

model=SVC()

t0=datetime.now()
model.fit(X_train,y_train)
print("train duration",datetime.now()-t0)

t0=datetime.now()
print("train score",model.score(X_train,y_train),"duration",datetime.now()-t0)

t0=datetime.now()
print("test score",model.score(X_test,Y_test),"duration",datetime.now()-t0)

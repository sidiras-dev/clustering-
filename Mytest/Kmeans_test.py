import numpy as np
import matplotlib.pyplot as plt

#configuration
D=2
K=3
N=300


#create the data
mu1=np.array([0,0])
mu2=np.array([5,5])
mu3=np.array([0,5])

X=np.zeros((N,D))
X[:100,:]=np.random.rand(100,D)+mu1
X[100:200,:]=np.random.rand(100,D)+mu2
X[200:,:]=np.random.rand(100,D)+mu3

Y=np.array([0]*100+[1]*100+[2]*100)

#Vusualize the data
plt.scatter(X[:,0], X[:,1], c=Y);

#how to get back the D sized vector when takein the mean
print(X.mean(axis=0).shape)

#find the mean of each cluster
means=np.zeros((K,D))
means[0,:]=X[Y==0].mean(axis=0)
means[1,:]=X[Y==1].mean(axis=0)
means[2,:]=X[Y==2].mean(axis=0)

#plot with the means with data
plt.scatter(X[:,0], X[:,1], c=Y); #same as before
plt.scatter(means[:,0],means[:,1],s=500,c='red',marker='*');
plt.show()
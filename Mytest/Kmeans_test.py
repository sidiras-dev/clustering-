
#exerise 1 
#  create 100 data point per cluser and find the cendroid
import numpy as np
import matplotlib.pyplot as plt

#configuration
D=2 #number of futhers
K=3 #number of clusters
N=300 # number of samples


#create the data
#3 means final cluster centers 
mu1=np.array([0,0])
mu2=np.array([5,5])
mu3=np.array([0,5])
#generate the data with each centrod (total 100 each cedroid have diferant generated source)
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




"""
**Exersise#2**

o Given: data(x), cluster means - find cluster idenities 

MEaninit a data point which cluster belong closseness measured in Euclidean or squared euclidean distance

example eyciladan distance 2<4 or 4<16 only the relatace distane matter
putting the visual into the code

1. Generate means array od sized KxD
2. user the N-300 D-2 K-3
3. Generate a data ,atrix X of size NxD
4. As the previous exerise generate daata k-menas clustering and the result can check visiual
5. outpout cluster indenties array of size N

Suggent  to avoid loops when using NumPy\ for loop are slow ,ussually you want tot vectorize your operation using built in functions perhams in numpy functions 

but for this exesize we do  with for loop
"""


#take the same data points as the code to line 22  meaninig from line 1 to line 22 we have the same code
#create new Y


#main loop
Y=np.zeros(N)
for n in range(N):
    closest_k=-1
    min_dist=float('inf')
    for k in range(K):
        d=(X[n]-means[k]).dot(X[n]-means[k])
        if d < min_dist:
            min_dist=d
            closest_k=k
    Y[n]=closest_k
#visualize the data
plt.scatter(X[:,0], X[:,1], c=Y);
plt.show()

            


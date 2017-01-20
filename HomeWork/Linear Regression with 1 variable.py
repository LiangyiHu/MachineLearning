import os
import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.mplot3d
os.chdir('C:\Users\mosimtec\Desktop\MachineLearning\mlclass-ex1-005\mlclass-ex1')
data=np.loadtxt('ex1data1.txt', delimiter=',')
theta=np.array([1.0,1.0])
print theta
alpha=0.3
times=50

def featurescaling(data):
    return (data-np.average(data,axis=0))/(data.max(axis=0)-data.min(axis=0))
def addx0(data):
    return np.hstack([np.ones([data[:,0].size, 1]),featurescaling(data)])
def costfunction(data, theta=np.ones([1,2])):
    return 1.0/(2*data[:,0].size)*np.sum((np.dot(data[:,(0,1)],theta.transpose())-data[:,2])**2)
def updatetheta(data, theta, alpha):
    newtheta=theta.copy()
    i=0
    for thetai in theta:
        newtheta[i]=thetai-alpha/(data[:,0].size)*(np.sum((np.dot(data[:,(0,1)],theta.transpose())-data[:,2])*data[:,i]))
        i+=1
    return newtheta
def mainframe(data, theta,alpha, times):
    data=addx0(data)
    a=np.array([])
    for i in range(times):
        theta=updatetheta(data,theta,alpha)
        print theta[0],theta[1], costfunction(data,theta)
        a=np.append(a,costfunction(data,theta))
    return a
def gradientdescent(data,theta,alpha,times):
    plt.plot(mainframe(data,theta,alpha,times))
    plt.show()
def threeDplot(data):
    x,y=np.mgrid[-2:2:40j,-1:3:40j]
    z=np.zeros((40,40),dtype=float)
    data=addx0(featurescaling(data))
    for i in range(40):
        for j in range(40):
            z[i,j]=costfunction(data, np.array([x[i,j],y[i,j]]))
    ax=plt.subplot(111,projection='3d')
    ax.plot_surface(x,y,z,rstride=1,cstride=1,cmap=plt.cm.coolwarm,alpha=0.8)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

threeDplot(data)
"""gradientdescent(data,theta,alpha,times)"""
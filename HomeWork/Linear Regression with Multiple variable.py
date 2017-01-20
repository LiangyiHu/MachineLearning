import os
import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.mplot3d

data=np.loadtxt('ex1data2.txt', delimiter=',')
theta=np.ones(data[0,:].size,dtype=float)
alpha=0.5
times=200

def featurescaling(data):
    return (data-np.average(data,axis=0))/(data.max(axis=0)-data.min(axis=0))
def addx0(data):
    return np.hstack([np.ones([data[:,0].size, 1]),featurescaling(data)])
def costfunction(data, theta=np.ones(data[0,:].size,dtype=float)):
    return 1.0/(2*data[:,0].size)*np.sum((np.dot(data[:,0:-1],theta.transpose())-data[:,-1])**2)
def updatetheta(data, theta, alpha):
    newtheta=theta.copy()
    i=0
    for thetai in theta:
        newtheta[i]=thetai-alpha/(data[:,0].size)*(np.sum((np.dot(data[:,0:-1],theta.transpose())-data[:,-1])*data[:,i]))
        i+=1
    return newtheta
def mainframe(data, theta,alpha, times):
    data=addx0(data)
    a=np.array([])
    print theta[0],theta[1],theta[2], costfunction(data,theta)
    for i in range(times):
        theta=updatetheta(data,theta,alpha)
        print theta[0],theta[1],theta[2], costfunction(data,theta)
        a=np.append(a,costfunction(data,theta))
    return a
def gradientdescent(data,theta,alpha,times):
    plt.plot(mainframe(data,theta,alpha,times))
    plt.show()
    
gradientdescent(data,theta,alpha,times)
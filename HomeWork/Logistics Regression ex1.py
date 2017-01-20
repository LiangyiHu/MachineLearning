# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.mplot3d
os.chdir('C:\Users\mosimtec\Desktop\MachineLearning\mlclass-ex2-005\mlclass-ex2')
data=np.loadtxt('ex2data1.txt', delimiter=',')
theta=np.ones(data[0,:].size,dtype=float)
theta=np.array([0.5,0.5,0.5])
alpha=10.0
times=500

def featurescaling(data):
    data[:,:-1]=(data[:,0:-1]-np.average(data[:,0:-1],axis=0))/(data[:,0:-1].max(axis=0)-data[:,0:-1].min(axis=0))
    return data
def processdata(data):
    return np.hstack((np.ones((data[:,0].size, 1),dtype=float),featurescaling(data)))
def costfunction(data, theta=np.ones(data[0,:].size,dtype=float)):
    return 1.0/data[:,0].size*np.sum(-data[:,-1]*np.log(1/(1+np.exp(-np.dot(data[:,:-1],theta.transpose()))))-(1-data[:,-1])*np.log(1-1/(1+np.exp(-np.dot(data[:,:-1],theta.transpose())))))
def updatetheta(data, theta, alpha):
    newtheta=theta.copy()
    i=0
    for thetai in theta:
        newtheta[i]=thetai-alpha/(data[:,0].size)*(np.sum((1/(1+np.exp(-np.dot(data[:,:-1],theta.transpose())))-data[:,-1])*data[:,i]))
        i+=1
    return newtheta
def mainframe(data, theta,alpha, times):
    data=processdata(data)
    a=np.array([])
    print theta[0],theta[1],theta[2], costfunction(data,theta)
    for i in range(times):
        theta=updatetheta(data,theta,alpha)
        print theta[0],theta[1],theta[2], costfunction(data,theta)
        a=np.append(a,costfunction(data,theta))
    return a,theta
def gradientdescent(data,theta,alpha,times):
    a,b=mainframe(data,theta,alpha,times)
    plt.plot(a)
    plt.show()
    return b
def showrawdata(data):
    plt.plot(data[data[:,2]==0][:,0],data[data[:,2]==0][:,1],'ro',data[data[:,2]==1][:,0],data[data[:,2]==1][:,1],'b^')
    plt.show()

    
gradientdescent(data,theta,alpha,times)

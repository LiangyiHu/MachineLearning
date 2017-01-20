# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.mplot3d

os.chdir('C:\Users\mosimtec\Desktop\MachineLearning\mlclass-ex2-005\mlclass-ex2')
data=np.loadtxt('ex2data2.txt', delimiter=',')
alpha=1.0
times=10
lamda=0
theta=np.ones(data[0,:].size,dtype=float)
magnitude=6
def featurescaling(data):
    newdata=data.copy()
    newdata[:,:-1]=(data[:,0:-1]-np.average(data[:,0:-1],axis=0))/(data[:,0:-1].max(axis=0)-data[:,0:-1].min(axis=0))
    return newdata
def processdata(data):
    return np.hstack((np.ones((data[:,0].size, 1),dtype=float),featurescaling(data)))
def costfunction(data,theta,lamda):
    return 1.0/data.shape[0]*(np.sum(-data[:,-1]*np.log(1/(1+np.exp(-np.dot(data[:,:-1],theta.transpose()))))-(1-data[:,-1])*np.log(1-1/(1+np.exp(-np.dot(data[:,:-1],theta.transpose())))))+lamda/2.0*np.sum(theta[1:]**2))
def updatetheta(data, theta, alpha,lamda):
    newtheta=theta.copy()
    i=0
    for thetai in theta:
        if i==0:
            newtheta[i]=thetai-alpha/(data.shape[0])*np.sum((1/(1+np.exp(-np.dot(data[:,:-1],theta.transpose())))-data[:,-1])*data[:,i])
            i+=1
        else:
            newtheta[i]=thetai-alpha/(data[:,0].size)*(np.sum((1/(1+np.exp(-np.dot(data[:,:-1],theta.transpose())))-data[:,-1])*data[:,i])+lamda*thetai)
            i+=1
    return newtheta
def mainframe(data, theta,alpha, times,lamda):
    newdata=data.copy()
    a=np.array([])
    for i in range(times):
        theta=updatetheta(newdata,theta,alpha,lamda)
        a=np.append(a,costfunction(newdata,theta,lamda))
    return a,theta
def gradientdescent(data,theta,alpha,times,lamda):
    a,b=mainframe(data,theta,alpha,times,lamda)
    plt.plot(a)
    plt.show()
    return b
def showrawdata(data):
    plt.plot(data[data[:,2]==0][:,0],data[data[:,2]==0][:,1],'ro',data[data[:,2]==1][:,0],data[data[:,2]==1][:,1],'b^')
    plt.show()
def populatefeature(inputmatrix,magnitude):
    outputmatrix=inputmatrix.copy()
    for level in range(2,magnitude+1):
        for i in range(level+1):
            outputmatrix=np.hstack((outputmatrix,((inputmatrix[:,0]**(level-i)*(inputmatrix[:,1]**i))).reshape(inputmatrix.shape[0],1)))
    return outputmatrix
def testnewdata(testdata,sample,theta,magnitude):
    maturetestdata=processarray(populatefeatureforarray(testdata,magnitude),sample)
    return 1.0/(1+np.exp(-(np.sum(maturetestdata*theta))))
def populatefeatureforarray(inputarray,magnitude):
    outputarray=inputarray.copy()
    for level in range(2,magnitude+1):
        for i in range(level+1):
            outputarray=np.append(outputarray,((inputarray[0]**(level-i)*(inputarray[1]**i))))
    return outputarray
def processarray(populatedarray, sample):
    return np.append(np.array([1]),(populatedarray-np.average(sample,axis=0))/(np.max(sample,axis=0)-np.min(sample,axis=0)))
    
populateddata=populatefeature(data[:,0:-1],magnitude)
maturedata=processdata(np.hstack((populateddata,data[:,-1].reshape(data.shape[0],1))))
theta=np.zeros(maturedata.shape[1]-1,dtype=float)
theta=mainframe(maturedata,theta,alpha,times,lamda)[1]
print testnewdata(np.array([0.5,0]),populateddata,theta,magnitude)

x,y=np.mgrid[-1:1:100j,-1:1:100j]
z=np.zeros((100,100),dtype=float)
for i in range(100):
        for j in range(100):
            z[i,j]=testnewdata(np.array([x[i,j],y[i,j]]),populateddata,theta,magnitude)
plt.contour(x,y,z,[0.2,0.5,0.8])
plt.plot(data[data[:,2]==0][:,0],data[data[:,2]==0][:,1],'ro',data[data[:,2]==1][:,0],data[data[:,2]==1][:,1],'b^')
plt.show()
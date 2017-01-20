import os
import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.mplot3d
os.chdir('C:\Users\mosimtec\Desktop\MachineLearning\mlclass-ex5-007\mlclass-ex5')
data=np.loadtxt('trainingset.csv', delimiter=',')
data=np.hstack((np.ones([data.shape[0],1]),data))
cvdata=np.loadtxt('CVset.csv',delimiter=',')
cvdata=np.hstack((np.ones([cvdata.shape[0],1]),cvdata))
alpha=1.0
times=100
lamda=0.0
def featurescaling(data,control):
    return np.hstack((data[:,0].reshape(data.shape[0],1),(data[:,1:]-np.average(control[:,1:],axis=0))/(control[:,1:].max(axis=0)-control[:,1:].min(axis=0))))
def costfunction(data, theta,lamda):
    return 1.0/(2*data.shape[0])*np.sum((np.dot(data[:,:-1],theta.transpose())-data[:,-1])**2)+lamda/(2*data.shape[0])*np.sum(theta[1:]**2)
def updatetheta(data, theta, alpha,lamda):
    newtheta=theta.copy()
    for i in range(theta.size):
        if i==0:
            newtheta[i]=1.0/(data.shape[0])*(np.sum((np.dot(data[:,:-1],theta.transpose())-data[:,-1])*data[:,i]))
        else:
            newtheta[i]=1.0/(data.shape[0])*(np.sum((np.dot(data[:,:-1],theta.transpose())-data[:,-1])*data[:,i]))+lamda/data.shape[0]*theta[i]
    return theta-alpha*newtheta
def gradientdescent(data, alpha, times,lamda):
    theta=np.ones([data.shape[1]-1])
    a=np.array([])
    for i in range(times):
        theta=updatetheta(data,theta,alpha,lamda)
        a=np.append(a,costfunction(data,theta,lamda))
    return a,theta
def checkapoint(input,theta):
    return theta[0]+input*theta[1]
def populatefeature(inputmatrix,magnitude):
    newmatrix=inputmatrix[:,1].copy()
    returnmatrix=inputmatrix.copy()
    for power in range(2,magnitude+1):
        newmatrix=(inputmatrix[:,1]**power).reshape(inputmatrix.shape[0],1)
        returnmatrix=np.hstack((np.hstack((returnmatrix[:,:-1],newmatrix)),returnmatrix[:,-1].reshape(inputmatrix.shape[0],1)))
    return returnmatrix
inputtraindata=populatefeature(data,3)
print inputtraindata

"""This is code for 2.1 learning curves:
inputtraindata=featurescaling(data,data)
inputcvdata=featurescaling(cvdata,data)
trainplot=np.array([])
cvplot=np.array([])
for i in range(2,inputtraindata.shape[0]):
    print i
    theta=gradientdescent(inputtraindata[:i,:],alpha,times,lamda)[1]
    trainplot=np.append(trainplot,costfunction(inputtraindata[:i,:],theta,lamda))
    cvplot=np.append(cvplot,costfunction(inputcvdata,theta,lamda))
plt.plot(trainplot)
plt.plot(cvplot)
plt.show()"""
import os
import matplotlib.pyplot as plt
import numpy as np

os.chdir('C:\Users\mosimtec\Desktop\MachineLearning\mlclass-ex3-007\mlclass-ex3')
datain=np.loadtxt('input.csv', delimiter=',')
dataout=np.loadtxt('output.csv', delimiter=',')
finaldata=np.hstack((np.hstack((np.ones((datain.shape[0],1)),datain)),dataout.reshape(dataout.shape[0],1)))
finaldata[finaldata[:,-1]==10,-1]=0.0
theta1=np.loadtxt('theta1.csv', delimiter=',')
theta2=np.loadtxt('theta2.csv', delimiter=',')
def costfunction(inputdata,theta1,theta2):
    output=np.zeros([10])
    for i in range(theta2.shape[0]):
        matrixy=inputdata[:,-1].copy()
        matrixy[matrixy[:]==0]=10
        if i==0:
            matrixy[matrixy[:]!=10]=0
            matrixy[matrixy[:]==10]=1
        else:
            matrixy[matrixy[:]!=i]=0
            matrixy[matrixy[:]==i]=1
        a1=np.vstack((np.ones([1,inputdata.shape[0]]),1.0/(1.0+np.exp(-np.dot(theta1,inputdata[:,:-1].transpose())))))
        output[i]=1.0/inputdata.shape[0]*np.sum(-(matrixy)*np.log(1.0/(1.0+np.exp(-np.dot(theta2[i,:],a1))))-(1-matrixy)*np.log(1.0-1.0/(1.0+np.exp(-np.dot(theta2[i,:],a1)))))
    return output.sum()

print costfunction(finaldata,theta1,theta2)
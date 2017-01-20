import os
import matplotlib.pyplot as plt
import numpy as np

os.chdir('C:\Users\mosimtec\Desktop\MachineLearning\mlclass-ex4-007\mlclass-ex4')
datain=np.loadtxt('input.csv', delimiter=',')
dataout=np.loadtxt('output.csv', delimiter=',')
finaldata=np.hstack((np.hstack((np.ones((datain.shape[0],1)),datain)),dataout.reshape(dataout.shape[0],1)))
finaldata[finaldata[:,-1]==10,-1]=0.0
lamda=1.0
times=500
alpha=2.5
def costfunction(inputdata,theta1,theta2,lamda):
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
    return output.sum()+lamda/(2*inputdata.shape[0])*(np.sum(theta1[:,1:]**2)+np.sum(theta2[:,1:]**2))
def randomizetheta(theta1shape0=25,theta1shape1=401,theta2shape0=10,theta2shape1=26):
    np.savetxt('theta1.csv',np.random.uniform(-0.12,0.12,(25,401)),delimiter=',')
    np.savetxt('theta2.csv',np.random.uniform(-0.12,0.12,(10,26)),delimiter=',')
def calculatea3(inputarray, theta1, theta2):
    a=1.0/(1.0+np.exp(-np.dot(theta2,np.append(1,calculatea2(inputarray,theta1,theta2).transpose()))))
    return a
def calculatea2(inputarray,theta1,theta2):
    return 1.0/(1.0+np.exp(-np.dot(theta1,inputarray[:-1].transpose())))
def getdefferentiate(finalmatrix,theta1,theta2):
    partialtheta1=np.zeros((25,401))
    partialtheta2=np.zeros((10,26))
    for i in range(finalmatrix.shape[0]):
        thisy=np.zeros(theta2.shape[0])
        thisy[finalmatrix[i,-1]]=1.0
        delta3=calculatea3(finalmatrix[i,:],theta1,theta2)-thisy
        delta2=np.dot(theta2[:,1:].transpose(),delta3)*gpiez2(finalmatrix[i],theta1)
        partialtheta1=partialtheta1+np.outer(delta2,finalmatrix[i,:-1])
        partialtheta2=partialtheta2+np.outer(delta3,np.append(1,calculatea2(finalmatrix[i,:],theta1,theta2)))
    partialtheta1=1.0/finalmatrix.shape[0]*partialtheta1+lamda/finalmatrix.shape[0]*np.hstack((np.zeros([theta1.shape[0],1]),theta1[:,1:]))
    partialtheta2=1.0/finalmatrix.shape[0]*partialtheta2+lamda/finalmatrix.shape[0]*np.hstack((np.zeros([theta2.shape[0],1]),theta2[:,1:]))
    return partialtheta1,partialtheta2
def gpiez2(inputarray, theta1):
    gz2=1.0/(1.0+np.exp(-np.dot(theta1,inputarray[:-1].transpose())))
    return gz2*(1-gz2)
def backpropagate(inputdata,theta1,theta2,alpha,times,lamda):
    cost=np.array([])
    for i in range(times):
        theta1=theta1-alpha*getdefferentiate(inputdata,theta1,theta2)[0]
        theta2=theta2-alpha*getdefferentiate(inputdata,theta1,theta2)[1]
        cost=np.append(cost,costfunction(inputdata,theta1,theta2,lamda))
        if i%10==0:
            print '%ith iteration has done' %(i+1)
    plt.plot(cost)
    plt.show()
    return theta1,theta2
theta1=np.loadtxt('theta1.csv', delimiter=',')
theta2=np.loadtxt('theta2.csv', delimiter=',')
theta1,theta2=backpropagate(finaldata,theta1,theta2,alpha,times,lamda)
np.savetxt('theta1_calculated.csv',theta1,delimiter=',')
np.savetxt('theta2_calculated.csv',theta2,delimiter=',')
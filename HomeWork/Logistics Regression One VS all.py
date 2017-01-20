import os
import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.mplot3d
import matplotlib.image as mpimg
import PIL

os.chdir('C:\Users\mosimtec\Desktop\MachineLearning\mlclass-ex3-007\mlclass-ex3')
datain=np.loadtxt('input.csv', delimiter=',')
dataout=np.loadtxt('output.csv', delimiter=',')
finaldata=np.hstack((np.hstack((np.ones((datain.shape[0],1)),datain)),dataout.reshape(dataout.shape[0],1)))

alpha=3.0
times=100
lamda=0.0
theta=np.zeros((10,finaldata.shape[1]-1),dtype=float)

def printsampledata(inputmatrix, row):
    sample=inputmatrix[row-1,:].reshape(20,20).transpose()
    plt.imshow(sample,cmap='binary')
    plt.show()
def costfunction(data,theta,lamda):
    return (np.sum(-data[:,-1]*np.log(1.0/(1.0+np.exp(-np.dot(data[:,:-1],theta.transpose()))))-(1-data[:,-1])*np.log(1-1.0/(1.0+np.exp(-np.dot(data[:,:-1],theta.transpose())))))+np.sum(theta[1:]**2)*lamda/2.0)*1.0/data.shape[0]
def updatetheta(data, theta, alpha,lamda):
    newtheta=theta.copy()
    i=0
    for thetai in theta:
        if i==0:
            newtheta[i]=thetai-alpha/(data.shape[0])*np.sum((1.0/(1.0+np.exp(-np.dot(data[:,:-1],theta.transpose())))-data[:,-1])*data[:,i])
            i+=1
        else:
            newtheta[i]=thetai-alpha/(data[:,0].size)*(np.sum((1.0/(1.0+np.exp(-np.dot(data[:,:-1],theta.transpose())))-data[:,-1])*data[:,i])+lamda*thetai)
            i+=1
    return newtheta
def mainframe(data, theta,alpha, times,lamda):
    a=np.array([])
    for i in range(times):
        theta=updatetheta(data,theta,alpha,lamda)
        a=np.append(a,costfunction(data,theta,lamda))
        print 'iteration %i has done' %(i+1)
    return a,theta
def gradientdescent(data,theta,alpha,times,lamda):
    a,b=mainframe(data,theta,alpha,times,lamda)
    """plt.plot(a)
    plt.show()"""
    return b
def OVAgradientdescent(inputmatrix,theta,alpha, times,lamda):
    for i in range(theta.shape[0]):
        if i==0:
            value=10
        else: value=i
        OVAdata=inputmatrix.copy()
        OVAdata[OVAdata[:,-1]!=value,-1]=0.0
        OVAdata[OVAdata[:,-1]==value,-1]=1.0
        print 'starts one-vs-all for %i -----------------------------------------' %(i)
        theta[i,:]=gradientdescent(OVAdata,theta[i,:],alpha,times,lamda)
    return theta
def testanewitem(inputarray,caltheta):
    record=np.array([])
    for i in range(caltheta.shape[0]):
        record=np.append(record,1.0/(1.0+np.exp(-(np.sum(inputarray[:-1]*caltheta[i,:])))))
    if record.max()<0.5:
        return False
    return record.argmax()==inputarray[-1]
def testallitem(inputdata,caltheta):
    rightguess,wrongguess=0,0
    for i in range(inputdata.shape[0]):
        if testanewitem(inputdata[i,:],caltheta):
            rightguess+=1
        else: wrongguess+=1
    return rightguess,wrongguess
def processanimage(image):
    return np.append(np.append(np.array([1.0]),image),np.array([0]))
    

"""np.savetxt('theta.csv',OVAgradientdescent(finaldata,theta,alpha,times,lamda),delimiter=',')"""
"""theta=np.loadtxt('theta.csv', delimiter=',')
finaldata[finaldata[:,-1]==10,-1]=0.0
print finaldata.dtype
print testallitem(finaldata,theta)"""

"""finaldata[finaldata[:,-1]!=10,-1]=0.0
finaldata[finaldata[:,-1]==10,-1]=1.0
gradientdescent(finaldata,theta[0,:],alpha,times,lamda)
increase times to see convergence line, get optimum time, incorporate image processing"""

"""theta=np.loadtxt('theta.csv', delimiter=',')
img=PIL.Image.open('newpic.png')
rsize=img.resize((20,20))
inputimg=1.0-np.average(np.asarray(rsize), axis=2)/255
inputimg=np.append(np.append(np.array([1.0]),inputimg),np.array([8.0]))
for i in range(401):
    print inputimg[i],finaldata[1800,i]
print testanewitem(inputimg,theta)"""
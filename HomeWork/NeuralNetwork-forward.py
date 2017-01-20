import os
import matplotlib.pyplot as plt
import numpy as np
import PIL

os.chdir('C:\Users\mosimtec\Desktop\MachineLearning\mlclass-ex4-007\mlclass-ex4')
datain=np.loadtxt('input.csv', delimiter=',')
dataout=np.loadtxt('output.csv', delimiter=',')
finaldata=np.hstack((np.hstack((np.ones((datain.shape[0],1)),datain)),dataout.reshape(dataout.shape[0],1)))
finaldata[finaldata[:,-1]==10,-1]=0.0
theta1=np.loadtxt('theta1_calculated.csv', delimiter=',')
theta2=np.loadtxt('theta2_calculated.csv', delimiter=',')
identifyunable=0
def convertit(inputarray):
    global identifyunable
    if inputarray.max()<0.5:
        identifyunable+=1
    return inputarray.argmax()
def calculateoutput(inputarray, theta1, theta2):
    a=np.append(1,1.0/(1.0+np.exp(-np.dot(theta1,inputarray[:-1].transpose()))))
    a=1.0/(1.0+np.exp(-np.dot(theta2,a.transpose())))
    return convertit(a)==inputarray[-1]
def statit(data,theta1,theta2):
    right,wrong=0,0
    for row in data:
        if calculateoutput(row,theta1,theta2):
            right+=1
        else: wrong+=1
    print right, wrong
def getoutput(inputarray, theta1, theta2):
    a=np.append(1,1.0/(1.0+np.exp(-np.dot(theta1,inputarray[:-1].transpose()))))
    a=1.0/(1.0+np.exp(-np.dot(theta2,a.transpose())))
    return a,convertit(a)
statit(finaldata,theta1,theta2)
img=PIL.Image.open('newpic.png')
rsize=img.resize((20,20))
inputimg=1.0-np.average(np.asarray(rsize), axis=2)/255
inputimg=np.append(np.append(np.array([1.0]),inputimg),np.array([8.0]))
print getoutput(inputimg,theta1,theta2)
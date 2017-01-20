import os
import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.mplot3d
os.chdir('C:\Users\mosimtec\Desktop\MachineLearning\mlclass-ex6-007\mlclass-ex6')
data=np.loadtxt('data1.csv', delimiter=',')
data=np.hstack((np.ones([data.shape[0],1]),data))
def showrawdata(data):
    plt.plot(data[data[:,-1]==0][:,1],data[data[:,-1]==0][:,2],'ro',data[data[:,-1]==1][:,1],data[data[:,-1]==1][:,2],'b^')
    plt.show()

showrawdata(data)
# -- coding: utf-8 --
"""
Created on Sun Feb 21 18:09:27 2021

@author: mahdis
"""
import numpy as np
import numpy as geek
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
from numpy import linalg as LA


#define
X1 = np.array([[4, 2], [2, 4], [2, 3], [3, 6], [4, 4]])
X2 = np.array([[9, 10], [6, 8], [9, 5], [8, 7], [10, 8]])
N1=len(X1)
N2=len(X2)

#mean
MeanX1 = np.sum(X1, axis=0)/N1
MeanX2 = np.sum(X2, axis=0)/N2
#print(MeanX1 , MeanX2)

#Cal_Cov
covX1 = np.cov(X1.T)
covX2 = np.cov(X2.T)
SW = covX1+covX2
#print(SW)

#Cal_SB
One = (MeanX1-MeanX2).T
Two = MeanX1-MeanX2
TwoR = Two.reshape(2,1)
def calcSB(x, y):
    return TwoR  * One

def SB_mat(X):
    return np.array([calcSB(X[0],X[0])])
SB= SB_mat(X1)

#two_Classes
MulSWSB = np.matmul(np.linalg.inv(SW),SB)
Two_Classes =  np.matmul(MulSWSB,np.identity(2))

#Wstar
w,vector = LA.eig(Two_Classes)

print(vector)
w2 =vector[0].T[0]
w1=vector[0].T[1]


# Plot
plt.scatter(X1.T[0],X1.T[1])
plt.scatter(X2.T[0],X2.T[1])
x = np.linspace(0, 4,11)
plt.plot(w2 + 0, w2,x, '-g')

plt.title('LDA projection vector with the highest eigen value = 12.2007')
plt.xlabel('X1')
plt.ylabel('X2')
plt.grid()
plt.show()
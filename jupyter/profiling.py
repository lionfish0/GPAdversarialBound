from GPAdversarialBound import getshiftbounds, getpred, plot2dB
import numpy as np
import matplotlib.pyplot as plt
ip = None
l = 1.0

#X = np.array([[0,0,0,0,0],[0,0,1,0,0],[1,0,0,0,0],[1,0,0,0,1],[1,0,0,1,0],[1,0,1,0,1],[0,1,1,0,1],[1,1,0,1,1],[0,1,1,1,1],[1,1,1,1,1]])*1.0


N = 20
D = 30
totalits = 50
X = np.random.rand(N,D)
Y = 2.0*(np.random.rand(N,1)>0.5)-1.0 #-1s and 1s
#Y = np.array([[-1,-1,-1,-1,-1,1,1,1,1,1]]).T*1.0
pos_allshifts,pos_dbinfo = getshiftbounds(X,Y,l=l,totalits=totalits)
neg_allshifts,pos_dbinfo = getshiftbounds(X,-Y,l=l,totalits=totalits)

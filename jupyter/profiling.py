from GPAdversarialBound import getshiftbounds, getpred, plot2dB
import numpy as np
import matplotlib.pyplot as plt
ip = None
l = 1.0
totalits = 20
X = np.array([[0,0,0,0,0],[0,0,1,0,0],[1,0,0,0,0],[1,0,0,0,1],[1,0,0,1,0],[1,0,1,0,1],[0,1,1,0,1],[1,1,0,1,1],[0,1,1,1,1],[1,1,1,1,1]])*1.0
Y = np.array([[-1,-1,-1,-1,-1,1,1,1,1,1]]).T*1.0
pos_allshifts,pos_dbinfo = getshiftbounds(X,Y,l=l,totalits=totalits)
neg_allshifts,pos_dbinfo = getshiftbounds(X,-Y,l=l,totalits=totalits)

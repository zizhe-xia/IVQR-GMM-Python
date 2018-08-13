'''
This file applies the GMM_IVQR method to the Fulton fish market data
'''

import numpy as np
import pandas as pd
from IVQR_GMM import IVQR_GMM
from math import log
from decimal import Decimal as Dec
from decimal import getcontext
getcontext().prec = 50


df = pd.read_csv('NYFishMarket.csv')
#print(df)
tots = df[['qty']]
#print(tots.values)
#print(tots)
covariate = df[['price']]
instruments = df[['stormy', 'mixed']]
#y = np.array([tots.values],dtype=np.dtype(Dec)).T
#w = np.array(np.concatenate((np.ones((tots.shape[0],1)), covariate.values), axis = 1), dtype=np.dtype(Dec))
#z = np.array(np.concatenate((np.ones((tots.shape[0],1)), instruments.values),axis=1), dtype=np.dtype(Dec))
y = np.array(tots.values)
w = np.concatenate((np.ones((tots.shape[0],1)), covariate.values), axis=1)
z = np.concatenate((np.ones((tots.shape[0],1)), instruments.values),axis=1)
#IVQR_MIO(y, w, Q, tau, T, abgap, bnd, method)
intercept = False
T = 15000
abgap = 1e-2
bnd = np.array([[6,13],[-1,2]])
for tau in [0.25,0.5,0.75]:
   print ('IV reg %g' % tau)
   print (IVQR_GMM(y, w, z, tau, T, abgap, bnd))
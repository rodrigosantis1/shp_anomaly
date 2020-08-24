# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 09:38:36 2020

@author: rodri
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


train = pd.read_csv('data/train.csv', parse_dates=['t'], 
                    index_col=0, usecols=['t','V1','V2','V3','V4'])
test  = pd.read_csv('data/test.csv', parse_dates=['t'], 
                    index_col=0, usecols=['t','V1','V2','V3','V4'])


mi = train.mean().values
cov = train.cov().values
inv_cov = np.linalg.inv(cov)

x = train.values
#x = test.values

y = np.full(len(x),0, dtype=x.dtype)
for i in range(len(x)):
    y[i] = (x[i]-mi).T.dot(inv_cov).dot((x[i]-mi))

plt.figure(figsize=(12,8))
plt.plot(y)

sns.distplot(train['V4'])

#
#    F = stats.f.ppf(CI, alpha, (n-alpha))
#    return (n**2 - 1)*alpha / (n*(n-alpha)) * F

p = 4
CI = 0.95

F = stats.f.ppf(CI,dfn=len(x), dfd=p)


lsc = 
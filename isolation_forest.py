# -*- coding: utf-8 -*-
"""
Intelligent fault diagnosis based on Isolation Forest algorithm for SHPs

@author: Santis, R. B.
@year: 2020 
"""

import pandas as pd
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Arial'],'size':12})

from sklearn.kernel_approximation import RBFSampler
from sklearn.decomposition import FastICA, PCA
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline

def hotelling_t2(x, eigen):
    y = np.full(len(x),0, dtype=x.dtype)
    D = np.linalg.inv(np.diag(eigen))
    for i in range(len(x)):
        y[i] = x[i].T.dot(D).dot(x[i])
    return y

def threshold_t2(n, alpha, CI=0.95):
    F = stats.f.ppf(CI, alpha, (n-alpha))
    return (n**2 - 1)*alpha / (n*(n-alpha)) * F

def temporal_distance(C_true, C_detect, max_p=None, div=1):
    ttc = np.full(len(C_true), 0)
    ctt = np.full(len(C_detect), 0)
    for i, anomaly in enumerate(C_true):
        delta = (C_detect - anomaly)
        ttc[i]  = min(abs(delta))
    for i, detection in enumerate(C_detect):
        delta = (C_true - detection)
        ctt[i] = min(abs(delta))
    return ttc.sum()/div, ctt.sum()/div

# Constants
N_RUN=3

models = [('PCA', make_pipeline(PCA())),
          ('KICA-PCA', make_pipeline(RBFSampler(n_components=100), FastICA(whiten=True), PCA(n_components=20))),
          ('Forest', IsolationForest(contamination=0.06,n_jobs=-1, max_samples=0.05, n_estimators=1000)) #random_state=15,
        ]

# Read data
train = pd.read_csv('data/train.csv', parse_dates=['t'], index_col=0)
test = pd.read_csv('data/test.csv', parse_dates=['t'], index_col=0)
ft = pd.read_csv('data/faults.csv', parse_dates=['t'],index_col=0)
df = pd.concat([train,test])

# Dataset 3D visualization
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(df['V6'],df['V5'],df[['V1','V2','V3','V4']].mean(axis=1),
                cmap='twilight_shifted', alpha=0.5)
ax.set_xlabel('Apparent Power'), ax.set_ylabel('LHU Inflow'), ax.set_zlabel('Average Vibration')
plt.show()

# Simmulation
results = []
for name, model in models:
    print(name)
    for run in range(N_RUN):
        np.random.seed(run)
        
        model.fit(train)
        if (name != 'Forest'):
            X_t  = model.transform(test)
            z = hotelling_t2(X_t, model['pca'].explained_variance_)
            th = threshold_t2(n=len(test), alpha=X_t.shape[1], CI=0.95)
            an = z > th
        else:
            z = model.score_samples(test)
            th = model.offset_
            an = (model.predict(test) == -1)
        
    
        score = pd.DataFrame(np.column_stack((z, np.repeat(th, len(z)), an)), 
                             columns=['score', 'threshold', 'anomaly'],
                             index=test.index)
        
        C_detect = (score.loc[an].index - df.index.min()).astype('timedelta64[m]').astype(np.int64)
        C_true =  (ft.index - df.index.min()).astype('timedelta64[m]').astype(np.int64)

        ttc, ctt = temporal_distance(C_true, C_detect, div=60) # maximum delay of 12 hours
        l = abs(len(C_true) - len(C_detect))
        
        print("{:.2f} {:.2f} {:.2f} {}".format(ttc+ctt, ttc, ctt, l))
    
        results.append(
        {   'run'           : run,
            'model'         : name,
            'TD'            : ttc+ctt,
            'TTC'           : ttc,
            'CTT'           : ctt,
            'l'             : l
        })
    
df_results = pd.DataFrame(results)
df_results_group = df_results.groupby('model').agg(['mean','std'])
df_results_group.to_csv('results.csv')


# Plot examples
samples = [('2018-12-16 23:30:00','2018-12-17 16:30:00'),
           ('2019-01-18 00:00:00','2019-01-18 18:00:00'),
           ('2018-10-17 19:30:00','2018-10-18 6:30:00'),]

from matplotlib.lines import Line2D

legend_elements = [Line2D([0], [0], color='blue', lw=1, ls='--', label='Threshold'),
                   Line2D([0], [0], ls='None', marker='.', color='b', label='Detection'),
                   Line2D([0], [0], ls='None', marker='x',color='r', label='Fault')]


for start, end in samples:
    f, ax = plt.subplots(figsize=(12,2))
    x = score.loc[start:end].resample('5T').mean()
    y = score.loc[score.anomaly == True].resample('5T').mean()
    ax = x['score'].plot(ax=ax, c='0.4',lw=1)
    ax.set_xlim(start, end)
    plt.axhline(y=th, c='blue',lw=1,ls='--')
    f = ft.loc[start:end]
    ax.scatter(y.index.values, th*y.anomaly.values, marker='.', c='blue')
    ax.scatter(f.index.values, np.repeat(th, len(f)), marker='x', c='red')
    plt.xlabel('Time')
    plt.ylabel('Score')
    lgd = plt.legend(handles=legend_elements, loc='lower center', framealpha=1)#, bbox_to_anchor=(1,1), loc="upper left")

    plt.show()
    
#    plt.savefig(start.split(' ')[0]+'.jpg', dpi=300,format='jpg', bbox_inches='tight')#, bbox_extra_artists=(lgd,), bbox_inches='tight')
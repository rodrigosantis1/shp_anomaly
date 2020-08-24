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

def temporal_distance2(C_true, C_detect, max_p=None, div=1):
    ttc = np.full(len(C_true), 0)
    ctt = np.full(len(C_detect), 0)
    for i, anomaly in enumerate(C_true):
        delta = (C_detect - anomaly)
        ttc[i]  = min(abs(delta))
    for i, detection in enumerate(C_detect):
        delta = (C_true - detection)
        ctt[i] = min(abs(delta))
    return ttc.sum()/div, ctt.sum()/div

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

models = [#('PCA', make_pipeline(PCA())),
          #('KICA-PCA', make_pipeline(RBFSampler(n_components=100), FastICA(whiten=True), PCA(n_components=20))),
          ('iForest', IsolationForest(contamination=0.06,n_jobs=-1, max_samples=2048, n_estimators=500)) #random_state=15,
        ]

# Read data
train = pd.read_csv('../data/train.csv', parse_dates=['t'], index_col=0)
test = pd.read_csv('../data/test.csv', parse_dates=['t'], index_col=0)
ft = pd.read_csv('../data/faults.csv', parse_dates=['t'],index_col=0)
df = pd.concat([train,test])

# Dataset 3D visualization
fig = plt.figure(figsize=(8,7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(df['V6'],df['V5'],df[['V1','V2','V3','V4']].mean(axis=1),
                cmap='twilight_shifted', alpha=0.5)
ax.set_xlabel('Apparent Power'), ax.set_ylabel('LHU Inflow'), ax.set_zlabel('Average Vibration')
ax.set_xlim(1000,6095)
ax.set_ylim(13.2,23.3)
ax.set_zlim(0.1,0.8)
plt.savefig('../img/dataset.jpg', dpi=600, format='jpg', bbox_inches='tight')

#2**7, 2**8, 2**9, 2**10, 2**11, 2**12, 2**13

# Simmulation
N_RUN=150
results = []
for name, model in models:
    print(name)
    for run in range(N_RUN):
        np.random.seed(run)
        
        #model.fit(train)
        if (name != 'iForest'):
            model.fit(train)
            X_t  = model.transform(test)
            z = hotelling_t2(X_t, model['pca'].explained_variance_)
            th = threshold_t2(n=len(test), alpha=X_t.shape[1], CI=0.95)
            an = z > th
        else:
            model.fit(train)
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
df_results_group.to_csv('../out/results.csv')


# Plot examples
samples = [('2018-12-16 23:30:00','2018-12-17 16:30:00'),
           ('2019-01-18 00:00:00','2019-01-18 18:00:00'),
           ('2018-10-17 19:30:00','2018-10-18 6:30:00'),]

from matplotlib.lines import Line2D

legend_elements = [Line2D([0], [0], color='blue', lw=1, ls='--', label='Threshold'),
                   Line2D([0], [0], ls='None', marker='.', color='b', label='Detection'),
                   Line2D([0], [0], ls='None', marker='x',color='r', label='Fault')]

f, axs = plt.subplots(3,figsize=(12,7))
for i, sample in enumerate(samples):
    start, end = sample[0], sample[1]
    x = score.loc[start:end].resample('5T').mean().abs()
    y = score.loc[score.anomaly == True].resample('5T').mean()
    th = x.threshold.values[0]
    x['score'].plot(ax=axs[i], c='0.4',lw=1)
    axs[i].set_xlim(start, end)
    axs[i].axhline(y=th, c='blue',lw=1,ls='--')
    f = ft.loc[start:end]
    axs[i].scatter(y.index.values, th*y.anomaly.values, marker='.', c='blue')
    axs[i].scatter(f.index.values, np.repeat(th, len(f)), marker='x', c='red')
    axs[i].set_xlabel('')
    axs[i].set_ylabel('Score')
    axs[i].spines['right'].set_visible(False)
    axs[i].spines['top'].set_visible(False)
    lgd = plt.legend(handles=legend_elements, loc='lower left', framealpha=1)
    plt.tight_layout()
plt.savefig('../img/anomalies.pdf', format='pdf', bbox_inches='tight')



f, ax = plt.subplots(figsize=(6,4))
x = abs(score.score.values)
mov_avg = abs(score.score.rolling(window=48).mean().values)
ax.plot(x, c='grey', lw=0.5)
ax.plot(mov_avg, c='green',lw=2)
#x[x<=th] = np.nan
#ax.plot(x, 'k.-')
ax.axhline(y=th, c='blue',ls='--',lw=1)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim(0,len(test))
ax.set_ylabel('Score')
ax.set_xlabel('Observation')
plt.legend(handles=[Line2D([0], [0], color='blue', lw=1, ls='--', label='Threshold'),
                    Line2D([0], [0], color='green', lw=2, ls='-', label='Moving Average (m=48)')])
plt.savefig('../img/score.pdf', format='pdf', bbox_inches='tight')


f, ax = plt.subplots(figsize=(6,5))
start = ft.index[0]
for end in ft.index[1:]:
    l = score[start:end].score.values
    #l = test[start:end]['V1'].cumsum().values #.drop(['V5','V6'],axis=1)
    #l = (l - np.mean(l)) / np.std(l)
    if (len(l) > 10):    
        ax.plot(-l[-15:], lw=1)
    start = end
ax.axhline(y=th, c='blue',ls='--',lw=1)

## NOT USED PLOTS


cmaps = ['Blues','Greens','Oranges','Reds']
cs = ['blue','green','orange','red']
markers = ['o','*','>']

fig = plt.figure(figsize=(8,7))
ax = fig.add_subplot(111, projection='3d')

for i, sample in enumerate(samples):
    start, end = sample[0], sample[1]
    df2 = df.loc[start:end]
    colors = plt.cm.jet(np.linspace(0,1,len(df2)))
    ax.plot(df2['V6'],df2['V5'],df2[['V1','V2','V3','V4']].mean(axis=1), c=cs[i], alpha=0.5)
    ax.scatter(df2['V6'],df2['V5'],df2[['V1','V2','V3','V4']].mean(axis=1),
                    c=np.arange(len(df2)), marker=markers[i],
                    cmap=cmaps[i], alpha=0.5)    
    #plt.figure()
ax.set_xlabel('Apparent Power'), ax.set_ylabel('LHU Inflow'), ax.set_zlabel('Average Vibration')
ax.set_xlim(1000,6095)
ax.set_ylim(13.2,23.3)
ax.set_zlim(0.1,0.8)
plt.savefig('../img/dataset2.jpg', dpi=600, format='jpg', bbox_inches='tight')






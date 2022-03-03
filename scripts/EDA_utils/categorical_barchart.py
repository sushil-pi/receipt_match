# -*- coding: utf-8 -*-
"""
@author: Sushil Kumar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Change this folder path with the path where the repo is cloned
root_folder = 'E:/tide/receipt_match'


def plot_insight1(data,destination):
    df=data
    fig, ax1 = plt.subplots()
    t = df.iloc[:,0]
    s1 = round((df.iloc[:,1]/sum(df.iloc[:,1]))*100, 2)
    ax1.bar(t, s1)
    ax1.set_xlabel(df.columns[0])
    for i, v in enumerate(s1):
        ax1.text(i-0.2, v - .25, str(v)+' %', color='black', fontweight='bold')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('% of Records', color='b')
    ax1.tick_params('y', colors='b')
  
    fig.tight_layout()
    plt.savefig(destination+df.columns[0]+'.png')    
    
def visualize(df,varlist,destination):
    if (not os.path.exists(destination)): 
        os.makedirs(destination)

    for x in varlist:
        
        if str(df[x].dtype).startswith(('int','float')) and df[x].nunique()>10:
            pass
        
        else:
            tmp = df[[x]]
            tmp.columns = ['xvar']
            tmp['flag'] = 1
            aggr = tmp.groupby('xvar').agg({'flag':['count']})
            aggr.reset_index(inplace=True)
            aggr.columns = [x,'total']
            aggr[x] = aggr[x].astype(str)
            aggr = aggr[[x,'total']]
            plot_insight1(data=aggr,destination=destination)
            
  
## Read data
df = pd.read_csv(root_folder + '/data/data_interview_test.csv', sep=':')


df = df[['DateMappingMatch', 'AmountMappingMatch',
       'DescriptionMatch', 'DifferentPredictedTime', 'TimeMappingMatch',
       'PredictedNameMatch', 'ShortNameMatch', 'DifferentPredictedDate',
       'PredictedAmountMatch', 'PredictedTimeCloseMatch']]

v = list(df.columns)            

visualize(df=df, varlist=v,destination= root_folder + '/insights/bar_chart/')

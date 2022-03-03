
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedShuffleSplit
import warnings
warnings.filterwarnings("ignore")

# Change this folder path with the path where the repo is cloned
root_folder = 'E:/tide/receipt_match'

#from sklearn.datasets import load_iris
def decile(x):
    decile10 = x.quantile(np.arange(0,1,0.1), interpolation='nearest')
    decile10.reset_index(drop=True,inplace=True)
    y= np.repeat(-1,len(x))
    for i in range(0,10):
        y[np.where(x>=decile10[i])] = i
    return y  

def OptiBin(df_input,min_buc_size_perc,n):
    X = df_input
    min_s = round((min_buc_size_perc/100)*len(X))
    X.columns = ['xvar','target']
    X=X.dropna(subset=['xvar'])
    m1= min(X['xvar'])
    cut_points = []
    tdf = [X]    
    dt = DecisionTreeClassifier(random_state=2019,max_depth=1,max_leaf_nodes=2,min_samples_leaf=min_s,min_impurity_decrease=0.0007)    
    for i in range(n):
        k=0
        for j in range((2**i)-1,(2**(i+1))-1):
            if i==0:
                dt.fit(tdf[j][['xvar']],tdf[j][['target']])
                cut_points.append(dt.tree_.threshold[0])        
            else:
                if j%2==1:
                    X = tdf[j-k-(2**(i-1))]
                    tdf.append(X[X['xvar'] <= cut_points[j-k-(2**(i-1))]])
                else: 
                    X = tdf[j-k-1-(2**(i-1))]
                    tdf.append(X[X['xvar'] > cut_points[j-k-1-(2**(i-1))]])
                    k=k+1
                if len(tdf[j])>0:     
                    dt.fit(tdf[j][['xvar']],tdf[j][['target']])
                    cut_points.append(dt.tree_.threshold[0]) 
                else: 
                    cut_points.append(-2.0)  
    cut_points.append(m1-1)                
    cut_points = np.array(cut_points)
    cut_points = cut_points[cut_points!=-2.0]
    cut_points = np.sort(cut_points)
    y= np.repeat(-1,len(df_input))
    for l in range(0,len(cut_points)):
        y[np.where(df_input['xvar']>cut_points[l])] = l
    return y 

   
    
def information_value(df,target,min_buc_size_perc,max_buc,opt_bin):
    varlist = list(df.columns)
    varlist.remove(target)
    v_name = []
    iv = []
    empty =[]
    datatype = []
    for x in varlist:
        
        if str(df[x].dtype).startswith(('int','float')) and df[x].nunique()>10:
            tmp = df[[x,target]]
            tmp.columns = ['xvar','target']
            nl = int(np.floor(np.log2(max_buc)))
            if opt_bin==1:
                tmp['deciles'] = OptiBin(df_input=tmp,min_buc_size_perc=min_buc_size_perc,n=nl)
            else:
                tmp['deciles'] = decile(x=tmp['xvar'])
            aggr = tmp.groupby('deciles').agg({'xvar':['min','max'],'target':['count','sum']})
            aggr.columns = ['min','max','total','bad']
            aggr['min'] = round(aggr['min'],2)
            aggr['max'] = round(aggr['max'],2)
            aggr[x] = np.where(aggr['min']==aggr['max'],aggr['min'],aggr['min'].map(str)+' - '+aggr['max'].map(str))
            aggr[x] = aggr[x].astype(str)
            aggr['good'] = aggr['total'] - aggr['bad']
            sum_bad = np.sum(aggr['bad'])
            sum_good = np.sum(aggr['good'])
            aggr['bad_prop'] = aggr['bad']/sum_bad
            aggr['good_prop'] = aggr['good']/sum_good
            aggr['woe'] = (aggr['bad_prop'] - aggr['good_prop'])*(np.log(aggr['bad_prop']/aggr['good_prop']))
            aggr['empty_bins'] = np.where((aggr['bad']==0) | (aggr['good']==0),1,0) 
            aggr['woe'] = np.where((aggr['bad']==0) | (aggr['good']==0),0,aggr['woe']) 
        
        else:
            tmp = df[[x,target]]
            tmp.columns = ['xvar','target']
            aggr = tmp.groupby('xvar').agg({'target':['count','sum']})
            aggr.reset_index(inplace=True)
            aggr.columns = [x,'total','bad']
            aggr[x] = aggr[x].astype(str)
            aggr['good'] = aggr['total'] - aggr['bad']
            sum_bad = np.sum(aggr['bad'])
            sum_good = np.sum(aggr['good'])
            aggr['bad_prop'] = aggr['bad']/sum_bad
            aggr['good_prop'] = aggr['good']/sum_good
            aggr['woe'] = (aggr['bad_prop'] - aggr['good_prop'])*(np.log(aggr['bad_prop']/aggr['good_prop']))
            aggr['empty_bins'] = np.where((aggr['bad']==0) | (aggr['good']==0),1,0) 
            aggr['woe'] = np.where((aggr['bad']==0) | (aggr['good']==0),0,aggr['woe']) 
            
        v_name.append(x)
        iv.append(np.sum(aggr['woe']))
        empty.append(np.sum(aggr['empty_bins']))
        datatype.append(str(df[x].dtype))  
        
    op=pd.DataFrame({"var":v_name,"iv":iv,"empty_bins":empty,"data_type":datatype})
    return op    
            
         
## Read data
df = pd.read_csv(root_folder + '/data/data_interview_test.csv', sep=':')
df['target'] = np.where(df['matched_transaction_id']==df['feature_transaction_id'], 1 , 0)
df = df[['DateMappingMatch', 'AmountMappingMatch',
       'DescriptionMatch', 'DifferentPredictedTime', 'TimeMappingMatch',
       'PredictedNameMatch', 'ShortNameMatch', 'DifferentPredictedDate',
       'PredictedAmountMatch', 'PredictedTimeCloseMatch', 'target']]


iv_df = information_value(df=df, target='target', min_buc_size_perc=1, max_buc=8, opt_bin=0)


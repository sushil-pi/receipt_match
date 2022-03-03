# -*- coding: utf-8 -*-
"""
@author: sushil
"""

import pandas as pd
import numpy as np
import joblib

# Change this folder path with the path where the repo is cloned
root_folder = 'E:/tide/receipt_match'

## Assuming matching vector is already calculated receipt id and transaction id and stored in data df
df = pd.read_csv(root_folder + '/data/data_interview_test.csv', sep=':')

lm_estimator = joblib.load(root_folder + '/trained_models/clf_receipt_match_model.pkl')

def match_batch_receipt_txn(data):
    try:
        input_data = data.copy()
        X = input_data[['DateMappingMatch', 'TimeMappingMatch','DescriptionMatch', 'ShortNameMatch', 'PredictedTimeCloseMatch', 'PredictedNameMatch']].values
        input_data['match_score'] = lm_estimator.predict_proba(X)[:, 1]
        input_data['match_status'] = np.where(input_data['match_score']>0.5, 1, 0)
        return input_data
    except:
        return 'Error'



def match_new_receipt(uploaded_receipt):
    try:
        tdf = df[df['receipt_id']==uploaded_receipt]
        X = tdf[['DateMappingMatch', 'TimeMappingMatch','DescriptionMatch', 'ShortNameMatch', 'PredictedTimeCloseMatch', 'PredictedNameMatch']].values
        tdf['match_score'] = lm_estimator.predict_proba(X)[:, 1]
        tdf = tdf[tdf['match_score']>0.5]
        if len(tdf)>0:
            output = tdf.sort_values(['match_score', 'feature_transaction_id'], ascending=[False, True])
        else:
            output = 'No Transactions Matched'
        return output
    except:
        return 'Error while searching'

# Get all the matching transactions by passing the receipt id
receipt_op = match_new_receipt('10,001')

# Generate output in batch mode
batch_op = match_batch_receipt_txn(df)

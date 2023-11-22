#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 17:16:21 2023

@author: dsimbandumwe
"""


import math
import numpy as np
import pandas as pd
from statsmodels.tools import eval_measures
from sklearn.metrics import mean_squared_error, r2_score







############################################################################
##
## Functions
##
############################################################################


def r2(actual: np.ndarray, predicted: np.ndarray):
    """ R2 Score """
    return r2_score(actual, predicted)

def adjr2(actual: np.ndarray, predicted: np.ndarray, rowcount: int, featurecount: int):
    """ R2 Score """
    return 1-(1-r2(actual,predicted))*(rowcount-1)/(rowcount-featurecount)

def mse(actual, predicted):
    return mean_squared_error(actual, predicted)

def rmse(actual, predicted):
    return math.sqrt(mse(actual, predicted))

def model_results(name, actual, predicted, rowcount, featurecount, r_df):
    #print('r-squared: ', round(r2(actual, predicted),4))
    #print('adj r-squared', round(adjr2(actual,predicted,rowcount, featurecount),4))
    #print('mse: ', round(mse(actual, predicted),4))
    #print('rmse: ', round(rmse(actual, predicted),4))
    #print('rmse: ', eval_measures.rmse(actual,predicted,axis=0))
    
    if (r_df is None):
        r_df = pd.DataFrame(columns = ['name','r_sq','adj_r_sq','mse','rmse'])
        
        
    new_row = { 'name' : name,
                'r_sq': r2(actual, predicted), 
                'adj_r_sq': adjr2(actual,predicted,rowcount, featurecount), 
                'mse': mse(actual, predicted),
                'rmse' :rmse(actual, predicted)}
    
    r_df = r_df.append(new_row, ignore_index=True)
    return round(r_df,4)
    
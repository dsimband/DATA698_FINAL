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
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss






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

def model_results(name, actual, predicted, aic, r_df):
    #print('r-squared: ', round(r2(actual, predicted),4))
    #print('adj r-squared', round(adjr2(actual,predicted,rowcount, featurecount),4))
    #print('mse: ', round(mse(actual, predicted),4))
    #print('rmse: ', round(rmse(actual, predicted),4))
    #print('rmse: ', eval_measures.rmse(actual,predicted,axis=0))
    
    if (r_df is None):
        r_df = pd.DataFrame(columns = ['name','mse','rmse','mape','mae','aic'])
        
        
    new_row = { 'name' : name,
                #'r_sq': r2(actual, predicted), 
                #'adj_r_sq': adjr2(actual,predicted,rowcount, featurecount), 
                'mse': mean_squared_error(actual, predicted),
                'rmse' :rmse(actual, predicted),
                'mape' :mean_absolute_percentage_error(actual, predicted),
                'mae' : mean_absolute_error(actual, predicted),
                'aic' : aic
                }
    
    r_df = r_df.append(new_row, ignore_index=True)
    return round(r_df,4)
  



def adf_test(timeseries):
    print("Results of Dickey-Fuller Test:")
    dftest = adfuller(timeseries, autolag="AIC")
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    print(dfoutput)

    



def kpss_test(timeseries):
    print("Results of KPSS Test:")
    kpsstest = kpss(timeseries, regression="c", nlags="auto")
    kpss_output = pd.Series(
        kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
    )
    for key, value in kpsstest[3].items():
        kpss_output["Critical Value (%s)" % key] = value
    print(kpss_output)

  
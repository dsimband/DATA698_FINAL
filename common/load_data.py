#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 17:11:22 2023

@author: dsimbandumwe
"""




import pandas as pd
import numpy as np
import math
from itertools import cycle
import statistics

#from datetime import datetime
import datetime


from fredapi import Fred
import pandas_datareader as pdr

import warnings



from common.load_features import get_recession, add_recession_feature, get_fed_chair, add_fed_chair

warnings.filterwarnings("ignore")




############################################################################
##
## COnfig and Data
##
############################################################################



fred = Fred(api_key='c0a3f23bdd23a65e6546b6d0e5f4d4a5')
rand_int = 12

#  Set start date
start_date = datetime.date(1940, 1, 1)
start_date_str = datetime.datetime.strftime(start_date, "%Y-%m-%d")

#  Federal Reserve Economic Data Service
data_source = 'fred'






############################################################################
##
## Functions
##
############################################################################


def load_taylor():
    
    
    # Variables
    target_inf = 2
    full_emp = 5
    alpha = 0.5
    beta = 0.5
    
    
    t1_df = pdr.DataReader(['FEDFUNDS','UNRATE','TB3MS'], data_source, start_date)
    t1_df.index.rename('observation_date', inplace=True)
    t1_df.reset_index(inplace=True)
    print('t1_df:', t1_df.shape)

    gdpc1_df = pd.read_csv('./data/GDPC1.csv', parse_dates=['observation_date'], skiprows=10)
    print('gdpc1_df:', gdpc1_df.shape)
    gdppot_df = pd.read_csv('./data/GDPPOT.csv', parse_dates=['observation_date'], skiprows=10)
    print('gdppot_df:', gdppot_df.shape)
    gdpdef_df = pd.read_csv('./data/GDPDEF.csv', parse_dates=['observation_date'], skiprows=10)
    print('gdpdef_df:', gdpdef_df.shape)
    cpi_df = pd.read_csv('./data/CPIAUCSL_PC1.csv', parse_dates=['observation_date'])
    print('cpi_df:', cpi_df.shape)
    holston_df = pd.read_csv('./data/Holston_Laubach_Williams_real_time_estimates.csv', parse_dates=['observation_date'], skiprows=5)
    print('holston_df:', holston_df.shape)
    

    
    taylor_df = t1_df.merge(gdpc1_df, how='outer', left_on='observation_date', right_on='observation_date')   \
                        .merge(gdppot_df, how='outer', left_on='observation_date', right_on='observation_date')   \
                        .merge(gdpdef_df, how='outer', left_on='observation_date', right_on='observation_date')   \
                        .merge(cpi_df, how='outer', left_on='observation_date', right_on='observation_date')      \
                        .merge(holston_df, how='outer', left_on='observation_date', right_on='observation_date')


    taylor_df.set_index('observation_date', inplace=True)
    taylor_df.index.rename('DATE', inplace=True)

    #taylor_df.dropna(inplace=True)
    taylor_df = taylor_df.resample('Q').mean()
    #taylor_df = taylor_df.resample('Q').mean()



    taylor_df['gap_inf'] = (taylor_df['GDPDEF_PC1'] - target_inf) 
    taylor_df['gap_gdp'] = (taylor_df['GDPC1'] - taylor_df['GDPPOT']) / taylor_df['GDPPOT'] * 100
    taylor_df['gap_ue'] = (full_emp - taylor_df['UNRATE'])



    #taylor_df = pd.DataFrame()
    taylor_df['ffef_tr'] = taylor_df['GDPDEF_PC1'] + 2    \
                            + beta * taylor_df['gap_inf']   \
                            + alpha * taylor_df['gap_gdp']


    #
    taylor_df['ffef_tr2'] = taylor_df['GDPDEF_PC1'] + 2    \
                            + beta * taylor_df['gap_inf']   \
                            + alpha * taylor_df['gap_ue']
    

    #taylor_df.rename(columns={'observation_date':'DATE'}, inplace=True)
    
    taylor_df.reset_index(inplace=True)
    r_df = get_recession()
    taylor_df['recession_flag'] = taylor_df['DATE'].apply(add_recession_feature, args=(r_df,))
    taylor_df.set_index('DATE',inplace=True)
    
    f_df = get_fed_chair()
    taylor_df = add_fed_chair(taylor_df, f_df)
    
    taylor_df['FEDFUNDS-1'] = taylor_df['FEDFUNDS'].shift(periods=1)
    taylor_df = taylor_df.query('DATE >= "1961-01-01" & DATE < "2023-09-30"')
    return taylor_df





def load_taylor1a():
    
    
    # Variables
    target_inf = 2
    full_emp = 5
    alpha = 0.5
    beta = 0.5
    const = 10
    
    
    t1_df = pdr.DataReader(['FEDFUNDS','UNRATE','TB3MS'], data_source, start_date)
    t1_df.index.rename('observation_date', inplace=True)
    t1_df.reset_index(inplace=True)
    print('t1_df:', t1_df.shape)

    gdpc1_df = pd.read_csv('./data/GDPC1.csv', parse_dates=['observation_date'], skiprows=10)
    print('gdpc1_df:', gdpc1_df.shape)
    gdppot_df = pd.read_csv('./data/GDPPOT.csv', parse_dates=['observation_date'], skiprows=10)
    print('gdppot_df:', gdppot_df.shape)
    gdpdef_df = pd.read_csv('./data/GDPDEF.csv', parse_dates=['observation_date'], skiprows=10)
    print('gdpdef_df:', gdpdef_df.shape)
    holston_df = pd.read_csv('./data/Holston_Laubach_Williams_real_time_estimates.csv', parse_dates=['observation_date'], skiprows=5)
    print('holston_df:', holston_df.shape)
    

    
    taylor_df = t1_df.merge(gdpc1_df, how='outer', left_on='observation_date', right_on='observation_date')   \
                        .merge(gdppot_df, how='outer', left_on='observation_date', right_on='observation_date')   \
                        .merge(gdpdef_df, how='outer', left_on='observation_date', right_on='observation_date')   \
                        .merge(holston_df, how='outer', left_on='observation_date', right_on='observation_date') 


    taylor_df.set_index('observation_date', inplace=True)
    taylor_df.index.rename('DATE', inplace=True)

    #taylor_df.dropna(inplace=True)
    taylor_df = taylor_df.resample('Q').mean()
    #taylor_df = taylor_df.resample('Q').mean()
    
    
    
    taylor_df['GDPC1_log'] = taylor_df['GDPC1'] + const
    taylor_df['GDPPOT_log'] = taylor_df['GDPPOT'] + const
    taylor_df['GDPDEF_PC1_log'] = taylor_df['GDPDEF_PC1'] + const
    taylor_df['UNRATE_log'] = taylor_df['UNRATE'] + const
    taylor_df['FEDFUNDS_log'] = taylor_df['FEDFUNDS'] + const
    
    
    taylor_df['GDPC1_log'] = taylor_df['GDPC1_log'].apply(math.log)
    taylor_df['GDPPOT_log'] = taylor_df['GDPPOT_log'].apply(math.log)
    taylor_df['GDPDEF_PC1_log'] = taylor_df['GDPDEF_PC1_log'].apply(math.log)
    taylor_df['UNRATE_log'] = taylor_df['UNRATE_log'].apply(math.log)
    taylor_df['FEDFUNDS_log'] = taylor_df['FEDFUNDS_log'].apply(math.log)
    



    #taylor_df['gap_inf'] = (taylor_df['GDPDEF_PC1'] - target_inf) 
    #taylor_df['gap_gdp'] = (taylor_df['GDPC1_log'] - taylor_df['GDPPOT_log'])
    #taylor_df['gap_ue'] = (full_emp - taylor_df['UNRATE'])
    #taylor_df['gap_gdp'] = (taylor_df['GDPC1'] - taylor_df['GDPPOT']) / taylor_df['GDPPOT'] * 100
    
    taylor_df['gap_inf'] = (taylor_df['GDPDEF_PC1_log'] - math.log(target_inf)) 
    taylor_df['gap_gdp'] = (taylor_df['GDPC1_log'] - taylor_df['GDPPOT_log'])
    taylor_df['gap_ue'] = (math.log(full_emp) - taylor_df['UNRATE_log'])



    #taylor_df = pd.DataFrame()
    taylor_df['ffef_tr'] = taylor_df['GDPDEF_PC1_log'] + math.log(2 + const)    \
                            + beta * taylor_df['gap_inf']   \
                            + alpha * taylor_df['gap_gdp']


    #
    taylor_df['ffef_tr2'] = taylor_df['GDPDEF_PC1_log'] + math.log(2 + const)    \
                            + beta * taylor_df['gap_inf']   \
                            + alpha * taylor_df['gap_ue']
    

    #taylor_df.rename(columns={'observation_date':'DATE'}, inplace=True)
    
    taylor_df.reset_index(inplace=True)
    r_df = get_recession()
    taylor_df['recession_flag'] = taylor_df['DATE'].apply(add_recession_feature, args=(r_df,))
    taylor_df.set_index('DATE',inplace=True)
    
    f_df = get_fed_chair()
    taylor_df = add_fed_chair(taylor_df, f_df)
    
    taylor_df['FEDFUNDS-1'] = taylor_df['FEDFUNDS'].shift(periods=1)
    taylor_df = taylor_df.query('DATE >= "1961-01-01" & DATE < "2023-09-30"')
    return taylor_df







def load_taylor2():
    
    # Variables
    target_inf = 2
    full_emp = 5
    alpha = 0.5
    beta = 0.5
    
    
    pcep_df = pd.read_csv('./data/PCEPILFE.csv', parse_dates=['observation_date'], skiprows=10)
    print('pcep_df:', pcep_df.shape)
    
    gdpdef_df = pd.read_csv('./data/GDPDEF.csv', parse_dates=['observation_date'], skiprows=10)
    print('gdpdef_df:', gdpdef_df.shape)
    
    holston_df = pd.read_csv('./data/Holston_Laubach_Williams_real_time_estimates.csv', parse_dates=['observation_date'], skiprows=5)
    print('holston_df:', holston_df.shape)

    t1_df = pdr.DataReader(['FEDFUNDS','PCEPILFE','GDPC1','GDPPOT','UNRATE','TB3MS'], data_source, start_date)
    t1_df.index.rename('observation_date', inplace=True)
    t1_df.reset_index(inplace=True)

    t1_df = t1_df.merge(pcep_df, how='outer', left_on='observation_date', right_on='observation_date')   \
                .merge(gdpdef_df, how='outer', left_on='observation_date', right_on='observation_date')  \
                .merge(holston_df, how='outer', left_on='observation_date', right_on='observation_date') 

    t1_df.set_index('observation_date', inplace=True)
    t1_df.index.rename('DATE', inplace=True)
    
    # calculate quarterly data
    t1_df = t1_df.resample('Q').mean()
    

    t1_df['rLR'] = t1_df['TB3MS'] - t1_df['GDPDEF_PC1']
    t1_df['GDPC1_log'] = t1_df['GDPC1'].apply(math.log)
    t1_df['GDPPOT_log'] = t1_df['GDPPOT'].apply(math.log)
    
    
    # Calculate Taylor Rules 1&2
    t1_df['gap_gdp'] = (t1_df['GDPC1_log'] - t1_df['GDPPOT_log'])
    t1_df['gap_ue'] = (full_emp - t1_df['UNRATE'])
    t1_df['gap_inf'] = (t1_df['PCEPILFE_CH1'] - target_inf)
    
    

    t1_df['ffef_tr'] = t1_df['rLR']  \
                            + t1_df['PCEPILFE_CH1']  \
                            + (alpha * t1_df['gap_gdp'])  \
                            + (beta * t1_df['gap_inf'])

    t1_df['ffef_tr2'] = t1_df['rLR']  \
                            + t1_df['PCEPILFE_CH1']  \
                            + (alpha * t1_df['gap_ue'])  \
                            + (beta * t1_df['gap_inf'])
    
    
    t1_df.reset_index(inplace=True)
    r_df = get_recession()
    t1_df['recession_flag'] = t1_df['DATE'].apply(add_recession_feature, args=(r_df,))
    t1_df.set_index('DATE',inplace=True)
    
    f_df = get_fed_chair()
    t1_df = add_fed_chair(t1_df, f_df)
    

    print('t1_df:', t1_df.shape)
    #t1_df.dropna(inplace=True)
    #t1_df.sort_index(inplace=True)
    t1_df['FEDFUNDS-1'] = t1_df['FEDFUNDS'].shift(periods=1)
    t1_df = t1_df.query('DATE >= "1961-01-01" & DATE < "2023-09-30"')
    
    return t1_df



def load_misery():
    
    t_df = pdr.DataReader(['FEDFUNDS','CPIAUCSL', 'UNRATE'], data_source, start_date)
    t_df.dropna(inplace=True)
    print('t_df:', t_df.shape)

    u_df = pd.read_csv('./data/CPIAUCSL_PC1.csv', parse_dates=['DATE'])
    u_df.set_index('DATE', inplace=True)
    print('u_df:', u_df.shape)

    misery_df = pd.merge(t_df,u_df, left_index=True, right_index=True )
    misery_df['m_index'] = (misery_df['CPIAUCSL_PC1']) + misery_df['UNRATE']
    print('misery_df:', misery_df.shape)  
    
    return misery_df



def time_split(df):
    t_df = df.copy()

    # Split Data
    splt_index = round(t_df.shape[0] * 0.8)
    train_df = t_df[:splt_index]
    print('train_df: ' , train_df.shape)

    end_index = splt_index - t_df.shape[0]
    test_df = t_df[end_index:]
    print('test_df: ' , test_df.shape)
    
    return train_df, test_df





    
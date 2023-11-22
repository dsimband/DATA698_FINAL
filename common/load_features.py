#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 17:17:00 2023

@author: dsimbandumwe
"""

import pandas as pd






############################################################################
##
## Functions
##
############################################################################



def get_fed_chair():
    df = pd.read_csv('./data/Fed_Chair.csv', parse_dates=['Start_Date','End_Date'])
    df['End_Date'].fillna(pd.Timestamp.now().normalize(), inplace=True)
    df = df[(df['Start_Date'] >= '1945-01-01')]
    return df

def add_fed_chair(df, chair_df):
    
    df.reset_index(inplace=True)

    df['chair_name'] = None
    df['chair_index'] = 0
    for i, row in chair_df.iterrows():
        df['chair_name'] = df['chair_name'].where((df['DATE'] < row['Start_Date']) | (df['DATE'] > row['End_Date']), row['Name'] )
        df['chair_index'] = df['chair_index'].where((df['DATE'] < row['Start_Date']) | (df['DATE'] > row['End_Date']), i )
    
    df.set_index('DATE',inplace=True)
    return df



def get_recession():
    recession_df = pd.read_csv('./data/FRED_Recession_Bars.csv', parse_dates=['Peak_Date','Trough_Date'])
    return recession_df[(recession_df['Peak_Date'] >= '1945-01-01')]


def add_recession_feature(date, recession_df):
    for i,row in recession_df.iterrows():
        if row['Peak_Date'] <= date <= row['Trough_Date']:
            return True
    return False

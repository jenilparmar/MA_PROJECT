# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 22:02:14 2024

@author: modim
"""
import pandas as pd
file = pd.read_csv('D:/MA_PROJECT/data/weatherAUS.csv')


file['Date'] = pd.to_datetime(file['Date'])
file['year'] = file['Date'].dt.year
file['month'] = file['Date'].dt.month
file['day'] = file['Date'].dt.day
file.drop(['Date'], axis = 1,inplace=True) 
file['RainToday'].replace({'No': 0, 'Yes': 1},inplace = True)
file['RainTomorrow'].replace({'No': 0, 'Yes': 1},inplace = True)

print(file.head())

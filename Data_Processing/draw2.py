# data analysis
import xlrd
import pandas as pd
import numpy as np
import random as rnd

# visulization 
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# data processing
from sklearn import preprocessing


df = pd.read_excel('tsdata.xlsx')
grouped = df.groupby('ID')
for i in range(210625,210849):
    dfx = grouped.get_group(i) #.to_csv('test1.csv',encoding='utf_8_sig'
    break





def normalization(dfx):
    range = np.max(dfx['']) - np.min(dfx[''])
    return (dfx[''] - np.min(dfx[''])) / range

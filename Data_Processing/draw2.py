# data analysis
from io import SEEK_SET
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





def normalization(dfn):
    range_n = np.max(dfn - np.min(dfn)

    return (dfn - np.min(dfn)) / range_n


# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 09:52:23 2021

@author: admin
"""

##########################################    TSfresh    ###################################################
import pandas as pd
# import pyecharts.options as opts
# from pyecharts.charts import Line
import numpy as np
import datetime
# from pyecharts.charts import Bar
import tsfresh
import matplotlib.pylab as plt
from tsfresh.examples.robot_execution_failures import download_robot_execution_failures,load_robot_execution_failures
from tsfresh.examples.robot_execution_failures import download_robot_execution_failures,load_robot_execution_failures
import matplotlib.pylab as plt
# import seaborn as sns
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters
from sklearn.tree import DecisionTreeClassifier
# from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from tsfresh.feature_extraction import extract_features, MinimalFCParameters
# settings = MinimalFCParameters()
# extracted_features_3 = extract_features(timeseries, column_id="id", column_sort="time",default_fc_parameters=settings)
# extracted_features_3.shape


# download_robot_execution_failures()
# timeseries, y = load_robot_execution_failures()
# timeseries = pd.read_csv('C:\\Users\THINKPAD\Desktop\\tsfresh\\load_robot_execution_failures_timeseries.csv')
# y = pd.read_csv('C:\\Users\THINKPAD\Desktop\\tsfresh\\load_robot_execution_failures_y.csv')
from tsfresh import extract_features

# df[df.id == 31][['time', 'F_x' ,'F_y',  'F_z',  'T_x',  'T_y',  'T_z']].plot(x='time',title='Success example(id 3)', figsize=(12, 6))
# df[df.id == 73][['time', 'F_x' ,'F_y',  'F_z',  'T_x',  'T_y', 'T_z']].plot(x='time',title='Failure example(id 20)', figsize=(12, 6))
# plt.show()
df = pd.read_excel(r'C:\Users\Luminous Isaac\Documents\GitHub\2021_IIAC_Competition\Data_Processing\Related_Data\tsdata_equal.xlsx')
#加入时间戳
df_ts=df.copy()
df_ts["Time"] = None
g = df_ts.groupby('ID')
# print(df_ts.groupby('ID').cumcount())
df_ts["Time"]=g.cumcount() + 1
df_ts.to_excel(r'C:\Users\Luminous Isaac\Documents\GitHub\2021_IIAC_Competition\Data_Processing\Related_Data\tsdata_带时间.xlsx',index=False,encoding='gb18030')

df=df_ts.copy()

#描述性统计
df_des=df.describe()
df_des.to_excel(r'C:\Users\Luminous Isaac\Documents\GitHub\2021_IIAC_Competition\Data_Processing\Related_Data\ts_描述性统计.xlsx',encoding='gb18030')

df=df[['ID',
 'Time',
 '中部跳动≤0.8',
 'Q处跳动≤0.6',
 'W3.DATAPOINT.X_R',
 'W3.DATAPOINT.X_I',
 'W3.DATAPOINT.Z_R',
 'W3.DATAPOINT.Z_L',
 'W3.DATAPOINT.X2_R',
 'W3.DATAPOINT.X2_L',
 'W3.DATAPOINT.X3_R',
 'W3.DATAPOINT.X3_L',
 'W3.DATAPOINT.FEED_ACT',
 'W3.DATAPOINT.FEED_SET',
 'W3.DATAPOINT.C_R',
 'W3.DATAPOINT.C_L',
 'W3.DATAPOINT.CODE_L1',
 'W3.DATAPOINT.CODE_L2',
 'W3.DATAPOINT.CODE_L3',
 'code1',
 'code2',
 'code3']]

df[['中部跳动≤0.8']]=df[['中部跳动≤0.8']].fillna(method='ffill')
df[['Q处跳动≤0.6']]=df[['Q处跳动≤0.6']].fillna(method='ffill')


 df.isnull().sum()
 ##插值法填充缺失值
df['W3.DATAPOINT.Z_R'] = df['W3.DATAPOINT.Z_R'].interpolate()
df['W3.DATAPOINT.Z_L'] = df['W3.DATAPOINT.Z_L'].interpolate()
df['W3.DATAPOINT.X3_L'] = df['W3.DATAPOINT.X3_L'].interpolate()

df.to_excel(r'C:\Users\Luminous Isaac\Documents\GitHub\2021_IIAC_Competition\Tsfreshts_处理后.xlsx',encoding='gb18030',index=False)
##################tiqu ########################
#
df=pd.read_excel(r'C:\Users\Luminous Isaac\Documents\GitHub\2021_IIAC_Competition\Tsfreshts_处理后.xlsx')
print(type(df))
print(df.dtypes)
df[['ID','Time','W3.DATAPOINT.C_L','code1','code2','code3']] = df[['ID','Time','W3.DATAPOINT.C_L','code1','code2','code3']].astype('float')



dfa = df[['ID','Time', 'W3.DATAPOINT.X_R',
 'W3.DATAPOINT.X_I',
 'W3.DATAPOINT.Z_R',
 'W3.DATAPOINT.Z_L',
 'W3.DATAPOINT.X2_R',
 'W3.DATAPOINT.X2_L',
 'W3.DATAPOINT.X3_R',
 'W3.DATAPOINT.X3_L',
 'W3.DATAPOINT.FEED_ACT',
 'W3.DATAPOINT.FEED_SET',
 'W3.DATAPOINT.C_R',
 'W3.DATAPOINT.C_L',
# 'W3.DATAPOINT.CODE_L1',
# 'W3.DATAPOINT.CODE_L2',
# 'W3.DATAPOINT.CODE_L3',
 'code1',
 'code2',
 'code3']]

def tiqu(i):
    y=df[[i,'ID']].drop_duplicates(subset=['ID'], keep='first')
    y.set_index(['ID'],inplace=True)
    extract_settings = ComprehensiveFCParameters()
    X = extract_features(dfa, column_id='ID', column_sort='Time', default_fc_parameters=extract_settings, impute_function=impute)
    X.to_excel('C:/Users/admin/Desktop/航空航天赛道/全部特征_'+i+'.xlsx')
    X_filtered = extract_relevant_features(dfa, y[i],column_id='ID', column_sort='Time', default_fc_parameters=extract_settings)
    X_filtered.info()
    features_filtered=select_features(X_filtered,y[i])
    # X_train, X_test, X_filtered_train, X_filtered_test, y_train, y_test = train_test_split(X, X_filtered, y, test_size=4)
    X_filtered.to_excel(r'C:\Users\Luminous Isaac\Documents\GitHub\2021_IIAC_Competition\Tsfresh提取特征_'+i+'.xlsx')
    y.to_excel(r'C:\Users\Luminous Isaac\Documents\GitHub\2021_IIAC_Competition\Tsfresh标签_'+i+'.xlsx')



tiqu('中部跳动≤0.8')
tiqu('Q处跳动≤0.6')

df_y=df[['中部跳动≤0.8','ID']].drop_duplicates(subset=['ID'], keep='first')
#y=df[['中部跳动≤0.8','ID']]
y=df_y.copy()
y.set_index(['ID'],inplace=True)
extract_settings = ComprehensiveFCParameters()
X = extract_features(dfa, column_id='ID', column_sort='Time',default_fc_parameters=extract_settings, impute_function=impute)
X.to_excel(r'C:\Users\Luminous Isaac\Documents\GitHub\2021_IIAC_Competition\Tsfresh全部特征_中部跳动.xlsx')
X_filtered = extract_relevant_features(dfa, y['中部跳动≤0.8'],column_id='ID', column_sort='Time', default_fc_parameters=extract_settings)
X_filtered.info()
features_filtered=select_features(X_filtered,y['中部跳动≤0.8'])
# X_train, X_test, X_filtered_train, X_filtered_test, y_train, y_test = train_test_split(X, X_filtered, y, test_size=4)
X_filtered.to_excel(r'C:\Users\Luminous Isaac\Documents\GitHub\2021_IIAC_Competition\Tsfresh提取特征_中部跳动.xlsx')
y.to_excel(r'C:\Users\Luminous Isaac\Documents\GitHub\2021_IIAC_Competition\Tsfresh标签_中部跳动.xlsx')

print(X.shape)
print(y.shape)
print(dfa.shape)

df8y=pd.read_excel(r'C:\Users\Luminous Isaac\Documents\GitHub\2021_IIAC_Competition\Tsfresh标签_中部跳动.xlsx')
df8x=pd.read_excel(r'C:\Users\Luminous Isaac\Documents\GitHub\2021_IIAC_Competition\Tsfresh全部特征_中部跳动.xlsx')
df8=pd.merge(df8y, df8x, how='inner', left_on='ID',right_on='Unnamed: 0')
df8.to_excel(r'C:\Users\Luminous Isaac\Documents\GitHub\2021_IIAC_Competition\Tsfresh回归数据_8.xlsx')


df6y=pd.read_excel(r'C:\Users\Luminous Isaac\Documents\GitHub\2021_IIAC_Competition\Tsfresh标签_Q处跳动≤0.6.xlsx')
df6x=pd.read_excel(r'C:\Users\Luminous Isaac\Documents\GitHub\2021_IIAC_Competition\Tsfresh全部特征_Q处跳动≤0.6.xlsx')
df6=pd.merge(df6y, df6x, how='inner', left_on='ID',right_on='Unnamed: 0')
df6.to_excel(r'C:\Users\Luminous Isaac\Documents\GitHub\2021_IIAC_Competition\Tsfresh回归数据_6.xlsx')
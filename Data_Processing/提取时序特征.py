# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 09:48:21 2021

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
df = pd.read_excel('C:/Users/admin/Desktop/航空航天赛道/tsdata.xlsx')
#加入时间戳
df_ts=df.copy()
df_ts["Time"] = None
g = df_ts.groupby('ID')
df_ts["Time"]=g.cumcount() + 1
df_ts.to_excel('C:/Users/admin/Desktop/航空航天赛道/tsdata_带时间.xlsx',index=False,encoding='gb18030')

df=df_ts.copy()

#描述性统计
df_des=df.describe()
df_des.to_excel('C:/Users/admin/Desktop/航空航天赛道/ts_描述性统计.xlsx',encoding='gb18030')

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

df.to_excel('C:/Users/admin/Desktop/航空航天赛道/ts_处理后.xlsx',encoding='gb18030',index=False)
##################tiqu ########################
#
df=pd.read_excel('C:/Users/admin/Desktop/航空航天赛道/ts_处理后.xlsx')
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
    X_filtered.to_excel('C:/Users/admin/Desktop/航空航天赛道/提取特征_'+i+'.xlsx')
    y.to_excel('C:/Users/admin/Desktop/航空航天赛道/标签_'+i+'.xlsx')



tiqu('中部跳动≤0.8')
tiqu('Q处跳动≤0.6')

df_y=df[['中部跳动≤0.8','ID']].drop_duplicates(subset=['ID'], keep='first')
#y=df[['中部跳动≤0.8','ID']]
y=df_y.copy()
y.set_index(['ID'],inplace=True)
extract_settings = ComprehensiveFCParameters()
X = extract_features(dfa, column_id='ID', column_sort='Time',default_fc_parameters=extract_settings, impute_function=impute)
X.to_excel('C:/Users/admin/Desktop/航空航天赛道/全部特征_中部跳动.xlsx')
X_filtered = extract_relevant_features(dfa, y['中部跳动≤0.8'],column_id='ID', column_sort='Time', default_fc_parameters=extract_settings)
X_filtered.info()
features_filtered=select_features(X_filtered,y['中部跳动≤0.8'])
# X_train, X_test, X_filtered_train, X_filtered_test, y_train, y_test = train_test_split(X, X_filtered, y, test_size=4)
X_filtered.to_excel('C:/Users/admin/Desktop/航空航天赛道/提取特征_中部跳动.xlsx')
y.to_excel('C:/Users/admin/Desktop/航空航天赛道/标签_中部跳动.xlsx')

print(X.shape)
print(y.shape)
print(dfa.shape)

df8y=pd.read_excel('C:/Users/admin/Desktop/航空航天赛道/标签_中部跳动.xlsx')
df8x=pd.read_excel('C:/Users/admin/Desktop/航空航天赛道/全部特征_中部跳动.xlsx')
df8=pd.merge(df8y, df8x, how='inner', left_on='ID',right_on='Unnamed: 0')
df8.to_excel('C:/Users/admin/Desktop/航空航天赛道/回归数据_8.xlsx')


df6y=pd.read_excel('C:/Users/admin/Desktop/航空航天赛道/标签_Q处跳动≤0.6.xlsx')
df6x=pd.read_excel('C:/Users/admin/Desktop/航空航天赛道/全部特征_Q处跳动≤0.6.xlsx')
df6=pd.merge(df6y, df6x, how='inner', left_on='ID',right_on='Unnamed: 0')
df6.to_excel('C:/Users/admin/Desktop/航空航天赛道/回归数据_6.xlsx')




##############画图###########

df2 = pd.read_excel(r'C:/Users/admin/Desktop/航空航天赛道/numbdata_2.xlsx')
df = pd.read_excel(r'C:/Users/admin/Desktop/航空航天赛道/tsdata_equal.xlsx')
df2=df2[['工件编号','Q处跳动','中部跳动']]
df=pd.merge(df, df2, left_on='ID',right_on='工件编号', how='left')
#########################################################################
print(df2.dtypes)
#df2[['Q处跳动']]=df2[['Q处跳动']].astype('str')

df2_T=df2[df2['Q处跳动'] <= 0.1]
df2_F=df2[df2['Q处跳动'] >= 0.6]

df2_T=df2_T[['工件编号','Q处跳动','中部跳动']]
df2_F=df2_F[['工件编号','Q处跳动','中部跳动']]


#df_T=df_T.dropna(axis=0,subset = ["工件编号"])   # 丢弃‘工件编号’这列中有缺失值的行 
#df_F=df_F.dropna(axis=0,subset = ["工件编号"])

g = df_F.groupby('ID')
df_F["TM"]=g.cumcount() + 1

g = df_T.groupby('ID')
df_T["TM"]=g.cumcount() + 1


df_f_pic = pd.DataFrame(columns=['Unnamed: 0', 'W3.DATAPOINT.X_R', 'W3.DATAPOINT.X_I',
       'W3.DATAPOINT.Z_R', 'W3.DATAPOINT.Z_L', 'W3.DATAPOINT.X2_R',
       'W3.DATAPOINT.X2_L', 'W3.DATAPOINT.X3_R', 'W3.DATAPOINT.X3_L',
       'W3.DATAPOINT.FEED_ACT', 'W3.DATAPOINT.FEED_SET', 'W3.DATAPOINT.C_R',
       'W3.DATAPOINT.C_L', 'code1', 'code2', 'code3', 'ID', 'TM', '工件编号',
       'Q处跳动', '中部跳动'])
for i in range(1,221):
    df_a=df_F[df_F['TM']==i].mean()
    df_f_pic=df_f_pic.append(df_a, ignore_index=True)
    
df_t_pic = pd.DataFrame(columns=['Unnamed: 0', 'W3.DATAPOINT.X_R', 'W3.DATAPOINT.X_I',
       'W3.DATAPOINT.Z_R', 'W3.DATAPOINT.Z_L', 'W3.DATAPOINT.X2_R',
       'W3.DATAPOINT.X2_L', 'W3.DATAPOINT.X3_R', 'W3.DATAPOINT.X3_L',
       'W3.DATAPOINT.FEED_ACT', 'W3.DATAPOINT.FEED_SET', 'W3.DATAPOINT.C_R',
       'W3.DATAPOINT.C_L', 'code1', 'code2', 'code3', 'ID', 'TM', '工件编号',
       'Q处跳动', '中部跳动'])
for i in range(1,221):
    df_a=df_T[df_T['TM']==i].mean()
    df_t_pic=df_t_pic.append(df_a, ignore_index=True)


负样本Q
# 取数据F
df_X_R = df_f_pic['W3.DATAPOINT.X_R']
df_X_I = df_f_pic['W3.DATAPOINT.X_I']
df_Z_R = df_f_pic['W3.DATAPOINT.X_I']
df_Z_L = df_f_pic['W3.DATAPOINT.Z_L']
df_X2_R = df_f_pic['W3.DATAPOINT.X2_R']
df_X2_L = df_f_pic['W3.DATAPOINT.X2_L']
df_X3_R = df_f_pic['W3.DATAPOINT.X3_R']
df_X3_L = df_f_pic['W3.DATAPOINT.X3_L']
df_FEED_ACT = df_f_pic['W3.DATAPOINT.FEED_ACT']
df_FEED_SET = df_f_pic['W3.DATAPOINT.FEED_SET']
df_C_R = df_f_pic['W3.DATAPOINT.C_R']

plt.rcParams['font.sans-serif'] = ['SimHei']
    # 获取时间戳
table = df_f_pic.TM

    # 图像大小
plt.figure(figsize=(15,9))

# 调整子图间距
plt.subplots_adjust(hspace =0.3)

    # 图像名
plt.suptitle("Q处跳动_负样本", fontsize=20, color='black')


# 绘图
plt.subplot(441)
plt.plot(table,df_X_R,color='r', label='X_R')
plt.xlabel('X_R')

plt.subplot(442)
plt.plot(table,df_X_I,color='y', label='X_I')
plt.xlabel('X_I')

plt.subplot(443)
plt.plot(table,df_Z_R,color='g', label='Z_R')
plt.xlabel('Z_R')

plt.subplot(444)
plt.plot(table,df_Z_L,color='c', label='Z_L')
plt.xlabel('Z_L')

plt.subplot(445)
plt.plot(table,df_X2_R,color='m', label='X2_R')
plt.xlabel('X2_R')

plt.subplot(446)
plt.plot(table,df_X2_L,color='k', label='X2_L')
plt.xlabel('X2_L')

plt.subplot(447)
plt.plot(table,df_X3_R,color='SkyBlue',label='X3_R')
plt.xlabel('X3_R')

plt.subplot(448)
plt.plot(table,df_X3_L, color='IndianRed', label='X3_L')
plt.xlabel('X3_L')

plt.subplot(449)
plt.plot(table,df_FEED_ACT, color='Purple', label='FEED_ACT')
plt.xlabel('FEED_ACT')

plt.subplot(4,4,10)
plt.plot(table,df_FEED_SET, color='Brown', label='FEED_SET')
plt.xlabel('FEED_SET')

plt.subplot(4,4,11)
plt.plot(table,df_C_R, color='Olive', label='C_R')
plt.xlabel('C_R')

plt.savefig('C:/Users/admin/Desktop/航空航天赛道/正负样本/Q处跳动_大于等于0.6.jpg',bbox_inches='tight')

正样本Q
# 取数据F
df_X_R = df_t_pic['W3.DATAPOINT.X_R']
df_X_I = df_t_pic['W3.DATAPOINT.X_I']
df_Z_R = df_t_pic['W3.DATAPOINT.X_I']
df_Z_L = df_t_pic['W3.DATAPOINT.Z_L']
df_X2_R = df_t_pic['W3.DATAPOINT.X2_R']
df_X2_L = df_t_pic['W3.DATAPOINT.X2_L']
df_X3_R = df_t_pic['W3.DATAPOINT.X3_R']
df_X3_L = df_t_pic['W3.DATAPOINT.X3_L']
df_FEED_ACT = df_t_pic['W3.DATAPOINT.FEED_ACT']
df_FEED_SET = df_t_pic['W3.DATAPOINT.FEED_SET']
df_C_R = df_t_pic['W3.DATAPOINT.C_R']

plt.rcParams['font.sans-serif'] = ['SimHei']
    # 获取时间戳
table = df_t_pic.TM

    # 图像大小
plt.figure(figsize=(15,9))

# 调整子图间距
plt.subplots_adjust(hspace =0.3)

    # 图像名
plt.suptitle("Q处跳动_正样本", fontsize=20, color='black')

# 绘图
plt.subplot(441)
plt.plot(table,df_X_R,color='r', label='X_R')
plt.xlabel('X_R')

plt.subplot(442)
plt.plot(table,df_X_I,color='y', label='X_I')
plt.xlabel('X_I')

plt.subplot(443)
plt.plot(table,df_Z_R,color='g', label='Z_R')
plt.xlabel('Z_R')

plt.subplot(444)
plt.plot(table,df_Z_L,color='c', label='Z_L')
plt.xlabel('Z_L')

plt.subplot(445)
plt.plot(table,df_X2_R,color='m', label='X2_R')
plt.xlabel('X2_R')

plt.subplot(446)
plt.plot(table,df_X2_L,color='k', label='X2_L')
plt.xlabel('X2_L')

plt.subplot(447)
plt.plot(table,df_X3_R,color='SkyBlue',label='X3_R')
plt.xlabel('X3_R')

plt.subplot(448)
plt.plot(table,df_X3_L, color='IndianRed', label='X3_L')
plt.xlabel('X3_L')

plt.subplot(449)
plt.plot(table,df_FEED_ACT, color='Purple', label='FEED_ACT')
plt.xlabel('FEED_ACT')

plt.subplot(4,4,10)
plt.plot(table,df_FEED_SET, color='Brown', label='FEED_SET')
plt.xlabel('FEED_SET')

plt.subplot(4,4,11)
plt.plot(table,df_C_R, color='Olive', label='C_R')
plt.xlabel('C_R')

plt.savefig('C:/Users/admin/Desktop/航空航天赛道/正负样本/Q处跳动_小于0.1.jpg',bbox_inches='tight')
  



##########################分割画图#######################
df2 = pd.read_excel(r'C:/Users/admin/Desktop/航空航天赛道/numbdata_2.xlsx')
df = pd.read_excel(r'C:/Users/admin/Desktop/航空航天赛道/tsdata_equal.xlsx')
df2=df2[['工件编号','Q处跳动','中部跳动']]
df=pd.merge(df, df2, left_on='ID',right_on='工件编号', how='left')

list1=[0.1,0.2,0.3,0.4,0.5,0.6,0.7]
for i in list1:
    
    df_F=df[(df['Q处跳动'] >= i-0.1) &(df['Q处跳动'] < i)]
           
    g = df_F.groupby('ID')
    df_F["TM"]=g.cumcount() + 1
        
    df_f_pic = pd.DataFrame(columns=['Unnamed: 0', 'W3.DATAPOINT.X_R', 'W3.DATAPOINT.X_I',
       'W3.DATAPOINT.Z_R', 'W3.DATAPOINT.Z_L', 'W3.DATAPOINT.X2_R',
       'W3.DATAPOINT.X2_L', 'W3.DATAPOINT.X3_R', 'W3.DATAPOINT.X3_L',
       'W3.DATAPOINT.FEED_ACT', 'W3.DATAPOINT.FEED_SET', 'W3.DATAPOINT.C_R',
       'W3.DATAPOINT.C_L', 'code1', 'code2', 'code3', 'ID', 'TM', '工件编号',
       'Q处跳动', '中部跳动'])
    for j in range(1,221):
        df_a=df_F[df_F['TM']==j].mean()
        df_f_pic=df_f_pic.append(df_a, ignore_index=True)
    
    # 取数据F
    df_X_R = df_f_pic['W3.DATAPOINT.X_R']
    df_X_I = df_f_pic['W3.DATAPOINT.X_I']
    df_Z_R = df_f_pic['W3.DATAPOINT.X_I']
    df_Z_L = df_f_pic['W3.DATAPOINT.Z_L']
    df_X2_R = df_f_pic['W3.DATAPOINT.X2_R']
    df_X2_L = df_f_pic['W3.DATAPOINT.X2_L']
    df_X3_R = df_f_pic['W3.DATAPOINT.X3_R']
    df_X3_L = df_f_pic['W3.DATAPOINT.X3_L']
    df_FEED_ACT = df_f_pic['W3.DATAPOINT.FEED_ACT']
    df_FEED_SET = df_f_pic['W3.DATAPOINT.FEED_SET']
    df_C_R = df_f_pic['W3.DATAPOINT.C_R']
    
    plt.rcParams['font.sans-serif'] = ['SimHei']
        # 获取时间戳
    table = df_f_pic.TM
    
        # 图像大小
    plt.figure(figsize=(15,9))
    
    # 调整子图间距
    plt.subplots_adjust(hspace =0.4)
    
        # 图像名
    title1 = str(i-0.1)
    title2 = str(i)
    plt.suptitle("Q处跳动值_" + title1 + "-" + title2 , fontsize=20, color='black')
    
    
    # 绘图
    plt.subplot(441)
    plt.plot(table,df_X_R,color='r', label='X_R')
    plt.xlabel('X_R')
    
    plt.subplot(442)
    plt.plot(table,df_X_I,color='y', label='X_I')
    plt.xlabel('X_I')
    
    plt.subplot(443)
    plt.plot(table,df_Z_R,color='g', label='Z_R')
    plt.xlabel('Z_R')
    
    plt.subplot(444)
    plt.plot(table,df_Z_L,color='c', label='Z_L')
    plt.xlabel('Z_L')
    
    plt.subplot(445)
    plt.plot(table,df_X2_R,color='m', label='X2_R')
    plt.xlabel('X2_R')
    
    plt.subplot(446)
    plt.plot(table,df_X2_L,color='k', label='X2_L')
    plt.xlabel('X2_L')
    
    plt.subplot(447)
    plt.plot(table,df_X3_R,color='SkyBlue',label='X3_R')
    plt.xlabel('X3_R')
    
    plt.subplot(448)
    plt.plot(table,df_X3_L, color='IndianRed', label='X3_L')
    plt.xlabel('X3_L')
    
    plt.subplot(449)
    plt.plot(table,df_FEED_ACT, color='Purple', label='FEED_ACT')
    plt.xlabel('FEED_ACT')
    
    plt.subplot(4,4,10)
    plt.plot(table,df_FEED_SET, color='Brown', label='FEED_SET')
    plt.xlabel('FEED_SET')
    
    plt.subplot(4,4,11)
    plt.plot(table,df_C_R, color='Olive', label='C_R')
    plt.xlabel('C_R')
    
    plt.savefig('C:/Users/admin/Desktop/航空航天赛道/正负样本/分割画图/Q处跳动值_'+ title1 +"-"+ title2 +'.jpg',bbox_inches='tight')

#########################跑比赛结果TsFresh提取一万多特征#########3#########
df = pd.read_excel('C:/Users/admin/Desktop/航空航天赛道/【批量下载】4. 航空航天赛道—航天动力training测试集-旋压单元数据处理结果(1).xlsx等/4. 航空航天赛道—航天动力/正式赛题_tsdata.xlsx')
#加入时间戳
df_ts=df.copy()
df_ts["Time"] = None
g = df_ts.groupby('ID')
df_ts["Time"]=g.cumcount() + 1
df_ts.to_excel('C:/Users/admin/Desktop/航空航天赛道/正式赛题_tsdata_带时间.xlsx',index=False,encoding='gb18030')

df=df_ts.copy()    

#描述性统计
df_des=df.describe()
df_des.to_excel('C:/Users/admin/Desktop/航空航天赛道/正式赛题_ts_描述性统计.xlsx',encoding='gb18030')

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
 'code1',
 'code2',
 'code3']]

 df.isnull().sum()
 ##插值法填充缺失值
fill0=['W3.DATAPOINT.X_R',
       'W3.DATAPOINT.X_I', 'W3.DATAPOINT.Z_R', 'W3.DATAPOINT.Z_L',
       'W3.DATAPOINT.X2_R', 'W3.DATAPOINT.X2_L', 'W3.DATAPOINT.X3_R',
       'W3.DATAPOINT.X3_L', 'W3.DATAPOINT.FEED_ACT', 'W3.DATAPOINT.FEED_SET',
       'W3.DATAPOINT.C_R', 'W3.DATAPOINT.C_L']
df=df.interpolate(subset=fill0)

df.to_excel('C:/Users/admin/Desktop/航空航天赛道/正式赛题_ts_处理后.xlsx',encoding='gb18030',index=False)




###########################正式赛题画图###########################################
#df2 = pd.read_excel(r'C:/Users/admin/Desktop/航空航天赛道/正式赛题/test_numbdata.xlsx')
#df = pd.read_excel(r'C:/Users/admin/Desktop/航空航天赛道/正式赛题/test_tsdata_equal.xlsx')
#df2=df2[['工件编号','Q处跳动','中部跳动']]
#df=pd.merge(df, df2, left_on='ID',right_on='工件编号', how='left')
#
#list1=df2['工件编号']
#for i in list1:
#    
##    df_F=df[(df['Q处跳动'] >= i-0.1) &(df['Q处跳动'] < i)]
#           
#    g = df.groupby('ID')
#    df["TM"]=g.cumcount() + 1
#        
#    
#    # 取数据F
#    df_X_R = df['W3.DATAPOINT.X_R']
#    df_X_I = df['W3.DATAPOINT.X_I']
#    df_Z_R = df['W3.DATAPOINT.X_I']
#    df_Z_L = df['W3.DATAPOINT.Z_L']
#    df_X2_R = df['W3.DATAPOINT.X2_R']
#    df_X2_L = df['W3.DATAPOINT.X2_L']
#    df_X3_R = df['W3.DATAPOINT.X3_R']
#    df_X3_L = df['W3.DATAPOINT.X3_L']
#    df_FEED_ACT = df['W3.DATAPOINT.FEED_ACT']
#    df_FEED_SET = df['W3.DATAPOINT.FEED_SET']
#    df_C_R = df['W3.DATAPOINT.C_R']
#    
#    plt.rcParams['font.sans-serif'] = ['SimHei']
#        # 获取时间戳
#    table = df.TM
#    
#        # 图像大小
#    plt.figure(figsize=(15,9))
#    
#    # 调整子图间距
#    plt.subplots_adjust(hspace =0.4)
#    
#        # 图像名
#    title = str(i)
#    plt.suptitle(title1, fontsize=20, color='black')
#    
#    
#    # 绘图
#    plt.subplot(441)
#    plt.plot(table,df_X_R,color='r', label='X_R')
#    plt.xlabel('X_R')
#    
#    plt.subplot(442)
#    plt.plot(table,df_X_I,color='y', label='X_I')
#    plt.xlabel('X_I')
#    
#    plt.subplot(443)
#    plt.plot(table,df_Z_R,color='g', label='Z_R')
#    plt.xlabel('Z_R')
#    
#    plt.subplot(444)
#    plt.plot(table,df_Z_L,color='c', label='Z_L')
#    plt.xlabel('Z_L')
#    
#    plt.subplot(445)
#    plt.plot(table,df_X2_R,color='m', label='X2_R')
#    plt.xlabel('X2_R')
#    
#    plt.subplot(446)
#    plt.plot(table,df_X2_L,color='k', label='X2_L')
#    plt.xlabel('X2_L')
#    
#    plt.subplot(447)
#    plt.plot(table,df_X3_R,color='SkyBlue',label='X3_R')
#    plt.xlabel('X3_R')
#    
#    plt.subplot(448)
#    plt.plot(table,df_X3_L, color='IndianRed', label='X3_L')
#    plt.xlabel('X3_L')
#    
#    plt.subplot(449)
#    plt.plot(table,df_FEED_ACT, color='Purple', label='FEED_ACT')
#    plt.xlabel('FEED_ACT')
#    
#    plt.subplot(4,4,10)
#    plt.plot(table,df_FEED_SET, color='Brown', label='FEED_SET')
#    plt.xlabel('FEED_SET')
#    
#    plt.subplot(4,4,11)
#    plt.plot(table,df_C_R, color='Olive', label='C_R')
#    plt.xlabel('C_R')
#    
#    plt.savefig('C:/Users/admin/Desktop/航空航天赛道/正负样本/分割画图/title1 +'.jpg',bbox_inches='tight')

#########################计算相似性##########################
df = pd.read_excel(r'C:/Users/admin/Desktop/航空航天赛道/正式赛题/test_tsdata_equal.xlsx')
df.rename(columns={'Unnamed: 0':'序号'},inplace=True)
df1 = pd.read_excel(r'C:/Users/admin/Desktop/航空航天赛道/tsdata_equal.xlsx')
df1.rename(columns={'Unnamed: 0':'序号'},inplace=True)
df_res=pd.DataFrame(columns = ['test','train','out'])
list1=df1['ID'].drop_duplicates()
list0=df['ID'].drop_duplicates()
l1 = []
l2 = []
l3 = []
for i in list0:
    print("i",i)
    for j in list1:
        print(j)
        df2=df[df['ID']==i]-df1[df1['ID']==j]#对应相减
        df3=abs(df2)#取绝对值
        df4=df3.sum().sum()#结果求和
        l1.append(i)
        l2.append(j)
        l3.append(df4)
        #df_res=df_res.append([i,j,df4],ignore_index=True)
df_res['test']  = l1
df_res['train']  = l2
df_res['out']  = l3 
print(df_res)
df_res.to_excel('C:/Users/admin/Desktop/航空航天赛道/正式赛题/xiangsixing.xlsx',index=False)

        

df_xiangsi=pd.read_excel(r'C:/Users/admin/Desktop/航空航天赛道/正式赛题/xiangsixing.xlsx')
        
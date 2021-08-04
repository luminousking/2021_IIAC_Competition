# data analysis
import xlrd
import pandas as pd
import numpy as np
import random as rnd

# visulization 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# %matplotlib inline

# data processing
from sklearn import preprocessing


df = pd.read_excel('tsdata_2.xlsx')
grouped = df.groupby('ID')
for i in range(210625,210849):
    dfx = grouped.get_group(i) #.to_csv('test1.csv',encoding='utf_8_sig'
    break

# 取数据
df_X_R = df['W3.DATAPOINT.X_R']
df_X_I = df['W3.DATAPOINT.X_I']
df_Z_R = df['W3.DATAPOINT.X_I']
df_Z_L = df['W3.DATAPOINT.Z_L']
df_X2_R = df['W3.DATAPOINT.X2_R']
df_X2_L = df['W3.DATAPOINT.X2_L']
df_X3_R = df['W3.DATAPOINT.X3_R']
df_X3_L = df['W3.DATAPOINT.X3_L']
df_FEED_ACT = df['W3.DATAPOINT.FEED_ACT']
df_FEED_SET = df['W3.DATAPOINT.FEED_SET']
df_C_R = df['W3.DATAPOINT.C_R']

table = dfx.index

def draw_picture(df):
    # 图像大小
    fig = plt.figure(figsize=(15,9), dpi=100)
    ax = fig.add_subplot(111)

 # X轴时间刻度格式 & 刻度显示
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(pd.date_range(table.index[0],table.index[-1],freq='H'), rotation=45)

    # 绘图
    ax.plot(table.index,df_X_R,color='r', label='X_R')
    ax.plot(table.index,df_X_I,color='y', label='X_I')
    ax.plot(table.index,df_Z_R,color='g', label='Z_R')
    ax.plot(table.index,df_Z_L,color='c', label='Z_L')
    ax.plot(table.index,df_X2_R,color='m', label='X2_R')
    ax.plot(table.index,df_X2_L,color='k', label='X2_L')
    ax.plot(table.index,df_X3_R,color='SkyBlue',label='X3_R')
    ax.plot(table.index,df_X3_L, color='IndianRed', label='X3_L')
    ax.plot(table.index,df_FEED_ACT, color='Purple', label='FEED_ACT')
    ax.plot(table.index,df_FEED_SET, color='Brown', label='FEED_SET')
    ax.plot(table.index,df_C_R, color='Olive', label='C_R')

    # 辅助线
    sup_line = [35 for i in range(480)]
    ax.plot(table.index, sup_line, color='black', linestyle='--', linewidth='1', label='辅助线')

    plt.xlabel('time_point', fontsize=14)    # X轴标签
    plt.ylabel("Speed", fontsize=16)         # Y轴标签
    ax.legend()                              # 图例
    plt.title("车速时序图", fontsize=25, color='black', pad=20)
    plt.gcf().autofmt_xdate()

    plt.show()


  













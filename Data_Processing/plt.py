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

df1 = pd.read_excel(r'C:\Users\Luminous Isaac\Documents\GitHub\2021_IIAC_Competition\Data_Processing\tsdata_2.xlsx')
grouped = df1.groupby('ID')

df2 = pd.read_excel(r'C:\Users\Luminous Isaac\Documents\GitHub\2021_IIAC_Competition\Data_Processing\numbdata_2.xlsx')

df3 = pd.read_csv(r'C:\Users\Luminous Isaac\Documents\GitHub\2021_IIAC_Competition\Data_Processing\index.csv')
a = 0

for i in range(0,167):
    dfx = grouped.get_group(df3['ID'][i]) 

    # 取数据
    df_X_R = dfx['W3.DATAPOINT.X_R']
    df_X_I = dfx['W3.DATAPOINT.X_I']
    df_Z_R = dfx['W3.DATAPOINT.X_I']
    df_Z_L = dfx['W3.DATAPOINT.Z_L']
    df_X2_R = dfx['W3.DATAPOINT.X2_R']
    df_X2_L = dfx['W3.DATAPOINT.X2_L']
    df_X3_R = dfx['W3.DATAPOINT.X3_R']
    df_X3_L = dfx['W3.DATAPOINT.X3_L']
    df_FEED_ACT = dfx['W3.DATAPOINT.FEED_ACT']
    df_FEED_SET = dfx['W3.DATAPOINT.FEED_SET']
    df_C_R = dfx['W3.DATAPOINT.C_R']

    # 获取时间戳
    table = dfx.index

    # 获取标题
 
    title1 = str(df2['工件编号'].tolist()[a])
    title2 = str(df2['中部跳动'][a])
    title3 = str(df2['Q处跳动'][a])

    # 图像大小
    plt.figure(figsize=(15,9))

    # 图像名
    plt.suptitle("Item number: " + title1 + "  Middle jump value: " + title2 + "  Q point jump value: " + title3, fontsize=15, color='black')

    # 绘图
    plt.subplot(441)
    plt.plot(table,df_X_R,color='r', label='X_R')

    plt.subplot(442)
    plt.plot(table,df_X_I,color='y', label='X_I')

    plt.subplot(443)
    plt.plot(table,df_Z_R,color='g', label='Z_R')

    plt.subplot(444)
    plt.plot(table,df_Z_L,color='c', label='Z_L')

    plt.subplot(445)
    plt.plot(table,df_X2_R,color='m', label='X2_R')

    plt.subplot(446)
    plt.plot(table,df_X2_L,color='k', label='X2_L')

    plt.subplot(447)
    plt.plot(table,df_X3_R,color='SkyBlue',label='X3_R')

    plt.subplot(448)
    plt.plot(table,df_X3_L, color='IndianRed', label='X3_L')

    plt.subplot(449)
    plt.plot(table,df_FEED_ACT, color='Purple', label='FEED_ACT')

    plt.subplot(4,4,10)
    plt.plot(table,df_FEED_SET, color='Brown', label='FEED_SET')

    plt.subplot(4,4,11)
    plt.plot(table,df_C_R, color='Olive', label='C_R')




    # X轴时间刻度格式 & 刻度显示（PS：我也不知道为什么加了这两条x-axis刻度就没了。。。）
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    # plt.xticks(pd.date_range(table[0],table[-1],freq='H'), rotation=45)

    # 出图！
    # plt.xlabel('time_point', fontsize=14)    # X轴标签
    # plt.ylabel("value", fontsize=16)         # Y轴标签
    # 图例
    # plt.show()
    plt.savefig('C:/Users/Luminous Isaac/Documents/GitHub/2021_IIAC_Competition/Data_Processing/pic/item_number{}.png'.format(df2['工件编号'].tolist()[a]))
    a = a + 1

    







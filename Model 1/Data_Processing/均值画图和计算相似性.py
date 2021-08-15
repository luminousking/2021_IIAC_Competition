# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 11:02:50 2021

@author: admin
"""
import pandas as pd
import numpy as np
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
    for j in list1:
        print("i:",i,"j:",j)
        df_i=df[df['ID']==i]
        df_i=df_i[['W3.DATAPOINT.X_R', 'W3.DATAPOINT.X_I', 'W3.DATAPOINT.Z_R',
       'W3.DATAPOINT.Z_L', 'W3.DATAPOINT.X2_R', 'W3.DATAPOINT.X2_L',
       'W3.DATAPOINT.X3_R', 'W3.DATAPOINT.X3_L', 'W3.DATAPOINT.FEED_ACT',
       'W3.DATAPOINT.FEED_SET', 'W3.DATAPOINT.C_R', 'W3.DATAPOINT.C_L',
       'code1', 'code2', 'code3']]
        df_j=df1[df1['ID']==j]
        df_j=df_j[['W3.DATAPOINT.X_R', 'W3.DATAPOINT.X_I', 'W3.DATAPOINT.Z_R',
       'W3.DATAPOINT.Z_L', 'W3.DATAPOINT.X2_R', 'W3.DATAPOINT.X2_L',
       'W3.DATAPOINT.X3_R', 'W3.DATAPOINT.X3_L', 'W3.DATAPOINT.FEED_ACT',
       'W3.DATAPOINT.FEED_SET', 'W3.DATAPOINT.C_R', 'W3.DATAPOINT.C_L',
       'code1', 'code2', 'code3']]
        df2=pd.DataFrame(np.mat(df_i)-np.mat(df_j))
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
df_res.to_excel('C:/Users/admin/Desktop/航空航天赛道/正式赛题/相似性.xlsx',index=False)
from sklearn import preprocessing
import numpy as np
import pandas as pd
import joblib
#ts数据归一化的列
tsnormal=[ 'W3.DATAPOINT.X_R', 'W3.DATAPOINT.X_I', 'W3.DATAPOINT.Z_R',
       'W3.DATAPOINT.Z_L', 'W3.DATAPOINT.X2_R',  'W3.DATAPOINT.X2_L',
       'W3.DATAPOINT.X3_R', 'W3.DATAPOINT.X3_L','W3.DATAPOINT.FEED_ACT',
       'W3.DATAPOINT.FEED_SET', 'W3.DATAPOINT.C_R', 'W3.DATAPOINT.C_L',
      'code1', 'code2', 'code3'] #,
#数值型数据需要归一化的列
numbnormal=['小头壁厚1', '小头壁厚1.1', '内孔', '斜坡壁厚1', '斜坡壁厚2',
       '大端壁厚1', '大端壁厚2', '大端内孔1', '大端内孔2', '总长']

tsdata=pd.read_excel('E:\IIAC\dataprocess\\tsdata.xlsx')
numbdata=pd.read_excel('E:\IIAC\dataprocess\\numbdata.xlsx')
#训练数据归一化
def train_data_normal():
    for i in tsnormal:
        min_max_scaler = preprocessing.MinMaxScaler()
        tsdata[[i]]=min_max_scaler.fit_transform(tsdata[[i]])
        joblib.dump(min_max_scaler, 'E:\IIAC\dataprocess\\tsnormal\\'+i)
    tsdata.to_excel('tsdata_2.xlsx')
    for x in numbnormal:
        print(x)
        min_max_scaler = preprocessing.MinMaxScaler()
        numbdata[[x]]=min_max_scaler.fit_transform(numbdata[[x]])
        joblib.dump(min_max_scaler, 'E:\IIAC\dataprocess\\datanormal\\'+x)
    numbdata.to_excel('numbdata_2.xlsx')
#测试数据集归一化
def test_data_normal(tsdata,numbata):
    for i in tsnormal:
        min_max_scaler  = joblib.load('E:\IIAC\dataprocess\\tsnormal\\'+i)
        tsdata[[i]]=min_max_scaler.fit_transform(tsdata[[i]])
    for x in numbnormal:
        min_max_scaler = joblib.load('E:\IIAC\dataprocess\\datanormal\\'+x)
        numbdata[[x]]=min_max_scaler.fit_transform(numbdata[[x]])
    return tsdata,numbdata



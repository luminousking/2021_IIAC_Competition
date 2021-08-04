import pandas as pd
import numpy as np
data=pd.read_excel('E:\IIAC\测试集-旋压单元数据处理结果(1).xlsx')
# print(data.describe())
data=data.replace('∞', np.nan)
data=data[data['工件编号']!=210836]
def tsdata(data):
    tsdata=data[['工件编号', 'W3.DATAPOINT.X_R', 'W3.DATAPOINT.X_I', 'W3.DATAPOINT.Z_R',
           'W3.DATAPOINT.Z_L', 'W3.DATAPOINT.X2_R', 'W3.DATAPOINT.X2_L',
           'W3.DATAPOINT.X3_R', 'W3.DATAPOINT.X3_L', 'W3.DATAPOINT.FEED_ACT','W3.DATAPOINT.FEED_SET',
           'W3.DATAPOINT.C_R', 'W3.DATAPOINT.C_L','W3.DATAPOINT.CODE_L1', 'W3.DATAPOINT.CODE_L2', 'W3.DATAPOINT.CODE_L3']]
    tsdata[['工件编号']]=tsdata[['工件编号']].fillna('00')
    pjcode=[]
    code=0
    for i in range(0,len(tsdata)):#
        if tsdata['工件编号'].tolist()[i]!='00':
            code=tsdata['工件编号'].tolist()[i]
            pjcode.append(code)
        else:
            pjcode.append(code)
    tsdata['ID']=pjcode

    content = tsdata['W3.DATAPOINT.CODE_L1'].unique().tolist() + tsdata['W3.DATAPOINT.CODE_L1'].unique().tolist() + \
              tsdata['W3.DATAPOINT.CODE_L1'].unique().tolist()
    content = list(set(content))
    print(content)
    l = {}
    i = 0
    for o in content:
        l[o] = i
        i += 1
    print(l)
    tsdata['code1'] = tsdata['W3.DATAPOINT.CODE_L1'].map(l)
    tsdata['code2'] = tsdata['W3.DATAPOINT.CODE_L2'].map(l)
    tsdata['code3'] = tsdata['W3.DATAPOINT.CODE_L3'].map(l)
    tsdata.to_excel('tsdata.xlsx')

# 提取时序数据
tsdata(data)

def numbdata(data):
    numbdata=data[['加工时间', '加工顺序', '工件编号', '小头壁厚5\n±0.03', 'Unnamed: 4', '内孔Φ39.4+0.02',
           '斜坡壁厚                       2.5±0.03', 'Unnamed: 7',
           '大端壁厚              2.45±0.03', 'Unnamed: 9', '大端内孔Φ69.84 0',
           'Unnamed: 11', '总长352±0.3', '中部跳动≤0.8', 'Q处跳动≤0.6']]

    numbdata=numbdata[numbdata['工件编号'].notna()]
    numbdata.columns=['加工时间', '加工顺序', '工件编号', '小头壁厚1', '小头壁厚1', '内孔',
           '斜坡壁厚1', '斜坡壁厚2',
           '大端壁厚1', '大端壁厚2', '大端内孔1',
           '大端内孔2', '总长', '中部跳动', 'Q处跳动']


    numbdata.to_excel('numbdata.xlsx')

#提取数值
numbdata(data)





import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os
import pandas as pd
import pyecharts.options as opts
from pyecharts.charts import Line

f = open('C:\Users\Luminous Isaac\Desktop\training','rb+')
# C:\Users\Luminous Isaac\Desktop\training
bb = pickle.load(f)


path = 'E:\\2021工业互联网算法大赛\三菱电梯\\training\训练用\Signal_Package\\' # f1=open('E:\\2021工业互联网算法大赛\三菱电梯\额外原始数据集\\520-LowFreqDataSet.pkl','rb+')
dd = pickle.load(f1)
for i in os.listdir(path):
    print(i)

def plot(path):
    for i in os.listdir(path):
        print(i)
        f = open(path+i,'rb+')
        bb = pickle.load(f)

        plt.figure()
        plt.plot(bb['主轴负载'])
        plt.plot(bb['主轴转速'])

        plt.show()
        plt.savefig('E:\\2021工业互联网算法大赛\三菱电梯\\training\训练用\pic\\'+'%s.png'%i)
        plt.close()



import matplotlib.pyplot as plt
import numpy as np

from pyts.datasets import load_gunpoint
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from pyts.transformation import ShapeletTransform

X = [[0, 3, 4, 3, 2, 1,6],
     [3,4,3],
      [3, 4, 3]]
y = [1, 1,0]
st = ShapeletTransform(n_shapelets=1, window_sizes=[3],remove_similar=False)
X_new = st.fit_transform(X, y)
# X_new = st.transform(X)
# # X_new.shape
st.shapelets_
# output
(4, 2)

# 搭建分类模型
X_train, X_test, y_train, y_test = load_gunpoint(return_X_y=True)
shapelet = ShapeletTransform(window_sizes=np.arange(10, 130, 3), random_state=42)
svc = LinearSVC()
clf = make_pipeline(shapelet, svc)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)

# output
0.9666666666666667

# 可视化数据集
plt.plot(X_test[0])
plt.plot(X_test[1])
plt.plot(X_test[2])
plt.plot(X_test[3])

# 预测
clf.predict([X_test[0],X_test[1],X_test[2],X_test[3]])

# output
array([1, 2, 2, 1])






from pyts.transformation import ShapeletTransform
X = [[0, 2, 3, 4, 3, 2, 1],
      [0, 1, 3, 4, 3, 4, 5],
      [2, 1, 0, 2, 1, 5, 4],
      [1, 2, 2, 1, 0, 3, 5]]
y = [0, 0, 1, 1]
st = ShapeletTransform(n_shapelets=2, window_sizes=[3])
X_new = st.fit_transform(X, y)

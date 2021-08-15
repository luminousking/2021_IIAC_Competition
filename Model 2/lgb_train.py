import os
import sys

sys.path.append('work')
from sklearn.model_selection import KFold
from lib.Data_Preprocessing import get_feature
import lightgbm as lgb
import pandas as pd
n_folds = 10
n_rounds = 20000
cats = []

# LGB参数
params_lgbm = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'learning_rate': 0.0001,
    'objective': 'rmse',
    'metric': 'rmse',
    'max_depth': -1,
    'n_jobs': 4,
    # 'feature_fraction': 0.8,
    # 'bagging_fraction': 0.8,
    # 'lambda_l2': 1,
    'verbose': -1
    # 'bagging_freq': 5
}

train = get_feature('C:/Users/Luminous Isaac/Documents/GitHub/2021_IIAC_Competition/Data_Processing/Related_Data/Train.xlsx')
test = get_feature(r'C:\Users\Luminous Isaac\Documents\GitHub\2021_IIAC_Competition\Data_Processing\Related_Data\Test.xlsx', mode='test')
target_name = 'CentralJumpmean'
test[target_name] = 0.
features_to_consider = train.drop(['worktime', 'workorder', 'workpieceNo', 'CentralJumpmean', 'QJumpmean'],
                                  axis=1).columns
kf = KFold(n_splits=n_folds, shuffle=True, random_state=2021)
for train_index, val_index in kf.split(train):
    trn_x = train.loc[train_index, features_to_consider]
    trn_y = train.loc[train_index, target_name]
    val_x = train.loc[val_index, features_to_consider]
    val_y = train.loc[val_index, target_name]
    train_data = lgb.Dataset(trn_x, label=trn_y, categorical_feature=cats)
    val_data = lgb.Dataset(val_x, label=val_y, categorical_feature=cats)
    model = lgb.train(params_lgbm,
                      train_data,
                      n_rounds,
                      valid_sets=[train_data, val_data],
                      verbose_eval=250,
                      early_stopping_rounds=500
                      )
    test[target_name] += model.predict(test[features_to_consider]) / n_folds

target_name = 'QJumpmean'
test[target_name] = 0.
kf = KFold(n_splits=n_folds, shuffle=True, random_state=2021)
for train_index, val_index in kf.split(train):
    trn_x = train.loc[train_index, features_to_consider]
    trn_y = train.loc[train_index, target_name]
    val_x = train.loc[val_index, features_to_consider]
    val_y = train.loc[val_index, target_name]
    train_data = lgb.Dataset(trn_x, label=trn_y, categorical_feature=cats)
    val_data = lgb.Dataset(val_x, label=val_y, categorical_feature=cats)
    model = lgb.train(params_lgbm,
                      train_data,
                      n_rounds,
                      valid_sets=[train_data, val_data],
                      verbose_eval=250,
                      early_stopping_rounds=500
                      )
    test[target_name] += model.predict(test[features_to_consider]) / n_folds

submit = test[['worktime', 'workorder', 'workpieceNo', 'SmallHeadThickness1', 'SmallHeadThickness2',
               'InnerHole', 'SlopeThickness1', 'SlopeThickness2', 'BigEndThickness1', 'BigEndThickness2',
               'BigEndHole1', 'BigEndHole2', 'TotalLength', 'CentralJumpmean', 'QJumpmean']]

submit.columns = ['加工时间', '加工顺序', '工件编号', '小头壁厚5\n±0.03', 'Unnamed: 4', '内孔Φ39.4+0.02',
                  '斜坡壁厚                       2.5±0.03', 'Unnamed: 7',
                  '大端壁厚              2.45±0.03', 'Unnamed: 9', '大端内孔Φ69.84 0',
                  'Unnamed: 11', '总长352±0.3', '中部跳动≤0.8', 'Q处跳动≤0.6']
submit.to_csv(r'C:\Users\Luminous Isaac\Documents\GitHub\2021_IIAC_Competition\Data_Processing\Related_Data\20210810.csv', index=False)

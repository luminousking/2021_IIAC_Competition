import pandas as pd
import numpy as np
from scipy.stats import kurtosis


def q2q(x):
    return x.quantile(0.75) - x.quantile(0.25)


def get_feature(filename, mode='Train'):
    df = pd.read_excel(filename)
    df.dropna(axis=1, how="all", inplace=True)
    df.fillna(method="ffill", inplace=True)
    if mode == 'Train':
        df.columns = ['worktime', 'workorder', 'workpieceNo', 'SmallHeadThickness1', 'SmallHeadThickness2',
                      'InnerHole', 'SlopeThickness1', 'SlopeThickness2', 'BigEndThickness1', 'BigEndThickness2',
                      'BigEndHole1', 'BigEndHole2', 'TotalLength', 'CentralJump', 'QJump', 'Note', 'TM',
                      'X_R', 'X_I', 'Z_R', 'Z_L', 'X2_R', 'X2_L', 'X3_R', 'X3_L', 'FEED_ACT', 'FEED_SET', 'SPINDLE_ACT',
                      'SPINDLE_SET', 'SPINDLE_PER', 'CODE_L1', 'CODE_L2', 'CODE_L3', 'C_R', 'C_L'
                      ]
    else:
        df.columns = ['worktime', 'workorder', 'workpieceNo', 'SmallHeadThickness1', 'SmallHeadThickness2',
                      'InnerHole', 'SlopeThickness1', 'SlopeThickness2', 'BigEndThickness1', 'BigEndThickness2',
                      'BigEndHole1', 'BigEndHole2', 'TotalLength', 'TM',
                      'X_R', 'X_I', 'Z_R', 'Z_L', 'X2_R', 'X2_L', 'X3_R', 'X3_L', 'FEED_ACT', 'FEED_SET', 'SPINDLE_ACT',
                      'SPINDLE_SET', 'SPINDLE_PER', 'CODE_L1', 'CODE_L2', 'CODE_L3', 'C_R', 'C_L'
                      ]
    df['diff'] = np.abs(df['X_R'].diff() / df['Z_R'].diff())
    df = df.query(f'diff<0.05&8<X_R<20')
    if mode == 'Train':
        fea_agg = {
            'CentralJump': ['mean'],
            'QJump': ['mean'],
            "X_R": ['mean', 'std', q2q, "max", "min", 'skew', kurtosis],
            "X_I": ['mean', 'std', q2q, "max", "min", 'skew', kurtosis],
            "Z_R": ['mean', 'std', q2q, "max", "min", 'skew', kurtosis],
            "Z_L": ['mean', 'std', q2q, "max", "min", 'skew', kurtosis],
            "X2_R": ['mean', 'std', q2q, "max", "min", 'skew', kurtosis],
            "X2_L": ['mean', 'std', q2q, "max", "min", 'skew', kurtosis],
            "X3_R": ['mean', 'std', q2q, "max", "min", 'skew', kurtosis],
            # "X3_L":['mean','std',q2q,"max","min",'skew',kurtosis],
            "FEED_ACT": ['mean'],
            "FEED_SET": ['mean'],
            "SPINDLE_ACT": ['mean'],
            "SPINDLE_SET": ['mean'],
            "SPINDLE_PER": ['mean'],
            "C_R": ['mean', 'std', q2q, "max", "min", 'skew', kurtosis],
            "C_L": ['mean', 'std', q2q, "max", "min", 'skew', kurtosis]
        }
    else:
        fea_agg = {
            "X_R": ['mean', 'std', q2q, "max", "min", 'skew', kurtosis],
            "X_I": ['mean', 'std', q2q, "max", "min", 'skew', kurtosis],
            "Z_R": ['mean', 'std', q2q, "max", "min", 'skew', kurtosis],
            "Z_L": ['mean', 'std', q2q, "max", "min", 'skew', kurtosis],
            "X2_R": ['mean', 'std', q2q, "max", "min", 'skew', kurtosis],
            "X2_L": ['mean', 'std', q2q, "max", "min", 'skew', kurtosis],
            "X3_R": ['mean', 'std', q2q, "max", "min", 'skew', kurtosis],
            # "X3_L":['mean','std',q2q,"max","min",'skew',kurtosis],
            "FEED_ACT": ['mean'],
            "FEED_SET": ['mean'],
            "SPINDLE_ACT": ['mean'],
            "SPINDLE_SET": ['mean'],
            "SPINDLE_PER": ['mean'],
            "C_R": ['mean', 'std', q2q, "max", "min", 'skew', kurtosis],
            "C_L": ['mean', 'std', q2q, "max", "min", 'skew', kurtosis]
        }
    feature = df.groupby(by=['worktime', 'workorder', 'workpieceNo', 'SmallHeadThickness1', 'SmallHeadThickness2',
                             'InnerHole', 'SlopeThickness1', 'SlopeThickness2', 'BigEndThickness1', 'BigEndThickness2',
                             'BigEndHole1', 'BigEndHole2', 'TotalLength']).agg(fea_agg).reset_index()
    feature.columns = ["".join(col) for col in feature.columns]
    return feature



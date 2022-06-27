from sklearn.metrics import recall_score, confusion_matrix
from cm import cm_analysis
import pandas as pd
import numpy as np

if __name__ == '__main__':
    label_df = pd.read_excel('dist/dev_results.xlsx', sep=',')
    true_list=label_df['true'].values.tolist()
    print(true_list)
    tc = label_df['two_channel'].values.tolist()
    print(tc)
    cr=label_df['crnn'].values.tolist()
    print(cr)
    xgb = label_df['xgboost'].values.tolist()
    print(xgb)
    cm_analysis(true_list, tc, f'exps/tc', sorted(set(true_list)))
    cm_analysis(true_list, xgb, f'exps/xgb', sorted(set(true_list)))
    cm_analysis(true_list, cr, f'exps/cr', sorted(set(true_list)))
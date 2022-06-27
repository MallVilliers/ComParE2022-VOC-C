import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")
import json
import pandas as pd
import module_xgb as xgb
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report, confusion_matrix,
                             recall_score)
import os
from svm import  make_dict_json_serializable,load_features
from sklearn.base import clone
from cm import cm_analysis
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV
import numpy as np

EMOTION_MAPPING = {
    'achievement': 0,
    'anger': 1,
    'fear': 2,
    'pain': 3,
    'pleasure': 4,
    'surprise': 5
}
EMOTION_MAPPING2 = {
    0 : 'achievement',
    1: 'anger',
    2: 'fear',
    3: 'pain',
    4: 'pleasure',
    5: 'surprise'
}


if __name__ == '__main__':
    (train_devel_names, train_devel_X, train_devel_y, _split), (
    test_names, test_X, test_y), feature_names = load_features \
        ('dist/features/opensmile/features.csv', 'dist/lab')
    train_X = train_devel_X[np.argwhere(_split == -1)].squeeze()
    train_y = train_devel_y[np.argwhere(_split == -1)].squeeze()
    devel_X = train_devel_X[np.argwhere(_split == 0)].squeeze()
    devel_y = train_devel_y[np.argwhere(_split == 0)].squeeze()
    train_devel_label=[]
    for index in range(len(train_devel_y)):
        train_devel_label.append(EMOTION_MAPPING[train_devel_y[index]])
    del_y=[]
    for index in range(len(devel_y)):
        del_y.append(EMOTION_MAPPING[devel_y[index]])
    del_y=np.array(del_y)
    tra_y=[]
    for index in range(len(train_y)):
        tra_y.append(EMOTION_MAPPING[train_y[index]])
    tra_y = np.array(tra_y)
    # devel_names = train_devel_names[np.argwhere(_split == 0)].squeeze()
    train_data = lgb.Dataset(train_X, label=tra_y)
    devel_data = lgb.Dataset(devel_X,label=del_y)
    # params_test = {'max_bin': range(5, 256, 10), 'min_data_in_leaf': range(1, 102, 10)}
    #
    # gsearch= GridSearchCV(
    #     estimator=lgb.LGBMClassifier(boosting_type='gbdt', objective='multiclass',  learning_rate=0.1,
    #                                  max_depth=5, num_leaves=10, bagging_fraction=0.8,
    #                                  feature_fraction=0.8),
    #     param_grid=params_test, scoring="recall_macro", cv=PredefinedSplit(_split), n_jobs=-1)
    # gsearch.fit(train_devel_X, train_devel_label)
    # best_estimator = gsearch.best_estimator_
    # estimator = clone(best_estimator, safe=False)
    # estimator.fit(train_X, tra_y)
    # devel_preds = estimator.predict(devel_X)
    # confusion_matrix_result = confusion_matrix(devel_preds, del_y)
    # predictions=[]
    # for x in devel_preds:
    #     predictions.append(np.argmax(x))
    # # print(predictions)
    # y_label=[]
    # for p in predictions:
    #     y_label.append(EMOTION_MAPPING2[p])
    # # print(y_label)
    # print('The confusion matrix result:\n', confusion_matrix_result)
    # cm_analysis(y_label, devel_y, f'dist/cm/cm_lgb', sorted(set(devel_y)))
    # uar_test = recall_score(devel_y, y_label, average="macro")
    # print(f"UAR: {uar_test:.2%} ")

    # print(train_data.label)
    num_round = 10
    param = {'subsample': 0.8, 'num_leaves': 31, 'num_trees': 100, 'objective': 'multiclass','num_class':6,'max_depth': 5 }
    bst = lgb.train(param, train_data, num_round, valid_sets=[devel_data])
    y_pred=bst.predict(devel_X)
    # print(y_pred)
    predictions = []
    for x in y_pred:
        predictions.append(np.argmax(x))
    # print(predictions)
    y_label=[]
    for p in predictions:
        y_label.append(EMOTION_MAPPING2[p])
    # print(y_label)
    confusion_matrix_result = confusion_matrix( y_label, devel_y)
    print('The confusion matrix result:\n', confusion_matrix_result)
    cm_analysis( devel_y,y_label, f'dist/cm/cm_lgb', sorted(set(devel_y)))
    uar_test = recall_score(devel_y,  y_label, average="macro")
    print(f"UAR: {uar_test:.2%} ")



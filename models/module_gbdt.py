import json
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report, confusion_matrix,
                             recall_score)
import os
from svm import  make_dict_json_serializable
from sklearn.base import clone
from cm import cm_analysis
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from svm import  make_dict_json_serializable,load_features


if __name__ == '__main__':
    (train_devel_names, train_devel_X, train_devel_y, _split), (test_names, test_X, test_y), feature_names =load_features\
        ('dist/features/opensmile/features.csv','dist/lab')
    train_X = train_devel_X[np.argwhere(_split == -1)].squeeze()
    train_y = train_devel_y[np.argwhere(_split == -1)].squeeze()
    devel_X = train_devel_X[np.argwhere(_split == 0)].squeeze()
    devel_y = train_devel_y[np.argwhere(_split == 0)].squeeze()
    devel_names = train_devel_names[np.argwhere(_split == 0)].squeeze()
    gbr = GradientBoostingClassifier(n_estimators=100, max_depth=6, min_samples_split=2, learning_rate=0.1)
    gbr.fit(train_X,train_y)
    y_pred=gbr.predict(devel_X)
    confusion_matrix_result = confusion_matrix(y_pred, devel_y)
    print('The confusion matrix result:\n', confusion_matrix_result)
    cm_analysis(devel_y, y_pred,f'dist/cm/cm_gbdt', sorted(set(devel_y)))
    uar_test = recall_score(devel_y,y_pred, average="macro")
    print(f"UAR: {uar_test:.2%} ")
    # svm_params = make_dict_json_serializable(grid_search.best_params_)
    # with open(os.path.join('exps/', "best_params.json"), "w") as f:
    #     json.dump(svm_params, f)

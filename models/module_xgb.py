import json
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report, confusion_matrix,
                             recall_score)
from xgboost import plot_tree
import matplotlib.pyplot as plt
import os
from svm import  make_dict_json_serializable
from sklearn.base import clone
from cm import cm_analysis
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV
import numpy as np
# 情绪-label对照表
EMOTION_MAPPING = {
    'achievement': 0,
    'anger': 1,
    'fear': 2,
    'pain': 3,
    'pleasure': 4,
    'surprise': 5
}
def load_features(feature_file, label_base):
    labels = pd.concat([pd.read_csv(f"{label_base}/{partition}.csv") for partition in ["train", "devel", "test"]])
    feature_delimiter = ";" if "opensmile" in feature_file else ","
    df = pd.read_csv(feature_file, delimiter=feature_delimiter, quotechar="'")
    joined_df = df.merge(labels, left_on=df.columns[0], right_on=labels.columns[0]).sort_values(by=df.columns[0])
    train_devel_df = joined_df[joined_df.filename.str.contains("train") | joined_df.filename.str.contains("devel")]
    test_df = joined_df[joined_df.filename.str.contains("test")]
    # subset of only specified groups and water rounds
    feature_names = list(df.columns[1:])
    train_devel_names = train_devel_df.values[:, 0].tolist()
    train_devel_features = train_devel_df[feature_names].values
    train_devel_labels = train_devel_df.values[:, -1]
    _split_indices = list(map(lambda x: -1 if "train" in x else 0, train_devel_names))
    train_devel_names = np.array(train_devel_names)
    train_devel_X = train_devel_features
    train_devel_y = np.array(train_devel_labels)
    split = np.array(_split_indices)
    test_names = test_df.values[:, 0].tolist()
    test_features = test_df[feature_names].values
    test_labels = test_df.values[:, -1]
    test_names = np.array(test_names)
    test_X = test_features
    test_y = np.array(test_labels)
    return (train_devel_names, train_devel_X, train_devel_y, split), (test_names, test_X, test_y), feature_names
if __name__ == '__main__':
    (train_devel_names, train_devel_X, train_devel_y, _split), (test_names, test_X, test_y), feature_names =load_features\
        ('dist/features/opensmile/features.csv','dist/lab')
    train_X = train_devel_X[np.argwhere(_split == -1)].squeeze()
    train_y = train_devel_y[np.argwhere(_split == -1)].squeeze()
    # print(train_y.shape)
    devel_X = train_devel_X[np.argwhere(_split == 0)].squeeze()
    devel_y = train_devel_y[np.argwhere(_split == 0)].squeeze()
    devel_names = train_devel_names[np.argwhere(_split == 0)].squeeze()
    # print(devel_X.shape)
    # print(devel_y.shape)
    # print(train_X.shape)
    # print(train_y.shape)

    # params = {
    #     'booster': 'gbtree',
    #     'objective': 'multi:softmax',
    #     'num_class': 3,
    #     'gamma': 0.1,
    #     'max_depth': 6,
    #     'lambda': 2,
    #     'subsample': 0.7,
    #     'colsample_bytree': 0.7,
    #     'min_child_weight': 3,
    #     'silent': 1,
    #     'eta': 0.1,
    #     'seed': 1000,
    #     'nthread': 4,
    # }
    # grid = {
    #      'max_depth':[3, 4, 5, 6, 7, 8, 9, 10],
    #     'learning_rate':[0.3,0.1],
    #     'min_child_weight':[1,3,4, 5],
    # 'n_estimators': [400, 500, 600, 700, 800],
    # 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],'subsample': [0.6, 0.7, 0.8, 0.9]}
    model=xgb.XGBClassifier(learning_rate=0.3,min_child_weight=1,max_depth=6)
    # model = xgb.XGBClassifier(learning_rate=0.3,max_depth=6,min_child_weight=1)
    # model.fit(train_X, train_y)
    # test_pred=model.predict(test_X)
    # print(type(test_pred))
    # df = pd.DataFrame({'pred': test_pred})
    # df.to_excel('exps/xgb.xlsx')
    # grid_search = GridSearchCV(
    #     estimator=model,
    #     param_grid=grid,
    #     scoring="recall_macro",
    #     n_jobs=-1,
    #     cv=PredefinedSplit(_split),
    #     refit=True,
    #     verbose=1,
    #     return_train_score=False,
    # )
    # grid_search.fit(train_devel_X, train_devel_y)
    # best_estimator = grid_search.best_estimator_
    # estimator = clone(best_estimator, safe=False)
    # estimator.fit(train_X, train_y)
    # devel_preds = estimator.predict(devel_X)
    model.fit(train_X, train_y)
    # pred_y = model.predict(devel_X)
    # confusion_matrix_result = confusion_matrix( pred_y , devel_y)
    # print('The confusion matrix result:\n', confusion_matrix_result)
    # cm_analysis(devel_y, pred_y  ,f'dist/cm/cm_xgb2', sorted(set(devel_y)))
    # uar_test = recall_score(devel_y,  pred_y  , average="macro")
    pred_p=model.predict_proba(devel_X)
    pred_p=pred_p.tolist()


    print(pred_p)
    # list转dataframe
    df = pd.DataFrame(pred_p)

    # 保存到本地excel
    df.to_excel("pred.xlsx", index=False)



    # def ceate_feature_map(features):
    #     outfile = open('dist/xgb.fmap', 'w')
    #     i = 0
    #     for feat in features:
    #         outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    #         i = i + 1
    #     outfile.close()
    #
    #
    # '''
    # X_train.columns在第一段代码中也已经设置过了。
    # 特别需要注意：列名字中不能有空格。
    # '''
    # ceate_feature_map(feature_names)
    # plot_tree(model, num_trees=0, fmap='dist/xgb.fmap')
    # fig = plt.gcf()
    # fig.set_size_inches(150, 100)
    # # plt.show()
    # fig.savefig('exps/tree.png')
    # print(f"UAR: {uar_test:.2%} ")
    # best_params = make_dict_json_serializable(grid_search.best_params_)
    # with open(os.path.join('exps/', "best_params.json"), "w") as f:
    #     json.dump(best_params, f)






import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeClassifier, Ridge, Lasso, LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
# from cuml.ensemble import RandomForestClassifier
# from cuml.ensemble import RandomForestRegressor
np.random.seed(0)
def relative_absolute_error(y_test, y_predict):
    y_test = np.array(y_test)
    y_predict = np.array(y_predict)
    error = np.sum(np.abs(y_test - y_predict)) / np.sum(np.abs(np.mean(
        y_test) - y_test))
    return error


def downstream_task_new(data, task_type, method='RF'):
    """ 在给定特征与标签的数据集上，根据任务类型与可选方法进行下游建模与五折交叉验证评估，返回相应评价指标的平均得分。函数会将最后一列视为标签列（转换为 float），其余列作为特征，并将特征列统一重命名为 feature_0...feature_n。对于部分任务分支，训练前会将特征中的无穷值与 NaN 替换为 0。

    支持的任务与评估方式：

    分类任务（task_type='cls'）：可选方法包括
    'SVC'：线性核支持向量机
    'LR'：逻辑回归（max_iter=1000）
    'DTC'：决策树
    'KNC'：K 近邻分类器（k=5）
    其他或默认：随机森林分类器 使用 StratifiedKFold(5) 进行分层五折交叉验证，返回 weighted F1 的平均值。
    回归任务（task_type='reg'）：使用随机森林回归器，KFold(5) 交叉验证，返回 1 - RAE（相对绝对误差）的平均值。
    二分类检测任务（task_type='det'）：使用 K 近邻分类器（k=5，n_jobs=128），StratifiedKFold(5) 交叉验证，返回 ROC-AUC 的平均值。
    多标签/多类别任务（task_type='mcls'）：使用 One-vs-Rest 的随机森林分类器（n_jobs=128），StratifiedKFold(5) 交叉验证，返回 micro F1 的平均值。
    排序任务（task_type='rank'）：未实现，返回 None。
    其他不支持的任务类型：返回 -1。
    Args: data (pandas.DataFrame): 输入数据表，最后一列为标签，可转换为 float；其余列为特征。 task_type (str): 任务类型，支持 'cls'、'reg'、'det'、'mcls'、'rank'。 method (str, optional): 分类任务的学习器选择，支持 'RF'（默认）、'SVC'、'LR'、'DTC'、'KNC'；对非分类任务忽略。

    Returns: float | None | int: - 分类（'cls'）：加权 F1 的五折平均值。 - 回归（'reg'）：1 - RAE 的五折平均值。 - 检测（'det'）：ROC-AUC 的五折平均值。 - 多分类（'mcls'）：micro F1 的五折平均值。 - 排序（'rank'）：None（未实现）。 - 其他任务类型：-1。

    备注: - 在分类与回归分支中，训练前会将特征中的无穷大与 NaN 替换为 0，以避免模型训练报错。 - 采用 5 折交叉验证（分层或普通），固定 random_state=0 且 shuffle=True（如适用），以获得稳定的平均指标。 
    """
    X = data.iloc[:, :-1]
    X.columns = [f'feature_{i}' for i in range(X.shape[1])]
    y = data.iloc[:, -1].astype(float)
    if task_type == 'cls':
        if method == 'SVC':
            clf = SVC(kernel='linear', random_state=0)
        elif method == 'LR':
            clf = LogisticRegression(random_state=0, max_iter=1000)
        elif method == 'DTC':
            clf = DecisionTreeClassifier(random_state=0)
        elif method == 'KNC':
            clf = KNeighborsClassifier(n_neighbors=5)
        else: # if method == 'RF':
            clf = RandomForestClassifier(random_state=0)
        f1_list = []
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            # 将无穷大和NaN值替换为0
            X_train[np.isinf(X_train)] = 0
            X_train[np.isnan(X_train)] = 0
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            f1_list.append(f1_score(y_test, y_predict, average='weighted'))
        return np.mean(f1_list)
    elif task_type == 'reg':
        kf = KFold(n_splits=5, random_state=0, shuffle=True)
        reg = RandomForestRegressor(random_state=0)
        rae_list = []
        for train, test in kf.split(X):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            # 将无穷大和NaN值替换为0
            X_train[np.isinf(X_train)] = 0
            X_train[np.isnan(X_train)] = 0
            reg.fit(X_train, y_train)
            y_predict = reg.predict(X_test)
            rae_list.append(1 - relative_absolute_error(y_test, y_predict))
        return np.mean(rae_list)
    elif task_type == 'det':
        knn = KNeighborsClassifier(n_neighbors=5, n_jobs=128)
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        ras_list = []
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            knn.fit(X_train, y_train)
            y_predict = knn.predict(X_test)
            ras_list.append(roc_auc_score(y_test, y_predict))
        return np.mean(ras_list)
    elif task_type == 'mcls':
        clf = OneVsRestClassifier(RandomForestClassifier(random_state=0, n_jobs=128))
        pre_list, rec_list, f1_list, auc_roc_score = [], [], [], []
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            f1_list.append(f1_score(y_test, y_predict, average='micro'))
        return np.mean(f1_list)
    elif task_type == 'rank':
        pass
    else:
        return -1

# 'RF', 'XGB', 'SVM', 'KNN', 'Ridge', 'LASSO', 'DT'
def downstream_task_by_method(data, task_type, method):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1].astype(float)
    if method == 'RF':
        if task_type == 'cls':
            model = RandomForestClassifier(random_state=0, n_jobs=128)
        elif task_type == 'mcls':
            model = OneVsRestClassifier(RandomForestClassifier(random_state=0), n_jobs=128)
        else:
            model = RandomForestRegressor(random_state=0, n_jobs=128)
    elif method == 'XGB':
        if task_type == 'cls':
            model = XGBClassifier(eval_metric='logloss', n_jobs=128)
        elif task_type == 'mcls':
            model = OneVsRestClassifier(XGBClassifier(eval_metric='logloss'), n_jobs=128)
        else:
            model = XGBRegressor(eval_metric='logloss', n_jobs=128)
    elif method == 'SVM':
        if task_type == 'cls':
            model = LinearSVC()
        elif task_type == 'mcls':
            model = LinearSVC()
        else:
            model = LinearSVR()
    elif method == 'KNN':
        if task_type == 'cls':
            model = KNeighborsClassifier(n_jobs=128)
        elif task_type == 'mcls':
            model = OneVsRestClassifier(KNeighborsClassifier(), n_jobs=128)
        else:
            model = KNeighborsRegressor(n_jobs=128)
    elif method == 'Ridge':
        if task_type == 'cls':
            model = RidgeClassifier()
        elif task_type == 'mcls':
            model = OneVsRestClassifier(RidgeClassifier(), n_jobs=128)
        else:
            model = Ridge()
    elif method == 'LASSO':
        if task_type == 'cls':
            model = LogisticRegression(penalty='l1',solver='liblinear', n_jobs=128)
        elif task_type == 'mcls':
            model = OneVsRestClassifier(LogisticRegression(penalty='l1',solver='liblinear'), n_jobs=128)
        else:
            model = Lasso()
    else:  # dt
        if task_type == 'cls':
            model = DecisionTreeClassifier()
        elif task_type == 'mcls':
            model = OneVsRestClassifier(DecisionTreeClassifier(), n_jobs=128)
        else:
            model = DecisionTreeRegressor()

    if task_type == 'cls':
        f1_list = []
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            model.fit(X_train, y_train)
            y_predict = model.predict(X_test)
            f1_list.append(f1_score(y_test, y_predict, average='weighted'))
        return np.mean(f1_list)
    elif task_type == 'reg':
        kf = KFold(n_splits=5, random_state=0, shuffle=True)
        rae_list = []
        for train, test in kf.split(X):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            model.fit(X_train, y_train)
            y_predict = model.predict(X_test)
            rae_list.append(1 - relative_absolute_error(y_test, y_predict))
        return np.mean(rae_list)
    elif task_type == 'mcls':
        clf = OneVsRestClassifier(RandomForestClassifier(random_state=0))
        pre_list, rec_list, f1_list, auc_roc_score = [], [], [], []
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            f1_list.append(f1_score(y_test, y_predict, average='micro'))
        return np.mean(f1_list)
    else:
        return -1


def downstream_task_by_method_std(data, task_type, method):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1].astype(float)
    if method == 'RF':
        if task_type == 'cls':
            model = RandomForestClassifier(random_state=0, n_jobs=128)
        elif task_type == 'mcls':
            model = OneVsRestClassifier(RandomForestClassifier(random_state=0), n_jobs=128)
        else:
            model = RandomForestRegressor(random_state=0, n_jobs=128)
    elif method == 'XGB':
        if task_type == 'cls':
            model = XGBClassifier(eval_metric='logloss', n_jobs=128)
        elif task_type == 'mcls':
            model = OneVsRestClassifier(XGBClassifier(eval_metric='logloss'), n_jobs=128)
        else:
            model = XGBRegressor(eval_metric='logloss', n_jobs=128)
    elif method == 'SVM':
        if task_type == 'cls':
            model = LinearSVC()
        elif task_type == 'mcls':
            model = LinearSVC()
        else:
            model = LinearSVR()
    elif method == 'KNN':
        if task_type == 'cls':
            model = KNeighborsClassifier(n_jobs=128)
        elif task_type == 'mcls':
            model = OneVsRestClassifier(KNeighborsClassifier(), n_jobs=128)
        else:
            model = KNeighborsRegressor(n_jobs=128)
    elif method == 'Ridge':
        if task_type == 'cls':
            model = RidgeClassifier()
        elif task_type == 'mcls':
            model = OneVsRestClassifier(RidgeClassifier(), n_jobs=128)
        else:
            model = Ridge()
    elif method == 'LASSO':
        if task_type == 'cls':
            model = LogisticRegression(penalty='l1',solver='liblinear', n_jobs=128)
        elif task_type == 'mcls':
            model = OneVsRestClassifier(LogisticRegression(penalty='l1',solver='liblinear'), n_jobs=128)
        else:
            model = Lasso()
    else:  # dt
        if task_type == 'cls':
            model = DecisionTreeClassifier()
        elif task_type == 'mcls':
            model = OneVsRestClassifier(DecisionTreeClassifier(), n_jobs=128)
        else:
            model = DecisionTreeRegressor()

    if task_type == 'cls':
        f1_list = []
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            model.fit(X_train, y_train)
            y_predict = model.predict(X_test)
            f1_list.append(f1_score(y_test, y_predict, average='weighted'))
        return np.mean(f1_list), np.std(f1_list)
    elif task_type == 'reg':
        kf = KFold(n_splits=5, random_state=0, shuffle=True)
        rae_list = []
        for train, test in kf.split(X):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            model.fit(X_train, y_train)
            y_predict = model.predict(X_test)
            rae_list.append(1 - relative_absolute_error(y_test, y_predict))
        return np.mean(rae_list), np.std(rae_list)
    elif task_type == 'mcls':
        clf = OneVsRestClassifier(RandomForestClassifier(random_state=0))
        pre_list, rec_list, f1_list, auc_roc_score = [], [], [], []
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            f1_list.append(f1_score(y_test, y_predict, average='micro'))
        return np.mean(f1_list), np.std(f1_list)
    else:
        return -1



def test_task_wo_cv(Dg, task='cls'):
    X = Dg.iloc[:, :-1]
    y = Dg.iloc[:, -1].astype(float)
    if task == 'cls':
        clf = RandomForestClassifier(random_state=0)
        pre_list, rec_list, f1_list, auc_roc_score = [], [], [], []
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            pre_list.append(precision_score(y_test, y_predict, average='weighted'))
            rec_list.append(recall_score(y_test, y_predict, average='weighted'))
            f1_list.append(f1_score(y_test, y_predict, average='weighted'))
            auc_roc_score.append(roc_auc_score(y_test, y_predict, average='weighted'))
            break
        return np.mean(pre_list), np.mean(rec_list), np.mean(f1_list), np.mean(auc_roc_score)
    elif task == 'reg':
        kf = KFold(n_splits=5, random_state=0, shuffle=True)
        reg = RandomForestRegressor(random_state=0)
        mae_list, mse_list, rae_list, rmse_list = [], [], [], []
        for train, test in kf.split(X):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
            reg.fit(X_train, y_train)
            y_predict = reg.predict(X_test)
            mae_list.append(1 - mean_absolute_error(y_test, y_predict))
            mse_list.append(1 - mean_squared_error(y_test, y_predict, squared=True))
            rae_list.append(1 - relative_absolute_error(y_test, y_predict))
            rmse_list.append(1 - mean_squared_error(y_test, y_predict, squared=False))
            break
        return np.mean(mae_list), np.mean(mse_list), np.mean(rae_list), np.mean(rmse_list)
    elif task == 'det':
        kf = KFold(n_splits=5, random_state=0, shuffle=True)
        knn_model = KNeighborsClassifier(n_neighbors=5)
        map_list = []
        f1_list = []
        ras = []
        recall = []
        for train, test in kf.split(X):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
            knn_model.fit(X_train, y_train)
            y_predict = knn_model.predict(X_test)
            map_list.append(average_precision_score(y_test, y_predict))
            f1_list.append(f1_score(y_test, y_predict, average='weighted'))
            ras.append(roc_auc_score(y_test, y_predict))
            recall.append(recall_score(y_test, y_predict, average='weighted'))
            break
        return np.mean(map_list), np.mean(f1_list), np.mean(ras), np.mean(recall)
    elif task == 'mcls':
        clf = OneVsRestClassifier(RandomForestClassifier(random_state=0))
        pre_list, rec_list, f1_list, maf1_list = [], [], [], []
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            pre_list.append(precision_score(y_test, y_predict, average='macro'))
            rec_list.append(recall_score(y_test, y_predict, average='macro'))
            f1_list.append(f1_score(y_test, y_predict, average='micro'))
            maf1_list.append(f1_score(y_test, y_predict, average='macro'))
            break
        return np.mean(pre_list), np.mean(rec_list), np.mean(f1_list), np.mean(maf1_list)
    elif task == 'rank':
        pass
    else:
        return -1

def test_task_new(Dg, task='cls'):
    X = Dg.iloc[:, :-1]
    y = Dg.iloc[:, -1].astype(float)
    if task == 'cls':
        clf = RandomForestClassifier(random_state=0)
        pre_list, rec_list, f1_list, auc_roc_score = [], [], [], []
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            pre_list.append(precision_score(y_test, y_predict, average='weighted'))
            rec_list.append(recall_score(y_test, y_predict, average='weighted'))
            f1_list.append(f1_score(y_test, y_predict, average='weighted'))
            auc_roc_score.append(roc_auc_score(y_test, y_predict, average='weighted'))
        return np.mean(pre_list), np.mean(rec_list), np.mean(f1_list), np.mean(auc_roc_score)
    elif task == 'reg':
        kf = KFold(n_splits=5, random_state=0, shuffle=True)
        reg = RandomForestRegressor(random_state=0)
        mae_list, mse_list, rae_list, rmse_list = [], [], [], []
        for train, test in kf.split(X):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
            reg.fit(X_train, y_train)
            y_predict = reg.predict(X_test)
            mae_list.append(1 - mean_absolute_error(y_test, y_predict))
            mse_list.append(1 - mean_squared_error(y_test, y_predict, squared=True))
            rae_list.append(1 - relative_absolute_error(y_test, y_predict))
            rmse_list.append(1 - mean_squared_error(y_test, y_predict, squared=False))
        return np.mean(mae_list), np.mean(mse_list), np.mean(rae_list), np.mean(rmse_list)
    elif task == 'det':
        kf = KFold(n_splits=5, random_state=0, shuffle=True)
        knn_model = KNeighborsClassifier(n_neighbors=5)
        map_list = []
        f1_list = []
        ras = []
        recall = []
        for train, test in kf.split(X):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
            knn_model.fit(X_train, y_train)
            y_predict = knn_model.predict(X_test)
            map_list.append(average_precision_score(y_test, y_predict))
            f1_list.append(f1_score(y_test, y_predict, average='weighted'))
            ras.append(roc_auc_score(y_test, y_predict))
            recall.append(recall_score(y_test, y_predict, average='weighted'))
        return np.mean(map_list), np.mean(f1_list), np.mean(ras), np.mean(recall)
    elif task == 'mcls':
        clf = OneVsRestClassifier(RandomForestClassifier(random_state=0))
        pre_list, rec_list, f1_list, maf1_list = [], [], [], []
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            pre_list.append(precision_score(y_test, y_predict, average='macro'))
            rec_list.append(recall_score(y_test, y_predict, average='macro'))
            f1_list.append(f1_score(y_test, y_predict, average='micro'))
            maf1_list.append(f1_score(y_test, y_predict, average='macro'))
        return np.mean(pre_list), np.mean(rec_list), np.mean(f1_list), np.mean(maf1_list)
    elif task == 'rank':
        pass
    else:
        return -1

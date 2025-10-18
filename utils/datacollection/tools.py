import numpy
import numpy as np
import pandas as pd
from collections import defaultdict

from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
    pairwise_distances,
)
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from scipy.special import expit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.feature_selection import SelectFromModel
from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_selection import mutual_info_regression
import torch
from sklearn.linear_model import RidgeClassifier
import torch.nn as nn
import torch.utils.data as Data
from utils.datacollection.logger import error, info
from utils.datacollection.Operation import add_binary, op_map, op_map_r
from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from xgboost import XGBClassifier

np.random.seed(0)
# from cuml.ensemble import RandomForestClassifier
# from cuml.ensemble import RandomForestRegressor


def cube(x):
    return x**3


def justify_operation_type(o):
    """将操作名称字符串转换为对应的数值运算函数或数据预处理器实例。

    支持的操作包括：

    数值函数：'sqrt'、'square'、'sin'、'cos'、'tanh'、'reciprocal'、'exp'、'log'、'cube'、'sigmoid'
    二元运算符：'+'（加）、'-'（减）、'*'（乘）、'/'（除）
    预处理器：'stand_scaler'（StandardScaler）、'minmax_scaler'（MinMaxScaler(feature_range=(-1, 1))）、'quan_trans'（QuantileTransformer(random_state=0)）
    对于不支持的操作名，将打印提示信息并返回原始输入。

    Args:
        o (str | Callable): 操作标识符字符串，或已存在的可调用对象/转换器实例。 - 若为字符串，将映射为对应的 numpy ufunc、Python 可调用函数，或 sklearn 预处理器实例。 - 若为可调用对象或实例且不在支持表中，则原样返回。

    Returns: Callable | object: 对应的 numpy ufunc/函数，或未拟合的 sklearn 预处理器实例。 若传入不受支持的字符串或非字符串对象，则返回原始输入。

    Raises: 无显式异常抛出。本函数在遇到不支持的操作名时仅打印提示，不抛出错误。
    """
    if o == "sqrt":
        o = np.sqrt
    elif o == "square":
        o = np.square
    elif o == "sin":
        o = np.sin
    elif o == "cos":
        o = np.cos
    elif o == "tanh":
        o = np.tanh
    elif o == "reciprocal":
        o = np.reciprocal
    elif o == "+":
        o = np.add
    elif o == "-":
        o = np.subtract
    elif o == "/":
        o = np.divide
    elif o == "*":
        o = np.multiply
    elif o == "stand_scaler":
        o = StandardScaler()
    elif o == "minmax_scaler":
        o = MinMaxScaler(feature_range=(-1, 1))
    elif o == "quan_trans":
        o = QuantileTransformer(random_state=0)
    elif o == "exp":
        o = np.exp
    elif o == "cube":
        o = cube
    elif o == "sigmoid":
        o = expit
    elif o == "log":
        o = np.log
    else:
        print("Please check your operation!")
    return o


# def feature_distance(features, y):
#     dis_mat = []
#     for i in range(features.shape[1]):
#         tmp = []
#         for j in range(features.shape[1]):
#             tmp.append(np.abs(mutual_info_regression(features[:, i].reshape
#                 (-1, 1), y) - mutual_info_regression(features[:, j].reshape
#                 (-1, 1), y))[0] / (mutual_info_regression(features[:, i].
#                 reshape(-1, 1), features[:, j].reshape(-1, 1))[0] + 1e-05))
#         dis_mat.append(np.array(tmp))
#     dis_mat = np.array(dis_mat)
#     return dis_mat


def eu_distance(feature, y):
    """计算两个特征向量之间的欧氏距离。

    Args:
        feature (array-like): 一维特征向量，形状为 (n_features,)。
        y (array-like): 一维特征向量，形状为 (n_features,)。

    Returns: numpy.ndarray: 距离矩阵，形状为 (1, 1)，包含两向量的欧氏距离。

    Raises: ValueError: 当输入无法被重塑为一维向量或两者维度不一致时。 TypeError: 当输入类型不被距离计算函数接受时。"""
    return pairwise_distances(
        feature.reshape(1, -1), y.reshape(1, -1), metric="euclidean"
    )


def feature_distance(features, y):
    """计算特征之间的两两欧氏距离矩阵

    给定一个形状为 (n_samples, n_features) 的特征矩阵，本函数将每一列视为一个特征向量， 计算所有特征向量之间的欧氏距离，返回形状为 (n_features, n_features) 的对称距离矩阵， 主对角线为 0。

    Args:
        features (array-like | torch.Tensor): 特征矩阵，形状为 (n_samples, n_features)。可为 NumPy 数组、可转为张量的序列或 PyTorch 张量。
        y (Any, optional): 未使用的参数，保留用于接口兼容。

    Returns: numpy.ndarray: 特征两两欧氏距离矩阵，形状为 (n_features, n_features)。

    Raises: TypeError: 当输入无法转换为 PyTorch 张量时抛出。 ValueError: 当输入不是二维数据或包含无效数值导致距离计算失败时抛出。
    """
    # dis_mat = []
    # for i in range(features.shape[1]):
    #     tmp = []
    #     for j in range(features.shape[1]):
    #         tmp.append(np.abs(eu_distance(features[:, i], features[:, j])))
    #     dis_mat.append(np.array(tmp))
    # dis_mat = np.array(dis_mat)
    r = torch.tensor(features)
    return torch.cdist(
        r.transpose(-1, 0),
        r.transpose(-1, 0),
        p=2.0,
        compute_mode="use_mm_for_euclid_dist_if_necessary",
    ).numpy()


def cluster_features(features, y, cluster_num=2):
    """使用层次聚类对特征进行聚类，并返回簇到样本索引的映射。

    该函数首先基于输入特征与标签计算样本间的预计算距离矩阵，随后采用单链接凝聚层次聚类（预计算距离）进行聚类。簇数量 k 取特征维度平方根的整数部分。最终返回每个簇对应的样本索引列表。

    Args:
        features (numpy.ndarray): 原始特征矩阵，形状为 (n_samples, n_features)。
        y (array-like): 与样本对应的标签或目标，用于距离度量的计算。
        cluster_num (int, optional): 期望的簇数。目前实现未使用该参数。默认值为 2。

    Returns: dict[int, list[int]]: 簇标签到样本索引列表的映射。

    Raises: ValueError: 当输入形状不合法、距离计算或聚类过程出错时可能抛出。"""
    k = int(np.sqrt(features.shape[1]))
    features = feature_distance(features, y)
    clustering = AgglomerativeClustering(
        n_clusters=k, affinity="precomputed", linkage="single"
    ).fit(features)
    labels = clustering.labels_
    clusters = defaultdict(list)
    for ind, item in enumerate(labels):
        clusters[item].append(ind)
    return clusters


def wocluster_features(features, y, cluster_num=2):
    """ 对特征进行“无聚类”分组的占位实现。

    该函数不执行实际聚类，而是为每个特征列分别创建一个分组。返回的映射中，键为特征列索引，值为仅包含该索引本身的列表。例如，若共有 n_features 个特征，则结果形如 {0: [0], 1: [1], ..., n_features-1: [n_features-1]}。参数 y 与 cluster_num 当前未被使用。

    Args: 
        features (array-like): 特征矩阵，形状为 (n_samples, n_features)，需支持 shape 属性。 
        y (array-like): 目标/标签数据，占位参数，未使用。 
        cluster_num (int, optional): 预期的聚类数，占位参数，未使用，默认为 2。

    Returns: dict[int, list[int]]: 从特征索引到分组索引列表的映射，每个列表仅包含与键相同的单个索引。

    Raises: AttributeError: 当 features 不具有 shape 属性时抛出。 """
    clusters = defaultdict(list)
    for ind, item in enumerate(range(features.shape[1])):
        clusters[item].append(ind)
    return clusters


class LinearAutoEncoder(nn.Module):

    def __init__(self, input, hidden, act=torch.relu):
        self.encoder = nn.Linear(input, hidden)
        self.encoder_act = act
        self.decoder = nn.Linear(hidden, input)
        self.decoder_act = act
        super().__init__()

    def forward(self, X):
        return self.decoder_act(self.decoder(self.encoder_act(self.encoder(X))))

    def generate(self, X):
        return self.encoder_act(self.encoder(X))


def Feature_GCN(X):
    """ 基于特征相关性的简单图卷积聚合，返回每个样本的聚合特征值

    该函数以特征两两绝对相关系数构建加权邻接矩阵，去除自环后进行按列与按行的归一化， 再加回单位阵形成带自环的权重矩阵 W。最后计算 X @ W 的行均值，得到每个样本的单一 聚合特征值。对于仅含一个特征的情况，相关矩阵为 1，直接返回该情形下的结果。此外， 相关矩阵中的 NaN 会被置为 0 以保证稳健性。

    Args: 
        X (pandas.DataFrame): 输入特征矩阵，形状为 (n_samples, n_features)。每列代表一个特征， 每行代表一个样本。

    Returns: numpy.ndarray: 长度为 n_samples 的一维数组，对应每个样本的相关性加权聚合特征值。

    Raises: ValueError: 当输入为空或无法计算相关矩阵时可能触发。 """
    """
    group feature 可能有一个cluster内元素为1的情况，这样corr - eye后返回的是一个零矩阵，故在这里设置为0时返回一个1.
    """
    corr_matrix = X.corr().abs()
    if len(corr_matrix) == 1:
        W = corr_matrix
    else:
        corr_matrix[np.isnan(corr_matrix)] = 0
        corr_matrix_ = corr_matrix - np.eye(len(corr_matrix), k=0)
        sum_vec = corr_matrix_.sum()
        for i in range(len(corr_matrix_)):
            corr_matrix_.iloc[:, i] = corr_matrix_.iloc[:, i] / sum_vec[i]
            corr_matrix_.iloc[i, :] = corr_matrix_.iloc[i, :] / sum_vec[i]
        W = corr_matrix_ + np.eye(len(corr_matrix), k=0)
    Feature = np.mean(np.dot(X.values, W.values), axis=1)
    return Feature


class AutoEncoder(nn.Module):

    def __init__(self, N_feature):
        self.N_feature = N_feature
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(self.N_feature, 16),
            nn.Tanh(),
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.Tanh(),
            nn.Linear(8, 16),
            nn.Tanh(),
            nn.Linear(16, self.N_feature),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def Feature_AE(X, gpu=-1):
    """ 使用简单自编码器对输入特征进行无监督表示学习，并返回每个样本在隐空间编码的均值作为一维特征

    该函数基于输入特征维度构建并训练一个自编码器（AutoEncoder）， 采用 MSE 损失与 Adam 优化器进行短暂训练（默认 10 个 epoch，batch_size=128，lr=0.005）。 训练完成后，对所有样本进行前向编码，并对每个样本的编码向量按维度求均值，得到一维的表示。

    Args: 
        X (pandas.DataFrame): 形状为 (n_samples, n_features) 的数值型数据框，将使用 X.values 转为张量进行训练与推理。 
        gpu (int, optional): 使用的 GPU 序号；-1 表示使用 CPU，非负整数表示对应的 CUDA 设备编号。默认值为 -1。

    Returns: 
        numpy.ndarray: 形状为 (n_samples,) 的一维数组，表示每个样本在自编码器隐空间中的均值编码。

    Raises: RuntimeError: 当指定的 GPU 不可用或 CUDA 初始化失败时由 PyTorch 抛出。 TypeError: 当 X 不是数值型二维数据（或不具备 .values 属性）导致张量转换失败时可能抛出。 """
    N_feature = X.shape[1]
    autoencoder = AutoEncoder(N_feature)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.005)
    loss_func = nn.MSELoss()
    X_tensor = torch.Tensor(X.values)
    if gpu >= 0:
        device = torch.device("cuda:" + str(gpu))
        autoencoder.to(device)
    else:
        device = torch.device("cpu")
    train_loader = Data.DataLoader(
        dataset=X_tensor, batch_size=128, shuffle=True, drop_last=False, num_workers=8
    )
    for epoch in range(10):
        for x in train_loader:
            b_x = x.view(-1, N_feature).float().to(device)
            encoded, decoded = autoencoder.forward(b_x)
            optimizer.zero_grad()
            loss = loss_func(decoded, b_x)
            loss.backward()
            optimizer.step()
    X_tensor.to(device)
    X_encoded = np.mean(autoencoder.forward(X_tensor)[0].cpu().detach().numpy(), axis=1)
    return X_encoded


def feature_state_generation(X, method="mds", gpu=-1):
    """ 根据输入特征生成特征状态表示，可选择多种方法并支持拼接组合

    支持的方法如下：

    "mds": 使用多维尺度缩放（MDS）生成状态
    "gcn": 使用图卷积网络（GCN）生成状态
    "ae": 使用自编码器（AE）生成状态
    "mds+ae": 将 AE 与 MDS 的状态按顺序拼接
    "mds+ae+gcn": 将 AE、GCN 与 MDS 的状态按顺序拼接
    Args: X (array-like): 输入特征数据，通常为形状为 (n_samples, n_features) 的数组或可转换为数组的对象。 method (str, optional): 特征状态生成方法，取值为 {"mds", "gcn", "ae", "mds+ae", "mds+ae+gcn"}。默认值为 "mds"。 gpu (int, optional): 在使用 "ae" 相关方法时指定 GPU 设备索引；-1 表示使用 CPU。默认值为 -1。

    Returns: numpy.ndarray: 生成的特征状态表示。当选择组合方法时，返回按顺序拼接后的向量或数组。

    Raises: Exception: 当 method 不在支持的方法集合中时抛出。 """
    if method == "mds":
        return _feature_state_generation_des(X)
    elif method == "gcn":
        return Feature_GCN(X)
    elif method == "ae":
        return Feature_AE(X, gpu)
    elif method == "mds+ae":
        return numpy.append(Feature_AE(X, gpu), _feature_state_generation_des(X))
    elif method == "mds+ae+gcn":
        state_mds = _feature_state_generation_des(X)
        state_gcn = Feature_GCN(X)
        state_ae = Feature_AE(X, gpu)
        return numpy.append(numpy.append(state_ae, state_gcn), state_mds)
    else:
        error("Wrong feature state method")
        raise Exception("Wrong feature state method")


def _feature_state_generation_des(X):
    """ 基于输入数据的描述性统计生成一维特征向量。函数对数值型数据计算基础的 8 项描述统计 （count、mean、std、min、25%、50%、75%、max），并对每一项统计结果再次进行描述统计， 将得到的数值按顺序展平并拼接为最终特征列表。所有缺失值在生成过程中会被填充为 0。

    Args: X (pandas.DataFrame): 含数值型列的数据集，将被转换为 float64 并参与统计计算。

    Returns: list[float]: 由多级描述统计结果拼接得到的一维特征向量。

    Raises: AttributeError: 当输入对象不支持 astype 或 describe 等方法时抛出。 ValueError: 当数据无法转换为 float64 类型时抛出。 IndexError: 当描述统计结果行数不足（非标准 8 项）导致索引越界时抛出。 """
    feature_matrix = []
    for i in range(8):
        feature_matrix = feature_matrix + list(
            X.astype(np.float64).describe().iloc[i, :].describe().fillna(0).values
        )
    return feature_matrix


def select_meta_cluster1(
    clusters, X, feature_names, epsilon, dqn_cluster, method="mds", gpu=-1
):
    """ 基于 epsilon-贪心策略从候选特征簇中选择一个最优簇。函数首先对全量特征生成状态嵌入，再对每个候选簇的特征子集生成动作嵌入，使用 dqn_cluster 的 Q 值评估进行打分并按 epsilon-贪心策略选择最终簇。返回所选簇的动作嵌入、全局状态嵌入、簇对应的特征子矩阵以及特征名。

    Args: 
        clusters (Mapping[Hashable, Iterable[int]]): 候选特征簇，键为簇标识，值为该簇包含的特征索引集合/列表。 
        X (numpy.ndarray): 输入特征矩阵，形状为 (n_samples, n_features)。 
        feature_names (Sequence[str]): 全量特征名序列，长度应与 X 的列数一致。 
        epsilon (float): 探索率，取值范围通常为 [0, 1]；以该概率随机选择簇，否则选择 Q 值最高的簇。 
        dqn_cluster: 具备 get_q_value(state_emb, action) 方法的估值模型对象，用于计算状态-动作对的 Q 值。 
        method (str, optional): 用于 feature_state_generation 的嵌入方法名称，默认为 "mds"。 
        gpu (int, optional): GPU 设备编号，-1 表示使用 CPU，默认为 -1。

    Returns: 
        tuple: 
            - action_emb (array-like): 所选簇对应的动作嵌入表示。 
            - state_emb (array-like): 基于全量特征生成的状态嵌入表示。 
            - f_cluster (numpy.ndarray): 所选簇对应的特征子矩阵，形状为 (n_samples, n_selected_features)。 
            - f_names (numpy.ndarray): 所选簇对应的特征名数组，dtype 为 str。

    Raises: ValueError: 当 clusters 为空，或 feature_names 的长度与 X 的特征数不一致时可能引发。 AttributeError: 当 dqn_cluster 未实现 get_q_value(state_emb, action) 接口时可能引发。 """
    state_emb = feature_state_generation(pd.DataFrame(X), method, gpu)
    q_vals, cluster_list, action_list = [], [], []
    for key, value in clusters.items():
        action = feature_state_generation(pd.DataFrame(X[:, list(value)]), method, gpu)
        action_list.append(action)
        q_value = dqn_cluster.get_q_value(state_emb, action).detach().numpy()[0]
        q_vals.append(q_value)
        cluster_list.append(key)
    if np.random.uniform() > epsilon:
        act_id = np.argmax(q_vals)
    else:
        act_id = np.random.randint(0, len(clusters))
    cluster_ind = cluster_list[act_id]
    f_cluster = X[:, list(clusters[cluster_ind])]
    action_emb = action_list[act_id]
    f_names = np.array(feature_names)[list(clusters[cluster_ind])]
    info("current select feature name : " + str(f_names))
    return action_emb, state_emb, f_cluster, f_names


def select_meta_cluster1_indice(
    clusters, X, feature_names, epsilon, dqn_cluster, method="mds", gpu=-1
):
    """ 基于epsilon-greedy策略从候选特征簇中选择一个簇，并返回对应的动作与状态嵌入及相关信息。
    整体特征矩阵通过指定方法生成全局状态嵌入；每个候选簇对应的特征子集生成动作嵌入；使用提供的DQN模型评估Q值并进行探索/利用选择。

    Args: 
        clusters (dict): 候选特征簇映射，键为簇标识，值为该簇包含的特征索引可迭代对象（如list或set）。 
        X (array-like): 输入特征矩阵，形状为 (n_samples, n_features)。 
        feature_names (array-like): 全部特征名称序列，长度应与 n_features 一致。 
        epsilon (float): 探索概率，取值范围通常为 [0, 1]；以该概率随机选择簇，否则选择Q值最大的簇。 
        dqn_cluster: 具有 get_q_value(state_emb, action_emb) 方法的DQN模型对象，返回可反向传播的张量Q值。 
        method (str, optional): 生成状态/动作嵌入的方法名称，例如 "mds"。默认值为 "mds"。 gpu (int, optional): 计算设备ID，-1表示使用CPU，非负整数表示对应GPU设备。默认值为 -1。

    Returns: tuple: - action_emb: 被选中特征簇对应的动作嵌入（类型由特征嵌入实现决定，如numpy数组或张量）。 - state_emb: 全局状态嵌入（类型由特征嵌入实现决定，如numpy数组或张量）。 - list[int]: 被选中特征簇的特征索引列表。 - array-like: 被选中特征簇的特征名称序列。
    """
    state_emb = feature_state_generation(pd.DataFrame(X), method, gpu)
    q_vals, cluster_list, action_list = [], [], []
    for key, value in clusters.items():
        action = feature_state_generation(pd.DataFrame(X[:, list(value)]), method, gpu)
        action_list.append(action)
        q_value = dqn_cluster.get_q_value(state_emb, action).detach().numpy()[0]
        q_vals.append(q_value)
        cluster_list.append(key)
    if np.random.uniform() > epsilon:
        act_id = np.argmax(q_vals)
    else:
        act_id = np.random.randint(0, len(clusters))
    cluster_ind = cluster_list[act_id]
    # f_cluster = X[:, list(clusters[cluster_ind])]
    action_emb = action_list[act_id]
    f_names = np.array(feature_names)[list(clusters[cluster_ind])]
    info("current select feature name : " + str(f_names))
    return action_emb, state_emb, list(clusters[cluster_ind]), f_names


def select_operation(
    f_cluster, operation_set, dqn_operation, steps_done, method="mds", gpu=-1
):
    op_state = feature_state_generation(pd.DataFrame(f_cluster), method, gpu)
    op_index = dqn_operation.choose_action(op_state, steps_done)
    op = operation_set[op_index]
    info("current select op : " + str(op))
    return op_state, op, op_index


def select_meta_cluster2(
    clusters,
    X,
    feature_names,
    f_cluster1,
    op_emb,
    epsilon,
    dqn_cluster,
    method="mds",
    gpu=-1,
):
    """ 基于特征聚类生成状态表示，并通过 DQN 策略从给定操作集合中选择一个操作

    Args: 
        f_cluster: 输入的特征聚类/特征集合，需可被 pandas.DataFrame 构造，通常为二维数组或可迭代对象 
        operation_set (Sequence): 可供选择的操作集合（列表或序列），需支持通过索引访问 
        dqn_operation: 具备 choose_action(state, steps_done) 方法的策略决策器/智能体，用于根据状态选择操作索引 
        steps_done (int): 已执行的步数，用于控制探索-利用策略或其他调度逻辑 
        method (str, optional): 状态构造方法名称，例如 "mds"；默认值为 "mds" 
        gpu (int, optional): 计算设备编号，-1 表示 CPU，非负整数表示对应的 GPU；默认值为 -1

    Returns: 
        tuple: 三元组 (op_state, op, op_index) 
            - op_state: 由 f_cluster 生成的状态表示（用于决策的特征张量/数组） 
            - op: 从 operation_set 中选定的操作 
            - op_index (int): 所选操作在 operation_set 中的索引

    Raises: IndexError: 当策略返回的索引超出 operation_set 范围时 ValueError: 当输入数据无法构造成有效的状态表示时 TypeError: 当 dqn_operation 不提供 choose_action 接口或参数类型不匹配时 
    """
    feature_emb = feature_state_generation(pd.DataFrame(f_cluster1), method, gpu)
    state_emb = torch.cat((torch.tensor(feature_emb), torch.tensor(op_emb)))
    q_vals, cluster_list, action_list = [], [], []
    for key, value in clusters.items():
        action = feature_state_generation(pd.DataFrame(X[:, list(value)]), method, gpu)
        action_list.append(action)
        q_value = dqn_cluster.get_q_value(state_emb, action).detach().numpy()[0]
        q_vals.append(q_value)
        cluster_list.append(key)
    act_id = np.random.randint(0, len(clusters))
    cluster_ind = cluster_list[act_id]
    f_cluster2 = X[:, list(clusters[cluster_ind])]
    action_emb = action_list[act_id]
    f_names = np.array(feature_names)[list(clusters[cluster_ind])]
    return action_emb, state_emb, f_cluster2, f_names


def select_meta_cluster2_indice(
    clusters,
    X,
    feature_names,
    f_cluster1,
    op_emb,
    epsilon,
    dqn_cluster,
    method="mds",
    gpu=-1,
):
    """ 基于给定的候选特征簇，从中随机选择一个簇并返回对应的动作嵌入、状态嵌入、被选簇的特征索引及特征名。 函数内部会先对当前状态（由已选簇特征与操作嵌入拼接而成）和每个候选簇的动作表示进行嵌入计算；虽然会计算对应的 Q 值，但当前实现采用纯随机策略进行动作选择（未使用 epsilon-greedy）。

    Args: 
        clusters (dict): 候选特征簇映射，键为簇标识，值为该簇包含的特征索引可迭代对象（如 list[int]）。 
        X (numpy.ndarray): 原始特征矩阵，形状为 (n_samples, n_features)。 
        feature_names (Sequence[str]): 与 X 列对齐的特征名序列，长度为 n_features。 
        f_cluster1 (array-like | pandas.DataFrame): 已选第一阶段特征簇的数据，用于构造状态嵌入。 
        op_emb (array-like): 操作（或策略）对应的嵌入向量，将与特征嵌入拼接形成状态。 
        epsilon (float): 探索率参数（当前实现未使用，保留作接口兼容）。 
        dqn_cluster: 提供 get_q_value(state_emb, action_emb) 方法的估值器/模型实例。 
        method (str): 特征嵌入的方式，传递给 feature_state_generation，默认 "mds"。 
        gpu (int): GPU 设备编号，-1 表示使用 CPU。

    Returns: tuple: - action_emb (array-like): 被随机选中簇对应的动作嵌入（由 feature_state_generation 产生）。 - state_emb (torch.Tensor): 当前状态嵌入，等于特征嵌入与操作嵌入的拼接结果。 - indices (list[int]): 被选中特征簇对应的特征索引列表。 - f_names (numpy.ndarray): 被选中特征簇对应的特征名数组。

    Raises: ValueError: 当 clusters 为空时无法进行随机选择。 TypeError: 当输入类型与预期不符（例如 X 或 feature_names 形状/类型不匹配）时可能抛出。 """
    feature_emb = feature_state_generation(pd.DataFrame(f_cluster1), method, gpu)
    state_emb = torch.cat((torch.tensor(feature_emb), torch.tensor(op_emb)))
    q_vals, cluster_list, action_list = [], [], []
    for key, value in clusters.items():
        action = feature_state_generation(pd.DataFrame(X[:, list(value)]), method, gpu)
        action_list.append(action)
        q_value = dqn_cluster.get_q_value(state_emb, action).detach().numpy()[0]
        q_vals.append(q_value)
        cluster_list.append(key)
    act_id = np.random.randint(0, len(clusters))
    cluster_ind = cluster_list[act_id]
    # f_cluster2 = X[:, list(clusters[cluster_ind])]
    action_emb = action_list[act_id]
    f_names = np.array(feature_names)[list(clusters[cluster_ind])]
    return action_emb, state_emb, list(clusters[cluster_ind]), f_names


def operate_two_features(f_cluster1, f_cluster2, op, op_func, f_names1, f_names2):
    """ 对两个特征矩阵按列进行二元组合运算，并生成对应的新特征名称。

    当两侧列数不一致时，从列数较多的一侧按需随机（可重复）采样列以与另一侧对齐，然后将对齐后的两组特征按列传入给定的二元运算函数进行计算；当列数相等时，按列一一对应计算。最终返回生成的特征矩阵及由原始名称通过 op 拼接形成的新名称列表。

    Args: 
        f_cluster1 (numpy.ndarray): 形状为 (n_samples, n_features1) 的特征矩阵。 
        f_cluster2 (numpy.ndarray): 形状为 (n_samples, n_features2) 的特征矩阵。 
        op (str): 用于拼接新特征名的操作符字符串（如 "+", "-", "*", "/" 等）。 
        op_func (Callable[[numpy.ndarray, numpy.ndarray], numpy.ndarray]): 对两个等列数矩阵进行按列二元运算的函数，需返回与输入列数一致的矩阵。 
        f_names1 (Sequence[str] 或 numpy.ndarray): 与 f_cluster1 列对应的名称序列，长度为 n_features1。 
        f_names2 (Sequence[str] 或 numpy.ndarray): 与 f_cluster2 列对应的名称序列，长度为 n_features2。

    Returns: Tuple[numpy.ndarray, List[str]]: - 运算得到的新特征矩阵，形状为 (n_samples, min(n_features1, n_features2))。 - 新特征名列表，长度与返回矩阵的列数一致，名称格式为 ""。

    Raises: ValueError: 当两个输入矩阵的样本数不一致时可能触发。 IndexError: 当名称序列长度与各自特征矩阵的列数不匹配时可能触发。 RuntimeError: 当 op_func 在计算过程中因输入不符合预期而失败时可能触发。 """
    if f_cluster1.shape[1] < f_cluster2.shape[1]:
        inds = np.random.randint(0, f_cluster2.shape[1], f_cluster1.shape[1])
        rand_fs = f_cluster2[:, inds]
        rand_names = f_names2[inds]
        f_generate = op_func(f_cluster1, rand_fs)
        final_name = [
            (str(f1_item) + op + str(rand_names[ind]))
            for ind, f1_item in enumerate(f_names1)
        ]
    elif f_cluster1.shape[1] > f_cluster2.shape[1]:
        inds = np.random.randint(0, f_cluster1.shape[1], f_cluster2.shape[1])
        rand_fs = f_cluster1[:, inds]
        rand_names = f_names1[inds]
        f_generate = op_func(rand_fs, f_cluster2)
        final_name = [
            (str(f1_item) + op + str(f_names2[ind]))
            for ind, f1_item in enumerate(rand_names)
        ]
    else:
        f_generate = op_func(f_cluster1, f_cluster2)
        final_name = [
            (str(f1_item) + op + str(f_names2[ind]))
            for ind, f1_item in enumerate(f_names1)
        ]
    return f_generate, final_name


def operate_two_features_new(f_cluster1, f_cluster2, op, op_func, f_names1, f_names2):
    """ 对两组特征簇按给定的二元运算逐对组合，生成新的特征矩阵及其对应的特征名称。

    遍历 f_cluster1 与 f_cluster2 的每一列，使用 op_func 对任意两列进行元素级计算， 将所有组合结果堆叠并转置为形状为 (n_samples, n_pairs) 的矩阵；同时基于 op 的映射与 对应列名生成每个组合特征的名称。

    Args: 
        f_cluster1 (numpy.ndarray): 第一组特征簇，形状为 (n_samples, n_features1)，按列表示特征。 
        f_cluster2 (numpy.ndarray): 第二组特征簇，形状为 (n_samples, n_features2)，按列表示特征。 
        op: 运算符标识，用于在内部映射表中查找对应的符号或名称以生成特征名。 op_func (Callable[[numpy.ndarray, numpy.ndarray], numpy.ndarray]): 对两个等长向量执行元素级二元运算的函数，返回与输入长度相同的一维数组。 
        f_names1 (Sequence[str]): 与 f_cluster1 各列一一对应的特征名称序列，长度应为 n_features1。 
        f_names2 (Sequence[str]): 与 f_cluster2 各列一一对应的特征名称序列，长度应为 n_features2。

    Returns: numpy.ndarray: 组合后的特征矩阵，形状为 (n_samples, n_features1 * n_features2)。 numpy.ndarray: 对应的组合特征名称数组，形状为 (n_features1 * n_features2,)。

    Raises: ValueError: 当 f_cluster1 与 f_cluster2 的样本数不一致时。 KeyError: 当 op 在运算符映射中不存在时。 TypeError: 当 op_func 的签名不符合要求或输入类型不兼容时。 IndexError: 当特征名称序列与对应特征簇的列数不匹配时。 """
    feas, feas_names = [], []
    for i in range(f_cluster1.shape[1]):
        for j in range(f_cluster2.shape[1]):
            feas.append(op_func(f_cluster1[:, i], f_cluster2[:, j]))
            feas_names.append(
                add_binary(op_map_r[op], str(f_names1[i]), str(f_names2[j]))
            )
    feas = np.array(feas)
    feas_names = np.array(feas_names)
    return feas.T, feas_names


# def insert_generated_feature_to_original_feas(feas, f):
#     y_label = pd.DataFrame(feas[feas.columns[len(feas.columns) - 1]])
#     y_label.columns = [feas.columns[len(feas.columns) - 1]]
#     feas = feas.drop(columns=feas.columns[len(feas.columns) - 1])
#     final_data = pd.concat([feas, f, y_label], axis=1)
#     return final_data
def insert_generated_feature_to_original_feas(feas, f):
    """ 将生成的新特征插入到原始特征集中，并保持原始标签列（最后一列）在末尾位置

    该函数默认把 feas 的最后一列视为标签列，其余列为原始特征； 会把输入的 f 转换为 DataFrame（若其不是），然后按列拼接：原始特征 + 新特征 + 标签列。

    Args: 
        feas (pandas.DataFrame): 原始特征与标签组成的数据集，最后一列应为标签列。 
        f (pandas.DataFrame | array-like): 待插入的新特征。若为 array-like，将被转换为 DataFrame。

    Returns: pandas.DataFrame: 按列拼接后的完整数据集，列顺序为“原始特征 + 新特征 + 原始标签”。

    Raises: ValueError: 当新特征行数与原始数据行数不一致，或拼接失败时可能抛出。 TypeError: 当 f 无法被转换为 DataFrame 时可能抛出。 """
    y_label = feas.iloc[:, -1]
    feas = feas.iloc[:, :-1]
    if not isinstance(f, DataFrame):
        f = pd.DataFrame(f)
    final_data = pd.concat([feas, f, y_label], axis=1)
    return final_data


def generate_next_state_of_meta_cluster1(
    X, y, dqn_cluster, cluster_num=2, method="mds", gpu=-1
):
    """ 基于特征聚类与Q值评估，生成下一步元聚类动作及对应状态表示

    该函数先对输入特征进行聚类，然后为整体特征生成状态嵌入， 并为每个聚类的特征子集生成候选“动作”嵌入，使用提供的DQN（或评估网络） 对“状态-动作”对进行Q值评估，选择Q值最高的聚类作为下一步动作。 最终返回被选中的动作嵌入、全局状态嵌入、被选中聚类对应的特征子矩阵以及全部聚类划分。

    Args: 
        X (array-like of shape (n_samples, n_features)): 输入特征矩阵。 
        y (array-like of shape (n_samples,) or (n_samples, ...)): 与样本对应的标签或辅助监督信号，用于特征聚类流程。 
        dqn_cluster (object): 评估网络/智能体，需实现 get_q_value_next_state(state_emb, action_emb) 方法， 返回可反向传播的张量，其第一维应包含单个样本的Q值。 
        cluster_num (int, optional): 聚类数目（特征被划分为多少个簇），默认值为 2。 
        method (str, optional): 状态/动作嵌入的生成方法名称，例如 "mds" 等，默认 "mds"。 
        gpu (int, optional): 计算设备ID，-1 表示使用 CPU，非负整数表示对应的 GPU ID，默认 -1。

    Returns: tuple: - act_emb (array-like): 被选中聚类的动作嵌入表示。 - state_emb (array-like): 全部特征的全局状态嵌入表示。 - f_cluster (array-like of shape (n_samples, n_selected_features)): 被选中聚类对应的特征子矩阵。 - clusters (dict[int, Iterable[int]]): 聚类结果映射，键为聚类标识，值为该簇内特征的列索引集合。

    Raises: RuntimeError: 当评估网络在计算Q值时发生错误（例如设备不匹配、前向传播失败）时可能抛出。 ValueError: 当输入维度或类型与预期不符导致聚类或嵌入生成失败时可能抛出。 """
    clusters = cluster_features(X, y, cluster_num)
    state_emb = feature_state_generation(pd.DataFrame(X), method, gpu)
    q_vals, cluster_list, action_list = [], [], []
    for key, value in clusters.items():
        action = feature_state_generation(pd.DataFrame(X[:, list(value)]), method, gpu)
        # 这里使用evalnet
        q_value = (
            dqn_cluster.get_q_value_next_state(state_emb, action).detach().numpy()[0]
        )
        q_vals.append(q_value)
        cluster_list.append(key)
        action_list.append(action)
    #     1. 这里直接argmax 之前是epi+argmax
    act_emb = action_list[np.argmax(q_vals)]
    act_ind = cluster_list[np.argmax(q_vals)]
    f_cluster = X[:, list(clusters[act_ind])]
    return act_emb, state_emb, f_cluster, clusters


def generate_next_state_of_meta_operation(
    f_cluster_, operation_set, dqn_operation, method="mds", gpu=-1
):
    """ 基于给定的特征簇生成当前元操作的状态表示，并使用 DQN 策略选择下一步操作。

    Args: 
        f_cluster_ (Iterable | pandas.DataFrame): 特征簇数据；将被转换为 DataFrame 用于状态构建。 
        operation_set (Sequence): 可选操作集合，需支持通过整数索引访问。 
        dqn_operation: 具备 choose_next_action(state) 方法的策略/代理，用于基于状态选择操作索引。 
        method (str, optional): 状态构建方法（例如 "mds"）。默认为 "mds"。 
        gpu (int, optional): GPU 设备编号；小于 0 表示使用 CPU。默认为 -1。

    Returns: tuple: - op_state (Any): 由 feature_state_generation 生成的操作状态表示。 - op (Any): 在 operation_set 中被选中的下一步操作。

    Raises: IndexError: 当返回的操作索引超出 operation_set 范围时。 TypeError: 当 operation_set 不可索引或 dqn_operation 不具备 choose_next_action 方法时。 ValueError: 当状态构建失败或生成的状态无效时。 """
    op_state = feature_state_generation(pd.DataFrame(f_cluster_), method, gpu)
    op_index = dqn_operation.choose_next_action(op_state)
    op = operation_set[op_index]
    return op_state, op


def generate_next_state_of_meta_cluster2(
    f_cluster_, op_emb_, clusters, X, dqn_cluster, method="mds", gpu=-1
):
    """ 基于当前元簇表示与操作嵌入，评估所有候选簇的动作价值，选择最优动作的表示，并生成下一状态表示

    该函数先对当前元簇特征与操作嵌入构造状态表示，然后遍历候选簇，将对应特征子集转换为动作表示，使用提供的 DQN 模型计算每个动作在下一状态下的 Q 值，最终返回 Q 值最高的动作表示及对应的状态表示。

    Args: 
        f_cluster_ (array-like or pandas.DataFrame): 当前元簇的特征集合，用于生成特征表示。 
        op_emb_ (array-like): 当前操作/算子的嵌入向量，用于与特征表示拼接构成状态。 
        clusters (dict): 候选簇映射，键为簇标识（如 int 或 str），值为该簇包含的特征索引可迭代对象。 
        X (numpy.ndarray): 原始特征矩阵，形状为 (n_samples, n_features)，用于按索引提取各候选簇的特征子集。 
        dqn_cluster (object): 具备 get_q_value_next_state(state_emb, action) 方法的 DQN 模型，用于评估动作价值。 
        method (str, optional): 特征/动作表示的生成方法（如降维/嵌入算法名），默认为 "mds"。 
        gpu (int, optional): 设备标识；-1 表示使用 CPU，非负整数表示对应的 GPU 编号，默认为 -1。

    Returns: tuple: - action_emb (array-like): 被选中的最优动作（候选簇）的表示向量。 - state_emb (torch.Tensor): 当前状态的表示向量，由元簇特征表示与操作嵌入拼接得到。

    Raises: ValueError: 当 candidates 列表为空（clusters 为空或无有效动作）导致无法取 argmax 时。 AttributeError: 当 dqn_cluster 未实现 get_q_value_next_state 方法时。 TypeError: 当输入数据类型不兼容，无法转换为张量或进行拼接时。 RuntimeError: 当张量形状不匹配或在计算图/设备上发生运行时错误时。 """
    feature_emb = feature_state_generation(pd.DataFrame(f_cluster_), method, gpu)
    state_emb = torch.cat((torch.tensor(feature_emb), torch.tensor(op_emb_)))
    q_vals, cluster_list, action_list = [], [], []
    for key, value in clusters.items():
        action = feature_state_generation(pd.DataFrame(X[:, list(value)]), method, gpu)
        action_list.append(action)
        q_value = (
            dqn_cluster.get_q_value_next_state(state_emb, action).detach().numpy()[0]
        )
        q_vals.append(q_value)
        cluster_list.append(key)
    action_emb = action_list[np.argmax(q_vals)]
    return action_emb, state_emb


def relative_absolute_error(y_test, y_predict):
    """ 计算相对绝对误差（Relative Absolute Error, RAE），定义为 sum(|y_true - y_pred|) / sum(|y_true - mean(y_true)|)。该指标用于衡量预测相对基准（以真实值均值作为简单模型）的误差大小，值越小表示模型相对基准更优。

    Args: y_test (array-like): 真实目标值序列，可为列表、元组或 numpy 数组。 y_predict (array-like): 预测值序列，与 y_test 等长。

    Returns: float: 相对绝对误差的标量值。 """
    y_test = np.array(y_test)
    y_predict = np.array(y_predict)
    error = np.sum(np.abs(y_test - y_predict)) / np.sum(
        np.abs(np.mean(y_test) - y_test)
    )
    return error


def downstream_task_new(data, task_type, metric_type, state_num=10):
    """ 基于给定的表格数据与任务类型，执行默认的下游评测并返回交叉验证得分。

    该函数假定数据集的最后一列为目标变量（y），其余列为特征（X），并依据 task_type 选择相应的基础模型与评价指标：

    "cls"（分类）：使用 RandomForestClassifier 与 StratifiedKFold(5) 评估，返回加权 F1 的 5 折平均值。
    "reg"（回归）：使用 RandomForestRegressor 与 KFold(5) 评估，返回 1 - RAE（相对绝对误差）的 5 折平均值。
    "det"（检测/二分类判别）：使用 KNeighborsClassifier(k=5) 与 StratifiedKFold(5) 评估，基于预测标签计算 ROC-AUC 的 5 折平均值。
    "rank"（排序）：当前未实现，返回 None。
    其他值：返回 -1。
    注意：

    metric_type 与 state_num 参数当前未被使用。
    交叉验证采用固定随机种子 random_state=0，确保结果可复现。
    目标列需与任务类型相匹配，例如 "det" 任务通常要求二分类目标。
    Args: data (pandas.DataFrame): 输入数据表，最后一列为目标变量，其余为特征。 task_type (str): 任务类型，可选值为 "cls"、"reg"、"det"、"rank"。 metric_type (str): 评价指标类型占位参数，当前未使用。 state_num (int): 状态数量占位参数，当前未使用，默认 10。

    Returns: float | None | int: - 对于 "cls"：返回加权 F1 的 5 折平均值。 - 对于 "reg"：返回 1 - RAE 的 5 折平均值。 - 对于 "det"：返回 ROC-AUC 的 5 折平均值。 - 对于 "rank"：返回 None（未实现）。 - 对于未识别的 task_type：返回 -1。

    Raises: ValueError: 当在某些折中目标变量只有单一类别时，roc_auc_score 可能抛出异常（"det" 任务）。 Exception: 来自底层模型拟合或指标计算的其他异常（例如数据中存在 NaN/Inf，或特征与目标维度不匹配）。 """
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    if task_type == "cls":
        clf = RandomForestClassifier(random_state=0)
        f1_list = []
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = (
                X.iloc[train, :],
                y.iloc[train],
                X.iloc[test, :],
                y.iloc[test],
            )
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            f1_list.append(f1_score(y_test, y_predict, average="weighted"))
        return np.mean(f1_list)
    elif task_type == "reg":
        kf = KFold(n_splits=5, random_state=0, shuffle=True)
        reg = RandomForestRegressor(random_state=0)
        rae_list = []
        for train, test in kf.split(X):
            X_train, y_train, X_test, y_test = (
                X.iloc[train, :],
                y.iloc[train],
                X.iloc[test, :],
                y.iloc[test],
            )
            reg.fit(X_train, y_train)
            y_predict = reg.predict(X_test)
            rae_list.append(1 - relative_absolute_error(y_test, y_predict))
        return np.mean(rae_list)
    elif task_type == "det":
        knn = KNeighborsClassifier(n_neighbors=5)
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        ras_list = []
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = (
                X.iloc[train, :],
                y.iloc[train],
                X.iloc[test, :],
                y.iloc[test],
            )
            knn.fit(X_train, y_train)
            y_predict = knn.predict(X_test)
            ras_list.append(roc_auc_score(y_test, y_predict))
        return np.mean(ras_list)
    elif task_type == "rank":
        pass
    else:
        return -1


def downstream_task(data, task_type, metric_type, state_num=10):
    """ 使用随机森林对给定数据执行下游任务评估。约定数据的最后一列为目标变量，其余列为特征。根据任务类型选择分类或回归模型，并在测试集上计算指定指标。

    Args: data (pandas.DataFrame): 包含特征与目标的数据集，最后一列为目标y，其余列为特征X。 task_type (str): 任务类型。'cls' 表示分类任务；'reg' 表示回归任务。 metric_type (str): 评估指标类型。 - 分类('cls'): 'acc' 准确率，'pre' 精确率，'rec' 召回率，'f1' 加权F1分数。 - 回归('reg'): 'mae' 平均绝对误差，'mse' 均方误差，'rae' 相对绝对误差的改写指标（返回 1 - RAE）。 state_num (int, optional): 训练/测试集划分的随机种子，默认值为 10。

    Returns: float: 测试集上的对应评估指标分数。

    Raises: ValueError: 当数据格式不符合要求（如目标列不可用）或 task_type/metric_type 与任务不匹配时，可能由底层库触发。 """


    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=state_num, shuffle=True
    )
    if task_type == "cls":
        clf = RandomForestClassifier(random_state=0).fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        if metric_type == "acc":
            return accuracy_score(y_test, y_predict)
        elif metric_type == "pre":
            return precision_score(y_test, y_predict)
        elif metric_type == "rec":
            return recall_score(y_test, y_predict)
        elif metric_type == "f1":
            return f1_score(y_test, y_predict, average="weighted")
    if task_type == "reg":
        reg = RandomForestRegressor(random_state=0).fit(X_train, y_train)
        y_predict = reg.predict(X_test)
        if metric_type == "mae":
            return mean_absolute_error(y_test, y_predict)
        elif metric_type == "mse":
            return mean_squared_error(y_test, y_predict)
        elif metric_type == "rae":
            return 1 - relative_absolute_error(y_test, y_predict)


def downstream_task_cross_validataion(data, task_type):
    """ 使用随机森林对给定数据执行 5 折交叉验证。

    分类任务使用加权 F1 分数进行评估。
    回归任务使用 1 - 相对绝对误差（Relative Absolute Error）的分数进行评估。 评估结果以数组形式打印到标准输出，不返回值。
    Args: data (pandas.DataFrame): 输入数据集，除最后一列外为特征列，最后一列为目标列。 task_type (str): 任务类型。"cls" 表示分类任务；"reg" 表示回归任务。

    Returns: None: 本函数仅打印交叉验证得分，不返回任何值。

    Raises: ValueError: 当数据格式不符合要求或模型/评估器参数无效时，底层库可能抛出该错误。 Exception: 来自底层 scikit-learn 在交叉验证或评分过程中产生的其他异常。 """
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    if task_type == "cls":
        clf = RandomForestClassifier(random_state=0)
        scores = cross_val_score(clf, X, y, cv=5, scoring="f1_weighted")
        print(scores)
    if task_type == "reg":
        reg = RandomForestRegressor(random_state=0)
        scores = 1 - cross_val_score(
            reg, X, y, cv=5, scoring=make_scorer(relative_absolute_error)
        )
        print(scores)


def test_task_new(Dg, task="cls", state_num=10):
    """ 对输入的数据集按指定任务类型执行 5 折交叉验证评估，并返回各折指标的均值

    该函数默认将输入 DataFrame 的最后一列视为标签 y，其余列为特征 X。 不同任务采用不同的基学习器与评价指标：

    分类任务（"cls"）使用 RandomForestClassifier，计算加权准确率、加权精确率、加权召回率与加权 F1 分数
    回归任务（"reg"）使用 RandomForestRegressor，计算 MAE、MSE 与相对绝对误差（RAE）
    检测任务（"det"）使用 KNeighborsClassifier，计算平均精度（Average Precision）、宏平均 F1 与 ROC-AUC
    排序任务（"rank"）尚未实现
    其他未知任务将返回 -1
    Args: Dg (pandas.DataFrame): 输入数据集，最后一列为标签，前面列为特征。 task (str): 任务类型，可选值为 "cls"、"reg"、"det"、"rank"。 state_num (int): 预留参数，当前未使用，默认为 10。

    Returns: tuple[float, float, float, float] | tuple[float, float, float] | int: - 当 task == "cls" 时，返回 (accuracy, precision_weighted, recall_weighted, f1_weighted) 的均值元组。 - 当 task == "reg" 时，返回 (mae, mse, rae) 的均值元组。 - 当 task == "det" 时，返回 (average_precision, f1_macro, roc_auc) 的均值元组。 - 当 task == "rank" 或其他未支持任务时，分别返回 None 或 -1（当前实现为 -1）。

    Raises: ValueError: 当输入数据不满足模型训练或评估要求（例如标签为空、特征维度不合法）时，底层库可能抛出相应异常。 """


    X = Dg.iloc[:, :-1]
    y = Dg.iloc[:, -1]
    if task == "cls":
        clf = RandomForestClassifier(random_state=0)
        acc_list, pre_list, rec_list, f1_list = [], [], [], []
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = (
                X.iloc[train, :],
                y.iloc[train],
                X.iloc[test, :],
                y.iloc[test],
            )
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            acc_list.append(accuracy_score(y_test, y_predict))
            pre_list.append(
                precision_score(y_test, y_predict, average="weighted", zero_division=0)
            )
            rec_list.append(
                recall_score(y_test, y_predict, average="weighted", zero_division=0)
            )
            f1_list.append(
                f1_score(y_test, y_predict, average="weighted", zero_division=0)
            )
        return np.mean(acc_list), np.mean(pre_list), np.mean(rec_list), np.mean(f1_list)
    elif task == "reg":
        kf = KFold(n_splits=5, random_state=0, shuffle=True)
        reg = RandomForestRegressor(random_state=0)
        mae_list, mse_list, rae_list = [], [], []
        for train, test in kf.split(X):
            X_train, y_train, X_test, y_test = (
                X.iloc[train, :],
                y.iloc[train],
                X.iloc[test, :],
                y.iloc[test],
            )
            reg.fit(X_train, y_train)
            y_predict = reg.predict(X_test)
            mae_list.append(mean_absolute_error(y_test, y_predict))
            mse_list.append(mean_squared_error(y_test, y_predict))
            rae_list.append(relative_absolute_error(y_test, y_predict))
        return np.mean(mae_list), np.mean(mse_list), np.mean(rae_list)
    elif task == "det":
        kf = KFold(n_splits=5, random_state=0, shuffle=True)
        knn_model = KNeighborsClassifier(n_neighbors=5)
        map_list = []
        f1_list = []
        ras = []
        for train, test in kf.split(X):
            X_train, y_train, X_test, y_test = (
                X.iloc[train, :],
                y.iloc[train],
                X.iloc[test, :],
                y.iloc[test],
            )
            knn_model.fit(X_train, y_train)
            y_predict = knn_model.predict(X_test)
            map_list.append(average_precision_score(y_test, y_predict))
            f1_list.append(f1_score(y_test, y_predict, average="macro"))
            ras.append(roc_auc_score(y_test, y_predict))
        return np.mean(map_list), np.mean(f1_list), np.mean(ras)
    elif task == "rank":
        pass
    else:
        return -1


def test_task(Dg, task="cls", state_num=10):
    """ 在给定数据集上执行分类或回归的快速基准评估。默认将数据集最后一列视为目标变量，其他列为特征， 按 8:2 划分训练/测试集，并使用随机森林分别计算相应任务的常用评估指标。

    Args: Dg (pandas.DataFrame): 输入数据集，最后一列为目标变量（y），其余列为特征（X）。 task (str): 任务类型，可选值为 "cls"（分类）或 "reg"（回归）。 state_num (int): 训练/测试划分的随机种子（random_state），用于保证可复现的切分。

    Returns: tuple[float, float, float, float] | tuple[float, float, float] | int: - 当 task == "cls" 时，返回 (accuracy, precision, recall, f1)，均为加权（weighted）多类别指标。 - 当 task == "reg" 时，返回 (MAE, RMSE, RAE)，分别为平均绝对误差、均方根误差、相对绝对误差。 - 当 task 既不是 "cls" 也不是 "reg" 时，返回 -1。

    Raises: ValueError: 当数据格式不符合要求（例如目标列不可用于拟合）或底层 scikit-learn 在拟合/预测过程中触发错误时可能抛出。 """
    X = Dg.iloc[:, :-1]
    y = Dg.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=state_num, shuffle=True
    )
    if task == "cls":
        clf = RandomForestClassifier(random_state=0).fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        acc = accuracy_score(y_test, y_predict)
        pre = precision_score(y_test, y_predict, average="weighted")
        rec = recall_score(y_test, y_predict, average="weighted")
        f1 = f1_score(y_test, y_predict, average="weighted")
        return acc, pre, rec, f1
    elif task == "reg":
        reg = RandomForestRegressor(random_state=0).fit(X_train, y_train)
        y_predict = reg.predict(X_test)
        return (
            mean_absolute_error(y_test, y_predict),
            mean_squared_error(y_test, y_predict, squared=False),
            relative_absolute_error(y_test, y_predict),
        )
    else:
        return -1


def overall_feature_selection(best_features, task_type):
    """ 基于整体特征拼接并使用模型驱动的特征选择，返回筛选后的数据集。 对回归任务使用 Lasso（L1 正则）与 SelectFromModel 进行稀疏选择； 对分类任务使用带 L1 惩罚的 LinearSVC 与 SelectFromModel 进行稀疏选择。 函数会在内部对筛选结果进行一次快速评测并记录相关指标日志。

    Args: best_features (Iterable[pandas.DataFrame] | Iterable[pandas.Series]]): 多个特征数据集合， 将按列拼接为一个整体数据集。拼接后的数据集中“最后一列”将被视为目标变量 y， 其余列为特征 X。各输入应具有相同行数并按样本对齐。 task_type (str): 任务类型，"reg" 表示回归任务，"cls" 表示分类任务。

    Returns: pandas.DataFrame: 由筛选后的特征列与原目标列按列拼接得到的新数据集，其中最后一列为目标变量。

    Notes: - 内部会调用 SelectFromModel 依据稀疏模型系数选择特征。 - 会调用 test_task_new 对筛选后的数据进行评测，并通过 info 记录指标： 回归任务记录 mae/mse/(1-rae)，分类任务记录 acc/prec/recall/f1。 - 仅支持 task_type 为 "reg" 或 "cls" 的场景。 """
    if task_type == "reg":
        data = pd.concat([fea for fea in best_features], axis=1)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        reg = linear_model.Lasso(alpha=0.1).fit(X, y)
        model = SelectFromModel(reg, prefit=True)
        X = X.loc[:, model.get_support()]
        new_data = pd.concat([X, y], axis=1)
        mae, mse, rae = test_task_new(new_data, task_type)
        info("mae: {:.3f}, mse: {:.3f}, 1-rae: {:.3f}".format(mae, mse, 1 - rae))
    elif task_type == "cls":
        data = pd.concat([fea for fea in best_features], axis=1)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        clf = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
        model = SelectFromModel(clf, prefit=True)
        X = X.loc[:, model.get_support()]
        new_data = pd.concat([X, y], axis=1)
        acc, pre, rec, f1 = test_task_new(new_data, task_type)
        info(
            "acc: {:.3f}, pre: {:.3f}, rec: {:.3f}, f1: {:.3f}".format(
                acc, pre, rec, f1
            )
        )
    return new_data

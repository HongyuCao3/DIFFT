import os
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append("./")
import torch
from logger import *
from utils.datacollection.Operation import (
    add_unary,
    operation_set,
    O1,
    O3,
    O2,
    sep_token,
)

os.environ["NUMEXPR_MAX_THREADS"] = "32"
os.environ["NUMEXPR_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
torch.set_num_threads(8)
info(torch.get_num_threads())
info(torch.__config__.parallel_info())
import warnings
import math
from nni.utils import merge_parameter
import nni
import random

torch.manual_seed(0)
warnings.filterwarnings("ignore")
warnings.warn("DelftStack")
warnings.warn("Do not show this message")
from sklearn.feature_selection import SelectKBest
from DQN import (
    DQN1,
    DQN2,
    DDQN1,
    DDQN2,
    DuelingDQN1,
    DuelingDDQN2,
    DuelingDQN2,
    DuelingDDQN1,
)
from tools import *
from task_mapping import (
    task_dict,
    task_type,
    base_path,
    task_measure,
    state_rep,
    support_rl_method,
)
import argparse


def init_param():
    parser = argparse.ArgumentParser(description="PyTorch Experiment")
    parser.add_argument(
        "--file-name", type=str, default="ap_omentum_ovary", help="data name"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        help="log level, check the _utils.logger",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="ng",
        help="ng/cls/reg/det/rank, if provided ng, the model will take the task type in config",
    )
    parser.add_argument(
        "--episodes", type=int, default=100, help="episodes for training"
    )
    parser.add_argument("--steps", type=int, default=15, help="steps for each episode")
    parser.add_argument(
        "--enlarge_num", type=int, default=3, help="feature space enlarge"
    )
    parser.add_argument("--memory", type=int, default=8, help="memory capacity")
    parser.add_argument("--eps_start", type=float, default=0.9, help="eps start")
    parser.add_argument("--eps_end", type=float, default=0.5, help="eps end")
    parser.add_argument("--eps_decay", type=int, default=100, help="eps decay")
    parser.add_argument("--index", type=float, default=0.5, help="file index")
    parser.add_argument("--state", type=int, default=0, help="random_state")
    parser.add_argument("--cluster_num", type=int, default=0, help="cluster_num")
    parser.add_argument("--a", type=float, default=0, help="a")
    parser.add_argument("--b", type=float, default=0, help="b")
    parser.add_argument("--c", type=float, default=0, help="c")
    parser.add_argument(
        "--rl-method", type=str, default="dqn", help="used reinforcement methods"
    )
    parser.add_argument(
        "--state-method",
        type=str,
        default="gcn",
        help="reinforcement state representation method",
    )
    parser.add_argument("--default-cuda", type=int, default=-1, help="the using cuda")
    # -c removing the feature clustering step of GRFG
    # -d using euclidean distance as feature distance metric in the M-clustering of GRFG
    # -b -u Third, we developed GRFG−𝑢 and GRFG−𝑏 by using random in the two feature generation scenarios
    parser.add_argument("--ablation-mode", type=str, default="", help="the using cuda")

    args, _ = parser.parse_known_args()
    return args


def model_train(param, nni):
    """使用强化学习进行特征生成与下游任务评估与优化的训练流程

    该函数从指定的 HDF 数据集读取数据，对原始特征进行归一化与命名重映射， 在多种特征操作集合（单目/双目/预处理）上通过分层策略（两阶段特征簇选择与操作选择） 结合多种 DQN 变体（DQN/DDQN/DuelingDQN/DuelingDDQN）进行迭代探索与学习， 以提升给定下游任务（回归/分类/检测）的性能指标。训练过程中可选地与 NNI 集成：

    周期性上报中间结果与最终结果
    使用试验 ID 区分输出 最终会将最优生成特征数据及探索轨迹写入本地临时目录。
    Args:
        param (dict): 训练与任务配置字典，主要键包括但不限于：
            - file_name (str): 数据集文件基名（从 base_path 读取 file_name.hdf）。
            - task (str): 下游任务类型，取值于已支持集合（如 'reg'、'cls'、'det' 或 'ng'）。
            - rl_method (str): 强化学习方法，取值于已支持集合（如 'dqn'、'ddqn'、'dueling_dqn'、'dueling_ddqn'）。
            - state_method (str): 状态表征方法（如 'gcn'、'ae'、'mds+ae'、'mds+ae+gcn'）。
            - default_cuda (str|int): 设备选择/标识。
            - episodes (int): 训练轮数。
            - steps (int): 每轮最大步数。
            - memory (int): 经验回放容量。
            - enlarge_num (int): 特征扩增上限倍数，用于控制特征数量上界。
            - eps_start (float): ε-greedy 初始探索率。 - eps_end (float): ε-greedy 最小探索率。
            - eps_decay (float): ε 衰减因子。
            - a (float): 奖励分配系数（簇选择1）。
            - b (float): 奖励分配系数（操作选择）。
            - c (float): 奖励分配系数（簇选择2）。
        nni (Optional[Any]): NNI 运行时对象；为 None 时本地运行，不进行 NNI 上报。

    Returns: None

    Raises:
        AssertionError: 当 state_method、rl_method 或 task 不在支持集合中时。
        FileNotFoundError: 当指定的 HDF 数据文件不存在时。
        KeyError: 当 HDF 文件键或期望的数据列缺失时。
        ValueError: 当数据预处理、特征生成或模型超参导致非法操作（如对非正数取对数、除以零等）时。

    副作用:
        - 从 base_path 读取 {file_name}.hdf 数据。
        - 在 ./tmp/{file_name}/ 下写出：
        - 最优特征数据 CSV（文件名包含 trial_id 与 best_per）。
        - 探索过程的最优轨迹 .bdata 与全量轨迹 .adata。
        - 使用日志输出训练进度与性能指标。
        - 在使用 NNI 时，上报中间与最终结果。
    """
    DEVICE = param["default_cuda"]
    STATE_METHOD = param["state_method"]
    assert STATE_METHOD in state_rep
    use_nni = True
    if nni is None:
        current_trial_name = "local"
        use_nni = False
    else:
        current_trial_name = nni.get_trial_id()
    D_OPT_PATH = "./tmp/" + param["file_name"] + "/"
    info("opt path is {}".format(D_OPT_PATH))
    always_best = []
    all = []

    # 读取数据
    data_path = base_path + param["file_name"] + ".hdf"
    info("read the data from {}".format(data_path))
    Dg = pd.read_hdf(data_path)

    # 检查数据
    assert param["rl_method"] in support_rl_method
    if param["task"] == "ng":
        task_name = task_dict[param["file_name"]]
    else:
        assert param["task"] in task_type
        task_name = param["task"]
    info("the task is performing " + task_name + " on _dataset " + param["file_name"])
    info("the chosen reinforcement learning method is " + param["rl_method"])
    measure = task_measure[task_name]
    info("the related measurement is " + measure)
    old_per = downstream_task_new(Dg, task_name, measure, state_num=22)
    info("done the base test with performance of {}".format(old_per))
    # if param['file_name'] == 'ap_omentum_ovary':
    #     # 0.7827625741773581 0.838308039423492
    #     k = 100
    #     selector = SelectKBest(mutual_info_regression, k=k).fit(Dg.iloc[:, :-1], Dg.iloc[:, -1])
    #     cols = selector.get_support()
    #     X_new = Dg.iloc[:, :-1].loc[:, cols]
    #     Dg = pd.concat([X_new, Dg.iloc[:, -1]], axis=1)
    #     Dg.columns = [str(i) for i in range(Dg.shape[1])]
    #     Dg.to_hdf(data_path, key='raw', mode='w')
    feature_names = list(Dg.columns)
    info("initialize the features...")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = Dg.values[:, :-1]
    X = scaler.fit_transform(X)
    feature_num = X.shape[1]
    y = Dg.values[:, -1]
    Dg = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1)
    Dg.columns = [str(i + len(operation_set) + 5) for i in range(len(feature_names))]
    feature_names = Dg.columns
    # O1 = ['sqrt', 'square', 'sin', 'cos', 'tanh', 'stand_scaler',
    #     'minmax_scaler', 'quan_trans', 'sigmoid', 'log', 'reciprocal']
    # O2 = ['+', '-', '*', '/']
    # O3 = ['stand_scaler', 'minmax_scaler', 'quan_trans']
    # operation_set = O1 + O2
    one_hot_op = pd.get_dummies(operation_set)
    operation_emb = defaultdict()
    for item in one_hot_op.columns:
        operation_emb[item] = one_hot_op[item].values
    EPISODES = param["episodes"]
    STEPS = param["steps"]
    STATE_DIM = 64
    ACTION_DIM = 64
    MEMORY_CAPACITY = param["memory"]
    OP_DIM = len(operation_set)
    FEATURE_LIMIT = Dg.shape[1] * param["enlarge_num"]
    print("feature limit is {}".format(FEATURE_LIMIT))
    N_ACTIONS = len(operation_set)
    dqn_cluster1 = None
    dqn_operation = None
    dqn_cluster2 = None
    info("initialize the model...")
    if STATE_METHOD == "gcn":
        STATE_DIM = X.shape[0]
        ACTION_DIM = X.shape[0]
    elif STATE_METHOD == "ae":
        STATE_DIM = X.shape[0]
        ACTION_DIM = X.shape[0]
    elif STATE_METHOD == "mds+ae":
        STATE_DIM = X.shape[0] + STATE_DIM
        ACTION_DIM = STATE_DIM
    elif STATE_METHOD == "mds+ae+gcn":
        STATE_DIM = 2 * X.shape[0] + STATE_DIM
        ACTION_DIM = STATE_DIM
        
    # 初始化模型
    if param["rl_method"] == "dqn":
        dqn_cluster1 = DQN1(
            STATE_DIM=STATE_DIM, ACTION_DIM=ACTION_DIM, MEMORY_CAPACITY=MEMORY_CAPACITY
        )
        dqn_operation = DQN2(
            N_STATES=STATE_DIM, N_ACTIONS=N_ACTIONS, MEMORY_CAPACITY=MEMORY_CAPACITY
        )
        dqn_cluster2 = DQN1(
            STATE_DIM=STATE_DIM + OP_DIM,
            ACTION_DIM=ACTION_DIM,
            MEMORY_CAPACITY=MEMORY_CAPACITY,
        )
    elif param["rl_method"] == "ddqn":
        dqn_cluster1 = DDQN1(
            STATE_DIM=STATE_DIM, ACTION_DIM=ACTION_DIM, MEMORY_CAPACITY=MEMORY_CAPACITY
        )
        dqn_operation = DDQN2(
            N_STATES=STATE_DIM, N_ACTIONS=N_ACTIONS, MEMORY_CAPACITY=MEMORY_CAPACITY
        )
        dqn_cluster2 = DDQN1(
            STATE_DIM=STATE_DIM + OP_DIM,
            ACTION_DIM=ACTION_DIM,
            MEMORY_CAPACITY=MEMORY_CAPACITY,
        )
    elif param["rl_method"] == "dueling_dqn":
        dqn_cluster1 = DuelingDQN1(
            STATE_DIM=STATE_DIM, ACTION_DIM=ACTION_DIM, MEMORY_CAPACITY=MEMORY_CAPACITY
        )
        dqn_operation = DuelingDQN2(
            N_STATES=STATE_DIM, N_ACTIONS=N_ACTIONS, MEMORY_CAPACITY=MEMORY_CAPACITY
        )
        dqn_cluster2 = DuelingDQN1(
            STATE_DIM=STATE_DIM + OP_DIM,
            ACTION_DIM=ACTION_DIM,
            MEMORY_CAPACITY=MEMORY_CAPACITY,
        )
    elif param["rl_method"] == "dueling_ddqn":
        dqn_cluster1 = DuelingDDQN1(
            STATE_DIM=STATE_DIM, ACTION_DIM=ACTION_DIM, MEMORY_CAPACITY=MEMORY_CAPACITY
        )
        dqn_operation = DuelingDDQN2(
            N_STATES=STATE_DIM, N_ACTIONS=N_ACTIONS, MEMORY_CAPACITY=MEMORY_CAPACITY
        )
        dqn_cluster2 = DuelingDDQN1(
            STATE_DIM=STATE_DIM + OP_DIM,
            ACTION_DIM=ACTION_DIM,
            MEMORY_CAPACITY=MEMORY_CAPACITY,
        )
    
    # 初始化环境
    base_per = old_per
    episode = 0
    step = 0
    best_per = old_per
    D_OPT = Dg
    best_features = []
    D_original = Dg.copy()
    steps_done = 0
    EPS_START = param["eps_start"]
    EPS_END = param["eps_end"]
    EPS_DECAY = param["eps_decay"]
    CLUSTER_NUM = 4
    duplicate_count = 0
    a, b, c = param["a"], param["b"], param["c"]
    info("initialize the model hyperparameter configure")
    info(
        "epsilon start with {}, end with {}, the decay is {}, the culster num is {}, the duplicate count is {}, the a, b, and c is set to {}, {}, and {}".format(
            EPS_START, EPS_END, EPS_DECAY, CLUSTER_NUM, duplicate_count, a, b, c
        )
    )
    info("the training start...")
    training_start_time = time.time()
    info("start training at " + str(training_start_time))
    best_step = -1
    best_episode = -1
    while episode < EPISODES:
        eps_start_time = time.time()
        step = 0
        Dg = D_original.copy()
        Dg_local = ""
        local_best = -999
        best_per_opt = []
        while step < STEPS:
            info(f"current feature is : {list(Dg.columns)}")
            step_start_time = time.time()
            steps_done += 1
            eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
                -1.0 * steps_done / EPS_DECAY
            )
            clusters = cluster_features(X, y, cluster_num=3)
            
            # 选择特征
            action_emb_c1, state_emb_c1, f_cluster1, f_names1 = select_meta_cluster1(
                clusters,
                Dg.values[:, :-1],
                feature_names,
                eps_threshold,
                dqn_cluster1,
                STATE_METHOD,
                DEVICE,
            )
            
            # 选择操作
            state_emb_op, op, op_index = select_operation(
                f_cluster1,
                operation_set,
                dqn_operation,
                steps_done,
                STATE_METHOD,
                DEVICE,
            )
            info("start operating in step {}".format(step))
            info("current op is " + str(op))
            if op in O1: # 一元操作
                op_sign = justify_operation_type(op)
                f_new, f_new_name = [], []
                if op == "sqrt":
                    for i in range(f_cluster1.shape[1]):
                        if np.sum(f_cluster1[:, i] < 0) == 0:
                            f_new.append(op_sign(f_cluster1[:, i]))
                            f_new_name.append(add_unary(op_map_r[op], f_names1[i]))
                    f_generate = np.array(f_new).T
                    final_name = f_new_name
                    if len(f_generate) == 0:
                        continue
                elif op == "reciprocal":
                    for i in range(f_cluster1.shape[1]):
                        if np.sum(f_cluster1[:, i] == 0) == 0:
                            f_new.append(op_sign(f_cluster1[:, i]))
                            f_new_name.append(add_unary(op_map_r[op], f_names1[i]))
                    f_generate = np.array(f_new).T
                    final_name = f_new_name
                    if len(f_generate) == 0:
                        continue
                elif op == "log":
                    for i in range(f_cluster1.shape[1]):
                        if np.sum(f_cluster1[:, i] <= 0) == 0:
                            f_new.append(op_sign(f_cluster1[:, i]))
                            f_new_name.append(add_unary(op_map_r[op], f_names1[i]))
                    f_generate = np.array(f_new).T
                    final_name = f_new_name
                    if len(f_generate) == 0:
                        continue
                elif op in O3:
                    f_generate = op_sign.fit_transform(f_cluster1)
                    final_name = [add_unary(op_map_r[op], f_n) for f_n in f_names1]
                else:
                    f_generate = op_sign(f_cluster1)
                    final_name = [add_unary(op_map_r[op], f_n) for f_n in f_names1]
            if op in O2: # element-wise operations
                op_emb = operation_emb[op]
                op_func = justify_operation_type(op)
                action_emb_c2, state_emb_c2, f_cluster2, f_names2 = (
                    select_meta_cluster2(
                        clusters,
                        Dg.values[:, :-1],
                        feature_names,
                        f_cluster1,
                        op_emb,
                        eps_threshold,
                        dqn_cluster2,
                        STATE_METHOD,
                        DEVICE,
                    )
                )
                if op == "/" and np.sum(f_cluster2 == 0) > 0:
                    continue
                f_generate, final_name = operate_two_features_new(
                    f_cluster1, f_cluster2, op, op_func, f_names1, f_names2
                )
                
            # 生成新特征
            if np.max(f_generate) > 1000:
                scaler = MinMaxScaler()
                f_generate = scaler.fit_transform(f_generate)
            f_generate = pd.DataFrame(f_generate)
            f_generate.columns = final_name
            public_name = np.intersect1d(np.array(Dg.columns), final_name)
            
            # 检查新特征是否重复
            if len(public_name) > 0:
                reduns = np.setxor1d(final_name, public_name)
                if len(reduns) > 0:
                    f_generate = f_generate[reduns]
                    Dg = insert_generated_feature_to_original_feas(Dg, f_generate)
                else:
                    continue
            else:
                Dg = insert_generated_feature_to_original_feas(Dg, f_generate)
                
            # 检查新特征是否超过限制
            if Dg.shape[1] > FEATURE_LIMIT:
                k = random.randint(feature_num, FEATURE_LIMIT)
                selector = SelectKBest(mutual_info_regression, k=k).fit(
                    Dg.iloc[:, :-1], Dg.iloc[:, -1]
                )
                cols = selector.get_support()
                X_new = Dg.iloc[:, :-1].loc[:, cols]
                Dg = pd.concat([X_new, Dg.iloc[:, -1]], axis=1)
            feature_names = list(Dg.columns)
            new_per = downstream_task_new(Dg, task_name, measure, state_num=0)
            if use_nni: # 使用NNI报告中间结果
                nni.report_intermediate_result(new_per)
            reward = new_per - old_per
            r_c1, r_op, r_c2 = ( # 计算reward
                param["a"] / 10 * reward / 3,
                param["b"] / 10 * reward / 3,
                param["c"] / 10 * reward / 3,
            )
            if new_per > best_per:# 记录最优结果
                always_best.append((Dg.columns, new_per, episode, step))
                best_episode = episode
                best_per = new_per
                D_OPT = Dg.copy()
            if new_per > local_best: # 记录局部最优结果
                local_best = new_per
                Dg_local = Dg.copy()
            all.append((Dg.columns, new_per, episode, step))
            old_per = new_per
            action_emb_c1_, state_emb_c1_, f_cluster_, clusters_ = (
                generate_next_state_of_meta_cluster1(# 生成下一个状态
                    Dg.values[:, :-1],
                    y,
                    dqn_cluster1,
                    cluster_num=CLUSTER_NUM,
                    method=STATE_METHOD,
                    gpu=DEVICE,
                )
            )
            state_emb_op_, op_ = generate_next_state_of_meta_operation(# 生成下一个操作状态
                f_cluster_,
                operation_set,
                dqn_operation,
                method=STATE_METHOD,
                gpu=DEVICE,
            )
            if op in O2:
                action_emb_c2_, state_emb_c2_ = generate_next_state_of_meta_cluster2(# 生成下一个操作状态
                    f_cluster_,
                    operation_emb[op_],
                    clusters_,
                    Dg.values[:, :-1],
                    dqn_cluster2,
                    method=STATE_METHOD,
                    gpu=DEVICE,
                )
                dqn_cluster2.store_transition(
                    state_emb_c2, action_emb_c2, r_c2, state_emb_c2_, action_emb_c2_
                )
            # 存储集群1的过渡
            dqn_cluster1.store_transition(
                state_emb_c1, action_emb_c1, r_c1, state_emb_c1_, action_emb_c1_
            )
            dqn_operation.store_transition(state_emb_op, op_index, r_op, state_emb_op_)
            if dqn_cluster1.memory_counter > dqn_cluster1.MEMORY_CAPACITY:
                dqn_cluster1.learn()
            if dqn_cluster2.memory_counter > dqn_cluster2.MEMORY_CAPACITY:
                dqn_cluster2.learn()
            if dqn_operation.memory_counter > dqn_operation.MEMORY_CAPACITY:
                dqn_operation.learn()

            info(
                "New performance is: {:.6f}, Best performance is: {:.6f} (e{}s{}) Base performance is: {:.6f}".format(
                    new_per, best_per, best_episode, best_step, base_per
                )
            )
            info("Episode {}, Step {} ends!".format(episode, step))
            best_per_opt.append(best_per)
            info(
                "Current spend time for step-{} is: {:.1f}s".format(
                    step, time.time() - step_start_time
                )
            )
            step += 1
        if episode != EPISODES - 1:
            best_features.append(pd.DataFrame(Dg_local.iloc[:, :-1]))
        else:
            best_features.append(Dg_local)
        episode += 1
        info(
            "Current spend time for episode-{} is: {:.1f}s".format(
                episode, time.time() - eps_start_time
            )
        )
        if episode % 5 == 0:
            info("Best performance is: {:.6f}".format(np.min(best_per_opt)))
            info("Episode {} ends!".format(episode))
    info("Total spend time for is: {:.1f}s".format(time.time() - training_start_time))
    info("Exploration ends!")
    
    # 评估最优结果
    info("Begin evaluation...")
    if task_name == "reg":
        mae0, rmse0, rae0 = test_task_new(D_original, task=task_name, state_num=0)
        mae1, rmse1, rae1 = test_task_new(D_OPT, task=task_name, state_num=0)
        if use_nni:
            nni.report_final_result(1 - rae1)
        info(
            "MAE on original is: {:.3f}, MAE on generated is: {:.3f}".format(mae0, mae1)
        )
        info(
            "RMSE on original is: {:.3f}, RMSE on generated is: {:.3f}".format(
                rmse0, rmse1
            )
        )
        info(
            "1-RAE on original is: {:.3f}, 1-RAE on generated is: {:.3f}".format(
                1 - rae0, 1 - rae1
            )
        )
    elif task_name == "cls":
        acc0, precision0, recall0, f1_0 = test_task_new(
            D_original, task=task_name, state_num=0
        )
        acc1, precision1, recall1, f1_1 = test_task_new(
            D_OPT, task=task_name, state_num=0
        )
        if use_nni:
            nni.report_final_result(f1_1)
        info(
            "Acc on original is: {:.3f}, Acc on generated is: {:.3f}".format(acc0, acc1)
        )
        info(
            "Pre on original is: {:.3f}, Pre on generated is: {:.3f}".format(
                precision0, precision1
            )
        )
        info(
            "Rec on original is: {:.3f}, Rec on generated is: {:.3f}".format(
                recall0, recall1
            )
        )
        info(
            "F-1 on original is: {:.3f}, F-1 on generated is: {:.3f}".format(f1_0, f1_1)
        )
    elif task_name == "det":
        map0, f1_0, ras0 = test_task_new(D_original, task=task_name, state_num=0)
        map1, f1_1, ras1 = test_task_new(D_OPT, task=task_name, state_num=0)
        if use_nni:
            nni.report_final_result(ras1)
        info(
            "Average Precision Score on original is: {:.3f}, Average Precision Score on generated is: {:.3f}".format(
                map0, map1
            )
        )
        info(
            "F1 Score on original is: {:.3f}, F1 Score on generated is: {:.3f}".format(
                f1_0, f1_1
            )
        )
        info(
            "ROC AUC Score on original is: {:.3f}, ROC AUC Score on generated is: {:.3f}".format(
                ras0, ras1
            )
        )
    else:
        error("wrong task name!!!!!")
        assert False
    info("Total using time: {:.1f}s".format(time.time() - training_start_time))
    D_OPT.to_csv(D_OPT_PATH + "/" + f"{current_trial_name}_{best_per}.csv")
    always_best_df = []
    with open(D_OPT_PATH + "/" + f"{current_trial_name}.bdata", "w") as f:
        for col_name, per, epi, step_ in always_best:
            col_name = [str(i) for i in list(col_name)]
            line = str.join(f",{str(sep_token)},", col_name) + f",{per},{epi},{step_}\n"
            f.write(line)
    with open(D_OPT_PATH + "/" + f"{current_trial_name}.adata", "w") as f:
        for col_name, per, epi, step_ in all:
            col_name = [str(i) for i in list(col_name)]
            line = str.join(f",{str(sep_token)},", col_name) + f",{per},{epi},{step_}\n"
            f.write(line)


if __name__ == "__main__":
    try:
        args = init_param()
        tuner_params = nni.get_next_parameter()
        trail_id = nni.get_trial_id()
        params = vars(merge_parameter(args, tuner_params))
        if not os.path.exists("./tmp"):
            os.mkdir("./tmp/")
        if not os.path.exists("./tmp/" + params["file_name"] + "/"):
            os.mkdir("./tmp/" + params["file_name"] + "/")
        start_time = str(time.asctime())
        debug(tuner_params)
        info(params)
        model_train(params, nni)
    except Exception as exception:
        error(exception)
        raise

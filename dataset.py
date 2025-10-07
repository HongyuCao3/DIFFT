import os
import sys

sys.path.append('./')

import pickle
from collections import namedtuple
from typing import List, Optional
import traceback
import numpy as np
import pandas
import pandas as pd
import torch
import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from utils.datacollection.logger import error, info
from utils.rlac_tools import test_task_new, downstream_task_new, downstream_task_by_method, \
    downstream_task_by_method_std

import warnings
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from utils.datacollection.Operation import *
from utils_ import chunkwise_seq
from torch.nn.utils.rnn import pad_sequence
warnings.filterwarnings('ignore')

base_path = './data'

TASK_DICT = {'airfoil': 'reg', 'amazon_employee': 'cls', 'ap_omentum_ovary': 'cls',
             'bike_share': 'reg', 'german_credit': 'cls', 'higgs': 'cls',
             'housing_boston': 'reg', 'ionosphere': 'cls', 'lymphography': 'cls',
             'messidor_features': 'cls', 'openml_620': 'reg', 'pima_indian': 'cls',
             'spam_base': 'cls', 'spectf': 'cls', 'svmguide3': 'cls',
             'uci_credit_card': 'cls', 'wine_red': 'cls', 'wine_white': 'cls',
             'openml_586': 'reg', 'openml_589': 'reg', 'openml_607': 'reg',
             'openml_616': 'reg', 'openml_618': 'reg', 'openml_637': 'reg',
             'smtp': 'det', 'thyroid': 'det', 'yeast': 'det', 'wbc': 'det', 'mammography': 'det', 'arrhythmia': 'cls',
             'nomao': 'cls', 'megawatt1': 'cls', 'activity': 'mcls', 'mice_protein': 'mcls', 'coil-20': 'mcls',
             'isolet': 'mcls', 'minist': 'mcls',
             'minist_fashion': 'mcls'
             }

MEASUREMENT = {
    'cls': ['precision', 'recall', 'f1_score', 'roc_auc'],
    'reg': ['mae', 'mse', 'rae', 'rmse'],
    'det': ['map', 'f1_score', 'ras', 'recall'],
    'mcls': ['precision', 'recall', 'mif1', 'maf1']
}

class CustomDataset(Dataset):
    def __init__(self, seqs, tabs, performances):
        assert len(seqs) == len(tabs) == len(performances), "Lengths of seqs, tabs, and performances must be equal"
        self.chunk_seqs, self.ind, self.max_chunk_num = chunkwise_seq(seqs)
        self.chunk_seqs = torch.tensor(self.chunk_seqs, dtype=torch.long)
        # self.seqs = torch.tensor(seqs, dtype=torch.long)
        self.seqs = seqs
        self.tabs = torch.tensor(tabs, dtype=torch.float)
        self.performances = torch.tensor(performances, dtype=torch.float)

    def __len__(self):
        return len(self.performances)
    
    def max_chunk_size(self):
        return self.chunk_seqs.size(1)
    
    # def max_chunk_num(self):
    #     return self.max_chunk_num

    def __getitem__(self, idx):
        return self.seqs[idx], self.tabs[idx], self.performances[idx], self.chunk_seqs[self.ind[idx][0]:self.ind[idx][1] + 1, :]

    # def __getitem__(self, idx):
    #     return {
    #         "seqs": self.seqs[idx], 
    #         "tabs": self.tabs[idx], 
    #         "performances": self.performances[idx], 
    #         "chunk_seqs": self.chunk_seqs[self.ind[idx][0]:self.ind[idx][1] + 1, :]
    #     }

class CustomDataloader(DataLoader):
    def __init__(self, dataset, batch_size=32, shuffle=True, pin_memory=True, drop_last=False, num_workers=10):
        super(CustomDataloader, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
            num_workers=num_workers,
            collate_fn=CustomDataloader.collate_fn  # 指定自定义 collate_fn
        )

    @staticmethod
    def collate_fn(batch):
        """
        处理 batch：
        - `seqs, tabs, performances` 为固定长度，正常 `stack`
        """

        # 解包 batch
        seqs, tabs, performances, chunk_seqs = zip(*batch)
        seqs = [torch.tensor(seq, dtype=torch.long) for seq in seqs]
        seqs = pad_sequence(seqs, batch_first=True, padding_value=0)  # (batch_size, max_seq_len)
        # 处理固定长度的数据（正常 `stack`）
        
        # seqs = torch.stack(seqs)  # (batch_size, ...)
        tabs = torch.stack(tabs)  # (batch_size, ...)
        performances = torch.stack(performances)  # (batch_size, ...)
 
        return {
            "seqs": seqs, 
            "tabs": tabs, 
            "performances": performances, 
            "chunk_seqs": list(chunk_seqs)
        }

class Data_Preprocessing(object):
    def __init__(self, task, task_type=None, dataset=None, batch_size=64, shuffle_time=10, infer_size=100, num_worker=10):
        """ 初始化数据集与数据加载器。根据给定任务名称与可选数据集，完成数据读取/预处理、训练/验证集划分与持久化，并构建用于训练、验证与推理的自定义 Dataset 与 Dataloader。同时计算序列/词表等元信息以及按块切分相关的统计值。

        该初始化流程包含以下步骤：

        加载原始数据（来自传入的 DataFrame 或基于任务名从本地 HDF 文件读取），并进行缺失值填充、列重命名等预处理
        自动推断或接收任务类型
        在无历史缓存时按比例划分训练/测试集，并将 raw/train/test 写入本地 history 目录的 HDF 文件；存在缓存时直接读取
        调用内部数据准备方法，获得训练/验证/推理所需的序列、表格特征与性能标签等张量/数组
        基于 CustomDataset/CustomDataloader 构建训练、验证与推理数据加载器
        计算并保存与按块切分相关的最大块大小与块数等统计
        Args: 
         - task (str): 任务名称。用于定位数据文件并作为 history 缓存的命名空间。 
         - task_type (Optional[Any]): 任务类型。当为 None 时，将通过全局映射表根据 task 自动推断。 
         - dataset (Optional[pandas.DataFrame]): 预先加载的数据集。若提供则跳过从本地 HDF 读取。 
         - batch_size (int, optional): DataLoader 批大小，默认 64。 shuffle_time (int, optional): 训练阶段的打乱次数或相关策略参数，默认 10。 infer_size (int, optional): 推理集采样或批量相关的大小参数，默认 100。 num_worker (int, optional): DataLoader 的工作线程数量，默认 10。

        Returns: None

        Raises: FileNotFoundError: 当 dataset 未提供且基于任务名的 HDF 文件不存在时。 KeyError: 当 task_type 需自动推断而 task 不在任务映射表中时，或读取 HDF 指定键失败时。 OSError: 本地持久化（写入 history HDF 文件）失败时。 ValueError: 数据划分、特征选择或下游数据转换过程中收到非法参数或数据格式不符合预期时。 TypeError: 传入 dataset 的类型或列格式不符合预期导致的类型错误。

        注:

        会在本地 base_path/history/{task}.hdf 下读写 'raw'、'train'、'test' 三个键。
        初始化后典型属性包括但不限于：
        training_data、validation_data、infer_data：对应阶段的 DataLoader
        train_seq/train_tab/train_perf、val_seq/val_tab/val_perf、infer_seq/infer_tab/infer_perf：数据张量/数组
        input_size、max_length、vocab_size、tab_len：数据规模与元信息
        max_chunk_size、max_chunk_num：按块切分的统计指标 """
        self.task_name = task
        self.batch_size = batch_size
        self.shuffle_time = shuffle_time
        self.infer_size = infer_size
        if task_type is None:
            self.task_type = TASK_DICT[self.task_name]
        else:
            self.task_type = task_type

        if dataset is None:
            data_path = os.path.join(base_path, self.task_name + '.hdf')
            original = pd.read_hdf(data_path)
        else:
            original = dataset
        col = np.arange(original.shape[1])
        self.col_names = original.columns
        original.columns = col
        self.original = original.fillna(value=0)
        y = self.original.iloc[:, -1]
        x = self.original.iloc[:, :-1]
        info('initialize the train and test dataset')
        self._check_path()
        if not os.path.exists(os.path.join(base_path, 'history', self.task_name + '.hdf')):
            if self.task_name == 'ap_omentum_ovary':
                Dg = pd.concat([x, y], axis=1)
    # 0.7827625741773581 0.838308039423492
                k = 100
                selector = SelectKBest(mutual_info_regression, k=k).fit(Dg.iloc[:, :-1], Dg.iloc[:, -1])
                cols = selector.get_support()
                X_new = Dg.iloc[:, :-1].loc[:, cols]
                Dg = pd.concat([X_new, Dg.iloc[:, -1]], axis=1)
                x = Dg.iloc[:, :-1]
                y = Dg.iloc[:, -1]
                # x = pd.concat([X_new, y], axis=1)
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                                random_state=0, shuffle=True)
            self.train = pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train)], axis=1)
            self.train.reset_index(drop=True, inplace=True)
            self.test = pd.concat([pd.DataFrame(X_test), pd.DataFrame(y_test)], axis=1)
            self.test.reset_index(drop=True, inplace=True)
            self.train.to_hdf(os.path.join(base_path, 'history', self.task_name + '.hdf'), key='train')
            self.test.to_hdf(os.path.join(base_path, 'history', self.task_name + '.hdf'), key='test')
            self.original.to_hdf(os.path.join(base_path, 'history', self.task_name + '.hdf'), key='raw')
        else:
            self.train = pd.read_hdf(os.path.join(base_path, 'history', self.task_name + '.hdf'), key='train')
            self.test = pd.read_hdf(os.path.join(base_path, 'history', self.task_name + '.hdf'), key='test')
        self.input_size = self.train.shape[0] # , self.infer_seq, self.infer_perf, self.infer_tab
        self.train_seq, self.train_perf, self.train_tab, self.max_length, self.vocab_size, self.val_seq, self.val_perf, self.val_tab, self.infer_seq, self.infer_perf, self.infer_tab = \
            self._get_data_from_local()
        self.tab_len = self.train_tab[0].shape[0]
        train_dataset = CustomDataset(self.train_seq, self.train_tab, self.train_perf)
        
        # 创建 DataLoader 对象
        self.training_data = CustomDataloader(
            dataset=train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            pin_memory=True, 
            drop_last=False, 
            num_workers=num_worker
        )
        val_dataset = CustomDataset(self.val_seq, self.val_tab, self.val_perf)
        self.validation_data = CustomDataloader(
            dataset=val_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            pin_memory=True, 
            drop_last=False, 
            num_workers=num_worker
        )
        infer_dataset = CustomDataset(self.infer_seq, self.infer_tab, self.infer_perf)
        self.infer_data = CustomDataloader(
            dataset=infer_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            pin_memory=True, 
            drop_last=False, 
            num_workers=num_worker
        )
        self.max_chunk_size = max(train_dataset.max_chunk_size(), val_dataset.max_chunk_size())
        self.max_chunk_num = max(train_dataset.max_chunk_num, val_dataset.max_chunk_num)
    def get_performance(self, data=None):
        if data is None:
            data = self.original
        return downstream_task_new(data, self.task_type)      
      
    def _get_data_from_local(self, file_base_path='./tmp'):
        """ 从本地缓存或原始文件构建数据集

        优先从缓存目录加载已处理的数据；若缓存不存在，则从本地原始数据文件解析序列、性能指标与表征特征，执行特征构建与数据拆分（训练/验证）及洗牌，生成推理集合，并将结果写入缓存。函数同时统计序列最大长度与词表大小。

        注意：

        缓存目录按任务名组织，包含训练、验证与推理三套数据（operation.list.pkl、performance.list.pkl、table.list.pkl 及其前缀 val_/infer_ 变体）。
        原始数据按任务名子目录读取，依据外部标志选择后缀（.adata 或 .bdata）。
        序列会基于分隔符进行段内归约与展开，训练/验证序列首尾分别添加起止标记（1 与 2）。
        推理集合按性能排序选取前若干样本的序列与特征。
        特征构建依赖类/实例方法与外部工具（如 Feature_GCN、split_list、show_ops、converge、op_post_seq 等）。
        Args: file_base_path (str): 原始数据根目录路径（包含按任务名组织的子目录）。默认为 './tmp'。

        Returns: tuple: 
            - train_seq (list[list[int]]): 训练集操作序列（含起止标记）。 
            - train_perf (list[list[float]]): 训练集性能指标列表（每条样本为长度为 1 的列表）。 
            - train_tab (list[ArrayLike]): 训练集表征特征（由 Feature_GCN 产生，数组/张量等）。 
            - max_length (int): 样本序列的最大长度（基于全量或缓存统计）。 
            - vocab_size (int): 词表最大索引值（基于全量或缓存统计）。 
            - val_seq (list[list[int]]): 验证集操作序列（含起止标记）。 
            - val_perf (list[list[float]]): 验证集性能指标列表。 
            - val_tab (list[ArrayLike]): 验证集表征特征。 
            - infer_seq (list[list[int]]): 推理集合操作序列（不加起止标记，或依实现而定）。 
            - infer_perf (list[list[float]]): 推理集合性能指标列表。 
            - infer_tab (list[ArrayLike]): 推理集合表征特征。

        Raises: FileNotFoundError: 原始数据目录或期望的原始数据文件不存在。 pickle.UnpicklingError: 从缓存读取 pickle 文件失败或缓存损坏。 ValueError: 原始数据解析过程中出现格式不符合预期的内容（如非整数/浮点转换失败）。 OSError: 读写缓存目录/文件时发生的系统级错误。 
        """
        name = self.task_name
        opt_path = f'{base_path}/history/{self.task_name}'
        opt_file = f'operation.list.pkl'
        performance_file = f'performance.list.pkl'
        tab_file = f'table.list.pkl'
        size = []
        train_seq = []
        train_perf = []
        train_tab = []
        val_seq = []
        val_perf = []
        val_tab = []
        overall_processed_seq = []
        overall_tab = []
        overall_performance = []
        infer_seq = []
        infer_perf = []
        infer_tab = []
        perf = []
        max_length = -1
        vocab_size = 0
        if os.path.exists(os.path.join(opt_path, opt_file)) and \
                os.path.exists(os.path.join(opt_path, performance_file)):
            info(f'have processed the data, load cache from local in :'
                 f'{opt_path}/performance.list.pkl and {opt_path}/operation.list.pkl')
            with open(os.path.join(opt_path, performance_file), 'rb') as f:
                train_perf= pickle.load(f)
            with open(os.path.join(opt_path, opt_file), 'rb') as f:
                train_seq = pickle.load(f)
            with open(os.path.join(opt_path, tab_file), 'rb') as f:
                train_tab = pickle.load(f)
            with open(os.path.join(opt_path, 'val_' + opt_file), 'rb') as f:
                val_seq = pickle.load(f)
            with open(os.path.join(opt_path, 'val_' + performance_file), 'rb') as f:
                val_perf = pickle.load(f)
            with open(os.path.join(opt_path, 'val_' + tab_file), 'rb') as f:
                val_tab = pickle.load(f)
            with open(os.path.join(opt_path, 'infer_' + opt_file), 'rb') as f:
                infer_seq = pickle.load(f)
            with open(os.path.join(opt_path, 'infer_' + performance_file), 'rb') as f:
                infer_perf = pickle.load(f)
            with open(os.path.join(opt_path, 'infer_' + tab_file), 'rb') as f:
                infer_tab = pickle.load(f)

            for i in tqdm.tqdm(train_seq, desc='calculate the max length in train'):
                size.append(len(i))
                if len(i) > max_length:
                    max_length = len(i)
                if vocab_size < max(i):
                    vocab_size = max(i)
            for i in tqdm.tqdm(val_seq, desc='calculate the max length in val'):
                size.append(len(i))
                if len(i) > max_length:
                    max_length = len(i)
                if vocab_size < max(i):
                    vocab_size = max(i)
        else:
            info('initial the data from local file')
            db_list = os.listdir(os.path.join(file_base_path, name))
            filtered_name = []
            if all:
                suffix = '.adata'
            else:
                suffix = '.bdata'
            for f in db_list:
                if f.__contains__(suffix):
                    filtered_name.append(os.path.join(file_base_path, name, f))
            if not os.path.exists(opt_path):
                os.mkdir(opt_path)
            for i in filtered_name:
                with open(i) as f:
                    lines = f.readlines()
                    for line in tqdm.tqdm(lines, desc=f'processing local data'):
                        seq = line.strip().split(',')
                        processed_seq = []
                        performance = []
                        tmp = []
                        for i in seq[:-3]:
                            if i == str(sep_token):  # split: setting to '4'
                                if len(tmp) == 1:
                                    # processed_seq.append(show_ops_r(converge(show_ops(tmp))))
                                    processed_seq.append(tmp.pop(0))
                                    processed_seq.append(int(i))
                                else:
                                    for tok in show_ops_r(converge(show_ops(tmp))):
                                        processed_seq.append(tok)
                                    processed_seq.append(int(i))
                                    tmp = []
                            else:
                                tmp.append(int(i))
                        if len(tmp) != 0:
                            for tok in show_ops_r(converge(show_ops(tmp))):
                                processed_seq.append(tok)
                        else:
                            processed_seq.pop(-1)
                        if max_length < len(processed_seq):
                            max_length = len(processed_seq)
                        if vocab_size < max(processed_seq):
                            vocab_size = max(processed_seq)
                        performance.append(float(seq[-3]))
                        overall_processed_seq.append(processed_seq)
                        split_seq = split_list(processed_seq)
                        tab_data = pd.DataFrame()
                        for i in split_seq:
                            op = show_ops_r(converge(show_ops(i)))
                            tab_data[str(i)] = op_post_seq(self.original, op)
                        data_representation = self.Feature_GCN(tab_data)
                        overall_tab.append(data_representation)
                        size.append(len(processed_seq))
                        overall_performance.append(performance)
                        perf.append(performance[0])


            infer_indices = np.argsort(perf, axis=0)[-self.infer_size:][::-1]
            infer_indices = infer_indices.astype(int)  # 确保 infer_indices 是整数数组
            infer_seq = [overall_processed_seq[i] for i in infer_indices]
            infer_tab = [overall_tab[i] for i in infer_indices]
            infer_perf = [overall_performance[i] for i in infer_indices]


            train_seq, val_seq, train_tab, val_tab, train_perf, val_perf = train_test_split(
                overall_processed_seq, overall_tab, overall_performance, test_size=0.2, random_state=42)

            # 对训练集和验证集分别执行 shuffle 和 padding 操作
            train_seq, train_tab, train_perf = self.shuffle(train_seq, train_tab, train_perf, shuffle_times=self.shuffle_time)
            val_seq, val_tab, val_perf = self.shuffle(val_seq, val_tab, val_perf, shuffle_times=self.shuffle_time)
            # infer_seq = [[1] + seq + [2] for seq in infer_seq]
            train_seq = [[1] + seq + [2] for seq in train_seq]
            val_seq = [[1] + seq + [2] for seq in val_seq]
            # overall_processed_seq, overall_tab, overall_performance =  \
            # self.shuffle(overall_processed_seq, overall_tab, overall_performance, shuffle_times=self.shuffle_time)
            # overall_processed_seq = self.padding(overall_processed_seq, max_length)
            # train_seq = self.padding(train_seq, max_length)
            # val_seq = self.padding(val_seq, max_length)
            with open(os.path.join(opt_path, opt_file), 'wb') as f:
                pickle.dump(train_seq, f)
            with open(os.path.join(opt_path, performance_file), 'wb') as f:
                pickle.dump(train_perf, f)
            with open(os.path.join(opt_path, tab_file), 'wb') as f:
                pickle.dump(train_tab, f)

            with open(os.path.join(opt_path, 'infer_' + opt_file), 'wb') as f:
                pickle.dump(infer_seq, f)
            with open(os.path.join(opt_path, 'infer_' + performance_file), 'wb') as f:
                pickle.dump(infer_perf, f)
            with open(os.path.join(opt_path, 'infer_' + tab_file), 'wb') as f:
                pickle.dump(infer_tab, f)

            with open(os.path.join(opt_path, 'val_' + opt_file), 'wb') as f:
                pickle.dump(val_seq, f)
            with open(os.path.join(opt_path, 'val_' + performance_file), 'wb') as f:
                pickle.dump(val_perf, f)
            with open(os.path.join(opt_path, 'val_' + tab_file), 'wb') as f:
                pickle.dump(val_tab, f)
        # overall_performance = choosed_performance
        # overall_processed_seq = choosed_seq
        return train_seq, train_perf, train_tab, max_length, vocab_size, val_seq, val_perf, val_tab , infer_seq, infer_perf, infer_tab
    def padding(self, seq_list, max_length):
        padded_seq_list = []
        for seq in seq_list:
            padded_seq = seq + [0] * (max_length - len(seq))
            padded_seq_list.append(padded_seq)
        return padded_seq_list
    def _check_path(self):
        if not os.path.exists(f'{base_path}/history/'):
            os.mkdir(f'{base_path}/history/')
        if not os.path.exists(f'{base_path}/history/{self.task_name}'):
            os.mkdir(f'{base_path}/history/{self.task_name}')
            
    def Feature_GCN(selef, X):
        """ 基于特征间皮尔逊相关性的简易特征图卷积（GCN）聚合，生成每个样本的单维特征表示

        该方法先计算输入特征矩阵各特征（列）之间的绝对相关系数矩阵，并在去除对角线后按列与行的总和进行归一化，再添加自环以形成加权邻接矩阵。
        随后将原始特征矩阵与该权重矩阵相乘，并对结果在特征维度上取均值，从而得到每个样本的聚合特征。

        Args: X (pandas.DataFrame): 特征矩阵，形状为 (n_samples, n_features)。每列表示一个特征，每行表示一个样本。

        Returns: numpy.ndarray: 聚合后的样本级特征，形状为 (n_samples,)，对应每个样本的单一数值表示。 
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
    
    def shuffle(self, seq, tab, performance, shuffle_times=10):
        augmented_seq, augmented_tab, augmented_performance = [], [], []

        for s, t, p in tqdm.tqdm(zip(seq, tab, performance), total=len(seq), desc="Shuffling"):
            split_seq = split_list(s)
            for seed in range(shuffle_times):
                np.random.seed(seed)
                np.random.shuffle(split_seq)
                new_seq = combine_list(split_seq)

                augmented_seq.append(new_seq)
                augmented_tab.append(t)
                augmented_performance.append(p)

        seq.extend(augmented_seq)
        tab.extend(augmented_tab)
        performance.extend(augmented_performance)

        return seq, tab, performance


        

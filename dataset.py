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


        

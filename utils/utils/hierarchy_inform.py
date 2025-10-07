import torch

"""
this class is used to init model trainer[embedding]

sos : root 0
"""


class HierarchyStructure:
    def __init__(self, layers, applyindices, aim, iam, hidm, heidm):
        """ 初始化层级标签信息与索引映射，用于构建按层次组织的全局/局部标签对照关系

        该初始化过程会：

        记录层数与每层的节点数量统计
        依据输入索引与映射关系计算总节点数与特殊标记（起始/填充）
        构建分层到全局标签的映射矩阵 hierarchical_label，其最后一层为全局身份映射（identity）
        Args: 
         - layers (int): 层级数（不包含附加的全局层）。 
         - applyindices (Iterable[int]): 需要参与统计/映射的索引集合（应用节点索引）。 
         - aim (Dict): applyid 到索引的映射（applyid_index_map）。 
         - iam (Dict): 索引到 applyid 的映射（index_applyid_map）。 
         - hidm (Dict[Any, Sequence[int]]): 局部（分层）标签映射，键为 applyid，值为包含层内局部标签 id 的序列（通常含首/末特殊标记）。 
         - heidm (Dict[Any, Sequence[int]]): 全局标签映射，键为 applyid，值为对应的全局标签序列（与 hidm 对齐）。

        Returns: None: 构造函数不返回值，但会初始化以下关键属性： 
            - layer_num (int): 层数 
            - num_per_layer (torch.LongTensor): 各层节点计数，形状为 [layer_num] 
            - num_nodes (int): 全部节点（含特殊标记）数量 
            - label_map (dict): 局部使用的标签索引缓存（初始化为空） 
            - applyid_index_map (Dict): 入参 aim 的引用 
            - index_applyid_map (Dict): 入参 iam 的引用 
            - hierarchical_label (torch.LongTensor): 分层到全局标签映射矩阵，形状为 [layer_num + 1, num_nodes] 
            - flat_init (bool): 是否已完成扁平化初始化标记 
            - hierarchical_init (bool): 是否已完成层级初始化标记 
            - sos_token (int): 起始标记 id（根节点） 
            - pad_token (int): 填充标记 id

        Raises: KeyError: 当在 heidm 或 iam 中查找缺失的 applyid/索引键时可能引发。 IndexError: 当写入 hierarchical_label 时索引越界可能引发。 ValueError: 当 hidm 与 heidm 的层级长度不一致或与 layers 不匹配时可能引发。 
        """
        self.layer_num = layers
        self.num_per_layer = torch.zeros(self.layer_num,
                                         dtype=torch.long) + 1
        self.num_nodes = len(applyindices) + self.layer_num + 1 # all labels + 4 eos + 1 padding
        self.label_map = dict()  # store the applyid's indice in each layer
        self.applyid_index_map = aim
        self.index_applyid_map = iam
        self.hierarchical_label = torch.zeros(self.layer_num + 1, self.num_nodes,
                                              dtype=torch.long)  # 存放每一层的标签所代表的global label
        self.flat_init = False
        self.hierarchical_init = False
        self.sos_token = 0  # <\sos> is root
        self.pad_token = 1
        for i in applyindices:
            ids = self.index_applyid_map[i]
            if ids == 'root':
                continue
            else:
                self.num_per_layer[int((len(ids) - 1) / 2 + 1) - 1] += 1
        # self.num_per_layer[-1] = self.num_nodes
        for applyid, local_labels in hidm.items(): # hid is local-wise label
            heid = heidm[applyid] # hmid is global-wise label
            for layer, local_label in enumerate(local_labels[1:-1]):
                # -1 => layer_wise eos
                # -2 => padding
                self.hierarchical_label[layer][local_label] = heid[layer + 1]
        # last layer is all nodes => for global use
        self.hierarchical_label[-1] = torch.arange(0, self.num_nodes) # adding last layer
        self.hierarchical_label[:-1, -1] = self.pad_token

# if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-i", "--id-label-map", type=str
    #                     , default='~/jupyter_base/HMTransformer/ds/data/label_map_2019',
    #                     help="label map, map the application id to the applyid code str")
    # parser.add_argument("-ii", "--indice-app-map", type=str
    #                     , default='~/jupyter_base/HMTransformer/ds/data/indice_id',
    #                     help="apply id map, map the applyid to the applyid index")
    # parser.add_argument("-ln", "--label-number", type=int, default=3545)
    #
    # # hierarchy data input
    # parser.add_argument("-hi", "--hierarchy-indice", type=str
    #                     , default='~/jupyter_base/HMTransformer/ds/data/hierarchical_indice_id')
    # parser.add_argument("-beta", "--beta", type=float, default=0.01,
    #                     help='to modified the weighted of global-wise train or layer-wise train')
    # args = parser.parse_args()
    # aim = dict()
    # iam = dict()
    # with open(args.indice_app_map, 'r') as f:
    #     mapping = [line.replace('\r', '').replace('\n', '').split('\t')
    #                for line in tqdm.tqdm(f, desc="Loading Indice apply id Map")]
    #     for line in mapping:
    #         indice = line[0]
    #         label = line[1]
    #         aim[label] = int(indice)
    #         iam[int(indice)] = label
    #
    # # vocab = ZHWordVocab('~/jupyter_base/science_embedding/data/corpus_2019')
    # vocab = ZHWordVocab.load_vocab('~/jupyter_base/HMTransformer/ds/data/vocab.save')
    # # vocab.save_vocab('~/jupyter_base/science_embedding/data/vocab.save')
    # ds = ZHSCIBERTDataset('~/jupyter_base/HMTransformer/ds/data/corpus_2019',
    #                       '~/jupyter_base/HMTransformer/ds/data/label_map_2019',
    #                       '~/jupyter_base/HMTransformer/ds/data/indice_id',
    #                       vocab=vocab, seq_len=50, neg_num=10,
    #                       hierarchical_path='~/jupyter_base/HMTransformer/ds/data/hierarchical_indice_id')
    #
    # aihs = HierarchyStructure(layers=4, applyindices=iam.keys(), aim=aim, iam=iam, heidm=ds.hemid_map,
    #                                  hidm=ds.hid_map)
    # # aihs.load_hierarchical_label(f=args.hierarchy_indice)
    # print(1)

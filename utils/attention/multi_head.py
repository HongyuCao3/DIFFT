import torch.nn as nn
from .single import Attention


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """ 执行多头注意力的前向传播。该过程包括将输入投影到多个头的子空间，计算缩放点积注意力， 然后将各头结果拼接并通过输出线性层映射回模型维度。

        Args: 
         - query (Tensor): 查询张量，形状为 (batch_size, query_len, d_model)。 
         - key (Tensor): 键张量，形状为 (batch_size, key_len, d_model)。 
         - value (Tensor): 值张量，形状为 (batch_size, value_len, d_model)。 
         - mask (Tensor, optional): 注意力掩码，可广播到形状 (batch_size, num_heads, query_len, key_len)。默认值为 None。

        Returns: Tensor: 多头注意力的输出，形状为 (batch_size, query_len, d_model)。 
        """
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [layer(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for layer, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)  # batch 20 hdim
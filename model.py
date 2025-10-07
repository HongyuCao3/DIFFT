import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, reduce
from transformers import BertModel


class SinCosPositionEmbedding(nn.Module):
    def __init__(self, hidden_size, max_seq_len=512):
        """ 初始化用于扩散模型的 Transformer 模块，构建表格条件编码、时间步嵌入与多层注意力块的计算图。该初始化过程包括：

        表格条件编码器（TabEncoder）
        输入与条件投影层
        由多个 BasicTransformerBlock 组成的堆栈
        层归一化与零初始化输出投影
        时间步位置嵌入与两层映射网络
        Args: 
         - in_channels (int): 输入特征维度（模型主干输入的通道数或特征数）。 
         - t_channels (int): 噪声时间步/扩散步嵌入的维度。 
         - hidden_channels (int): Transformer 隐层维度（注意力与前馈网络的通用维度）。 
         - context_channels (int, optional): 条件上下文特征维度，用于跨注意力的上下文输入。默认值为 128。 
         - depth (int, optional): Transformer 块的层数。默认值为 1。 
         - n_heads (int, optional): 多头自/交叉注意力的头数。默认值为 8。 
         - dropout (float, optional): Dropout 比例。默认值为 0.0。 
         - tab_len (int, optional): 表格条件输入长度，用于 TabEncoder。默认值为 267。 
         - out_channels (int | None, optional): 输出特征维度（或类别数）。为 None 时输出维度等于输入维度。默认值为 None。

        Returns: None 
        """
        super(SinCosPositionEmbedding, self).__init__()
        self.hidden_size = hidden_size

        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2) * (-math.log(10000.0) / hidden_size)
        )

        pe = torch.zeros(max_seq_len, hidden_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: Tensor (seq_len, batch_size, hidden_size)
        """
        seq_len = x.size(0)
        return (
            self.pe[:seq_len, :].unsqueeze(1).to(x.device)
        )  # (seq_len, 1, hidden_size)


class TransEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        dropout,
        num_layers,
        latent_dim,
        tokenizer,
        max_seq_len=512,
        pretrained_model_name="bert-base-uncased",
    ):
        """ 初始化基于 Transformer 的序列编码器，包含正余弦位置编码、VAE 潜变量层以及用于评分的 MLP 评估头

        Args: 
         - vocab_size (int): 词表大小，通常用于构建或校验嵌入层的规模。 
         - hidden_size (int): Transformer 的隐藏维度（d_model），同时用于位置编码与线性层维度。 
         - dropout (float): Transformer 编码层中的丢弃率。 
         - num_layers (int): TransformerEncoder 的层数。 
         - latent_dim (int): VAE 潜变量维度，用于均值与对数方差的线性映射。 
         - tokenizer: 词嵌入模块或可调用对象，将作为内部的 embedding 使用（如 nn.Embedding 或兼容实现）。 
         - max_seq_len (int, optional): 最大序列长度，用于生成位置编码。默认值为 512。 
         - pretrained_model_name (str, optional): 预训练模型名称，占位以便兼容或扩展预训练资源。默认值为 "bert-base-uncased"。

        Returns: None """
        super(TransEncoder, self).__init__()
        self.pad_token = 0
        self.embedding = tokenizer
        self.position_embedding = SinCosPositionEmbedding(hidden_size, max_seq_len)
        self.adaptive_pooling = nn.AdaptiveAvgPool1d(20)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=int(hidden_size / 64),
            dropout=dropout,
            dim_feedforward=hidden_size * 4,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )
        self.pad_token = 0
        self.norm1 = nn.LayerNorm(hidden_size)
        # VAE layers
        self.fc_mean = nn.Linear(hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)

        # Evaluator MLP
        self.evaluator = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def vae_reparameterize(self, mean, logvar):
        """ 使用变分自编码器（VAE）的重参数化技巧从高斯分布中采样潜变量，以保证采样过程对均值和方差参数可导。

        该方法将对数方差转换为标准差，并采用 z = μ + σ ⊙ ε（ε ~ N(0, I)）进行可微分采样，从而支持端到端的反向传播。

        Args: 
         - mean (torch.Tensor): 潜变量高斯分布的均值，形状与 logvar 相同。 
         - logvar (torch.Tensor): 潜变量高斯分布的对数方差（log σ^2），形状与 mean 相同。

        Returns: torch.Tensor: 采样得到的潜变量张量，形状与 mean 相同。 """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, seq):
        """ 执行模型的前向传播。首先对输入序列进行嵌入并添加位置编码，随后通过 Transformer 编码器获取上下文表征并进行归一化。基于编码结果预测潜变量的均值与对数方差，经由 VAE 重参数化得到采样的潜变量；同时对序列级聚合表征进行评估，最后返回相关结果。

        Args: 
         - seq (torch.Tensor): 输入的 token 序列张量，形状为 (seq_len, batch)，通常为整型索引，其中等于 pad_token 的位置将用于生成 padding mask。

        Returns: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
            - z (torch.Tensor): 通过重参数化采样得到的潜变量，形状为 (batch, seq_len, latent_dim)。 
            - mean (torch.Tensor): 潜变量分布的均值，形状为 (seq_len, batch, latent_dim) 或与编码张量一致的时间批次布局。 
            - logvar (torch.Tensor): 潜变量分布的对数方差，形状与 mean 相同。 
            - evaluation (torch.Tensor): 基于序列级（对 batch 维度求平均后）表征的评估输出，形状通常为 (seq_len, eval_dim)，具体取决于 evaluator 的实现。 
            - encoded (torch.Tensor): Transformer 编码后的归一化表征，形状为 (seq_len, batch, hidden_dim)。

        Raises: ValueError: 当输入张量维度与期望不匹配或设备/数据类型不兼容时可能抛出（取决于具体模块实现）。 """
        seq_len, batch_size = seq.shape
        pos_ids = (
            torch.arange(seq_len, device=seq.device)
            .unsqueeze(1)
            .expand(seq_len, batch_size)
        )  # (seq_len, batch)

        embedded = self.embedding(seq)  # ( batch,seq_len, hidden)

        embedded = embedded + self.position_embedding(embedded)

        embedded = embedded.permute(1, 0, 2)  # (seq_len, batch, hidden)
        encoded = self.transformer_encoder(
            embedded, src_key_padding_mask=(seq == self.pad_token)
        )  # (batch, seq_len,  hidden)
        encoded = F.normalize(encoded, dim=-1)
        # encoded = self.norm1(encoded)  # (seq_len, batch, hidden)

        mean = self.fc_mean(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.vae_reparameterize(mean, logvar)  # (batch, seq, N)
        evaluation = self.evaluator(z.mean(dim=0))
        return z, mean, logvar, evaluation, encoded


class TransDecoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        dropout,
        num_layers,
        latent_dim,
        tokenizer,
        max_chunk_len,
        max_chunk_num,
        args,
    ):
        """ 初始化 Transformer 解码器模块，用于基于潜在表示进行序列生成/重建。该模块包含可学习位置嵌入、Transformer 解码堆叠、特征投影与池化、以及按任务自适应的输出分类头。

        Args: vocab_size (int): 词表大小，用于生成器输出的类别数量（特定任务下也用于最终分类头）。 hidden_size (int): 解码器隐藏维度（d_model），同时决定多头注意力中头数为 hidden_size/64。 dropout (float): Transformer 解码层中的 dropout 概率。 num_layers (int): Transformer 解码器堆叠层数。 latent_dim (int): 输入潜在特征维度，将被投影到解码器隐藏维度。 tokenizer: 分词器实例，用于标记映射（内部使用）。 max_chunk_len (int): 单个chunk的最大长度，用于构造位置嵌入张量的长度维度。 max_chunk_num (int): chunk数量上限，用于构造位置嵌入张量的序列维度与chunk预测头的输出维度。 args: 运行参数对象；当 args.task_name == "ap_omentum_ovary" 时，输出分类头维度为 vocab_size，否则为 100。

        Returns: None """
        super(TransDecoder, self).__init__()
        self.tokenizer = tokenizer
        self.eos = 4
        self.bos = 1
        self.pad_token = 0
        self.adaptive_pooling = nn.AdaptiveAvgPool1d(20)
        self.pos_embedding = nn.Parameter(
            torch.randn(max_chunk_num, max_chunk_len, hidden_size)
        )
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=int(hidden_size / 64),
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            self.decoder_layer, num_layers=num_layers
        )
        self.sincos_encode = self.sincos_encoding(1000, hidden_size)
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        self.chunk_pre = nn.Sequential(
            nn.Linear(64 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, max_chunk_num + 1),
        )
        self.pool = nn.AdaptiveAvgPool1d(64)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)

        self.chunk_max = (128,)
        # self.subfeat_len_max = max,
        self.subfeat_dim = (128,)
        self.decoder_hidden_size = hidden_size
        self.feat_proj = nn.Linear(latent_dim, self.decoder_hidden_size)
        if args is not None and args.task_name == "ap_omentum_ovary":
            self.out_cls = nn.Linear(self.decoder_hidden_size, vocab_size)
        else:
            self.out_cls = nn.Linear(self.decoder_hidden_size, 100)
        # self.register_buffer('sincos_encoding', self.sincos_encoding(max_chunk_len, hidden_size))

    def get_eos_embed(self):
        return self.tokenizer(torch.tensor([self.eos])).to(self.tokenizer.weight.device)

    def get_bos_embed(self, batch_size):
        # return self.tokenizer(torch.tensor([self.eos])).to(self.tokenizer.weight.device)
        bos_idx = torch.full(
            (batch_size, 1), self.bos, device=self.tokenizer.weight.device
        )
        return self.tokenizer(bos_idx)

    def sincos_encoding(self, max_seq_len, hidden_size):
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size)
        )

        pe = torch.zeros(max_seq_len, hidden_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def forward(self, z, target_seq):

        # target_seq : (batch_size_chunk,  subfeat_len_max]])
        # padding_mask.value [Bchunk, subfeat_lenth]

        device = z.device
        # target_seq_index= target_seq
        # batch_size*chunk,subfeat_len_max = target_seq.size()
        # batch_size, _,_ = z.size()
        ##############################################
        z = z.permute(1, 2, 0)
        z = self.adaptive_pooling(z)  # seq, B, C
        z = z.permute(2, 0, 1)
        ##############################################
        memory = self.feat_proj(z)  # [B,500,dim]  -> [B*chunk_num,500,dim]
        # memory = self.norm1(memory)
        target_seq_chunk = []
        memory_expanded_list = []
        tgt_key_padding_mask = []
        for i, chunk in enumerate(target_seq):

            cleaned_chunk = []
            for row in chunk:
                filtered_row = row[row != 4]
                if len(filtered_row) > 0:
                    cleaned_chunk.append(filtered_row)
            chunk = torch.stack(cleaned_chunk, dim=0)
            tgt_key_padding_mask.extend(chunk == self.pad_token)
            chunk_embed = self.tokenizer(chunk.to(device))  # [chunk, seq, subfeat]
            chunk_embed = (
                chunk_embed
                + self.pos_embedding[
                    : chunk_embed.size(0), : chunk_embed.size(1), : chunk_embed.size(2)
                ]
            )
            target_seq_chunk.extend(chunk_embed)
            memory_expanded_list.extend(
                memory[:, i, :].unsqueeze(0).repeat_interleave(chunk.size(0), 0)
            )
        tgt_key_padding_mask = torch.stack(tgt_key_padding_mask, dim=0).to(device)
        target_seq_ = torch.stack(target_seq_chunk, dim=0)
        memory_expanded = torch.stack(memory_expanded_list, dim=0)

        z_pool = self.pool(z.permute(1, 2, 0)).permute(2, 0, 1)  # (b, 10, d)
        z_flat = rearrange(z_pool, "s b d -> b (s d)")  # (b, 10*d)
        chunk_num_logits = self.chunk_pre(z_flat)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            target_seq_.size(1)
        ).to(z.device)

        output = self.transformer_decoder(
            tgt=target_seq_,
            memory=memory_expanded,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            # tgt_is_causal=True
        )  # [(B*chunk_max), subfeat_len_max, subfeat_dim]

        output_logits = self.out_cls(output)  # [B*chunk, subfeat_len, max_feat_index]

        return output_logits, chunk_num_logits

    def generate(self, z, max_len=64, chunk_num=None):

        device = z.device
        z_pool = self.pool(z.permute(1, 2, 0)).permute(2, 0, 1)  # (b, 10, d)
        z_flat = rearrange(z_pool, "s b d -> b (s d)")  # (b, 10*d)
        if chunk_num is None:
            chunk_num_logits = self.chunk_pre(z_flat)
            chunk_num = chunk_num_logits.argmax(dim=-1).clamp(min=1).max().item()
        z = z.permute(1, 2, 0)
        z = self.adaptive_pooling(z)
        z = z.permute(2, 0, 1)

        memory = self.feat_proj(z)
        generated_seq = []
        tgt = self.get_bos_embed(chunk_num)
        tgt = tgt + self.pos_embedding[: tgt.size(0), : tgt.size(1), : tgt.size(2)]
        memory = memory.repeat_interleave(chunk_num, dim=1).permute(1, 0, 2)
        # tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(device)

        for t in range(1, max_len):
            output = self.transformer_decoder(
                tgt=tgt,
                memory=memory,
                # tgt_mask=tgt_mask
                # tgt_key_padding_mask=tgt_key_padding_mask
            )
            output = self.out_cls(output[:, -1, :])
            # output[:, 1] =
            next_token = output.argmax(dim=-1).unsqueeze(1)
            tgt = torch.cat([tgt, self.tokenizer(next_token)], dim=1)
            generated_seq.append(next_token.squeeze(1).cpu().numpy())
        generated_seq = torch.tensor(generated_seq).T

        return generated_seq


class TabEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, num_layers):
        super(TabEncoder, self).__init__()

        self.layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        current_size = input_size
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(current_size, hidden_size))
            self.norm_layers.append(nn.LayerNorm(hidden_size))
            current_size = hidden_size

        self.final_layer = nn.Linear(hidden_size, hidden_size)
        self.final_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        for layer, norm in zip(self.layers, self.norm_layers):
            residual = x
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)
            if residual.shape == x.shape:  # 维度匹配时才加残差
                x = x + residual
            x = norm(x)

        x = self.final_layer(x)
        x = self.final_norm(x)
        # x = F.normalize(x, dim=-1)  # 需要单位长度归一化时打开

        return x


class TransformerVAE(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size=512,
        dropout=0.1,
        num_layers=2,
        latent_dim=128,
        max_chunk_len=50,
        max_chunk_num=128,
        tab_len=1000,
        args=None,
    ):
        super(TransformerVAE, self).__init__()
        self.tokenizer = nn.Embedding(vocab_size, hidden_size)
        self.encoder = TransEncoder(
            vocab_size, hidden_size, dropout, num_layers, latent_dim, self.tokenizer
        )
        self.decoder = TransDecoder(
            vocab_size,
            hidden_size,
            dropout,
            num_layers,
            latent_dim,
            self.tokenizer,
            max_chunk_len,
            max_chunk_num,
            args,
        )
        self.tab_encoder = TabEncoder(tab_len, hidden_size, dropout, num_layers)

    def forward(self, seq, tab=None, chunk=None):
        if tab is not None:
            tab_emb = self.tab_encoder(tab)
            z, mean, logvar, evaluation, seq_emb = self.encoder(seq)
            logits, chunk_num = self.decoder(z, chunk)
            return (
                logits,
                mean,
                logvar,
                evaluation,
                seq_emb.mean(dim=0),
                tab_emb,
                chunk_num,
            )
        else:
            z, mean, logvar, evaluation, seq_emb = self.encoder(seq)
            logits, chunk_num = self.decoder(z, chunk)
            return logits, mean, logvar, evaluation, seq_emb.mean(dim=0), chunk_num

    def encode(self, seq, tab=None):
        z, mean, logvar, evaluation, seq_emb = self.encoder(seq)
        if tab is not None:
            tab_emb = self.tab_encoder(tab)
            return z, mean, logvar, evaluation, seq_emb, tab_emb
        else:
            return z, mean, logvar, evaluation, seq_emb

    def decode(self, z, chunk=None):
        logits, chunk_num = self.decoder(z, chunk)
        return logits, chunk_num

    def evaluate(self, z):
        z = z.mean(dim=0)
        evaluation = self.encoder.evaluator(z)
        return evaluation

    def generate(self, z, chunk_num=None):

        return self.decoder.generate(z, chunk_num=chunk_num)

    def generate_test(self, seq):
        z, mean, logvar, evaluation, seq_emb = self.encoder(seq)

        return self.decoder.generate(z)

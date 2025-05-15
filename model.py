import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, reduce
from transformers import BertModel

class SinCosPositionEmbedding(nn.Module):
    def __init__(self, hidden_size, max_seq_len=512):
        super(SinCosPositionEmbedding, self).__init__()
        self.hidden_size = hidden_size

        position = torch.arange(max_seq_len).unsqueeze(1)  
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * (-math.log(10000.0) / hidden_size))

        pe = torch.zeros(max_seq_len, hidden_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        x: Tensor (seq_len, batch_size, hidden_size)
        """
        seq_len = x.size(0)
        return self.pe[:seq_len, :].unsqueeze(1).to(x.device)  # (seq_len, 1, hidden_size)

class TransEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout, num_layers, latent_dim, tokenizer, max_seq_len=512, pretrained_model_name='bert-base-uncased'):
        super(TransEncoder, self).__init__()
        self.pad_token = 0
        self.embedding = tokenizer
        self.position_embedding = SinCosPositionEmbedding(hidden_size, max_seq_len)
        self.adaptive_pooling = nn.AdaptiveAvgPool1d(20)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=int(hidden_size/64), dropout=dropout,dim_feedforward=hidden_size*4)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.pad_token = 0
        self.norm1= nn.LayerNorm(hidden_size)
        # VAE layers
        self.fc_mean = nn.Linear(hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)

        # Evaluator MLP
        self.evaluator = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def vae_reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    


    def forward(self, seq):
        seq_len, batch_size = seq.shape
        pos_ids = torch.arange(seq_len, device=seq.device).unsqueeze(1).expand(seq_len, batch_size)  # (seq_len, batch)
        

        embedded = self.embedding(seq)  # ( batch,seq_len, hidden)

        embedded = embedded + self.position_embedding(embedded) 


        embedded = embedded.permute(1, 0, 2)  # (seq_len, batch, hidden)
        encoded = self.transformer_encoder(embedded, src_key_padding_mask=(seq == self.pad_token))  # (batch, seq_len,  hidden)
        encoded = F.normalize(encoded, dim=-1)
        # encoded = self.norm1(encoded)  # (seq_len, batch, hidden)

        mean = self.fc_mean(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.vae_reparameterize(mean, logvar) # (batch, seq, N)
        evaluation = self.evaluator(z.mean(dim=0))
        return z, mean, logvar, evaluation, encoded

class TransDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout, num_layers, latent_dim, tokenizer, max_chunk_len, max_chunk_num,args):
        super(TransDecoder, self).__init__()
        self.tokenizer = tokenizer
        self.eos = 4
        self.bos = 1
        self.pad_token = 0
        self.adaptive_pooling = nn.AdaptiveAvgPool1d(20)
        self.pos_embedding = nn.Parameter(torch.randn(max_chunk_num, max_chunk_len, hidden_size))
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=int(hidden_size/64), dropout=dropout,batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.sincos_encode = self.sincos_encoding(1000, hidden_size)
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        self.chunk_pre = nn.Sequential(
            nn.Linear(64*hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, max_chunk_num + 1)
        )
        self.pool = nn.AdaptiveAvgPool1d(64)
        self.norm1= nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        
        self.chunk_max = 128,
        # self.subfeat_len_max = max,
        self.subfeat_dim = 128,
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
        bos_idx = torch.full((batch_size,1), self.bos, device=self.tokenizer.weight.device)
        return self.tokenizer(bos_idx)

    
    def sincos_encoding(self, max_seq_len, hidden_size):
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        
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
        memory = self.feat_proj(z) # [B,500,dim]  -> [B*chunk_num,500,dim]
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
            chunk_embed = chunk_embed + self.pos_embedding[:chunk_embed.size(0), :chunk_embed.size(1), :chunk_embed.size(2)]
            target_seq_chunk.extend(chunk_embed)
            memory_expanded_list.extend(memory[:,i,:].unsqueeze(0).repeat_interleave(chunk.size(0), 0))
        tgt_key_padding_mask = torch.stack(tgt_key_padding_mask, dim=0).to(device)
        target_seq_ = torch.stack(target_seq_chunk, dim=0)
        memory_expanded = torch.stack(memory_expanded_list, dim=0)

        z_pool = self.pool(z.permute(1,2,0)).permute(2,0,1)  # (b, 10, d)
        z_flat = rearrange(z_pool, 's b d -> b (s d)')  # (b, 10*d)
        chunk_num_logits = self.chunk_pre(z_flat)


        tgt_mask = nn.Transformer.generate_square_subsequent_mask(target_seq_.size(1)).to(z.device)


        output = self.transformer_decoder(
            tgt=target_seq_,
            memory=memory_expanded,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            # tgt_is_causal=True
        )  # [(B*chunk_max), subfeat_len_max, subfeat_dim]

        output_logits = self.out_cls(output) # [B*chunk, subfeat_len, max_feat_index] 

        return output_logits, chunk_num_logits
    
    def generate(self, z, max_len = 64, chunk_num = None):
        
        device = z.device
        z_pool = self.pool(z.permute(1,2,0)).permute(2,0,1)  # (b, 10, d)
        z_flat = rearrange(z_pool, 's b d -> b (s d)')  # (b, 10*d)
        if chunk_num is None:
            chunk_num_logits = self.chunk_pre(z_flat)
            chunk_num = chunk_num_logits.argmax(dim=-1).clamp(min=1).max().item()
        z = z.permute(1, 2, 0)  
        z = self.adaptive_pooling(z) 
        z = z.permute(2, 0, 1)  

        memory = self.feat_proj(z)
        generated_seq = []
        tgt = self.get_bos_embed(chunk_num)
        tgt = tgt + self.pos_embedding[:tgt.size(0), :tgt.size(1), :tgt.size(2)]
        memory = memory.repeat_interleave(chunk_num, dim=1).permute(1, 0, 2)
        # tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(device)

        for t in range(1, max_len):
            output = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            # tgt_mask=tgt_mask
            # tgt_key_padding_mask=tgt_key_padding_mask
            )  
            output = self.out_cls(output[:,-1, :])
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
    def __init__(self, vocab_size, hidden_size=512, dropout=0.1, num_layers=2, latent_dim=128, max_chunk_len=50, max_chunk_num=128, tab_len=1000, args = None):
        super(TransformerVAE, self).__init__()
        self.tokenizer = nn.Embedding(vocab_size, hidden_size)
        self.encoder = TransEncoder(vocab_size, hidden_size, dropout, num_layers, latent_dim, self.tokenizer)
        self.decoder = TransDecoder(vocab_size, hidden_size, dropout, num_layers, latent_dim, self.tokenizer, max_chunk_len, max_chunk_num, args)
        self.tab_encoder = TabEncoder(tab_len, hidden_size, dropout, num_layers)
        
    def forward(self, seq, tab = None, chunk = None):
        if tab is not None:
            tab_emb = self.tab_encoder(tab)
            z, mean, logvar, evaluation, seq_emb = self.encoder(seq)
            logits, chunk_num = self.decoder(z, chunk)
            return logits, mean, logvar, evaluation, seq_emb.mean(dim=0), tab_emb, chunk_num 
        else:
            z, mean, logvar, evaluation, seq_emb = self.encoder(seq)
            logits, chunk_num = self.decoder(z, chunk)
            return logits, mean, logvar, evaluation, seq_emb.mean(dim=0), chunk_num
        
    def encode(self, seq, tab = None):
        z, mean, logvar, evaluation, seq_emb = self.encoder(seq)
        if tab is not None:
            tab_emb = self.tab_encoder(tab)
            return z, mean, logvar, evaluation, seq_emb, tab_emb
        else:
            return z, mean, logvar, evaluation, seq_emb
        
    def decode(self, z, chunk = None):
        logits, chunk_num = self.decoder(z, chunk)
        return logits, chunk_num
    
    def evaluate(self, z):
        z = z.mean(dim=0)
        evaluation = self.encoder.evaluator(z)
        return evaluation
    
    def generate(self, z, chunk_num = None):

        return self.decoder.generate(z, chunk_num=chunk_num)
    
    def generate_test(self,seq):
        z, mean, logvar, evaluation, seq_emb = self.encoder(seq)
        
        return self.decoder.generate(z)


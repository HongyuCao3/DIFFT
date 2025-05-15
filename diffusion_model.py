from transformers import AutoConfig
# from transformers import BertEncoder
from transformers.models.bert.modeling_bert import BertEncoder, BertModel
import torch
import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from model import TabEncoder
from diffusion_model_utils import *
from transformers import get_cosine_schedule_with_warmup

def WarmUpCosine_Scheduler(optimizer, max_lr, min_lr, max_epoch, warmup_steps=40):

    t = warmup_steps # warmup steps
    T = max_epoch # max epoch
    n_t = max_lr - min_lr # max - min lr
    
    lambda1 = lambda epoch: (0.9*epoch / t+0.1) if epoch < t else  0.1  if n_t * (1+math.cos(math.pi*(epoch - t)/(T-t)))<0.1 else n_t * (1+math.cos(math.pi*(epoch - t)/(T-t)))

    return th.optim.lr_scheduler.LambdaLR(optimizer, lambda1)

def warmup_stable_decay(optimizer, max_steps, warmup_ratio=0.05, decay_ratio=0.2):

    n_warmup = max_steps * warmup_ratio
    n_decay = max_steps * decay_ratio  
    def inner(step):
        if step < n_warmup:
            return step / n_warmup  
        elif step <= max_steps - n_decay:
            return 1  
        else:
            t = (step - (max_steps - n_decay)) / n_decay  # (0 -> 1)
            t = 1 - t**0.5  # (1 -> 0)
        return t
    scheduler = th.optim.lr_scheduler.LambdaLR(optimizer, inner)
    return scheduler

def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


class TransformerDM(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """

    def __init__(self, in_channels, t_channels, hidden_channels, context_channels=128,
                 depth=1, n_heads=8, dropout=0.,tab_len=267, out_channels=None):
        super(TransformerDM, self).__init__()
        self.in_channels = in_channels
        d_head = hidden_channels // n_heads
        self.t_channels = t_channels
        
        self.cond_tab_encoder = TabEncoder(tab_len, hidden_channels, dropout, 4)

        self.proj_in = nn.Linear(in_channels, hidden_channels, bias=False)
        self.cond_proj_in = nn.Linear(context_channels, context_channels, bias=False)

        self.transformer_blocks = nn.ModuleList(
            # [nn.TransformerDecoderLayer(d_model=hidden_channels, nhead=8, dropout=dropout)
            #     for _ in range(depth)]
            [BasicTransformerBlock(hidden_channels, n_heads, d_head, dropout=dropout, context_dim=context_channels)
                for _ in range(depth)]
        )

        self.norm = nn.LayerNorm(hidden_channels)

        if out_channels is None:
            self.proj_out = zero_module(nn.Linear(hidden_channels, in_channels, bias=False))
        else:
            self.num_cls = out_channels
            self.proj_out = zero_module(nn.Linear(hidden_channels, out_channels, bias=False))

        self.map_noise = PositionalEmbedding(t_channels)

        self.map_layer0 = nn.Linear(in_features=t_channels, out_features=hidden_channels)
        self.map_layer1 = nn.Linear(in_features=hidden_channels, out_features=hidden_channels)

    def forward(self, x, t, cond=None):
        cond = self.cond_tab_encoder(cond)

        t_emb = self.map_noise(t)[:, None]
        t_emb = F.silu(self.map_layer0(t_emb))
        t_emb = F.silu(self.map_layer1(t_emb))

        x = self.proj_in(x)
        cond = self.cond_proj_in(cond)

        for block in self.transformer_blocks:
            x = block(x, t_emb, context=cond)
        
        x = self.norm(x)

        x = self.proj_out(x)
        return x

# 测试warmup cosine scheduler
if __name__ == "__main__":
    class Args:
        def __init__(self):
            self.max_train_steps = 1000

    args = Args()
    model = nn.Linear(10, 10)
    optimizer = th.optim.Adam(model.parameters(), lr=0.0001)
    # scheduler = WarmUpCosine_Scheduler(optimizer, 0.005, 0.00001, 1000, warmup_steps=50)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=20, num_training_steps=100)

    for step in range(100):
        optimizer.step()
        scheduler.step()
        print(f"Step {step + 1}: Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
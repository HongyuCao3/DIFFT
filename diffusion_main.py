import argparse
import os
import sys
import pandas

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append("./")
from utils.datacollection.logger import info, error
import warnings
from torchinfo import summary

warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans
import random
import sys
from typing import List
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch import Tensor
from model import *
from dataset import *

# from feature_env import base_path
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn.functional as F
from diffusion_model import *
from pipeline_diffusion import DiffTipeline
from diffusers import DDPMScheduler
import time

parser = argparse.ArgumentParser()
parser.add_argument(
    "--task_name",
    type=str,
    choices=[
        "airfoil",
        "amazon_employee",
        "ap_omentum_ovary",
        "german_credit",
        "higgs",
        "housing_boston",
        "ionosphere",
        "lymphography",
        "messidor_features",
        "openml_620",
        "pima_indian",
        "spam_base",
        "spectf",
        "svmguide3",
        "uci_credit_card",
        "wine_red",
        "wine_white",
        "openml_586",
        "openml_589",
        "openml_607",
        "openml_616",
        "openml_618",
        "mice_protein",
        "openml_637",
    ],
    default="spectf",
)
parser.add_argument("--exp_name", type=str, default="default")
parser.add_argument("--hidden_size", type=int, default=512)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--latent_dim", type=int, default=512)
parser.add_argument("--dropout", type=float, default=0.05)
parser.add_argument("--lr", type=float, default=0.0002)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--load_epoch", type=int, default=0)
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--pre_epochs", type=int, default=601)
parser.add_argument("--add_origin", type=bool, default=True)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--shuffle_time", type=int, default=2)
parser.add_argument("--num_worker", type=int, default=16)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--accumulation_steps", type=int, default=32)
parser.add_argument("--infer_size", type=int, default=300)
parser.add_argument("--tab_len", type=int, default=1000)

parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument("--max_seq_len", type=int, default=512)
parser.add_argument("--use_reward", type=int, default=100)
parser.add_argument("--diff_hidden_size", type=int, default=512)
parser.add_argument("--diff_num_layers", type=int, default=8)
parser.add_argument("--diff_num_step", type=int, default=20)
parser.add_argument(
    "--vae_load_path",
    type=str,
    default="data/history/spectf_ldm/test_v2_256_100Rbest/model/vae.pt",
)
parser.add_argument(
    "--ldm_load_path",
    type=str,
    default="data/history/spectf_ldm/test_v2_256_100Rbest/model/ldm.pt",
)
parser.add_argument("--prediction_type", type=str, default="epsilon")
parser.add_argument("--snr_gamma", type=float, default=0)
parser.add_argument("--loss_type", type=str, default="mse")
parser.add_argument("--test", action="store_true", default=False)
parser.add_argument("--guidance_scale", type=float, default=0)
parser.add_argument("--infer_func", type=str, default="cls")  # reg cls
parser.add_argument("--infer_method", type=str, default="RF")  # svc lr dtc knc

args = parser.parse_args()


def pre_training(ldm, vae, training_data, validation_data, infer_data, args):
    """使用变分自编码器（VAE）提供的潜空间表示，预训练条件扩散模型（LDM）

    本函数以 VAE 的编码结果作为扩散模型的训练目标，通过 DDPM 噪声调度器对潜变量添加噪声， 根据指定的预测类型与损失函数优化 LDM。支持断点续训、余弦退火学习率调度、梯度累积、 验证评估与模型检查点保存。

    Args:
        ldm (torch.nn.Module): 待训练的潜空间扩散模型，需实现前向接口 (noisy_z, timesteps, cond) -> noise_pred。
        vae (torch.nn.Module): 仅用于编码的 VAE 模型，需实现 encode(seq, tab) -> (z0, mean, logvar, evaluation, seq_emb, tab_emb)。
        training_data (Iterable): 训练数据迭代器/数据加载器，batch 至少包含键 "seqs", "tabs", "performances", "chunk_seqs"。
        validation_data (Iterable): 验证数据迭代器/数据加载器，用于周期性验证。
        infer_data (Iterable): 推理评估数据集，用于训练前/过程中的推理指标统计。
        args (argparse.Namespace): 运行配置，关键字段包括但不限于：
            - gpu (int): 使用的 CUDA 设备索引。
            - lr (float): 初始学习率。
            - pre_epochs (int): 预训练总轮数。
            - resume (bool): 是否从 checkpoint_last.pth 恢复训练。
            - model_path (str): 模型与检查点保存目录。
            - task_path (str): 日志保存目录。
            - prediction_type (str): 目标类型，支持 "epsilon"、"v_prediction"、"sample"。
            - loss_type (str): 损失类型，支持 "l1"、"mse"/"l2"。
            - guidance_scale (float): 条件引导强度，>0 时以约 0.1 概率屏蔽条件。
            - accumulation_steps (int): 梯度累积步数。
            - batch_size (int): 记录/归一化时使用的批大小。
            - snr_gamma (float): SNR 加权超参，0 表示不使用 SNR 加权。

    Returns: torch.nn.Module: 传入的 VAE 实例（未被训练，仅用于编码，原样返回）。

    Raises: ValueError: 当 args.prediction_type 不在 {"epsilon", "v_prediction", "sample"} 中， 或 args.loss_type 不在 {"l1", "mse", "l2"} 中时抛出。

    Notes: - 优化器：Adam；学习率调度：带预热的余弦调度（预热约为 pre_epochs 的 10%）。 - 噪声调度器：DDPMScheduler（scaled_linear β 调度，1000 步）。 - 每轮保存 ldm_last.pt 与 checkpoint_last.pth；每 10 轮额外保存 ldm_{epoch}.pt； 当验证集最优时保存 ldm_best_val.pt。 - 该函数会原地更新 ldm 的参数，并在训练/验证过程中记录日志与写入权重文件。
    """
    device = int(args.gpu)
    ldm.train()
    vae.eval()
    criterion = nn.MSELoss()
    start_epoch = 0
    best_val = 9999
    best_acc = 0
    val_loss = 9999
    infer_acc = 0
    optimizer = torch.optim.Adam(ldm.parameters(), lr=args.lr)
    for group in optimizer.param_groups:
        group["initial_lr"] = args.lr
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.pre_epochs // 10,
        num_training_steps=args.pre_epochs + 100,
        last_epoch=start_epoch - 1,
    )
    if args.resume:
        ckpt = torch.load(
            os.path.join(args.model_path, f"checkpoint_last.pth"),
            map_location=torch.device("cuda"),
        )
        start_epoch = ckpt["epoch"] + 1
        ldm.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        print_log(
            f"Resuming {start_epoch - 1} epoch from {os.path.join(args.model_path, f'checkpoint_last.pt')}",
            args.task_path,
        )
        best_val = ckpt["best"]["best_val"]
        best_acc = ckpt["best"]["best_acc"]

    print_log(
        f"Best validation loss: {best_val}, Best accuracy: {best_acc} from checkpoint",
        args.task_path,
    )
    print_log(
        f"Start pre-training from epoch {start_epoch} to {args.pre_epochs}, start lr is {scheduler.get_last_lr()}",
        args.task_path,
    )

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        clip_sample=False,
        beta_schedule="scaled_linear",
        beta_start=0.00085,
        beta_end=0.012,
        prediction_type=args.prediction_type,
    )
    # val_loss = valid(vae, ldm, validation_data, device, args)
    infer_acc = infer(vae, ldm, infer_data, device, args)

    total_time = 0
    optimizer.zero_grad()
    for epoch in range(start_epoch, args.pre_epochs):
        cost_time = 0

        train_loss = 0
        optimizer.zero_grad()
        for i, batch in enumerate(training_data):
            seq = batch["seqs"].to(device)
            tab = batch["tabs"].to(device)
            performance = batch["performances"].to(device)
            chunk = batch["chunk_seqs"]  # .to(device)

            with torch.no_grad():
                z0, mean, logvar, evaluation, seq_emb, tab_emb = vae.encode(seq, tab)

            start_time = time.time()
            # z0: seqlen,B_64,C_128  tab: B_64,C_128
            z0 = z0.permute(1, 0, 2).contiguous()

            if random.random() < 0.1 and args.guidance_scale > 0:
                tab = torch.zeros_like(tab).float().to(device)

            cond = tab.unsqueeze(1)
            # get noise
            noise = torch.randn_like(z0).to(device)
            bs = z0.shape[0]

            # get timestep
            timesteps = torch.randint(
                0,
                noise_scheduler.num_train_timesteps,
                (bs,),
                device=device,
            ).long()
            noisy_z = noise_scheduler.add_noise(z0, noise, timesteps)
            noise_pred = ldm(noisy_z, timesteps, cond=cond)

            # get target

            if args.prediction_type == "epsilon":
                target = noise
            elif args.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(z0, noise, timesteps)
            elif args.prediction_type == "sample":
                target = z0
            else:
                raise ValueError(
                    f"Prediction Type: {args.prediction_type} not supported."
                )

            if args.snr_gamma == 0:
                if args.loss_type == "l1":
                    loss = F.l1_loss(noise_pred, target, reduction="mean")
                elif args.loss_type in ["mse", "l2"]:
                    loss = F.mse_loss(noise_pred, target, reduction="mean")
                else:
                    raise ValueError(
                        f"Loss Type: {args.loss_type.loss_type} not supported."
                    )
            else:
                snr = compute_snr(noise_scheduler, timesteps)
                mse_loss_weights = torch.stack(
                    [snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1
                ).min(dim=1)[0]
                if args.prediction_type == "epsilon":
                    mse_loss_weights = mse_loss_weights / snr
                elif args.prediction_type == "v_prediction":
                    mse_loss_weights = mse_loss_weights / (snr + 1)

                if args.loss_type == "l1":
                    loss = F.l1_loss(noise_pred, target, reduction="none")
                elif args.loss_type in ["mse", "l2"]:
                    loss = F.mse_loss(noise_pred, target, reduction="none")
                else:
                    raise ValueError(f"Loss Type: {args.loss_type} not supported.")
                loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                loss = loss.mean()
            # get loss
            loss.backward()
            cost_time += time.time() - start_time
            if (i + 1) % args.accumulation_steps == 0 or (i + 1) == len(training_data):
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item()
            if i % (args.accumulation_steps * 5) == 0:
                print_log(
                    f"Training Epoch [{epoch}] Batch [{i}/{len(training_data)}] Loss: [{(loss.item() / args.batch_size):.4f}] LR: [{optimizer.param_groups[0]['lr']:.6f}]",
                    args.task_path,
                )
        scheduler.step()

        print_log(
            f"Training Epoch [{epoch}] Loss: [{(train_loss / (len(training_data) * args.batch_size)):.4f}] LR: [{optimizer.param_groups[0]['lr']:.6f}] Time: [{cost_time:.4f}], save in {os.path.join(args.model_path, f'ldm_last.pt')}",
            args.task_path,
        )
        torch.save(ldm.state_dict(), os.path.join(args.model_path, f"ldm_last.pt"))
        torch.save(
            {
                "epoch": epoch,
                "model": ldm.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best": {"best_val": best_val, "best_acc": best_acc},
            },
            os.path.join(args.model_path, f"checkpoint_last.pth"),
        )
        if epoch % 10 == 0 and epoch != 0:
            val_loss = valid(vae, ldm, validation_data, device, args, epoch)
            torch.save(
                ldm.state_dict(), os.path.join(args.model_path, f"ldm_{epoch}.pt")
            )

        if val_loss < best_val:
            best_val = val_loss
            print_log(
                f"Training Epoch {epoch} get best val_loss {best_val},\
                saving {os.path.join(args.model_path, f'ldm_best_val.pt')}",
                args.task_path,
            )
            torch.save(
                ldm.state_dict(), os.path.join(args.model_path, f"ldm_best_val.pt")
            )
    print_log(
        f"Total training time: {total_time:.2f} seconds, {(total_time/epoch):.2f} seconds per epoch",
        args.task_path,
    )
    return vae


def valid(vae, ldm, validation_data, device, args, epoch=0):
    """在验证集上评估扩散模型的重构误差

    该函数构建基于 DDPM 调度器的采样管线，使用给定的 VAE 将输入序列与条件编码为潜变量 z0， 随后通过扩散管线生成预测潜变量 z_predict，并以均方误差（MSE）衡量二者差异。遍历整个验证集后， 返回批量归一化的平均损失。函数会记录验证开始与最终损失日志，并将模型置于评估模式与无梯度上下文中。 在一定概率下（10% 且 guidance_scale > 0）会将条件 tab 置零以进行无条件/弱条件评估。

    Args:
        vae (torch.nn.Module): 变分自编码器模型，需实现 encode(seq, tab) 接口并返回潜变量及相关统计量。
        ldm (torch.nn.Module): 扩散/去噪网络（例如 UNet），由采样管线在各时间步调用进行预测。
        validation_data (Iterable[Dict[str, torch.Tensor]] | torch.utils.data.DataLoader): 验证数据迭代器， 其批数据需包含键 "seqs"、"tabs"、"performances"、"chunk_seqs" 等，其中 "seqs" 与 "tabs" 将被用于编码。
        device (torch.device | str): 运行设备（例如 "cuda" 或 "cpu"）。
        args (argparse.Namespace): 运行配置，需包含以下字段：
            - task_path (str): 日志输出路径。
            - use_reward (bool): 是否在管线中启用目标引导/奖励。
            - guidance_scale (float): 引导系数，>0 时启用条件引导。
            - diff_num_step (int): 采样步数。
            - batch_size (int): 验证批大小，用于归一化平均损失。
        epoch (int, optional): 当前验证轮次编号，仅用于日志记录。默认为 0。

    Returns: float: 验证集的平均 MSE 损失（按批大小与迭代次数归一化）。

    Raises: RuntimeError: 当模型前向或张量设备、形状不匹配导致底层计算失败时可能抛出。"""
    print_log(f"Validation Epoch [{epoch}] Start", args.task_path)
    pipeline = DiffTipeline(
        vae,
        DDPMScheduler(
            num_train_timesteps=1000,
            clip_sample=False,
            beta_start=0.00085,
            beta_end=0.012,
            steps_offset=1,
            rescale_betas_zero_snr=True,
            beta_schedule="scaled_linear",
        ),
        target_guidance=args.use_reward,
    )
    generator = torch.Generator(device="cuda").manual_seed(0)
    vae.eval()
    ldm.eval()
    criterion = nn.MSELoss()
    loss = 0

    with torch.no_grad():
        for i, batch in enumerate(validation_data):
            seq = batch["seqs"].to(device)
            tab = batch["tabs"].to(device)
            performance = batch["performances"].to(device)
            chunk = batch["chunk_seqs"]  # .to(device)

            with torch.no_grad():
                z0, mean, logvar, evaluation, seq_emb, tab_emb = vae.encode(seq, tab)

            z0 = z0.permute(1, 0, 2).contiguous()
            if random.random() < 0.1 and args.guidance_scale > 0:
                tab = torch.zeros_like(tab).float().to(device)

            cond = tab.unsqueeze(1)

            # z, mean, logvar, evaluation, seq_emb, tab_emb = vae.encode(seq, tab)
            z_predict, z_list, _ = pipeline(
                ldm,
                z0.shape,
                cond,
                steps=args.diff_num_step,
                generator=generator,
                guidance_scale=args.guidance_scale,
                device=device,
                use_reward=args.use_reward,
            )
            # x = vae.decode(z0)

            loss += criterion(z_predict.float(), z0.float()).item()
    loss = loss / (len(validation_data) * args.batch_size)
    print_log(f"Validation Epoch [{epoch}] Loss: {loss:.4f}", args.task_path)
    return loss


def infer(vae, ldm, data, device, args):
    """使用扩散式生成管线在给定条件下生成候选特征操作序列，对原始特征表进行扩展并在下游任务上评估改进效果。函数会遍历数据批次，通过条件引导的潜空间扩散模型生成序列，利用 VAE 将序列解码为可应用的特征操作，动态构造新的特征列并评估精度提升；若取得更高精度则保存对应的扩展数据表。

    Args:
        vae: 变分自编码器，用于在潜空间与可执行的特征操作序列之间进行编码/解码，需包含 generate 等接口。
        ldm: 潜空间扩散模型（Latent Diffusion Model），作为去噪网络参与扩散采样步骤。
        data: 输入数据与基表的二元组 (loader, df)。其中：
            - loader: 可迭代的数据加载器，单个 batch 为字典，至少包含键：
            - "seqs": 特征操作序列的张量表示，用于条件/对齐；
            - "tabs": 样本级条件（表格特征）张量，用于条件引导；
            - "performances": 性能或指标相关张量（用于潜在的评估/引导）；
            - "chunk_seqs": 片段化的序列信息（实现细节依赖模型）。
            - df: pandas.DataFrame，原始特征表（最后一列为标签列，其他列为特征）。
        device: 计算设备（如 "cuda" 或 torch.device），用于张量与模型推理。
        args: 运行配置，需包含但不限于以下字段：
            - infer_func: 下游评估函数名或标识，用于 downstream_task_new 调用；
            - infer_method: 评估方法或策略标识；
            - max_seq_len: 生成序列的最大步数/长度；
            - latent_dim: 潜向量维度；
            - batch_size: 推理批大小；
            - guidance_scale: 条件引导强度（>0 时启用，部分步骤会以一定概率置零条件以增强鲁棒性）；
            - diff_num_step: 扩散采样步数；
            - use_reward: 是否使用目标/奖励引导；
            - task_path: 任务输出目录，用于日志与结果文件保存。

    Returns: float: 在当前推理过程中达到的最高下游任务精度（或相应指标）。

    Raises: OSError: 当在保存改进后的特征表为 CSV 时，目录不存在或无写权限等导致的文件系统错误。 RuntimeError: 当模型推理或张量设备迁移过程中出现异常时可能抛出。 ValueError: 当输入数据格式不符合预期（如 batch 缺失必要键、data 结构不匹配）时可能抛出。
    """
    data, df = data
    max_acc = downstream_task_new(df, args.infer_func)
    print_log(
        f"Infer Start, Original accuracy:{max_acc} ({args.max_seq_len})", args.task_path
    )
    y = df.iloc[:, -1]
    #################################################################################################
    pipeline = DiffTipeline(
        vae,
        DDPMScheduler(
            num_train_timesteps=1000,
            clip_sample=False,
            beta_start=0.00085,
            beta_end=0.012,
            steps_offset=1,
            rescale_betas_zero_snr=True,
            beta_schedule="scaled_linear",
        ),
        target_guidance=args.use_reward,
    )
    generator = torch.Generator(device="cuda").manual_seed(0)
    shape = (args.batch_size, args.max_seq_len, args.latent_dim)
    #################################################################################################
    # print(y)
    df = df.iloc[:, :-1]
    df.columns = [str(i) for i in range(df.shape[1])]
    vae.eval()
    ldm.eval()
    total_time = 0
    with torch.no_grad():
        # for batch in data:
        for i, batch in enumerate(data):
            seq = batch["seqs"]
            tab = batch["tabs"]
            performance = batch["performances"]
            chunk = batch["chunk_seqs"]
            seq = seq.to(device)
            tab = tab.to(device)
            performance = performance.to(device)
            sample_time = time.time()
            ################################################################################################
            if random.random() < 0.1 and args.guidance_scale > 0:
                tab = torch.zeros_like(tab).float().to(device)
            # condition 待定
            cond = tab.unsqueeze(1)
            # z, mean, logvar, evaluation, seq_emb, tsab_emb = vae.encode(seq, tab)
            if cond.shape[0] != shape[0]:
                shape = (cond.shape[0], args.max_seq_len, args.latent_dim)
                continue
            z_predict, z_list, _ = pipeline(
                ldm,
                shape,
                cond,
                steps=args.diff_num_step,
                generator=generator,
                guidance_scale=args.guidance_scale,
                device=device,
                use_reward=args.use_reward,
            )

            generated_seq = vae.generate(z_predict.permute(1, 0, 2).contiguous())
            ################################################################################################
            sample_time = time.time() - sample_time
            total_time += sample_time

            if i % 50 == 0:
                print_log(
                    f"Infer Batch [{i}/{len(data)}] Sample Time: {sample_time:.4f} seconds",
                    args.task_path,
                )
            new_df = df
            for i in generated_seq:
                try:
                    idx = (i == 4).nonzero(as_tuple=True)[0][0].item()
                    feat = i[:idx].cpu().numpy()
                    new_df[" ".join(show_ops(feat))] = op_post_seq(df, feat)
                except:
                    continue
            new_df = new_df.replace([np.inf, -np.inf], np.nan)
            new_df = new_df.dropna(axis=1)
            new_df = new_df.clip(lower=-1e5, upper=1e5)
            # print('New df', new_df.columns)
            new_acc = downstream_task_new(
                pd.concat([new_df, y], axis=1),
                args.infer_func,
                method=args.infer_method,
            )
            if new_acc > max_acc:
                # print('----------------------------------')
                print_log(
                    f"New accuracy: {new_acc} ({args.max_seq_len})", args.task_path
                )
                # 保存new_df
                new_df.to_csv(
                    os.path.join(
                        args.task_path,
                        f"infer_{args.infer_func}_{args.max_seq_len}.csv",
                    ),
                    index=False,
                )
                # print('----------------------------------')
                max_acc = new_acc
    print_log(
        f"Infer Finished, Total Sample Time: {total_time:.4f} seconds, Single Sample Time: {(total_time/(i.item()*args.batch_size)):.4f} seconds",
        args.task_path,
    )
    return max_acc


def main():
    if not torch.cuda.is_available():
        print_log("No GPU found!")
        sys.exit(1)
    # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpu)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.enabled = True
    cudnn.benchmark = False
    cudnn.deterministic = True
    device = int(args.gpu)
    args.task_path = os.path.join(
        BASE_DIR,
        "data/history",
        f"{args.task_name}_ldm",
        ("test_" if args.test else "") + f"{args.exp_name}",
    )
    args.model_path = os.path.join(
        BASE_DIR,
        "data/history",
        f"{args.task_name}_ldm",
        ("test_" if args.test else "") + f"{args.exp_name}",
        "model",
    )
    os.makedirs(args.task_path, exist_ok=True)
    os.makedirs(args.model_path, exist_ok=True)
    # print args
    if args.task_name.startswith("openml_"):
        args.infer_func = "reg"
    for arg, value in vars(args).items():
        print_log(f"{arg}: {value}", args.task_path)
    # check if the task name start with openml
    dataset = Data_Preprocessing(
        task=args.task_name,
        batch_size=args.batch_size,
        shuffle_time=args.shuffle_time,
        infer_size=args.infer_size,
    )
    training_data = dataset.training_data
    validation_data = dataset.validation_data
    infer_data = (dataset.infer_data, dataset.test)
    max_length = dataset.max_length
    max_chunk_size = dataset.max_chunk_size
    max_chunk_num = dataset.max_chunk_num
    args.tab_len = dataset.tab_len
    print_log(
        f"Max length in training data is:{max_length}, vocab size is: {dataset.vocab_size}, device is {device}",
        args.task_path,
    )
    print_log(f"Model path:{args.model_path}", args.task_path)
    vae = TransformerVAE(
        vocab_size=dataset.vocab_size + 1,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        num_layers=args.num_layers,
        latent_dim=args.latent_dim,
        max_chunk_len=max_chunk_size,
        max_chunk_num=max_chunk_num,
        tab_len=args.tab_len,
        args=args,
    ).to(device)
    vae.load_state_dict(
        torch.load(args.vae_load_path, map_location=torch.device("cuda"))
    )  # load the pre-trained vae model

    ldm = TransformerDM(
        in_channels=args.latent_dim,
        t_channels=256,
        context_channels=args.latent_dim,
        hidden_channels=args.diff_hidden_size,
        depth=args.diff_num_layers,
        dropout=args.dropout,
        tab_len=args.tab_len,
        out_channels=None,
    ).to(device)

    if args.test:
        print_log("Start Infering", args.task_path)
        ldm.load_state_dict(
            torch.load(args.ldm_load_path, map_location=torch.device("cuda"))
        )
        print_log(f"Load ldm model from {args.ldm_load_path}", args.task_path)
        torch.save(ldm.state_dict(), os.path.join(args.model_path, f"ldm.pt"))
        torch.save(vae.state_dict(), os.path.join(args.model_path, f"vae.pt"))
        infer(vae, ldm, infer_data, device, args)
        print_log("Infering Finished", args.task_path)
    else:
        print_log("Start pre-training", args.task_path)
        ldm = pre_training(
            ldm, vae, training_data, validation_data, infer_data, args=args
        )


if __name__ == "__main__":
    main()

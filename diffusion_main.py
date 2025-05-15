import argparse
import os
import sys
import pandas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append('./')
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
import torch.utils
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
parser.add_argument('--task_name', type=str, choices=['airfoil', 'amazon_employee',
                                                        'ap_omentum_ovary', 'german_credit',
                                                        'higgs', 'housing_boston', 'ionosphere',
                                                        'lymphography', 'messidor_features', 'openml_620',
                                                        'pima_indian', 'spam_base', 'spectf', 'svmguide3',
                                                        'uci_credit_card', 'wine_red', 'wine_white', 'openml_586',
                                                        'openml_589', 'openml_607', 'openml_616', 'openml_618','mice_protein',
                                                        'openml_637'], default='spectf')
parser.add_argument('--exp_name', type=str, default='default')
parser.add_argument('--hidden_size', type=int, default=512)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--latent_dim', type=int, default=512)
parser.add_argument('--dropout', type=float, default=0.05)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--load_epoch', type=int, default=0)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--pre_epochs', type=int, default=601)
parser.add_argument('--add_origin', type=bool, default=True)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--shuffle_time', type=int, default=2)
parser.add_argument('--num_worker', type=int, default=16)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--accumulation_steps', type=int, default=32)
parser.add_argument('--infer_size', type=int, default=300)
parser.add_argument('--tab_len', type=int, default=1000)

parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--max_seq_len', type=int, default=512)
parser.add_argument('--use_reward', type=int, default=100)
parser.add_argument('--diff_hidden_size', type=int, default=512)
parser.add_argument('--diff_num_layers', type=int, default=8)
parser.add_argument('--diff_num_step', type=int, default=20)
parser.add_argument('--vae_load_path', type=str, default="data/history/spectf_ldm/test_v2_256_100Rbest/model/vae.pt")
parser.add_argument('--ldm_load_path', type=str, default="data/history/spectf_ldm/test_v2_256_100Rbest/model/ldm.pt")
parser.add_argument('--prediction_type', type=str, default='epsilon')
parser.add_argument('--snr_gamma', type=float, default=0)
parser.add_argument('--loss_type', type=str, default='mse')
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--guidance_scale', type=float, default=0)
parser.add_argument('--infer_func', type=str, default='cls') # reg cls 
parser.add_argument('--infer_method', type=str, default='RF') # svc lr dtc knc 

args = parser.parse_args()

def pre_training(ldm, vae, training_data, validation_data, infer_data, args):
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
    scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=args.pre_epochs//10, \
                num_training_steps=args.pre_epochs + 100, last_epoch=start_epoch-1)
    if args.resume:
        ckpt = torch.load(os.path.join(args.model_path, f'checkpoint_last.pth'), map_location=torch.device("cuda"))
        start_epoch = ckpt['epoch'] + 1
        ldm.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        print_log(f"Resuming {start_epoch - 1} epoch from {os.path.join(args.model_path, f'checkpoint_last.pt')}",args.task_path)
        best_val = ckpt['best']['best_val']
        best_acc = ckpt['best']['best_acc']
        
    print_log(f"Best validation loss: {best_val}, Best accuracy: {best_acc} from checkpoint", args.task_path)
    print_log(f"Start pre-training from epoch {start_epoch} to {args.pre_epochs}, start lr is {scheduler.get_last_lr()}", args.task_path)
    
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, clip_sample=False, beta_schedule = "scaled_linear", 
                                        beta_start = 0.00085, beta_end = 0.012, prediction_type = args.prediction_type)
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
            chunk = batch["chunk_seqs"]# .to(device)
            
            with torch.no_grad():
                z0, mean, logvar, evaluation, seq_emb, tab_emb = vae.encode(seq,tab)
            
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
                raise ValueError(f"Prediction Type: {args.prediction_type} not supported.")
            
            if args.snr_gamma == 0:
                if args.loss_type == "l1":
                    loss = F.l1_loss(noise_pred, target, reduction="mean")
                elif args.loss_type in ["mse", "l2"]:
                    loss = F.mse_loss(noise_pred, target, reduction="mean")
                else:
                    raise ValueError(f"Loss Type: {args.loss_type.loss_type} not supported.")
            else:
                snr = compute_snr(noise_scheduler, timesteps)
                mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
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
                print_log(f"Training Epoch [{epoch}] Batch [{i}/{len(training_data)}] Loss: [{(loss.item() / args.batch_size):.4f}] LR: [{optimizer.param_groups[0]['lr']:.6f}]", args.task_path)
        scheduler.step()  

        print_log(f"Training Epoch [{epoch}] Loss: [{(train_loss / (len(training_data) * args.batch_size)):.4f}] LR: [{optimizer.param_groups[0]['lr']:.6f}] Time: [{cost_time:.4f}], save in {os.path.join(args.model_path, f'ldm_last.pt')}", args.task_path)
        torch.save(ldm.state_dict(), os.path.join(args.model_path, f'ldm_last.pt'))
        torch.save({
            'epoch': epoch,
            'model': ldm.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best':{'best_val': best_val, 'best_acc': best_acc}, 
        }, os.path.join(args.model_path, f'checkpoint_last.pth'))
        if epoch % 10 == 0 and epoch != 0 :
            val_loss = valid(vae, ldm, validation_data, device, args, epoch)
            torch.save(ldm.state_dict(), os.path.join(args.model_path, f'ldm_{epoch}.pt'))
            
        if val_loss < best_val:
            best_val = val_loss
            print_log(f"Training Epoch {epoch} get best val_loss {best_val},\
                saving {os.path.join(args.model_path, f'ldm_best_val.pt')}", args.task_path)
            torch.save(ldm.state_dict(), os.path.join(args.model_path, f'ldm_best_val.pt'))
    print_log(f"Total training time: {total_time:.2f} seconds, {(total_time/epoch):.2f} seconds per epoch", args.task_path)
    return vae

def valid(vae, ldm, validation_data, device, args, epoch = 0):
    print_log(f'Validation Epoch [{epoch}] Start', args.task_path)
    pipeline = DiffTipeline(vae, DDPMScheduler(num_train_timesteps=1000, clip_sample=False, beta_start = 0.00085, beta_end = 0.012,\
        steps_offset = 1, rescale_betas_zero_snr = True, beta_schedule = "scaled_linear"), target_guidance=args.use_reward)
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
            chunk = batch["chunk_seqs"]# .to(device)
            
            with torch.no_grad():
                z0, mean, logvar, evaluation, seq_emb, tab_emb = vae.encode(seq,tab)
            
            z0 = z0.permute(1, 0, 2).contiguous()
            if random.random() < 0.1 and args.guidance_scale > 0:
                tab = torch.zeros_like(tab).float().to(device)
            
            cond = tab.unsqueeze(1)
            
            # z, mean, logvar, evaluation, seq_emb, tab_emb = vae.encode(seq, tab)
            z_predict, z_list, _ = pipeline(ldm, z0.shape, cond, steps=args.diff_num_step, generator=generator, guidance_scale=args.guidance_scale, device=device, use_reward=args.use_reward)
            # x = vae.decode(z0)

            loss += criterion(z_predict.float(), z0.float()).item()
    loss = loss / (len(validation_data) * args.batch_size)
    print_log(f'Validation Epoch [{epoch}] Loss: {loss:.4f}', args.task_path)
    return loss

def infer(vae, ldm, data, device, args):
    data, df = data
    max_acc = downstream_task_new(df, args.infer_func)
    print_log(f'Infer Start, Original accuracy:{max_acc} ({args.max_seq_len})', args.task_path)
    y = df.iloc[:, -1]
    #################################################################################################
    pipeline = DiffTipeline(vae, DDPMScheduler(num_train_timesteps=1000, clip_sample=False, beta_start = 0.00085, beta_end = 0.012,\
        steps_offset = 1, rescale_betas_zero_snr = True, beta_schedule = "scaled_linear"), target_guidance=args.use_reward)
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
            z_predict, z_list, _ = pipeline(ldm, shape, cond, steps=args.diff_num_step, generator=generator, guidance_scale=args.guidance_scale, device=device, use_reward=args.use_reward)
            
            generated_seq = vae.generate(z_predict.permute(1, 0, 2).contiguous())
            ################################################################################################
            sample_time = time.time() - sample_time
            total_time += sample_time
            
            if i % 50 == 0:
                print_log(f'Infer Batch [{i}/{len(data)}] Sample Time: {sample_time:.4f} seconds', args.task_path)
            new_df = df
            for i in generated_seq:
                try:
                    idx = (i == 4).nonzero(as_tuple=True)[0][0].item() 
                    feat = i[:idx].cpu().numpy()
                    new_df[' '.join(show_ops(feat))] = op_post_seq(df, feat)
                except:
                    continue
            new_df = new_df.replace([np.inf, -np.inf], np.nan)
            new_df = new_df.dropna(axis=1)
            new_df = new_df.clip(lower=-1e5, upper=1e5)  
            # print('New df', new_df.columns)
            new_acc = downstream_task_new(pd.concat([new_df,y], axis=1), args.infer_func, method = args.infer_method)
            if new_acc > max_acc:
                # print('----------------------------------')
                print_log(f'New accuracy: {new_acc} ({args.max_seq_len})', args.task_path)
                # 保存new_df
                new_df.to_csv(os.path.join(args.task_path, f'infer_{args.infer_func}_{args.max_seq_len}.csv'), index=False)
                # print('----------------------------------') 
                max_acc = new_acc
    print_log(f'Infer Finished, Total Sample Time: {total_time:.4f} seconds, Single Sample Time: {(total_time/(i.item()*args.batch_size)):.4f} seconds', args.task_path)
    return max_acc

def main():
    if not torch.cuda.is_available():
        print_log('No GPU found!')
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
    args.task_path = os.path.join(BASE_DIR, 'data/history', f'{args.task_name}_ldm',('test_' if args.test else '') + f'{args.exp_name}')
    args.model_path = os.path.join(BASE_DIR, 'data/history', f'{args.task_name}_ldm',('test_' if args.test else '') + f'{args.exp_name}','model')
    os.makedirs(args.task_path, exist_ok=True)
    os.makedirs(args.model_path, exist_ok=True)
    # print args
    if args.task_name.startswith('openml_'):
        args.infer_func = 'reg' 
    for arg, value in vars(args).items():
        print_log(f"{arg}: {value}", args.task_path)
    # check if the task name start with openml
    dataset = Data_Preprocessing(task=args.task_name, batch_size=args.batch_size, shuffle_time=args.shuffle_time, infer_size=args.infer_size)
    training_data = dataset.training_data
    validation_data = dataset.validation_data
    infer_data = (dataset.infer_data, dataset.test)
    max_length = dataset.max_length
    max_chunk_size = dataset.max_chunk_size
    max_chunk_num = dataset.max_chunk_num
    args.tab_len = dataset.tab_len
    print_log(f'Max length in training data is:{max_length}, vocab size is: {dataset.vocab_size}, device is {device}', args.task_path)
    print_log(f'Model path:{args.model_path}', args.task_path)
    vae = TransformerVAE(vocab_size=dataset.vocab_size + 1, hidden_size=args.hidden_size,\
                            dropout=args.dropout, num_layers=args.num_layers, latent_dim=args.latent_dim, max_chunk_len=max_chunk_size, \
                                max_chunk_num=max_chunk_num, tab_len=args.tab_len, args = args).to(device)
    vae.load_state_dict(torch.load(args.vae_load_path, map_location=torch.device("cuda")))  # load the pre-trained vae model
    
    ldm = TransformerDM(in_channels=args.latent_dim, t_channels=256, context_channels=args.latent_dim, hidden_channels=args.diff_hidden_size, depth=args.diff_num_layers,\
        dropout=args.dropout, tab_len=args.tab_len, out_channels=None).to(device)
    
    if args.test:
        print_log("Start Infering",args.task_path)
        ldm.load_state_dict(torch.load(args.ldm_load_path, map_location=torch.device("cuda")))
        print_log(f"Load ldm model from {args.ldm_load_path}",args.task_path)
        torch.save(ldm.state_dict(), os.path.join(args.model_path, f'ldm.pt'))
        torch.save(vae.state_dict(), os.path.join(args.model_path, f'vae.pt'))
        infer(vae, ldm, infer_data, device, args)
        print_log("Infering Finished", args.task_path)
    else:
        print_log("Start pre-training",args.task_path)
        ldm = pre_training(ldm, vae, training_data, validation_data, infer_data, args=args)



if __name__ == '__main__':
    main()
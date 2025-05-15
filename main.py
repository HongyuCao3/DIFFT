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
import torch.nn.utils as nn_utils
import torch.optim.lr_scheduler as lr_scheduler
import time
parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, choices=['airfoil', 'amazon_employee',
                                                        'ap_omentum_ovary', 'german_credit',
                                                        'higgs', 'housing_boston', 'ionosphere',
                                                        'lymphography', 'messidor_features', 'openml_620',
                                                        'pima_indian', 'spam_base', 'spectf', 'svmguide3',
                                                        'uci_credit_card', 'wine_red', 'wine_white', 'openml_586',
                                                        'openml_589', 'openml_607', 'openml_616', 'openml_618', 'mice_protein',
                                                        'openml_637'], default='ionosphere')
parser.add_argument('--hidden_size', type=int, default=512)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--latent_dim', type=int, default=512)
parser.add_argument('--max_seq_len', type=int, default=50)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--load_epoch', type=int, default=0)
parser.add_argument('--epochs', type=int, default=600)
parser.add_argument('--pre_epochs', type=int, default=800)
parser.add_argument('--add_origin', type=bool, default=True)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--keyword', type=str, default='')
parser.add_argument('--shuffle_time', type=int, default=0)
parser.add_argument('--accumulation_steps', type=int, default=64)
parser.add_argument('--infer_size', type=int, default=300)
args = parser.parse_args()
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
def info_nce_loss(seq_emb, tab_emb, temperature=0.07):
    # emb shape: (batch_size, embedding_dim)
    batch_size = seq_emb.size(0)


    logits = torch.mm(seq_emb, tab_emb.t()) / temperature
    logits_ts = torch.mm(tab_emb, seq_emb.t()) / temperature

    labels = torch.arange(batch_size).to(seq_emb.device)


    loss_seq2tab = F.cross_entropy(logits, labels)
    # loss_tab2seq = F.cross_entropy(logits.t(), labels)
    loss_tab2seq = F.cross_entropy(logits_ts, labels)

    loss = (loss_seq2tab + loss_tab2seq) / 2
    return loss

def pre_training(vae, training_data, validation_data, load_epoch):
    device = int(args.gpu)
    # optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr)
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    reconstruction_criterion = torch.nn.CrossEntropyLoss(ignore_index=0)  
    performance_criterion = torch.nn.MSELoss()  
    chunk_criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(load_epoch, args.pre_epochs+1):
        vae.train()
        time0 = time.time()
        total_reconstruction_loss = 0
        total_performance_loss = 0
        total_alignment_loss = 0
        total_kl_loss = 0
        total_chunk_loss = 0
        correct = 0
        total_chunk = 0
        chunk_size_gt_distribution = {}
        optimizer.zero_grad()
        for i, batch in enumerate(((training_data))):

            seq = batch["seqs"]
            tab = batch["tabs"]
            performance = batch["performances"]
            chunk = batch["chunk_seqs"] 
            
            seq = seq.to(device)
            chunk_size_gt = (seq == 4).sum(dim=1) + 1
            tab = tab.to(device)
            performance = performance.to(device)
            unique, counts = torch.unique(chunk_size_gt, return_counts=True)
            for u, c in zip(unique.cpu().numpy(), counts.cpu().numpy()):
                if u in chunk_size_gt_distribution:
                    chunk_size_gt_distribution[u] += c
                else:
                    chunk_size_gt_distribution[u] = c
            logits, mean, logvar, evaluation, seq_emb, tab_emb, chunk_size = vae(seq, tab, chunk)

            
            merged_target_seq_index = [sub_target_seq.squeeze(0) for sub_target_seq in chunk]
            merged_target_seq_index = torch.cat(merged_target_seq_index, dim=0).to(device)
            merged_target_seq_index = merged_target_seq_index[:, 1:] 
            reconstruction_loss = reconstruction_criterion(
                logits.view(-1, logits.size(-1)), merged_target_seq_index.contiguous().view(-1)  
            )


            performance_loss = performance_criterion(evaluation, performance)

            chunk_loss = chunk_criterion(chunk_size, chunk_size_gt)
            kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / seq.size(0)

            
            loss = chunk_loss  + reconstruction_loss + performance_loss +  0.001 * kl_loss

            loss.backward()

            if (i + 1) % args.accumulation_steps == 0 or (i + 1) == len(training_data):
                optimizer.step()
                optimizer.zero_grad()  
            chunk_pre = chunk_size.argmax(dim=1)
            correct += (chunk_pre == chunk_size_gt).sum().item()
            total_chunk += chunk_size_gt.size(0)
            total_reconstruction_loss += reconstruction_loss.item()
            total_performance_loss += performance_loss.item()
            total_kl_loss += kl_loss.item()
            total_chunk_loss += chunk_loss.item()
        time1 = time.time()
        print(f"Epoch {epoch} Time: {time1 - time0:.2f}s")
        if epoch % 1 == 0:
            # print("Chunk Size GT Distribution:")
            # for size, count in sorted(chunk_size_gt_distribution.items()):
            #     print(f"Size {size}: {count} occurrences")
            # accuracy = valid(vae, training_data, device)
            # print(f'Validation Accuracy: {accuracy:.4f}')
            # print('loss:', loss.item())
            info(f"Pre-training Epoch {epoch}: Reconstruction Loss = {total_reconstruction_loss / len(training_data)}, Performance Loss = {total_performance_loss / len(training_data)}, Alignment Loss = {total_alignment_loss / len(training_data)}, Chunk Loss = {total_chunk_loss/len(training_data)}, KL Loss = {total_kl_loss / len(training_data)}, Accuracy = {correct / total_chunk}")
        if epoch % 100 == 0 and epoch > 0:
            accuracy = valid(vae, validation_data, device)
            # print(f'Validation Accuracy: {accuracy:.4f}')
            torch.save(vae.state_dict(), os.path.join(BASE_DIR, 'data/history', args.task_name, 'model', f'vae_{epoch}{args.keyword}.pt'))

    return vae

def valid(vae, validation_data, device):

    vae.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in validation_data:

            seq = batch["seqs"]
            tab = batch["tabs"]
            performance = batch["performances"]
            chunk = batch["chunk_seqs"] 
            seq = seq.to(device)
            tab = tab.to(device)
            performance = performance.to(device)

            chunk_size_gt = (seq == 4).sum(dim=1) + 1

            logits, mean, logvar, evaluation, seq_emb, tab_emb, chunk_size = vae(seq, tab, chunk)
            correct += (chunk_size.argmax(dim=1) == chunk_size_gt).sum().item()
            total += chunk_size_gt.size(0)

    accuracy = correct / total
    print(f'Validation Accuracy: {accuracy:.4f}')
    return accuracy

    
def infer(vae, data, device, df):
    max_acc = downstream_task_new(df, 'cls')
    print('Original accuracy:', max_acc)
    y = df.iloc[:, -1]
    print(y.isnull().sum()) 
    # print(y)
    df = df.iloc[:, :-1]
    df.columns = [str(i) for i in range(df.shape[1])]
    vae.eval()
    with torch.no_grad():
        for batch in data:
            seq = batch["seqs"]
            tab = batch["tabs"]
            performance = batch["performances"]
            chunk = batch["chunk_seqs"]
            seq = seq.to(device)
            tab = tab.to(device)
            performance = performance.to(device)
            generated_seq = vae.generate_test(seq)
            # shoud be vae.generate in diffusion
            new_df = df
            for i in generated_seq:
                try:
                    idx = (i == 4).nonzero(as_tuple=True)[0][0].item()  #
                    feat = i[:idx].cpu().numpy()
                    new_df[' '.join(show_ops(feat))] = op_post_seq(df, feat)
                except:
                    continue
            new_df = new_df.replace([np.inf, -np.inf], np.nan)    
            new_df = new_df.dropna(axis=1)
            new_acc = downstream_task_new(pd.concat([new_df,y], axis=1), 'reg')
            if new_acc > max_acc:
                print('----------------------------------')
                print(f'New accuracy: {new_acc}')
                max_acc = new_acc

    return 0


def main():
    if not torch.cuda.is_available():
        info('No GPU found!')
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
    info(f"Args = {args}")
    dataset = Data_Preprocessing(task=args.task_name, batch_size=args.batch_size, shuffle_time=args.shuffle_time, infer_size=args.infer_size)
    training_data = dataset.training_data
    validation_data = dataset.validation_data
    infer_data = dataset.infer_data
    max_length = dataset.max_length
    max_chunk_size = dataset.max_chunk_size
    max_chunk_num = dataset.max_chunk_num
    data = dataset.test
    tab_len = dataset.tab_len
    info(f'Max length in training data is:{max_length}, vocab size is: {dataset.vocab_size}')
    model_path = os.path.join(BASE_DIR, 'data/history', args.task_name, 'model')
    info(f'Model path:{model_path}')
    vae = TransformerVAE(vocab_size=dataset.vocab_size + 1, hidden_size=args.hidden_size,\
                            dropout=args.dropout, num_layers=args.num_layers, latent_dim=args.latent_dim, max_chunk_len=max_chunk_size, max_chunk_num=max_chunk_num, tab_len=tab_len).to(device)
    if args.load_epoch > 0 and os.path.exists(os.path.join(model_path, f'vae_{args.load_epoch}{args.keyword}.pt')):
        vae.load_state_dict(torch.load(os.path.join(model_path, f'vae_{args.load_epoch}{args.keyword}.pt')))
        vae.eval()
        info(f"Load vae model from epoch {args.load_epoch}{args.keyword}")
    if args.load_epoch < args.pre_epochs:
        info("Start pre-training")
        os.makedirs(model_path, exist_ok=True)
        vae = pre_training(vae, training_data, validation_data, args.load_epoch)
    print('Start infer')
    # valid(vae, validation_data, device)
    infer(vae, infer_data, device, data)

    



if __name__ == '__main__':
    main()
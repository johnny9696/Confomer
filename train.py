import os
import json
import argparse
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from util import get_hps, get_wav, get_text_loader, log_scaler, log_model
from Conformer import Conformer
from data_loader import MelLangCollate, MelLangLoader

import torch.multiprocessing as mp

import warnings

def main():
    args=argparse.ArgumentParser()
    args.add_argument('-c','--config',type=str, default='./config.json')
    path=args.parse_args()
    hps=get_hps(path.config)
    torch.manual_seed(1234)
    hps.n_gpus = torch.cuda.device_count()
  
    hps.batch_size=int(float(hps.train.batch_size)/float(hps.n_gpus))
    if hps.n_gpus>1:
        mp.spawn(train_and_eval,nprocs=hps.n_gpus,args=(hps.n_gpus,hps, ))
    else:   
        train_and_eval(0,hps.n_gpus,hps)

def train_and_eval(rank, n_gpus, hps):
    global global_step

    if hps.n_gpus>1:
        os.environ["MASTER_ADDR"]="localhost"
        os.environ["MASTER_PORT"]="12355"
        dist.init_process_group(backend='nccl',init_method='env://',world_size=n_gpus,rank=rank)
    
    device=torch.device("cuda:{:d}".format(rank))
    print(device)
    collate_fn=MelLangCollate()
    train_data=MelLangLoader(hps.dataset.training_data, hps)
    validation_data=MelLangLoader(hps.dataset.validation_data, hps)

    train_loader=DataLoader(train_data,num_workers=1,shuffle=False, batch_size=int(hps.train.batch_size),
    pin_memory=True, drop_last=True, collate_fn=collate_fn)
    validation_loader=DataLoader(validation_data, num_workers=1, shuffle=False, batch_size=int(hps.train.batch_size), 
    pin_memory=True, drop_last=True, collate_fn=collate_fn)

    model=Conformer(n_mels=int(hps.dataset.n_mels), n_class= int(hps.model.n_class), 
    encoder_dim=int(hps.model.encoder_dim), expantion_factor=int(hps.model.expantion_factor), 
    kernel_size=int(hps.model.kernel_size), num_attention_head=int(hps.model.num_attention_head),
    dropout_p=float(hps.model.dropout_p), n_Conf_block=int(hps.model.n_Conf_block)).to(device)

    if hps.n_gpus>1:
        print("Multi GPU Setting Start")
        model=DistributedDataParallel(model,device_ids=[rank]).to(device)
        print("Multi GPU Setting Finish")

    optimizer=optim.Adam(model.parameters(), lr=float(hps.train.learning_rate), 
    betas=(float(hps.train.beta1), float(hps.train.beta2)), eps=float(hps.train.eps), weight_decay=0, amsgrad=False, foreach=None, 
    maximize=False, capturable=False)

    scheduler=optim.lr_scheduler.OneCycleLR(optimizer, max_lr=float(hps.train.max_lr), steps_per_epoch=hps.batch_size,epochs=int(hps.train.epochs))

    epoch_str=1
    global_step=0

    for epoch in range(epoch_str, int(hps.train.epochs)):
        train(rank, device, epoch, hps, model, train_loader,optimizer)
        evaluate(rank, device,  epoch, hps, model, validation_loader)
        print("one epoch Finbished")



def train(rank, device, epoch, hps, model, train_loader, optimizer):
    global global_step

    criterion=nn.CrossEntropyLoss()
    model.train()
    for batch_idx, (mel_padded, label) in enumerate(train_loader):
        mel_padded=mel_padded.to(device)
        label=label.to(device)

        optimizer.zero_grad()
        output=model(mel_padded)
        loss=criterion(output,label)
        loss.backward()
        optimizer.step()

        global_step += 1



def evaluate(rank, device, epoch,hps, model, validation_loader, optimizer):
    global global_step
    
    criterion=nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        for batch_idx, (mel_padded,label) in enumerate(validation_loader):
            mel_padded=mel_padded.to(device)
            label=label.to(device)

            output=model(mel_padded)
            loss=criterion(output,label)

            global_step += 1



if  __name__=="__main__":
    warnings.simplefilter(action="ignore",category=FutureWarning)
    main()
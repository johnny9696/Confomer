import os
import json
import argparse
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from util import get_hps, get_wav, get_text_loader, log_scalar, log_model, logger_start, label2length_label
from symbols import symbols, vec2text
from Conformer import Conformer
from data_loader import MelTextCollate, MelTextLoader
from commons import Adam

import torch.multiprocessing as mp

import warnings

def main():
    args=argparse.ArgumentParser()
    args.add_argument('-c','--config',type=str, default='./configs/config_text.json')
    path=args.parse_args()
    hps=get_hps(path.config)
    torch.manual_seed(1234)
    hps.n_gpus = torch.cuda.device_count()
  
    hps.batch_size=int(hps.train.batch_size/hps.n_gpus)


    if hps.n_gpus>1:
        mp.spawn(train_and_eval,nprocs=hps.n_gpus,args=(hps.n_gpus,hps, ))
    else:   
        train_and_eval(0,hps.n_gpus,hps)

def train_and_eval(rank, n_gpus, hps):
    global global_step

    #tensorboard setting
    writer=logger_start(hps)

    if hps.n_gpus>1:
        os.environ["MASTER_ADDR"]="localhost"
        os.environ["MASTER_PORT"]="12355"
        dist.init_process_group(backend='nccl',init_method='env://',world_size=n_gpus,rank=rank)
    
    device=torch.device("cuda:{:d}".format(rank))
    collate_fn=MelTextCollate(n_class=hps.model.n_class)
    train_data=MelTextLoader(hps.dataset.training_data, hps)
    validation_data=MelTextLoader(hps.dataset.validation_data, hps)

    train_loader=DataLoader(train_data,num_workers=4,shuffle=False, batch_size=hps.train.batch_size,
    pin_memory=True, drop_last=True, collate_fn=collate_fn)
    validation_loader=DataLoader(validation_data, num_workers=4, shuffle=False, batch_size=hps.train.batch_size, 
    pin_memory=True, drop_last=True, collate_fn=collate_fn)

    model=Conformer(n_mels=hps.dataset.n_mels, n_class= len(symbols), 
    encoder_dim=hps.model.encoder_dim, expantion_factor=hps.model.expantion_factor, 
    kernel_size=hps.model.kernel_size, num_attention_head=hps.model.num_attention_head,
    dropout_p=hps.model.dropout_p, n_Conf_block=hps.model.n_Conf_block).to(device)

    if rank ==0:
        print(symbols)

    if hps.n_gpus>1:
        if rank ==0 :
            print("Multi GPU Setting Start")
        model=DistributedDataParallel(model,device_ids=[rank]).to(device)
        if rank ==0:
            print("Multi GPU Setting Finish")
    sample_audio=torch.rand(2,80,488)
    length=torch.zeros(488,dtype=torch.long)
    log_model(writer, (sample_audio , length),model)

    optimizer=Adam(model.parameters(),d_model=hps.model.encoder_dim, warmup_step=hps.train.warmup_step, lr=hps.train.learning_rate, 
    betas=(hps.train.beta1, hps.train.beta2), eps=hps.train.eps, weight_decay=0, amsgrad=False, foreach=None, 
    maximize=False, capturable=False,scheduler=hps.train.scheduler) 

    


    epoch_str=1
    global_step=0

    for epoch in range(epoch_str, int(hps.train.epochs)):
        train(rank, device, epoch, hps, model, train_loader,optimizer, writer)
        evaluate(rank, device,  epoch, hps, model, validation_loader,optimizer,writer)
        print("one epoch Finbished")
        torch.save(model, hps.train.log_path+'/epoch'+str(epoch)+'.pth')
    writer.close()



def train(rank, device, epoch, hps, model, train_loader, optimizer,writer):
    global global_step
    global_loss=0.0

    criterion=nn.CTCLoss(zero_infinity=True).to(device)
    model.train()
    torch.autograd.set_detect_anomaly(True)
    for batch_idx, (mel_padded, input_length, label, target_length) in enumerate(train_loader):
        mel_padded=mel_padded.to(device)
        label=label.to(device)
        target_length=target_length.to(device)
        input_length=input_length.to(device)
        """
        #save model structure on tensorboard
        if global_step ==1 and rank == 0 :
            log_model(writer, model, mel_padded[0])
        """
        #print(mel_padded,label,input_length,target_length)
        optimizer.zero_grad()
        output,output_length=model(mel_padded, input_length)
        batch, seq_length, on_class=output.size()
        output=output.view((seq_length,batch,on_class))
        print(label.size(), output.size())
        loss=criterion(output, label, output_length, target_length)
        loss.backward()
        optimizer.step()
        global_loss += float(loss.item())
        #calculate accuracy
        _, output= torch.max(output,2)
        #label_=label2length_label(label, output.size(1), hps).to(device)    
        #log tensorboard
        if batch_idx % int(hps.train.log_step) ==0 and rank == 0 :
            _label=label.detach().cpu()
            _output=output.detach().cpu()
            _label=_label.tolist()
            _output=_output.tolist()
            t_label=vec2text(_label[0])
            t_output=vec2text(_output[0])
            print(_label[0],':',"t",t_label)
            print(_output[0],':',t_output)
            log_scalar(writer,'train/loss',global_step,loss)
            log_scalar(writer,'train/global_loss',global_step,global_loss/100)
            #log_scalar(writer,'train/accuracy',global_step, correct/total )
            log_scalar(writer,"Learning_rate",global_step, optimizer.get_lr())
            global_loss=0.0
            print("Steps >> {} train loss : {} ".format(global_step, float(loss.item())))
            print("Learning rate>> {} : {} ".format(global_step, optimizer.get_lr()))

        global_step += 1

    writer.flush()



def evaluate(rank, device, epoch, hps, model, validation_loader, optimizer, writer):
    global global_step
    global_loss=0.0

    criterion=nn.CTCLoss().to(device)
    model.eval()
    
    for batch_idx, (mel_padded, input_length, label, target_length) in enumerate(validation_loader):
        mel_padded=mel_padded.to(device)
        label=label.to(device)
        target_length=target_length.to(device)
        input_length=input_length.to(device)
        """
        #save model structure on tensorboard
        if global_step ==1 and rank == 0 :
            log_model(writer, model, mel_padded[0])
        """
        optimizer.zero_grad()
        output,output_length=model(mel_padded, input_length)
        loss=criterion(output.transpose(1,0), label, output_length, target_length)
        optimizer.step()
        global_loss += float(loss.item())
        #calculate accuracy
        _, output= torch.max(output,2)
        #label_=label2length_label(label, output.size(1), hps).to(device)
    
        #total += output.size(1)
        #correct += (label_==output).sum().item()

        #log tensorboard
        if batch_idx % int(hps.train.log_step) ==0 :
            _label=label.detach().cpu()
            _output=output.detach().cpu()
            _label=_label.tolist()
            _output=_output.tolist()
            t_label=vec2text(_label[0])
            t_output=vec2text(_output[0])
            print(t_label, t_output)
            log_scalar(writer,'eval/loss',global_step,loss)
            log_scalar(writer,'eval/global_loss',global_step,global_loss/100)
            #log_scalar(writer,'eval/accuracy',global_step, correct/total )
            global_loss=0.0
            print("Steps >> {} validation loss : {} ".format(global_step, float(loss.item())))

        global_step += 1

    writer.flush()



if  __name__=="__main__":
    warnings.simplefilter(action="ignore",category=FutureWarning)

    import sys

    main()
import os
import math
import torch
from torch import nn, optim
import numpy as np

class Adam():
    def __init__(self, params, d_model, lr=1e-3, betas=(0.9,0.999), eps=1e-8 , weight_decay=0,scheduler=None, 
    warmup_step=4000, amsgrad=False, foreach=None, capturable=False, fused=False, maximize=False):
        
        self.lr=lr
        self.step_=1
        self.warmup_step=warmup_step
        self.d_model=d_model
        self.scheduler=scheduler
        self.cur_lr = lr*self.cal_lr()

        self.optim=optim.Adam(params, lr=1e-3, betas=betas, eps=eps, weight_decay=weight_decay,amsgrad=amsgrad, foreach=foreach, capturable=capturable,maximize=maximize)

    def cal_lr(self):
        lr=1.0
        if self.scheduler=='noam':
            lr=self.d_model**(-0.5)*min(self.step_**(-0.5),self.step_*(self.warmup_step**(-1.5)))
        return lr

    def update_learning_rate(self):
        self.step_ +=1
        if self.scheduler=='noam':
            self.cur_lr= self.lr * self.cal_lr()

        for group in self.optim.param_groups:
            group['lr']=self.cur_lr
    
    def get_lr(self):
        return self.cur_lr

    def load_state_dict(self,d):
        self.optim.load_state_dict(d)

    def state_dict(self):
        self.optim.state_dict()

    def step(self):
        self.optim.step()
        #self.update_learning_rate()

    def zero_grad(self):
        self.optim.zero_grad()
    

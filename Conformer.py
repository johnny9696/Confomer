from doctest import OutputChecker
from tkinter import E
import torch
import torch.nn as nn
import torch import Tensor
import attention.MHSA as MHSA

class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        return x*self.sigmoid(x)

class GLU(nn.Module):
    def __init__(self, dim) :
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs):
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()

class Convolution_module(nn.Module):
    def __init__(self,
    input_channnel,
    output_channel,
    kernel_size,
    dropout_p,
    padding=0,
    stride=1,
    ):
        self.Layer_norm=nn.LayerNorm()
        self.P_conv=nn.Conv1d(in_channels=input_channnel,out_channels=output_channel,kernel_size=1,stride=stride,padding=padding,bias=True)
        self.D_conv=nn.Conv1d(in_channels=input_channnel,out_channel=output_channel,kernel_size=kernel_size,groups=input_channnel,stride=stride,padding=padding,bias=False)
        self.batch_norm=nn.BatchNorm1d(input_channnel)
        self.GLU=GLU(dim=1)
        self.swish=Swish()
        self.dropout=nn.Dropout(p=dropout_p)
    def forward(self,x):
        orgi_x=torch.detach(x)
        x=self.Layer_norm(x)
        x=self.P_conv(x)
        x=self.GLU(x)
        x=self.D_conv(x)
        x=self.batch_norm(x)
        x=self.swish(x)
        x=self.P_conv(x)
        x=self.dropout(x)
        return x+orgi_x


class Feed_forward_module(nn.Module):
    def __init__(self,hidden_channels,expansion_factor,dropout_p):
        self.Layernorm=nn.LayerNorm
        self.Linear1=nn.Linear(hidden_channels,hidden_channels*expansion_factor)
        self.swish=Swish()
        self.Linear2=nn.Linear(hidden_channels*expansion_factor,hidden_channels)
        self.dropout=nn.Dropout(p=dropout_p)
    def forward(self,x):
        """
        x : [batch, h, hidden_channels]
        """
        x=self.Layernorm(x) #[batch, h, hidden_channels]
        x=self.Linear1(x) # [batch, h, hidden_channels*Expansion_factor]
        x=self.swish(x)
        x=self.dropout(x)
        x=self.Linear2(x) # [batch, h, hidden_channels]
        x=self.dropout(x)
        return x

class  MHSA_module(nn.Module):
    def __init__(self,
    dropout_p,
    d_model,
    num_head):
        self.Layer_norm=nn.LayerNorm()
        self.dropout=nn.Dropout(p=dropout_p)
        self.MHSA=MHSA(d_model, num_head)
    def forward(self,x):
        x=self.Layer_norm(x)
        x=self.MHSA(x)
        x=self.dropout(x)
        return x

class convolution_subsampling(nn.Module):
    def __init__(self,
    in_channels,
    out_channels):
        self.sequential=nn.sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=2),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=2),
            nn.ReLU()
        )
    def forward(self,x):
        """
        Input : Mel Spectrogram Shape [batch, n_Mels, Frames]
        After Unsqueeze(1) [batch, 1 , n_Mels , Frames]
        After SubSampling [batch, out_channels , t , h ]
        To Use Linear Function Need to change to 2D
        [batch, out_channels*t,h] -> Transpose [batch,h,out_channels*t]
        t=((((n_mels-3)//2+1)-3)//2)+1
        """
        output = self.sequential(x.unsqueeze(1))
        batch,channels,length,height=output.size()
        output=output.contigous().view(batch,channels*length,height)
        output=output.permute(0,2,1)
        return output

class Conformer_block(nn.Module):
    def __init__(self,
    dropout_p,
    expansion_factor,
    hidden_channels):
        self.feed_forward = Feed_forward_module(hidden_channels=hidden_channels,expansion_factor=expansion_factor,dropout_p=dropout_p)
        self.MHSA = MHSA_module()
        self.convolution = Convolution_module()
        self.Layernorm = nn.Layernorm()

    def residual_connect(self,x,orig_x):
        x = x+orig_x
        return x,x
    
    def forward(self,x):
        """
        Block Input Shape :  [batch, h, hidden_channels]
        After Feed forward : [batch, h, hidden_channels]
        """
        orig_x = x
        x = self.feed_forward(x)
        orgi_x, x = self.residual_connect(torch.div(x,0.5),orgi_x)
        x=self.MHSA(x)
        orgi_x,x=self.residual_connect(x,orgi_x)
        x=self.convolution(x) 
        orgi,x=self.residual_connect(x,orgi_x)
        x=self.feed_forward(x)
        orgi_x,x=self.residual_connect(torch.div(x,0.5),orgi_x)
        x=self.Layernorm(x)
        return x

class Conformer(nn.Module):
    def __init__(self,
    dropout_p=0.4,
    n_Conf_block=6,
    n_mels,
    n_class,
    hidden_channel,
    input_channel,):
        self.Conv_sub = convolution_subsampling(in_channels=1,out_channels=hidden_channel) #[b,h,out_channels*t]
        self.Linear = nn.Linear(hidden_channel*((((n_mels-3)//2+1)-3)//2+1), hidden_channel) #[batch, h, hidden_channels]
        self.dropout = nn.dropout(p=dropout_p) # [batch, h, hidden_channels]
        self.block = Conformer_block()
        conformer_block = []
        for i in n_Conf_block:
            conformer_block.append(self.block)
        self.conformer_block = nn.Sequencial(*conformer_block)
        self.FClayer = nn.Linear(,n_class)
    
    def forward(self,x):
        #input x [b,n_mels,frames]
        x=self.Conv_sub(x)
        x=self.Linear(x)
        x=self.dropout(x)
        x=self.conformer_block(x)
        x=self.FClayer(x)
        return x

import torch
import torch.nn as nn
import torch import Tensor

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
    def __init__(self,encoder_dim,expansion_factor,dropout_p):
        self.Layernorm=nn.LayerNorm
        self.Linear=nn.Linear(encoder_dim,encoder_dim*expansion_factor)
        self.swish=Swish()
        self.dropout=nn.Dropout(p=dropout_p)
    def forward(self,x):
        x=self.Layernorm(x)
        x=self.Linear(x)
        x=self.swish(x)
        x=self.dropout(x)
        x=self.Linear(x)
        x=self.dropout(x)
        return x

class  MHSA(nn.Module):
    def __init__(self):

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
        output=self.sequential(x.unsqueeze(1))




class Conformer_block(nn.Module):
    def __init__(self):
        self.feed_forward=Feed_forward_module()
        self.MHSA=MHSA()
        self.convolution=Convolution_module()
        self.Layernorm=nn.Layernorm()

    def residual_connect(self,x,orig_x):
        x=x+orig_x
        return x,x
    
    def forward(self,x):
        orig_x=x
        x=self.feed_forward(x)
        orgi_x,x=self.residual_connect(x,orgi_x)
        x=self.MHSA(x)
        orgi_x,x=self.residual_connect(x,orgi_x)
        x=self.convolution(x)
        orgi,x=self.residual_connect(x,orgi_x)
        x=self.feed_forward(x)
        orgi_x,x=self.residual_connect(x,orgi_x)
        x=self.Layernorm(x)
        return x

class Conformer(nn.Module):
    def __init__(self,
    dropout_p=0.4,
    n_Conf_block=6,
    n_class,
    hidden_channel,
    input_channel,):
        self.Conv_sub=convolution_subsampling()
        self.Linear=nn.Linear()
        self.dropout=nn.dropout(p=dropout_p)
        self.block=Conformer_block()
        conformer_block=[]
        for i in n_Conf_block:
            conformer_block.append(self.block)
        self.conformer_block=nn.Sequencial(*conformer_block)
        self.FClayer=nn.Linear(,n_class)
    
    def forward(self,x):
        #input x [b,]
        x=self.Conv_sub(x)
        x=self.Linear(x)
        x=self.dropout(x)
        x=self.conformer_block(x)
        x=self.FClayer(x)
        return x

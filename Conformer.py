from doctest import OutputChecker
from json import encoder
from tkinter import E
import torch
import torch.nn as nn
from torch import Tensor
from attention import MHSA

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
    expansion_factor,
    padding=0,
    stride=1,
    ):
        super(Convolution_module,self).__init__()
        self.expansion_factor=expansion_factor
        self.Layer_norm=nn.LayerNorm(input_channnel)
        self.P_conv1=nn.Conv1d(in_channels=input_channnel,out_channels=output_channel*expansion_factor,kernel_size=1,stride=stride,padding=padding,bias=True)
        self.P_conv2=nn.Conv1d(in_channels=input_channnel,out_channels=output_channel,kernel_size=1,stride=stride,padding=padding,bias=True)
        self.D_conv=nn.Conv1d(in_channels=output_channel,out_channels=output_channel,kernel_size=kernel_size,groups=input_channnel,stride=stride,padding=(kernel_size-1)//2,bias=True)
        self.batch_norm=nn.BatchNorm1d(input_channnel)
        self.GLU=GLU(dim=1)
        self.swish=Swish()
        self.dropout=nn.Dropout(p=dropout_p)
    def forward(self,x):
        orgi_x=torch.detach(x)
        x=self.Layer_norm(x)
        x=x.transpose(1,2)
        x=self.P_conv1(x)
        x=self.GLU(x)
        x=self.D_conv(x)
        x=self.batch_norm(x)
        x=self.swish(x)
        x=self.P_conv2(x)
        x=self.dropout(x)
        x=x.transpose(1,2)
        return x+orgi_x

class Feed_forward_module(nn.Module):
    def __init__(self,hidden_channels,expansion_factor,dropout_p):
        super(Feed_forward_module,self).__init__()
        self.Layernorm=nn.LayerNorm(hidden_channels)
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
        super(MHSA_module,self).__init__()
        self.Layer_norm=nn.LayerNorm(d_model)
        self.dropout=nn.Dropout(p=dropout_p)
        self.MHSA=MHSA(d_model = d_model, num_heads=num_head)
    def forward(self,x):
        x=self.Layer_norm(x)
        x=self.MHSA(x)
        x=self.dropout(x)
        return x

class convolution_subsampling(nn.Module):
    def __init__(self,
    in_channels,
    out_channels):
        super(convolution_subsampling,self).__init__()
        self.sequential=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=2),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=2),
            nn.ReLU()
        )
    def forward(self,x, input_length):
        """
        Input : Mel Spectrogram Shape [batch, n_Mels, Frames]
        After Unsqueeze(1) [batch, 1 , n_Mels , Frames]
        After SubSampling [batch, out_channels , t , h ]
        To Use Linear Function Need to change to 2D
        [batch, out_channels*t,h] -> Transpose [batch,h,out_channels*t]
        t=((((frames-3)//2+1)-3)//2)+1
        input_length is [n_frames, ....]
        """
        output = self.sequential(x.unsqueeze(1))
        batch,channels,length,height=output.size()  
        output=output.contiguous().view(batch,channels*length,height)
        output=output.permute(0,2,1)
        output_length=input_length>>2
        output_length -= 1
        return output, output_length

class Conformer_block(nn.Module):
    def __init__(self,
    dropout_p,
    expansion_factor,
    kernel_size,
    num_head,
    encoder_dim):
        super(Conformer_block,self).__init__()
        self.feed_forward = Feed_forward_module(hidden_channels= encoder_dim,expansion_factor=expansion_factor,dropout_p=dropout_p)
        self.MHSA = MHSA_module(d_model = encoder_dim, num_head = num_head , dropout_p = dropout_p)
        self.convolution = Convolution_module(input_channnel = encoder_dim, output_channel = encoder_dim,expansion_factor=expansion_factor, kernel_size = kernel_size, dropout_p = dropout_p, padding=0, stride=1)
        self.Layernorm = nn.LayerNorm(encoder_dim)

    def residual_connect(self, x, orig_x):
        x = x+orig_x
        return x,x
    
    def forward(self,x):
        """
        Block Input Shape :  [batch, h, hidden_channels]
        After Feed forward : [batch, h, hidden_channels]
        """
        org_x = x
        x = self.feed_forward(x)
        org_x, x = self.residual_connect(torch.div(x,0.5), org_x)
        x=self.MHSA(x)
        org_x,x=self.residual_connect(x,org_x)
        x=self.convolution(x)
        org,x=self.residual_connect(x,org_x)
        x=self.feed_forward(x)
        org_x,x=self.residual_connect(torch.div(x,0.5),org_x)
        x=self.Layernorm(x)
        return x


class Single_LSTM_Decoder(nn.Module):
    def __init__(self,
    input_size,
    hidden_size,
    num_layers=3,
    bias=True
    ):
        super(Single_LSTM_Decoder,self).__init__()
        self.LSTM=nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,bias=bias, batch_first=True)
    def forward(self,x):
        return self.LSTM(x)

class Conformer(nn.Module):
    def __init__(self,
    n_mels,
    n_class,
    encoder_dim,
    expantion_factor,
    kernel_size,
    num_attention_head,
    dropout_p=0.4,
    n_Conf_block=6):
        super(Conformer,self).__init__()
        self.Conv_sub = convolution_subsampling(in_channels=1,out_channels=encoder_dim) #[b,h,out_channels*t]
        self.Linear = nn.Linear(encoder_dim*((((n_mels-3)//2+1)-3)//2+1), encoder_dim) #[batch, h, hidden_channels]
        self.dropout = nn.Dropout(p=dropout_p) # [batch, h, hidden_channels]
        self.block = Conformer_block(encoder_dim=encoder_dim, dropout_p=dropout_p , expansion_factor=expantion_factor ,kernel_size=kernel_size,
        num_head=num_attention_head)
        conformer_block = []
        for i in range(n_Conf_block):
            conformer_block.append(self.block)
        self.conformer_block = nn.Sequential(*conformer_block)
        self.single_LSTM=Single_LSTM_Decoder(input_size=encoder_dim,hidden_size=n_class)
        #self.FClayer = nn.Linear(encoder_dim, n_class, bias = False)
    
    def forward(self,x, input_length):
        #input x [b,n_mels,frames]
        x, output_length = self.Conv_sub(x,input_length)
        x=self.Linear(x)
        x=self.dropout(x)
        x=self.conformer_block(x)
        x,_=self.single_LSTM(x)
        #x=self.FClayer(x)
        x= nn.functional.log_softmax(x, dim = 2)
        return x, output_length
from turtle import position
import torch
import torch.nn as nn
from torch import Tensor
import math
import torch.nn.functional as F

#position Embedding Matrix
class position_encoding(nn.Module):
    """
    Positional Encoding proposed in "Attention Is All You Need".
    Since transformer contains no recurrence and no convolution, in order for the model to make
    use of the order of the sequence, we must add some positional information.
    "Attention Is All You Need" use sine and cosine functions of different frequencies:
        PE_(pos, 2i)    =  sin(pos / power(10000, 2i / d_model))
        PE_(pos, 2i+1)  =  cos(pos / power(10000, 2i / d_model))
    """
    def __init__(self, d_model: int = 512, max_len: int = 10000) -> None:
        super(position_encoding, self).__init__()
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, length: int) -> Tensor:
        return self.pe[:, :length]

class Relative_position_encoding_attention(nn.Module):
    def __init__(self, d_model, num_heads, dropout_p):
        super(Relative_position_encoding_attention,self).__init__()

        self.d_model = d_model
        self.d_head = int(d_model/num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(d_model)

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.pos_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p = dropout_p)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform(self.u_bias)
        torch.nn.init.xavier_uniform(self.v_bias)

        self.out_proj=nn.Linear(d_model,d_model)
        
    def forward(self, query, key, value, position_emb, mask = None):
        batch_size = value.size(0)
        query = self.query_proj(query).contiguous().view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).contiguous().view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.value_proj(value).contiguous().view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        position_emb=self.pos_proj(position_emb).view(batch_size, -1, self.num_heads, self.d_head)

        content_score = torch.matmul((query+self.u_bias).transpose(1,2),key.transpose(2,3))
        pos_score = torch.matmul((query+self.v_bias).transpose(1,2),position_emb.permute(0,2,3,1))
        pos_score = self.relative_shift(pos_score)
        score = (content_score+pos_score)/self.sqrt_dim

        if mask is not None:
            mask = mask.unsqueeze(1)
            score.masked_fill_(mask, -1e9)
        
        atten = F.softmax(score,-1)
        atten = self.dropout(atten)

        context = torch.matmul(atten,value).transpose(1,2)
        context = context.contiguous().view(batch_size, -1 , self.d_model)

        return self.out_proj(context)

    def relative_shift(self, pos_score):
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)

        return pos_score

class MHSA(nn.Module):
    def __init__(self, d_model, num_heads, dropout_p = 0.1):
        super(MHSA,self).__init__()
        self.PE=position_encoding(d_model)
        self.attention = Relative_position_encoding_attention(d_model, num_heads, dropout_p)

    def forward(self,x, mask=None):
        batch_size, seq_length , _ =x.size()
        pe=self.PE(seq_length)
        pe= pe.repeat(batch_size, 1, 1)
        x=self.attention(x, x, x, position_emb = pe, mask = mask)
        return x
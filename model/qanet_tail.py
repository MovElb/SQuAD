import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def norm(x, eps=1e-6):
    mean = x.mean(-1, keepdim=True)
    std = x.std(-1, keepdim=True)
    return (x - mean) / (std + eps)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kern_sz, dim=1, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        if dim == 1:
            self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=kern_sz, 
                                            groups=in_ch, padding=kern_sz // 2, bias=bias)
            self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, 
                                            kernel_size=1, padding=0, bias=bias)
        elif dim == 2:
            self.depthwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=kern_sz, 
                                            groups=in_ch, padding=kern_sz // 2, bias=bias)
            self.pointwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, 
                                            kernel_size=1, padding=0, bias=bias)
        else:
            raise Exception("Wrong dimension for Depthwise Separable Convolution!")

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))
    
    
class Highway(nn.Module):
    def __init__(self, n_layers, dim):
        super(Highway, self).__init__()
        self.n = n_layers
        self.linear = nn.ModuleList([nn.Linear(dim, dim).to(device) for _ in range(self.n)])
        self.gate = nn.ModuleList([nn.Linear(dim, dim).to(device) for _ in range(self.n)])
        self.relu = nn.ReLU()

    def forward(self, x):
        for i in range(self.n):
            gate = F.sigmoid(self.gate[i](x))
            nonlinear = self.relu(self.linear[i](x))
            x = gate * nonlinear + (1 - gate) * x
        return x
    
    
class ScaledDotProductAttention(nn.Module):
    def __init__(self, size, p=0.1):
        super(ScaledDotProductAttention, self).__init__()

        self.scaling = 1 / (np.sqrt(size))
        self.dropout = nn.Dropout(p)
        
    
    def forward(self, q, k, v, mask=None):
        attention = torch.bmm(q, k.transpose(1, 2)) * self.scaling
        
        if mask is not None:
            attention.data.masked_fill_(mask, -float('inf'))
            
        attention = F.softmax(attention, dim=2)
        return torch.bmm(self.dropout(attention), v)
    
    
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, h_size, k_size, v_size, p=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_heads = n_heads
        self.h_size = h_size

        [self.q_proj, self.k_proj, self.v_proj] = [nn.Parameter(torch.empty(n_heads, h_size, size, device=device))
                                                   for size in [k_size, k_size, v_size]]
        for param in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.xavier_normal_(param.data)

        self.attention = ScaledDotProductAttention(k_size, p)

        self.out = nn.Linear(n_heads * v_size, h_size).to(device)
        self.layer_norm = nn.LayerNorm(h_size).to(device)

        self.dropout = nn.Dropout(p)
        
        
    def repeat_n_heads(self, input):
        return input.repeat(self.n_heads, 1, 1).view(self.n_heads, -1, self.h_size)
    
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q_len = q.size(1)
        seq_len = k.size(1)

        residual = q

        q, k, v = [self.repeat_n_heads(var) for var in [q, k, v]]

        q = self.proj_heads(q, self.q_proj, q_len)
        k = self.proj_heads(k, self.k_proj, seq_len)
        v = self.proj_heads(v, self.v_proj, seq_len)

        if mask is not None:
            mask = mask.repeat(self.n_heads, 1, 1)

        result = self.attention(q, k, v, mask)
        result = torch.split(result, batch_size, dim=0)
        result = torch.cat(result, dim=-1)

        result = self.out(result)
        result = self.dropout(result)

        return self.layer_norm(result + residual)

    
    @staticmethod
    def proj_heads(input, projection, len):

        proj_size = projection.size(2)
        return torch.bmm(input, projection).view(-1, len, proj_size)


class SelfAttention(nn.Module):
    def __init__(self, n_heads, input_dim):
        super(SelfAttention, self).__init__()
        self.d_k = input_dim // n_heads
        self.multihead = MultiHeadAttention(n_heads, input_dim, self.d_k, self.d_k)
        
        
    def forward(self, context, questions, mask=None):
        return self.multihead(context, questions, questions, mask)
    
    
class Embedding(nn.Module):
    def __init__(self, char_emb_size, d_model, kernel_sz, n_highway, dropout_c=0.05, dropout_w=0.1):
        super(Embedding, self).__init__()
        self.drop_c = nn.Dropout(p=dropout_c)
        self.conv2d = DepthwiseSeparableConv(char_emb_size, d_model, kernel_sz, dim=2, bias=True)
        self.relu = nn.ReLU()
        self.conv1d = DepthwiseSeparableConv(word_emb_size + d_model, d_model, kernel_sz, bias=True)
        self.drop_w = nn.Dropout(p=dropout_w)
        self.high = Highway(n_highway, d_model)

    def forward(self, ch_emb, wd_emb):
        wd_emb = wd_emb.transpose(1, 2)
        ch_emb = ch_emb.permute(0, 3, 1, 2)
        
        wd_emb = self.drop_w(wd_emb)
        ch_emb = self.drop_c(ch_emb)
        
        ch_emb = self.conv2d(ch_emb)
        ch_emb = self.relu(ch_emb)
        
        ch_emb, _ = torch.max(ch_emb, dim=3) 
        wd_ch = torch.cat([wd_emb, ch_emb], dim=1)
        
        wd_ch = self.conv1d(wd_ch).transpose(1, 2)
        wd_ch = self.highway(wd_ch)
        return wd_ch
    
    
class EncoderBlock(nn.Module):
    def __init__(self, batch_size, d_model, conv_num, n_heads, kern_sz, p=0.1):
        super(EncoderBlock, self).__init__()
        self.convs = nn.ModuleList([DepthwiseSeparableConv(d_model, d_model, kern_sz) for _ in range(conv_num)])
        self.self_att = SelfAttention(n_heads, d_model)
        self.W = nn.Linear(d_model, d_model).to(device)
        self.relu = nn.ReLU()
        self.dropout = p

    def forward(self, x):
        out = self.pos_encoding(x)
        out = res = out.transpose(1, 2)
        
        for i, conv in enumerate(self.convs):
            out = norm(out)
            out = conv(out)
            out = self.relu(out)
            out = res + out
            if (i + 1) % 2 == 0:
                out = F.dropout(out, p=self.dropout, training=self.training)
            res = out
            
        res = out = norm(out).transpose(1, 2)
        out = self.self_att(out, out)
        out = res + out
        out = F.dropout(out, p=self.dropout, training=self.training)
        res = out
        out = norm(out)
        out = self.W(out)
        out = self.relu(out)
        out = res + out
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out
    
    
    def pos_encoding(self, x):
        _, max_len, model_dim = x.shape
        encoding = np.array([
            [pos / np.power(10000, 2 * i / model_dim) for i in range(model_dim)]
            if pos != 0 else np.zeros(model_dim) for pos in range(max_len)])

        encoding[1:, 0::2] = np.sin(encoding[1:, 0::2])
        encoding[1:, 1::2] = np.cos(encoding[1:, 1::2])
        return x + torch.from_numpy(encoding).float().to(device)
    
    
class ContextQueryAttention(nn.Module):
    def __init__(self, d_model, p=0.1):
        super(ContextQueryAttention, self).__init__()
        self.W = nn.Linear(3 * d_model, 1, bias=False).to(device)
        self.dropout = nn.Dropout(p)

    def forward(self, C, Q):
        (_, n, _) = C.size()
        (_, m, _) = Q.size()
        
        C_ = C.transpose(0, 1)
        Q_ = Q.transpose(0, 1)
        S = torch.einsum('cbd,kbd->ckdb', [C_, Q_]).transpose(2, 3)
        S = torch.cat([C_.unsqueeze(1).repeat(1, m, 1, 1), 
                       Q_.unsqueeze(0).repeat(n, 1, 1, 1),
                       S], 
                      dim=-1)

        S = self.W(S).squeeze(dim=-1).permute(2, 0, 1)
        
        S_ = F.softmax(S, dim=2)
        S__ = F.softmax(S, dim=1)
        S_mul_12 = torch.bmm(S_, S__.transpose(1, 2))
        A = torch.bmm(S_, Q)
        B = torch.bmm(S_mul_12, C)
        output = torch.cat([C, A, C * A, C * B], dim=-1)
        output = self.dropout(output)
        return output
    
    
class AnswerStartEnd(nn.Module):
    def __init__(self, d_model):
        super(AnswerStartEnd, self).__init__()
        self.W0 = nn.Linear(2 * d_model, 1).to(device)
        self.W1 = nn.Linear(2 * d_model, 1).to(device)
        
    
    def forward(self, M0, M1, M2):
        cat_M01 = torch.cat([M0, M1], dim=-1)
        cat_M02 = torch.cat([M0, M2], dim=-1)
        Y0 = self.W0(cat_M01).squeeze(dim=-1)
        Y1 = self.W1(cat_M02).squeeze(dim=-1)
        
        if self.training:
            p1 = F.log_softmax(Y0, dim=1)
            p2 = F.log_softmax(Y1, dim=1)
        else:
            p1 = F.softmax(Y0, dim=1)
            p2 = F.softmax(Y1, dim=1)
        return p1, p2
    
    
    
class QANet_tail(nn.Module):
    def __init__(self, opt):
        super(QANet_tail, self).__init__()
        inp_sz = 2 * opt['hidden_size']
        if opt['concat_rnn_layers']:
            inp_sz *= opt['doc_layers']
            
        self.num_blocks = opt['num_blocks']
        
        self.resize_cont = DepthwiseSeparableConv(inp_sz, opt['hidden_size'], opt['emb_enc_kersize'])
        self.resize_ques = DepthwiseSeparableConv(inp_sz, opt['hidden_size'], opt['emb_enc_kersize'])
        
        self.cont_emb_enc = EncoderBlock(opt['batch_size'], opt['hidden_size'], 
                                         opt['emb_enc_convs'], opt['n_heads'], opt['emb_enc_kersize'])
        self.ques_emb_enc = EncoderBlock(opt['batch_size'], opt['hidden_size'], 
                                         opt['emb_enc_convs'], opt['n_heads'], opt['emb_enc_kersize'])
        self.mod_enc = EncoderBlock(opt['batch_size'], opt['hidden_size'], 
                                    opt['mod_enc_convs'], opt['n_heads'], opt['mod_enc_kersize'])
        self.cont_quer_atten = ContextQueryAttention(opt['hidden_size'])
        self.downscale_cq = DepthwiseSeparableConv(4 * opt['hidden_size'], opt['hidden_size'], 
                                                opt['emb_enc_kersize'])
        self.answ = AnswerStartEnd(opt['hidden_size'])
        
    def forward(self, cont_hidden, cont_mask, ques_hidden, ques_mask):
        cont_hidden = self.resize_cont(cont_hidden.transpose(1, 2)).transpose(1, 2)
        ques_hidden = self.resize_ques(ques_hidden.transpose(1, 2)).transpose(1, 2)
        
        cont_enc = self.cont_emb_enc(cont_hidden)
        ques_enc = self.ques_emb_enc(ques_hidden)
        
        cont_ques_atten = self.cont_quer_atten(cont_enc, ques_enc).transpose(1, 2)
        cont_ques_atten = self.downscale_cq(cont_ques_atten).transpose(1, 2)
        
        
        M0 = cont_ques_atten
        
        p1, p2 = self.answ(M0, M0, M0)
        return p1, p2

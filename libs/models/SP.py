import torch
import torch.nn as nn
import numpy as np

from .graph.graph import Graph
from .graph.tools import k_adjacency, normalize_adjacency_matrix, get_adjacency_matrix


class MultiScale_GraphConv(nn.Module):
    def __init__(self,
                 num_scales,  # 13
                 in_channels,
                 out_channels,
                 dataset,
                 disentangled_agg=True,
                 use_mask=True,
                 dropout=0,
                 activation='relu'):
        super().__init__()

        self.graph = Graph(labeling_mode='spatial', layout=dataset)  # get graph
        neighbor = self.graph.neighbor  # in and out
        self.num_scales = num_scales  # 13

        if dataset == 'LARA':
            A_binary = get_adjacency_matrix(neighbor, 19)
        elif dataset == 'TCG-15':
            A_binary = get_adjacency_matrix(neighbor, 17)
        else:
            A_binary = get_adjacency_matrix(neighbor, 25)

        if disentangled_agg:  # 13-hop
            A_powers = [k_adjacency(A_binary, k, with_self=True) for k in range(num_scales)]  #  13 （V，V）
            A_powers = np.concatenate([normalize_adjacency_matrix(g) for g in A_powers])  # （13×V，V）
        else:
            A_powers = [A_binary + np.eye(len(A_binary)) for k in range(num_scales)]
            A_powers = [normalize_adjacency_matrix(g) for g in A_powers]
            A_powers = [np.linalg.matrix_power(g, k) for k, g in enumerate(A_powers)]
            A_powers = np.concatenate(A_powers)

        self.A_powers = torch.Tensor(A_powers)
        self.A_binary = torch.Tensor(A_binary)
        self.use_mask = use_mask
        if use_mask:
            # NOTE: the inclusion of residual mask appears to slow down training noticeably
            self.A_res = nn.init.uniform_(nn.Parameter(torch.Tensor(self.A_powers.shape)), -1e-6, 1e-6)  # B

        self.mlp = MLP(in_channels * num_scales, [out_channels], dropout=dropout, activation=activation)

        self.CTRGCN = CTRGC(out_channels, out_channels)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, joint_graph):
        x = x.transpose(2, 3)  # n,c,v,t->n,c,t,v
        N, C, T, V = x.shape
        self.A_powers = self.A_powers.to(x.device)  # (13*v,v)
        self.A_binary = self.A_binary.to(x.device)  # (v,v)
        A = self.A_powers.to(x.dtype)  # TO float32
        if self.use_mask:
            A = A + self.A_res.to(x.dtype)  #add B

        support = torch.einsum('vu,nctu->nctv', A, x)  # （n,c,t,13*v）
        support = support.view(N, C, T, self.num_scales, V)  # （n,c,t,13,v）
        support = support.permute(0, 3, 1, 2, 4).contiguous().view(N, self.num_scales * C, T, V)  ##（n,13*c,t,v）
        x = self.mlp(support)  #（n,c,t,v）

        out = self.CTRGCN(x, joint_graph, self.A_binary, self.alpha, self.beta)
        out = self.bn(out)
        out = out + x
        out = self.relu(out)

        return out


class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels != 64:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        # self.soft = nn.Softmax(dim = -2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, joint_graph, A=None, alpha=1, beta=1):
        x1, x2, x3 = self.conv1(x), self.conv2(x), self.conv3(x)

        # temporal
        xt = self.tanh(x1.mean(1).unsqueeze(-1) - x2.mean(1).unsqueeze(-2))  # N,T,V,V
        xt = xt * beta + joint_graph.unsqueeze(0).unsqueeze(0)
        # xt = xt * beta + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,T,V,V
        xt = torch.einsum('ntuv,nctv->nctu', xt, x3)

        xc = self.tanh(x1.mean(-2).unsqueeze(-1) - x2.mean(-2).unsqueeze(-2))  # N,C,V,V
        xc = self.conv4(xc) * alpha + joint_graph.unsqueeze(0).unsqueeze(0)
        # xc = self.conv4(xc) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V
        xc = torch.einsum('ncuv,nctv->nctu', xc, x3)

        return xt + xc


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu', dropout=0):
        super().__init__()
        channels = [in_channels] + out_channels #[12*13,64]
        self.layers = nn.ModuleList()
        for i in range(1, len(channels)):
            if dropout > 0.001:
                self.layers.append(nn.Dropout(p=dropout))
            self.layers.append(nn.Conv2d(channels[i-1], channels[i], kernel_size=1))
            self.layers.append(nn.BatchNorm2d(channels[i]))
            self.layers.append(activation_factory(activation)) #relu

    def forward(self, x):
        # Input shape: (N,C,T,V)
        for layer in self.layers:
            x = layer(x) #CONV+batchnorm+relu
        return x

def activation_factory(name, inplace=True):
    if name == 'relu':
        return nn.ReLU(inplace=inplace)
    elif name == 'leakyrelu':
        return nn.LeakyReLU(0.2, inplace=inplace)
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'linear' or name is None:
        return nn.Identity()
    else:
        raise ValueError('Not supported activation:', name)
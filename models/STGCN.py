import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from models.AbstractTrafficStateModel import AbstractTrafficStateModel


class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in, c_out, 1)  # filter=(1,1)

    def forward(self, x):  # x: (batch_size, feature_dim(c_in), input_length, num_nodes)
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            return F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in, 0, 0])
        return x  # return: (batch_size, c_out, input_length-1+1, num_nodes-1+1)


class TemporalConvLayer(nn.Module):
    def __init__(self, kt, c_in, c_out, act="relu"):
        super(TemporalConvLayer, self).__init__()
        self.kt = kt
        self.act = act
        self.c_out = c_out
        self.align = Align(c_in, c_out)
        if self.act == "GLU":
            self.conv = nn.Conv2d(c_in, c_out * 2, (kt, 1), 1)
        else:
            self.conv = nn.Conv2d(c_in, c_out, (kt, 1), 1)

    def forward(self, x):
        """
        :param x: (batch_size, feature_dim(c_in), input_length, num_nodes)
        :return: (batch_size, c_out, input_length-kt+1, num_nodes)
        """
        x_in = self.align(x)[:, :, self.kt - 1:, :]  # (batch_size, c_out, input_length-kt+1, num_nodes)
        if self.act == "GLU":
            # x: (batch_size, c_in, input_length, num_nodes)
            x_conv = self.conv(x)
            # x_conv: (batch_size, c_out * 2, input_length-kt+1, num_nodes)  [P Q]
            return (x_conv[:, :self.c_out, :, :] + x_in) * torch.sigmoid(x_conv[:, self.c_out:, :, :])
            # return P * sigmoid(Q) shape: (batch_size, c_out, input_length-kt+1, num_nodes)
        if self.act == "sigmoid":
            return torch.sigmoid(self.conv(x) + x_in)  # residual connection
        return torch.relu(self.conv(x) + x_in)  # residual connection


class SpatioConvLayer(nn.Module):
    def __init__(self, ks, c_in, c_out, lk, device):
        super(SpatioConvLayer, self).__init__()
        self.Lk = lk
        self.theta = nn.Parameter(torch.FloatTensor(c_in, c_out, ks).to(device))  # kernel: C_in*C_out*ks
        self.b = nn.Parameter(torch.FloatTensor(1, c_out, 1, 1).to(device))
        self.align = Align(c_in, c_out)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def forward(self, x):
        # Lk: (Ks, num_nodes, num_nodes)
        # x:  (batch_size, c_in, input_length, num_nodes)
        # x_c: (batch_size, c_in, input_length, Ks, num_nodes)
        # theta: (c_in, c_out, Ks)
        # x_gc: (batch_size, c_out, input_length, num_nodes)
        x_c = torch.einsum("knm,bitm->bitkn", self.Lk, x)  # delete num_nodes(n)
        x_gc = torch.einsum("iok,bitkn->botn", self.theta, x_c) + self.b  # delete Ks(k) c_in(i)
        x_in = self.align(x)  # (batch_size, c_out, input_length, num_nodes)
        return torch.relu(x_gc + x_in)  # residual connection


class STConvBlock(nn.Module):
    def __init__(self, ks, kt, n, c, p, lk, device):
        super(STConvBlock, self).__init__()
        self.tconv1 = TemporalConvLayer(kt, c[0], c[1], "GLU")
        self.sconv = SpatioConvLayer(ks, c[1], c[1], lk, device)
        self.tconv2 = TemporalConvLayer(kt, c[1], c[2])
        self.ln = nn.LayerNorm([n, c[2]])
        self.dropout = nn.Dropout(p)

    def forward(self, x):  # x: (batch_size, feature_dim/c[0], input_length, num_nodes)
        x_t1 = self.tconv1(x)    # (batch_size, c[1], input_length-kt+1, num_nodes)
        x_s = self.sconv(x_t1)   # (batch_size, c[1], input_length-kt+1, num_nodes)
        x_t2 = self.tconv2(x_s)  # (batch_size, c[2], input_length-kt+1-kt+1, num_nodes)
        x_ln = self.ln(x_t2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.dropout(x_ln)


class FullyConvLayer(nn.Module):
    def __init__(self, c, out_dim):
        super(FullyConvLayer, self).__init__()
        self.conv = nn.Conv2d(c, out_dim, 1)  # c,self.output_dim,1

    def forward(self, x):
        return self.conv(x)


class OutputLayer(nn.Module):
    def __init__(self, c, t, n):
        super(OutputLayer, self).__init__()
        self.tconv1 = TemporalConvLayer(t, c, c, "GLU")
        self.ln = nn.LayerNorm([n, c])
        self.tconv2 = TemporalConvLayer(1, c, c, "GLU")  # kernel=1*1

    def forward(self, x):
        # (batch_size, input_dim(c), T, num_nodes)
        x_t1 = self.tconv1(x)
        # (batch_size, input_dim(c), 1, num_nodes)
        x_ln = self.ln(x_t1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # (batch_size, input_dim(c), 1, num_nodes)
        x_t2 = self.tconv2(x_ln).squeeze(2)
        # (batch_size, input_dim(c), num_nodes)
        return x_t2


class STGCN(AbstractTrafficStateModel):
    def __init__(self, config):
        super().__init__(config)

        self.st_conv1 = STConvBlock(config.Ks, config.Kt, config.num_nodes,
                                    config.blocks[0], config.keep_prob, config.Lk, config.device)
        self.st_conv2 = STConvBlock(config.Ks, config.Kt, config.num_nodes,
                                    config.blocks[1], config.keep_prob, config.Lk, config.device)
        self.output = OutputLayer(config.blocks[1][2], config.n_his - len(config.blocks) * 2
                                  * (config.Kt - 1), config.num_nodes)

    def encoder(self, x):
        # x: (batch_size, num_nodes, input_length, feature_dim)
        x = x.permute(0, 3, 2, 1)  # (batch_size, feature_dim, input_length, num_nodes)
        x_st1 = self.st_conv1(x)   # (batch_size, c[2](64), input_length-kt+1-kt+1, num_nodes)
        x_st2 = self.st_conv2(x_st1)  # (batch_size, c[2](128), input_length-kt+1-kt+1-kt+1-kt+1, num_nodes)
        outputs = self.output(x_st2)  # (batch_size, output_length(12), num_nodes)
        outputs = outputs.permute(0, 2, 1)  # (batch_size, num_nodes, output_dim(128))
        return outputs

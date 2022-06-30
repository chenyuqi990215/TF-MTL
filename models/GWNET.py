import torch
import torch.nn as nn
import torch.nn.functional as F
from models.AbstractTrafficStateModel import AbstractTrafficStateModel
import numpy as np
from utils.graph import asym_adj

class NConv(nn.Module):
    def __init__(self):
        super(NConv, self).__init__()

    def forward(self, x, adj):
        x = torch.einsum('ncvl,vw->ncwl', (x, adj))
        return x.contiguous()


class Linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(Linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class GCN(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(GCN, self).__init__()
        self.nconv = NConv()
        c_in = (order*support_len+1)*c_in
        self.mlp = Linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class GWNET(AbstractTrafficStateModel):
    def __init__(self,  config):
        super().__init__(config)
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=self.config.feature_dim,
                                    out_channels=self.config.residual_channels,
                                    kernel_size=(1, 1))

        self.adj_mx = [asym_adj(self.config.adj_mx), asym_adj(np.transpose(self.config.adj_mx))]
        self.supports = [torch.tensor(i).to(self.config.device) for i in self.adj_mx]
        if self.config.randomadj:
            self.aptinit = None
        else:
            self.aptinit = self.supports[0]
        if self.config.aptonly:
            self.supports = None

        receptive_field = self.config.output_dim

        self.supports_len = 0
        if self.supports is not None:
            self.supports_len += len(self.supports)

        if self.config.gcn_bool and self.config.addaptadj:
            if self.aptinit is None:
                if self.supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(self.config.num_nodes, 10).to(self.config.device),
                                             requires_grad=True).to(self.config.device)
                self.nodevec2 = nn.Parameter(torch.randn(10, self.config.num_nodes).to(self.config.device),
                                             requires_grad=True).to(self.config.device)
                self.supports_len += 1
            else:
                if self.supports is None:
                    self.supports = []
                m, p, n = torch.svd(self.aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(self.config.device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(self.config.device)
                self.supports_len += 1

        for b in range(self.config.blocks):
            additional_scope = self.config.kernel_size - 1
            new_dilation = 1
            for i in range(self.config.layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=self.config.residual_channels,
                                                   out_channels=self.config.dilation_channels,
                                                   kernel_size=(1, self.config.kernel_size), dilation=new_dilation))
                # print(self.filter_convs[-1])
                self.gate_convs.append(nn.Conv1d(in_channels=self.config.residual_channels,
                                                 out_channels=self.config.dilation_channels,
                                                 kernel_size=(1, self.config.kernel_size), dilation=new_dilation))
                # print(self.gate_convs[-1])
                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=self.config.dilation_channels,
                                                     out_channels=self.config.residual_channels,
                                                     kernel_size=(1, 1)))
                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=self.config.dilation_channels,
                                                 out_channels=self.config.skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(self.config.residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.config.gcn_bool:
                    self.gconv.append(GCN(self.config.dilation_channels, self.config.residual_channels,
                                          self.config.dropout, support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=self.config.skip_channels,
                                    out_channels=self.config.end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=self.config.end_channels,
                                    out_channels=self.config.output_window,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.receptive_field = receptive_field

    def encoder(self, x):
        inputs = x.transpose(1, 2)
        inputs = inputs.transpose(1, 3)  # (batch_size, feature_dim, num_nodes, input_window)
        inputs = nn.functional.pad(inputs, (1, 0, 0, 0))  # (batch_size, feature_dim, num_nodes, input_window+1)

        in_len = inputs.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(inputs, (self.receptive_field-in_len, 0, 0, 0))
        else:
            x = inputs
        x = self.start_conv(x)  # (batch_size, residual_channels, num_nodes, self.receptive_field)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.config.gcn_bool and self.config.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.config.blocks * self.config.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*
            # (dilation, init_dilation) = self.dilations[i]
            # residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # (batch_size, residual_channels, num_nodes, self.receptive_field)
            # dilated convolution
            filter = self.filter_convs[i](residual)
            # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
            gate = torch.sigmoid(gate)
            x = filter * gate
            # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
            # parametrized skip connection
            s = x
            # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
            s = self.skip_convs[i](s)
            # (batch_size, skip_channels, num_nodes, receptive_field-kernel_size+1)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except(Exception):
                skip = 0
            skip = s + skip
            # (batch_size, skip_channels, num_nodes, receptive_field-kernel_size+1)
            if self.config.gcn_bool and self.supports is not None:
                # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
                if self.config.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x, self.supports)
                # (batch_size, residual_channels, num_nodes, receptive_field-kernel_size+1)
            else:
                # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
                x = self.residual_convs[i](x)
                # (batch_size, residual_channels, num_nodes, receptive_field-kernel_size+1)
            # residual: (batch_size, residual_channels, num_nodes, self.receptive_field)
            x = x + residual[:, :, :, -x.size(3):]
            # (batch_size, residual_channels, num_nodes, receptive_field-kernel_size+1)
            x = self.bn[i](x)
        x = F.relu(skip)
        # (batch_size, skip_channels, num_nodes, self.output_dim)
        x = F.relu(self.end_conv_1(x))
        # (batch_size, end_channels, num_nodes, self.output_dim)
        x = self.end_conv_2(x).squeeze(-1).transpose(-1, -2)
        # (batch_size, num_nodes, output_window)
        return x
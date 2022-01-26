from torch import nn
import torch
import torch.nn.functional as F
import torch.nn.init as init
import math


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


class SpatioConvLayer(nn.Module):
    def __init__(self, ks, c_in, c_out, lk, device):
        super(SpatioConvLayer, self).__init__()
        self.Lk = lk
        self.theta = nn.Parameter(torch.FloatTensor(c_in, c_out, ks).to(device))  # kernel: C_in*C_out*ks
        self.b = nn.Parameter(torch.FloatTensor(1, c_out, 1).to(device))
        self.align = Align(c_in, c_out)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def forward(self, x):
        x = x.transpose(-1, -2)
        x_c = torch.einsum("knm,bim->bikn", self.Lk, x)  # delete num_nodes(n)
        x_gc = torch.einsum("iok,bikn->bon", self.theta, x_c) + self.b  # delete Ks(k) c_in(i)
        x_in = self.align(x.unsqueeze(1)).squeeze()  # (batch_size, c_out, num_nodes)
        return torch.relu(x_gc + x_in).transpose(-1, -2)  # residual connection


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    图注意力层
    """

    def __init__(self, in_features, out_features, dropout=0.2, alpha=0.01):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features  # 节点表示向量的输入特征维度
        self.out_features = out_features  # 节点表示向量的输出特征维度
        self.dropout = dropout  # dropout参数
        self.alpha = alpha  # leakyrelu激活的参数

        # 定义可训练参数，即论文中的W和a
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # xavier初始化
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  # xavier初始化

        # 定义leakyrelu激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp):
        """
        implement of global graph attention layer
        inp: input_fea [N, in_features]  in_features表示节点的输入特征向量元素个数
        """
        h = torch.matmul(inp, self.W)  # [B, N, out_features]
        N = h.size()[1]  # N 图的节点数
        B = h.size()[0]  # B batch_size

        a_input = torch.cat([h.repeat(1, 1, N).view(B, N * N, -1), h.repeat(1, N, 1)], dim=-1).view(B, N, N, 2 * self.out_features)
        # [B, N, N, 2*out_features]
        attention = self.leakyrelu(torch.matmul(a_input, self.a).squeeze())
        # [B, N, N, 1] => [B, N, N] 图注意力的相关系数（未归一化）

        attention = F.softmax(attention, dim=-1)  # softmax形状保持不变 [N, N]，得到归一化的注意力权重！
        attention = F.dropout(attention, self.dropout)  # dropout，防止过拟合
        h_prime = torch.matmul(attention, h)  # [B, N, N].[B, N, out_features] => [B, N, out_features]

        return F.elu(h_prime)

def get_gnn_model(config):
    if 'GAT' in config.gnn:
        return GraphAttentionLayer(config.out_dim, config.out_dim)
    elif 'GCN' in config.gnn:
        return SpatioConvLayer(config.Ks, config.out_dim, config.out_dim, config.Lk, config.device)
    else:
        raise NotImplementedError


class AGCLSTM(nn.Module):
    '''
    Attention Enhanced Graph Convolutional LSTM Network
    ref: https://arxiv.org/abs/1902.09130
    '''
    def __init__(self, input_sz, hidden_sz, config):
        super(AGCLSTM, self).__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        self.num_nodes = config.num_nodes

        # i_t
        self.W_i = get_gnn_model(config)
        self.U_i = get_gnn_model(config)
        self.b_i = nn.Parameter(torch.Tensor(hidden_sz))

        # f_t
        self.W_f = get_gnn_model(config)
        self.U_f = get_gnn_model(config)
        self.b_f = nn.Parameter(torch.Tensor(hidden_sz))

        # c_t
        self.W_c = get_gnn_model(config)
        self.U_c = get_gnn_model(config)
        self.b_c = nn.Parameter(torch.Tensor(hidden_sz))

        # o_t
        self.W_o = get_gnn_model(config)
        self.U_o = get_gnn_model(config)
        self.b_o = nn.Parameter(torch.Tensor(hidden_sz))

        self.pool = nn.Linear(self.num_nodes, 1)
        self.lin_h = nn.Linear(self.hidden_size, self.hidden_size)
        self.lin_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.lin_u = nn.Linear(self.hidden_size, 1)
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def gnn_forward(self, x, model):
        sz = x.size()
        x = x.reshape(-1, self.num_nodes, self.hidden_size)
        return model(x).reshape(*sz)

    def f_att(self, h):
        '''
        :param h: [batch_size * num_nodes, hidden_size]
        :return:
        '''
        sz = h.size()
        h = h.reshape(-1, self.num_nodes, self.hidden_size)

        q = F.relu(self.pool(h.transpose(-1, -2)).squeeze(-1)).unsqueeze(-2)  # [batch_size, 1, hidden_size]
        q = q.repeat(1, self.num_nodes, 1)  # [batch_size, num_nodes, hidden_size]

        a = F.sigmoid(self.lin_u(F.tanh(self.lin_h(h) + self.lin_q(q))))
        return (h * a).reshape(*sz)


    def forward(self,
                x,
                init_states=None):

        """
        assumes x.shape represents (batch_size, sequence_size, input_size)
        """
        bs, seq_sz, _ = x.size()
        hidden_seq = []

        if init_states is None:
            h_t, c_t = (
                torch.zeros(bs, self.hidden_size).to(x.device),
                torch.zeros(bs, self.hidden_size).to(x.device),
            )
        else:
            h_t, c_t = init_states

        for t in range(seq_sz):
            x_t = x[:, t, :]

            i_t = torch.sigmoid(self.gnn_forward(x_t, self.W_i) + self.gnn_forward(h_t, self.U_i) + self.b_i)
            f_t = torch.sigmoid(self.gnn_forward(x_t, self.W_f) + self.gnn_forward(h_t, self.U_f) + self.b_f)
            g_t = torch.tanh(self.gnn_forward(x_t, self.W_c) + self.gnn_forward(h_t, self.U_c) + self.b_c)
            o_t = torch.sigmoid(self.gnn_forward(x_t, self.W_o) + self.gnn_forward(h_t, self.U_o) + self.b_o)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            h_t = self.f_att(h_t) + h_t

            hidden_seq.append(h_t)

        # reshape hidden_seq p/ retornar
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)
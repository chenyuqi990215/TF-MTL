import torch
from torch import nn
import torch.nn.functional as F
import sys
sys.path.append('../')
from models.HetETA_utils.graph_func import HetCheb
from models.HetETA_utils.build_graph import HetroRoadGraph
from models.HetETA_utils.map import RoadNetworkMap

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
    def __init__(self, kt, c_in, c_out, act="GTU"):
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


class HetETAConvBlock(nn.Module):
    def __init__(self, kt, c, p, lk, eh, et, device):
        super(HetETAConvBlock, self).__init__()
        self.tconv1 = TemporalConvLayer(kt, c[0], c[1], "GLU")
        self.sconvh = HetCheb(c[1], c[1], lk, eh, device=device)
        self.sconvt = HetCheb(c[1], c[1], lk, et, device=device)
        self.dropout = nn.Dropout(p)

    def forward(self, x, hg, tg):  # x: (batch_size, feature_dim/c[0], input_length, num_nodes)
        x_t1 = self.tconv1(x)    # (batch_size, c[1], input_length-kt+1, num_nodes)
        x_s = []
        for b in range(x_t1.size(0)):
            for t in range(x_t1.size(2)):
                x_in = x_t1[b, :, t, :].transpose(0, 1)
                x_in = {'segment': x_in}
                x_out = self.sconvh(hg, x_in)
                x_s.append(x_out['segment'].unsqueeze(0))
        x_s = torch.cat(x_s, dim=0).reshape(x_t1.size(0), x_t1.size(2), x_t1.size(3), -1)
        x_s = x_s.permute(0, 3, 1, 2)
        x_t = []
        for b in range(x_t1.size(0)):
            for t in range(x_t1.size(2)):
                x_in = x_t1[b, :, t, :].transpose(0, 1)
                x_in = {'segment': x_in}
                x_out = self.sconvt(tg, x_in)
                x_t.append(x_out['segment'].unsqueeze(0))
        x_t = torch.cat(x_t, dim=0).reshape(x_t1.size(0), x_t1.size(2), x_t1.size(3), -1)
        x_t = x_t.permute(0, 3, 1, 2)
        x_ln = torch.cat((x_s, x_t), dim=1)  # (batch_size, 2*c[1], input_length-kt+1, num_nodes)
        return self.dropout(x_ln)

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class HetFlow(nn.Module):
    def __init__(self, hid_dim, map_root, traj_path, zone_range, device, n_input=6, feature_dim=1, unit_length=50):
        super(HetFlow, self).__init__()
        self.hetro_road = HetroRoadGraph(map_root, zone_range, traj_path, unit_length)
        self.hg = self.hetro_road.road_hg.to(device)
        self.tg = self.hetro_road.traj_hg.to(device)
        config = AttrDict()
        config.Kt = 3
        config.Lk = 3
        config.device = device
        config.blocks = [[feature_dim, 8], [2 * 8, 11]]
        config.keep_prob = 0.5


        self.st_conv1 = HetETAConvBlock(config.Kt, config.blocks[0], config.keep_prob, config.Lk,
                                        self.hg.etypes, self.tg.etypes, config.device)
        self.st_conv2 = HetETAConvBlock(config.Kt, config.blocks[1], config.keep_prob, config.Lk,
                                        self.hg.etypes, self.tg.etypes, config.device)
        self.out = nn.Linear(11 * 2 * (n_input - 2 * (config.Kt - 1)), hid_dim)

    def encoder(self, x):
        # x: (batch_size, num_nodes, input_length, feature_dim)
        x = x.permute(0, 3, 2, 1)  # (batch_size, feature_dim, input_length, num_nodes)
        print(x.size())
        x_st1 = self.st_conv1(x, self.hg, self.tg)   # (batch_size, c[2](64), input_length-kt+1, num_nodes)
        print(x_st1.size())
        x_st2 = self.st_conv2(x_st1, self.hg, self.tg)  # (batch_size, c[2](128), input_length-kt+1-kt+1, num_nodes)
        x_st2 = x_st2.permute(0, 3, 1, 2)
        x_st2 = x_st2.reshape(x_st2.size(0), x_st2.size(1), -1)
        outputs = self.out(x_st2)  # (batch_size, num_nodes, hidden dim)
        return outputs

    def forward(self, x, traj=None, traj_len=None):
        return self.encoder(x)


class HetETA(HetFlow):
    def __init__(self, hid_dim, map_root, traj_path, zone_range, device, n_input=6, n_out=1, feature_dim=1, unit_length=50, max_speed=120):
        super(HetETA, self).__init__(hid_dim, map_root, traj_path, zone_range, device, n_input, feature_dim, unit_length)
        self.fc_last = nn.Linear(hid_dim, 1)
        self.fc_flow = nn.Linear(hid_dim, n_out * feature_dim)
        self.device = device
        self.feature_dim = feature_dim
        self.n_out = n_out
        self.max_speed = max_speed
        self.hid_dim = hid_dim
        self.rn = RoadNetworkMap(map_root, zone_range=zone_range, unit_length=unit_length)
        self.dist = [0 for _ in range(self.rn.valid_edge_cnt)]
        for rid in self.rn.valid_edge:
            self.rn.valid_edge[rid] = self.rn.edgeDis[rid]
        self.dist = torch.tensor(self.dist).to(device).reshape(-1, 1)

    def forward(self, x, traj=None, traj_len=None):
        '''
        for traffic flow task:
        :param x: [batch_size, num_nodes, n_input, feature_dim]
        :param traj: None
        :param traj_len: None
        :return: [batch_size, num_nodes, n_out, feature_dim]

        for ETA:
        :param x: [1, num_nodes, n_input, feature_dim]
        :param traj: [batch_size, max_traj_len]
        :param traj_len: [...] list of int
        :return: [batch_size]
        '''
        if traj is None:
            x = self.encoder(x)
            x = self.fc_flow(x).reshape(x.size(0), x.size(1), self.n_out, self.feature_dim)
            return x
        else:
            bs = traj.size(0)
            max_traj_len = traj.size(1)
            mask = torch.zeros(bs, max_traj_len).to(self.device)
            for i in range(bs):
                mask[i, :traj_len[i]] = 1
            fea = self.encoder(x).reshape(-1, self.hid_dim)
            traj_input = torch.index_select(fea, dim=0, index=traj.reshape(-1))
            traj_input = traj_input.reshape(bs, max_traj_len, -1)
            traj_output = self.fc_last(traj_input).squeeze(-1)

            dd = torch.index_select(self.dist, dim=0, index=traj.reshape(-1))
            dd = dd.reshape(bs, max_traj_len)

            tt = dd / (traj_output * self.max_speed + 1e-6)
            tt = (tt * mask).sum(dim=-1)
            return tt

import numpy as np

if __name__ == "__main__":
    map_root = '/nas/user/cyq/TrajectoryRecovery/roadnet/Chengdu/'
    zone_range = [30.655347, 104.039711, 30.730157, 104.127151]
    traj_path = '/nas/user/cyq/TrajectoryRecovery/train_data_final/Chengdu/valid/valid_output.txt'

    model = HetFlow(64, map_root, traj_path, zone_range, "cpu")
    x = torch.rand(1, 8781, 6, 1)
    print(model(x).size())

    model = HetETA(64, map_root, traj_path, zone_range, "cpu")
    traj = torch.randint(low=0, high=8781, size=(64, 100))
    traj_len = np.random.randint(low=0, high=100, size=(64))
    print(model(x, traj, traj_len).size())
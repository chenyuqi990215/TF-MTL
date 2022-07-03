from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import random

import networkx as nx
import numpy as np
from gensim.models import Word2Vec


class Graph():
    def __init__(self, nx_G, is_directed, p, q):
        self.G = nx_G
        self.is_directed = is_directed
        self.p = p
        self.q = q

    def node2vec_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0],
                                               alias_edges[(prev, cur)][1])]
                    walk.append(next)
            else:
                break

        return walk

    def simulate_walks(self, num_walks, walk_length):
        '''
        Repeatedly simulate random walks from each node.
        '''
        G = self.G
        walks = []
        nodes = list(G.nodes())
        for walk_iter in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

        return walks

    def get_alias_edge(self, src, dst):
        '''
        Get the alias edge setup lists for a given edge.
        '''
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]['weight'] / p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight'] / q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''
        G = self.G
        is_directed = self.is_directed

        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}

        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return


def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to
    https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)

    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


def learn_embeddings(walks, dimensions, window_size, iter):
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(
        walks, vector_size=dimensions, window=window_size, min_count=0, sg=1,
        workers=8, epochs=iter)
    return model


class Node2Vec():
    def __init__(self, args, adj_mx):
        self.cache_path = args.cache_path
        self.num_nodes = args.num_nodes
        self.D = args.dim
        self.SE_config = {'is_directed': True, 'p': 2, 'q': 1, 'num_walks': 100,
                          'walk_length': 80, 'dimensions': self.D, 'window_size': 10,
                          'iter': 1}
        self.adj_mx = adj_mx
        self._generate_SE()


    def _generate_SE(self):
        if not os.path.exists(self.cache_path):
            #   SE: [N, D]([N, K * d])
            nx_G = nx.from_numpy_matrix(self.adj_mx, create_using=nx.DiGraph())
            G = Graph(nx_G, self.SE_config['is_directed'], self.SE_config['p'], self.SE_config['q'])
            G.preprocess_transition_probs()
            walks = G.simulate_walks(self.SE_config['num_walks'], self.SE_config['walk_length'])
            model = learn_embeddings(walks, self.SE_config['dimensions'],
                                         self.SE_config['window_size'], self.SE_config['iter'])
            model.wv.save_word2vec_format(self.cache_path)

        SE = np.zeros(shape=(self.num_nodes, self.SE_config['dimensions']), dtype=np.float32)
        f = open(self.cache_path, mode='r')
        lines = f.readlines()
        for line in lines[1:]:
            temp = line.split(' ')
            index = int(temp[0])
            SE[index] = temp[1:]
        print(SE.shape)
        self.SE = SE

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def generate_se(config, cache_path):
    config.cache_path = cache_path
    node2vec = Node2Vec(config, config.adj_mx)
    config.SE = node2vec.SE
    return config

class FC(nn.Module):  # is_training: self.training
    def __init__(self, input_dims, units, activations, bn, bn_decay, device, use_bias=True):
        super(FC, self).__init__()
        self.input_dims = input_dims
        self.units = units
        self.activations = activations
        self.bn = bn
        self.bn_decay = bn_decay
        self.device = device
        self.use_bias = use_bias
        self.layers = self._init_layers()

    def _init_layers(self):
        ret = nn.Sequential()
        units, activations = self.units, self.activations
        if isinstance(units, int):
            units, activations = [units], [activations]
        elif isinstance(self.units, tuple):
            units, activations = list(units), list(activations)
        assert type(units) == list
        index = 1
        input_dims = self.input_dims
        for num_unit, activation in zip(units, activations):
            if self.use_bias:
                basic_conv2d = nn.Conv2d(input_dims, num_unit, (1, 1), stride=1, padding=0, bias=True)
                nn.init.constant_(basic_conv2d.bias, 0)
            else:
                basic_conv2d = nn.Conv2d(input_dims, num_unit, (1, 1), stride=1, padding=0, bias=False)
            nn.init.xavier_normal_(basic_conv2d.weight)
            ret.add_module('conv2d' + str(index), basic_conv2d)
            if activation is not None:
                if self.bn:
                    decay = self.bn_decay if self.bn_decay is not None else 0.1
                    basic_batch_norm = nn.BatchNorm2d(num_unit, eps=1e-3, momentum=decay)
                    ret.add_module('batch_norm' + str(index), basic_batch_norm)
                ret.add_module('activation' + str(index), activation())
            input_dims = num_unit
            index += 1
        return ret

    def forward(self, x):
        # x: (N, H, W, C)
        x = x.transpose(1, 3).transpose(2, 3)  # x: (N, C, H, W)
        x = self.layers(x)
        x = x.transpose(2, 3).transpose(1, 3)  # x: (N, H, W, C)
        return x


class SpatialAttention(nn.Module):
    def __init__(self, K, D, bn, bn_decay, device):
        super(SpatialAttention, self).__init__()
        self.K = K
        self.D = D
        self.bn = bn
        self.bn_decay = bn_decay
        self.device = device
        self.input_query_fc = FC(input_dims=2 * self.D, units=self.D, activations=nn.ReLU,
                                 bn=self.bn, bn_decay=self.bn_decay, device=self.device)
        self.input_key_fc = FC(input_dims=2 * self.D, units=self.D, activations=nn.ReLU,
                               bn=self.bn, bn_decay=self.bn_decay, device=self.device)
        self.input_value_fc = FC(input_dims=2 * self.D, units=self.D, activations=nn.ReLU,
                                 bn=self.bn, bn_decay=self.bn_decay, device=self.device)
        self.mha = nn.MultiheadAttention(self.D, num_heads=K, batch_first=True)
        self.output_fc = FC(input_dims=self.D, units=self.D, activations=None,
                            bn=self.bn, bn_decay=self.bn_decay, device=self.device)

    def forward(self, x, ste):
        '''
        spatial attention mechanism
        x:      (batch_size, num_step, num_nodes, D)
        ste:    (batch_size, num_step, num_nodes, D)
        return: (batch_size, num_step, num_nodes, D)
        '''
        batch_size = x.size(0)
        num_step = x.size(1)
        num_nodes = x.size(2)

        x = torch.cat((x, ste), dim=-1)
        # (batch_size, num_step, num_nodes, D)
        query = self.input_query_fc(x)
        key = self.input_key_fc(x)
        value = self.input_value_fc(x)
        # (batch_size, num_step, num_nodes, d)
        query = query.reshape(batch_size * num_step, num_nodes, -1)
        key = key.reshape(batch_size * num_step, num_nodes, -1)
        value = value.reshape(batch_size * num_step, num_nodes, -1)

        x, _ = self.mha(query, key, value)
        x = x.reshape(batch_size, num_step, num_nodes, -1)
        x = self.output_fc(x)  # (batch_size, num_steps, num_nodes, D)
        return x


class TemporalAttention(nn.Module):
    def __init__(self, K, D, bn, bn_decay, device, mask=True):
        super(TemporalAttention, self).__init__()
        self.K = K
        self.D = D
        self.bn = bn
        self.bn_decay = bn_decay
        self.device = device
        self.mask = mask
        self.input_query_fc = FC(input_dims=2 * self.D, units=self.D, activations=nn.ReLU,
                                 bn=self.bn, bn_decay=self.bn_decay, device=self.device)
        self.input_key_fc = FC(input_dims=2 * self.D, units=self.D, activations=nn.ReLU,
                               bn=self.bn, bn_decay=self.bn_decay, device=self.device)
        self.input_value_fc = FC(input_dims=2 * self.D, units=self.D, activations=nn.ReLU,
                                 bn=self.bn, bn_decay=self.bn_decay, device=self.device)
        self.mha = nn.MultiheadAttention(self.D, K, batch_first=True)
        self.output_fc = FC(input_dims=self.D, units=self.D, activations=None,
                            bn=self.bn, bn_decay=self.bn_decay, device=self.device)

    def forward(self, x, ste):
        '''
        temporal attention mechanism
        x:      (batch_size, num_step, num_nodes, D)
        ste:    (batch_size, num_step, num_nodes, D)
        return: (batch_size, num_step, num_nodes, D)
        '''
        batch_size = x.size(0)
        num_step = x.size(1)
        num_nodes = x.size(2)

        x = torch.cat((x, ste), dim=-1)
        # (batch_size, num_step, num_nodes, D)
        query = self.input_query_fc(x)
        key = self.input_key_fc(x)
        value = self.input_value_fc(x)

        query = query.transpose(1, 2).reshape(batch_size * num_nodes, num_step, -1)
        key = key.transpose(1, 2).reshape(batch_size * num_nodes, num_step, -1)
        value = value.transpose(1, 2).reshape(batch_size * num_nodes, num_step, -1)

        mask = torch.ones((num_step, num_step), device=self.device)
        mask = torch.tril(mask)
        x, _ = self.mha(query, key, value, attn_mask=mask)

        x = x.reshape(batch_size, num_nodes, num_step, -1).transpose(1, 2)
        x = self.output_fc(x)  # (batch_size, output_length, num_nodes, D)
        return x


class GatedFusion(nn.Module):
    def __init__(self, D, bn, bn_decay, device):
        super(GatedFusion, self).__init__()
        self.D = D
        self.bn = bn
        self.bn_decay = bn_decay
        self.device = device
        self.HS_fc = FC(input_dims=self.D, units=self.D, activations=None,
                        bn=self.bn, bn_decay=self.bn_decay, device=self.device, use_bias=False)
        self.HT_fc = FC(input_dims=self.D, units=self.D, activations=None,
                        bn=self.bn, bn_decay=self.bn_decay, device=self.device, use_bias=True)
        self.output_fc = FC(input_dims=self.D, units=[self.D, self.D], activations=[nn.ReLU, None],
                            bn=self.bn, bn_decay=self.bn_decay, device=self.device)

    def forward(self, HS, HT):
        '''
        gated fusion
        HS:     (batch_size, num_step, num_nodes, D)
        HT:     (batch_size, num_step, num_nodes, D)
        return: (batch_size, num_step, num_nodes, D)
        '''
        XS = self.HS_fc(HS)
        XT = self.HT_fc(HT)
        z = torch.sigmoid(torch.add(XS, XT))
        H = torch.add(torch.multiply(z, HS), torch.multiply(1 - z, HT))
        H = self.output_fc(H)
        return H


class STAttBlock(nn.Module):
    def __init__(self, K, D, bn, bn_decay, device, mask=True):
        super(STAttBlock, self).__init__()
        self.K = K
        self.D = D
        self.d = self.D / self.K
        self.bn = bn
        self.bn_decay = bn_decay
        self.device = device
        self.mask = mask
        self.sp_att = SpatialAttention(K=self.K, D=self.D, bn=self.bn, bn_decay=self.bn_decay, device=self.device)
        self.temp_att = TemporalAttention(K=self.K, D=self.D, bn=self.bn, bn_decay=self.bn_decay, device=self.device)
        self.gated_fusion = GatedFusion(D=self.D, bn=self.bn, bn_decay=self.bn_decay, device=self.device)

    def forward(self, x, ste):
        HS = self.sp_att(x, ste)
        HT = self.temp_att(x, ste)
        H = self.gated_fusion(HS, HT)
        return torch.add(x, H)


class TransformAttention(nn.Module):
    def __init__(self, K, D, bn, bn_decay, device):
        super(TransformAttention, self).__init__()
        self.K = K
        self.D = D
        self.d = self.D / self.K
        self.bn = bn
        self.bn_decay = bn_decay
        self.device = device
        self.input_query_fc = FC(input_dims=self.D, units=self.D, activations=nn.ReLU,
                                 bn=self.bn, bn_decay=self.bn_decay, device=self.device)
        self.input_key_fc = FC(input_dims=self.D, units=self.D, activations=nn.ReLU,
                               bn=self.bn, bn_decay=self.bn_decay, device=self.device)
        self.input_value_fc = FC(input_dims=self.D, units=self.D, activations=nn.ReLU,
                                 bn=self.bn, bn_decay=self.bn_decay, device=self.device)
        self.mha = nn.MultiheadAttention(self.D, K, batch_first=True)
        self.output_fc = FC(input_dims=self.D, units=self.D, activations=None,
                            bn=self.bn, bn_decay=self.bn_decay, device=self.device)

    def forward(self, x, ste1, ste2):
        '''
        transform attention mechanism
        x:      (batch_size, input_length, num_nodes, D)
        ste_1:  (batch_size, input_length, num_nodes, D)
        ste_2:  (batch_size, output_length, num_nodes, D)
        return: (batch_size, output_length, num_nodes, D)
        '''
        # query: (batch_size, output_length, num_nodes, D)
        # key:   (batch_size, input_length, num_nodes, D)
        # value: (batch_size, input_length, num_nodes, D)
        batch_size = x.size(0)
        num_nodes = x.size(2)
        input_step = x.size(1)
        output_step = ste2.size(1)

        query = self.input_query_fc(ste2)
        key = self.input_key_fc(ste1)
        value = self.input_value_fc(x)

        query = query.transpose(1, 2).reshape(batch_size * num_nodes, output_step, -1)
        key = key.transpose(1, 2).reshape(batch_size * num_nodes, input_step, -1)
        value = value.transpose(1, 2).reshape(batch_size * num_nodes, input_step, -1)

        x, _ = self.mha(query, key, value)
        x = x.reshape(batch_size, num_nodes, output_step, -1).transpose(1, 2)
        x = self.output_fc(x)  # (batch_size, output_length, num_nodes, D)
        return x


class STEmbedding(nn.Module):
    def __init__(self, D, bn, bn_decay, device):
        super(STEmbedding, self).__init__()
        self.D = D
        self.bn = bn
        self.bn_decay = bn_decay
        self.device = device
        self.SE_fc = FC(input_dims=self.D, units=self.D, activations=None,
                        bn=self.bn, bn_decay=self.bn_decay, device=self.device)
        self.TE_fc = FC(input_dims=self.D, units=self.D,
                        activations=None, bn=self.bn, bn_decay=self.bn_decay, device=self.device)

    def forward(self, SE, TE):
        '''
        spatio-temporal embedding
        SE:     (num_nodes, D)
        TE:     (input_length+output_length, D)
        retrun: (batch_size, input_length+output_length, num_nodes, D)
        '''
        SE = SE.unsqueeze(0).unsqueeze(0)  # (1, 1, num_nodes, D)
        SE = self.SE_fc(SE)
        TE = TE.unsqueeze(1).unsqueeze(0)  # (1, input_length+output_length, 1, D)
        TE = self.TE_fc(TE)
        return torch.add(SE, TE)

"""
>>> from models.GMAN import *
>>> adj_mx = np.random.randint(low=0, high=2, size=(8000, 8000))
>>> model = GMAN(64, adj_mx, "cuda:0", './SE.pkl').cuda()
>>> x = torch.rand(1, 6, 8000, 1).cuda()
>>> model(x).size()
torch.Size([1, 1, 8000, 1])
"""
class GMAN(nn.Module):
    def __init__(self, hid_dim, adj_mx, device, cache_path, n_input=6, n_out=1, feature_dim=1):

        # get data feature
        super().__init__()
        config = AttrDict()
        config.L = 1
        config.K = 8
        config.bn = True
        config.bn_decay = 0.1
        config.D = hid_dim
        config.n_pred = n_out
        config.adj_mx = adj_mx
        config.n_his = n_input
        config.device = device
        config.feature_dim = feature_dim
        config.num_nodes = adj_mx.shape[0]
        config.output_dim = n_out
        config.dim = hid_dim
        config = generate_se(config, cache_path)
        self.config = config

        # define the model structure
        self.SE = config.SE
        self.input_fc = FC(input_dims=config.feature_dim, units=[config.D, config.D], activations=[nn.ReLU, None],
                           bn=config.bn, bn_decay=config.bn_decay, device=config.device)
        self.st_embedding = STEmbedding(D=config.D, bn=config.bn, bn_decay=config.bn_decay,
                                        device=config.device)

        temporal_emb_shape = (config.n_his + config.n_pred, config.D)
        self.temporal_emb = torch.randn(temporal_emb_shape, requires_grad=True)
        self.temporal_emb = torch.nn.Parameter(self.temporal_emb, requires_grad=True)

        self.encoder = nn.ModuleList()
        for _ in range(config.L):
            self.encoder.append(
                STAttBlock(K=config.K, D=config.D, bn=config.bn, bn_decay=config.bn_decay, device=config.device))
        self.trans_att = TransformAttention(K=config.K, D=config.D, bn=config.bn, bn_decay=config.bn_decay,
                                            device=config.device)
        self.decoder = nn.ModuleList()
        for _ in range(config.L):
            self.decoder.append(
                STAttBlock(K=config.K, D=config.D, bn=config.bn, bn_decay=config.bn_decay, device=config.device))
        # self.decoder = STAttBlock(K=self.K, D=self.D, bn=self.bn, bn_decay=self.bn_decay,
        #                           device=self.device)
        self.output_fc_1 = FC(input_dims=config.D, units=config.D, activations=nn.ReLU,
                              bn=config.bn, bn_decay=config.bn_decay, device=config.device, use_bias=True)
        self.output_fc_2 = FC(input_dims=config.D, units=[config.output_dim], activations=[None],
                              bn=config.bn, bn_decay=config.bn_decay, device=config.device, use_bias=True)

    def forward(self, x):
        # x  : (batch_size, input_length, num_nodes, feature_dim)
        # ret: (batch_size, output_length, num_nodes, output_dim)
        # handle data

        SE = torch.from_numpy(self.SE).to(device=self.config.device)
        TE = self.temporal_emb

        # create network
        # input
        x = self.input_fc(x)  # (batch_size, input_length, num_nodes, D)
        # STE
        ste = self.st_embedding(SE, TE)
        ste = ste.repeat(x.size()[0], 1, 1, 1)

        ste_p = ste[:, :self.config.n_his]  # (batch_size, input_length, num_nodes, D)
        ste_q = ste[:, self.config.n_his:]  # (batch_size, output_length, num_nodes, D)
        # encoder
        for encoder_layer in self.encoder:
            x = encoder_layer(x, ste_p)
        # transAtt
        x = self.trans_att(x, ste_p, ste_q)  # (batch_size, output_length, num_nodes, D)
        # decoder
        for decoder_layer in self.decoder:
            x = decoder_layer(x, ste_q)
        # output
        x = F.dropout(x, p=0.1)
        x = self.output_fc_1(x)
        x = F.dropout(x, p=0.1)
        x = self.output_fc_2(x)
        return x

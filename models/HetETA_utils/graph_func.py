"""Torch Module for Chebyshev Spectral Graph Convolution layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv, HeteroGraphConv
from dgl import broadcast_nodes, function as fn
import dgl
class ChebConv(nn.Module):
    r"""Chebyshev Spectral Graph Convolution layer from `Convolutional
    Neural Networks on Graphs with Fast Localized Spectral Filtering
    <https://arxiv.org/pdf/1606.09375.pdf>`__

    .. math::
        h_i^{l+1} &= \sum_{k=0}^{K-1} W^{k, l}z_i^{k, l}

        Z^{0, l} &= H^{l}

        Z^{1, l} &= \tilde{L} \cdot H^{l}

        Z^{k, l} &= 2 \cdot \tilde{L} \cdot Z^{k-1, l} - Z^{k-2, l}

        \tilde{L} &= 2\left(I - \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}\right)/\lambda_{max} - I

    where :math:`\tilde{A}` is :math:`A` + :math:`I`, :math:`W` is learnable weight.


    Parameters
    ----------
    in_feats: int
        Dimension of input features; i.e, the number of dimensions of :math:`h_i^{(l)}`.
    out_feats: int
        Dimension of output features :math:`h_i^{(l+1)}`.
    k : int
        Chebyshev filter size :math:`K`.
    activation : function, optional
        Activation function. Default ``ReLu``.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.
    """

    def __init__(self,
                 in_feats,
                 # out_feats,
                 k,
                 activation=F.relu,
                 bias=True,
                 device='cpu'):
        super(ChebConv, self).__init__()
        self._k = k
        self._in_feats = in_feats
        # self._out_feats = out_feats
        self.activation = activation
        self.device = device
        # self.linear = nn.Linear(k * in_feats, out_feats, bias)

    def forward(self, graph, feat, lambda_max=None):
        r"""Compute ChebNet layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.
        lambda_max : list or tensor or None, optional.
            A list(tensor) with length :math:`B`, stores the largest eigenvalue
            of the normalized laplacian of each individual graph in ``graph``,
            where :math:`B` is the batch size of the input graph. Default: None.

            If None, this method would set the default value to 2.
            One can use :func:`dgl.laplacian_lambda_max` to compute this value.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """
        def unnLaplacian(feat, D_invsqrt, graph):
            """ Operation Feat * D^-1/2 A D^-1/2 """
            graph.ndata['h'] = feat * D_invsqrt
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return graph.ndata.pop('h') * D_invsqrt

        if lambda_max is None:
            lambda_max = [2] * graph.batch_size

        with graph.local_scope():
            D_invsqrt = th.pow(graph.in_degrees().float().clamp(
                min=1), -0.5).unsqueeze(-1).to(self.device)

            if isinstance(lambda_max, list):
                lambda_max = th.Tensor(lambda_max).to(self.device)
            if lambda_max.dim() == 1:
                lambda_max = lambda_max.unsqueeze(-1)  # (B,) to (B, 1)

            # broadcast from (B, 1) to (N, 1)
            lambda_max = broadcast_nodes(graph, lambda_max)
            re_norm = 2. / lambda_max

            # X_0 is the raw feature, Xt refers to the concatenation of X_0, X_1, ... X_t
            Xt = X_0 = feat

            # X_1(f)
            if self._k > 1:
                h = unnLaplacian(X_0, D_invsqrt, graph)
                X_1 = - re_norm * h + X_0 * (re_norm - 1)
                # Concatenate Xt and X_1
                Xt = th.cat((Xt, X_1), 1)

            # Xi(x), i = 2...k
            for _ in range(2, self._k):
                h = unnLaplacian(X_1, D_invsqrt, graph)
                X_i = - 2 * re_norm * h + X_1 * 2 * (re_norm - 1) - X_0
                # Concatenate Xt and X_i
                Xt = th.cat((Xt, X_i), 1)
                X_1, X_0 = X_i, X_1

            # linear projection
            # h = self.linear(Xt)

            # activation
            if self.activation:
                Xt = self.activation(Xt)

        return Xt


class ChebAttn(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 k,
                 activation=F.relu,
                 device='cpu'):
        super(ChebAttn, self).__init__()
        self.attn = GATConv(in_feats=in_feats,
                            out_feats=in_feats,
                            num_heads=1,
                            feat_drop=0.0,
                            attn_drop=0.0,
                            residual=False,
                            activation=None,
                            allow_zero_in_degree=True)
        self.cheb = ChebConv(in_feats=in_feats,
                             k=k,
                             activation=None,
                             device=device)
        self.lin = nn.Linear(k * in_feats, out_feats)
        self.activation = activation
        self.k = k
        self.device = device

    def forward(self, graph, feat):
        # TODO: only support hetero-edge (in_type, relation, out_type) with in_type==out_type
        feat = feat[0]
        cx = self.cheb(graph, feat)  # [num_nodes, k * in_feats]
        ax = []
        for i in range(0, self.k):
            k_graph = dgl.transform.khop_graph(graph, i).to(self.device)
            ax.append(self.attn(k_graph, feat).reshape(feat.size(0), -1))
        ax = th.cat(ax, dim=-1)  # [num_nodes, k * in_feat]
        x = cx * ax
        x = self.lin(x)
        if self.activation is not None:
            x = self.activation(x)

        return x

class HetCheb(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 k,
                 rel_names,
                 activation=F.relu,
                 device='cpu'):
        super(HetCheb, self).__init__()
        self.conv = HeteroGraphConv({
            rel: ChebAttn(in_feats=in_feats,
                          out_feats=out_feats,
                          k=k,
                          activation=activation,
                          device=device)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, feat):
        h = self.conv(graph, feat)
        # h = {k: v for k, v in h.items()}
        # h = self.conv2(graph, h)
        return h

import numpy as np
import torch
if __name__ == "__main__":
    n_users = 1000
    n_follows = 3000
    n_hetero_features = 10
    n_user_classes = 5
    n_max_clicks = 10

    follow_src = np.random.randint(0, n_users, n_follows)
    follow_dst = np.random.randint(0, n_users, n_follows)

    hetero_graph = dgl.heterograph({
        ('user', 'follow', 'user'): (follow_src, follow_dst),
        ('user', 'followed-by', 'user'): (follow_dst, follow_src)})

    hetero_graph.nodes['user'].data['feature'] = torch.randn(n_users, 6, 24, n_hetero_features)

    model = HetCheb(n_hetero_features, 20, 3, hetero_graph.etypes)

    user_feats = hetero_graph.nodes['user'].data['feature']

    # user_feats [1000,10]
    node_features = {'user': user_feats}

    h_dict = model(hetero_graph, node_features)
    print(h_dict['user'].size())

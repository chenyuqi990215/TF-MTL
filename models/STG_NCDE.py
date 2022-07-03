import torch
import torch.nn.functional as F
import torch.nn as nn
from models.STG_NCDE_utils import controldiffeq
from models.STG_NCDE_utils.vector_fields import *

'''
>>> from models.STG_NCDE import *
>>> model = make_model(64, 8000, "cuda:0")
>>> x = torch.rand(1, 6, 8000, 1).cuda()
>>> model = model.cuda()
>>> model(x).size()
torch.Size([1, 1, 8000, 1])
'''

class NeuralGCDE(nn.Module):
    def __init__(self, args, func_f, func_g, input_channels, hidden_channels, output_channels, initial, device, atol,
                 rtol, solver, times):
        super(NeuralGCDE, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = input_channels
        self.hidden_dim = hidden_channels
        self.output_dim = output_channels
        self.horizon = args.horizon
        self.num_layers = args.num_layers

        self.default_graph = args.default_graph
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)

        self.func_f = func_f
        self.func_g = func_g
        self.solver = solver
        self.atol = atol
        self.rtol = rtol

        # predictor
        self.end_conv = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

        self.init_type = 'fc'
        if self.init_type == 'fc':
            self.initial_h = torch.nn.Linear(self.input_dim, self.hidden_dim)
            self.initial_z = torch.nn.Linear(self.input_dim, self.hidden_dim)
        elif self.init_type == 'conv':
            self.start_conv_h = nn.Conv2d(in_channels=input_channels,
                                          out_channels=hidden_channels,
                                          kernel_size=(1, 1))
            self.start_conv_z = nn.Conv2d(in_channels=input_channels,
                                          out_channels=hidden_channels,
                                          kernel_size=(1, 1))
        self.times = times

    def forward(self, x):
        # source: B, T_1, N, D
        # target: B, T_2, N, D
        # supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)

        times = self.times
        coeffs = controldiffeq.natural_cubic_spline_coeffs(times, x.transpose(1, 2))
        spline = controldiffeq.NaturalCubicSpline(times, coeffs)
    
        if self.init_type == 'fc':
            h0 = self.initial_h(spline.evaluate(times[0]))
            z0 = self.initial_z(spline.evaluate(times[0]))
        elif self.init_type == 'conv':
            h0 = self.start_conv_h(spline.evaluate(times[0]).transpose(1, 2).unsqueeze(-1)).transpose(1, 2).squeeze()
            z0 = self.start_conv_z(spline.evaluate(times[0]).transpose(1, 2).unsqueeze(-1)).transpose(1, 2).squeeze()

        z_t = controldiffeq.cdeint_gde_dev(dX_dt=spline.derivative,  # dh_dt
                                           h0=h0,
                                           z0=z0,
                                           func_f=self.func_f,
                                           func_g=self.func_g,
                                           t=times,
                                           method=self.solver,
                                           atol=self.atol,
                                           rtol=self.rtol)

        # init_state = self.encoder.init_hidden(source.shape[0])
        # output, _ = self.encoder(source, init_state, self.node_embeddings)      #B, T, N, hidden
        # output = output[:, -1:, :, :]                                   #B, 1, N, hidden
        z_T = z_t[-1:, ...].transpose(0, 1)

        # CNN based predictor
        output = self.end_conv(z_T)  # B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)  # B, T, N, C

        return output


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def model_config(config):
    config.g_type = "agc"
    config.embed_dim = 10
    config.num_layers = 3
    config.cheb_k = 2
    config.solver = "rk4"
    config.default_graph = True
    return config


def make_model(hid_dim, num_nodes, device, n_input=6, n_out=1):
    """
    :param data: dict contains train, valid, test
    :return:
    """
    args = AttrDict()
    args = model_config(args)
    args.hid_dim = hid_dim
    args.hid_hid_dim = hid_dim
    args.input_dim = 1
    args.output_dim = 1
    args.horizon = n_out
    args.num_nodes = num_nodes
    args.device = device


    times = torch.linspace(0, n_input - 1, n_input).to(device)
    vector_field_f = FinalTanh_f(input_channels=args.input_dim, hidden_channels=args.hid_dim,
                                 hidden_hidden_channels=args.hid_hid_dim,
                                 num_hidden_layers=args.num_layers)
    vector_field_g = VectorField_g(input_channels=args.input_dim, hidden_channels=args.hid_dim,
                                   hidden_hidden_channels=args.hid_hid_dim,
                                   num_hidden_layers=args.num_layers, num_nodes=args.num_nodes, cheb_k=args.cheb_k,
                                   embed_dim=args.embed_dim,
                                   g_type=args.g_type)
    model = NeuralGCDE(args, func_f=vector_field_f, func_g=vector_field_g, input_channels=args.input_dim,
                       hidden_channels=args.hid_dim,
                       output_channels=args.output_dim, initial=True,
                       device=args.device, atol=1e-9, rtol=1e-7, solver=args.solver, times=times)
    return model

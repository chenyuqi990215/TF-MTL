from torch import nn
from models.AbstractTrafficStateModel import AbstractTrafficStateModel
from utils.evaluator import masked_mae
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
        # x:  (batch_size, c_in, num_nodes)
        # x_c: (batch_size, c_in, Ks, num_nodes)
        # theta: (c_in, c_out, Ks)
        # x_gc: (batch_size, c_out, input_length, num_nodes)
        # return: (batch_size, c_out, num_nodes)
        x_c = torch.einsum("knm,bim->bikn", self.Lk, x)  # delete num_nodes(n)
        x_gc = torch.einsum("iok,bikn->bon", self.theta, x_c) + self.b  # delete Ks(k) c_in(i)
        x_in = self.align(x.unsqueeze(1)).squeeze()  # (batch_size, c_out, num_nodes)
        return torch.relu(x_gc + x_in)  # residual connection


class AbstractModel(nn.Module):
    def __init__(self, model: AbstractTrafficStateModel, config, scalar):
        super(AbstractModel, self).__init__()
        self.model = model
        self.config = config
        self.scalar = scalar

    def forward(self, batch):
        pass

    def predict(self, batch):
        pass

    def calculate_loss(self, batch):
        pass


class BaseLine(AbstractModel):
    def __init__(self, model: AbstractTrafficStateModel, config, scalar):
        super(BaseLine, self).__init__(model, config, scalar)
        self.fc = nn.Linear(config.enc_dim, config.n_pred)

    def forward(self, batch):
        enc = self.model.encoder(batch['anchor_x'])
        return self.fc(enc)

    def predict(self, batch):
        return self.forward(batch)

    def calculate_loss(self, batch):
        y_predicted = self.forward(batch)
        y_true = self.scaler.inverse_transform(batch['anchor_y'])
        return masked_mae(y_predicted, y_true, 0.0)


class Triplet(AbstractModel):
    def __init__(self, model: AbstractTrafficStateModel, config, scalar):
        super(Triplet, self).__init__(model, config, scalar)
        self.fc = nn.Linear(config.enc_dim, config.n_pred)

    def forward(self, batch):
        enc_anchor = self.model.encoder(batch['anchor_x'])
        enc_positive = self.model.encoder(batch['positive_x'])
        enc_negative = self.model.encoder(batch['negative_x'])
        pred = self.fc(enc_anchor)
        return enc_anchor, enc_positive, enc_negative, pred

    def predict(self, batch):
        return self.forward(batch)[3]

    def calculate_loss(self, batch):
        enc_anchor, enc_positive, enc_negative, y_predicted = self.forward(batch)
        y_true = self.scaler.inverse_transform(batch['anchor_y'])
        loss_supervised = masked_mae(y_predicted, y_true, 0.0)

        enc_anchor = F.normalize(enc_anchor, dim=-1)
        enc_positive = F.normalize(enc_positive, dim=-1)
        enc_negative = F.normalize(enc_negative, dim=-1)
        sim_anchor_positive = torch.sum(enc_anchor * enc_positive, dim=-1)  # [batch_size, num_nodes]
        sim_anchor_negative = torch.sum(enc_anchor * enc_negative, dim=-1)  # [batch_size, num_nodes]
        loss_triplet = torch.mean(torch.max(sim_anchor_negative + self.config.margin - sim_anchor_positive, 0))

        return loss_supervised + self.config.beta * loss_triplet

class EncDec(AbstractModel):
    def __init__(self, model: AbstractTrafficStateModel, config, scalar):
        super(EncDec, self).__init__(model, config, scalar)
        self.lstm = nn.LSTM(hidden_size=config.out_dim, input_size=config.enc_dim, batch_first=True)
        self.fc = nn.Linear(config.out_dim, 1)
        self.gnn = SpatioConvLayer(config.Ks, config.out_dim, config.out_dim, config.Lk, config.device)

    def decoder(self, enc):
        '''
        :param enc: (batch_size, num_nodes, enc_dim)
        :return: (batch_size, num_nodes, n_pred, out_dim)
        '''
        h = torch.zeros(1, enc.size()[0] * enc.size()[1], self.config.out_dim).to(self.config.device)
        c = torch.zeros(1, enc.size()[0] * enc.size()[1], self.config.out_dim).to(self.config.device)

        outputs = []
        input_embed = enc.reshape(enc.size()[0] * enc.size()[1], -1)
        for di in range(self.config.n_pred):
            input_embed = input_embed.unsqueeze(1)
            decoder_output, (h, c) = self.lstm(input_embed, (h, c))  # [batch_size * num_nodes, 1, out_dim]
            decoder_output = decoder_output.reshape(enc.size()[0], enc.size()[1], -1)
            outputs.append(decoder_output.unsqueeze(2))  # (batch_size, num_nodes, 1, out_dim)
            decoder_output = decoder_output.permute(0, 2, 1)
            decoder_output = self.gnn(decoder_output)
            input_embed = decoder_output.reshape(enc.size()[0] * enc.size()[1], -1)
        return torch.cat(outputs, dim=2)

    def forward(self, batch):
        enc = self.model.encoder(batch['anchor_x'])
        dec = self.decoder(enc)
        return self.fc(dec).squeeze()

    def predict(self, batch):
        return self.forward(batch)

    def calculate_loss(self, batch):
        y_predicted = self.forward(batch)
        y_true = self.scaler.inverse_transform(batch['anchor_y'])
        return masked_mae(y_predicted, y_true, 0.0)


class DecTriplet(EncDec):
    def __init__(self, model: AbstractTrafficStateModel, config, scalar):
        super(DecTriplet, self).__init__(model, config, scalar)

    def forward(self, batch):
        enc_anchor = self.model.encoder(batch['anchor_x'])
        enc_positive = self.model.encoder(batch['positive_x'])
        enc_negative = self.model.encoder(batch['negative_x'])
        dec_anchor = self.decoder(enc_anchor)
        pred = self.fc(dec_anchor).squeeze()
        return enc_anchor, enc_positive, enc_negative, pred

    def triplet_loss(self, sim_anchor, sim_positive, sim_negative):
        sim_anchor_positive = F.kl_div(sim_anchor, sim_positive, reduction='sum')  # (batch_size, num_nodes)
        sim_anchor_negative = F.kl_div(sim_anchor, sim_negative, reduction='sum')  # (batch_size, num_nodes)
        print(sim_anchor_positive.size())
        return torch.mean(torch.max(sim_anchor_negative + self.config.margin - sim_anchor_positive, 0))

    def calculate_loss(self, batch):
        enc_anchor, enc_positive, enc_negative, pred = self.forward(batch)
        sim_anchor = torch.matmul(enc_anchor, enc_anchor.transpose(-1, -2))
        sim_anchor = torch.softmax(sim_anchor, dim=-1)  # (batch_size, num_nodes, num_nodes)

        sim_positive = torch.matmul(enc_positive, enc_positive.transpose(-1, -2))
        sim_positive = torch.softmax(sim_positive, dim=-1)  # (batch_size, num_nodes, num_nodes)

        sim_negative = torch.matmul(enc_negative, enc_negative.transpose(-1, -2))
        sim_negative = torch.softmax(sim_negative, dim=-1)  # (batch_size, num_nodes, num_nodes)
        # TODO


class Config:
    def __init__(self):
        self.blocks = [[1, 32, 64], [64, 32, 128]]
        self.n_pred = 12
        self.n_his = 12
        self.num_nodes = 1125
        self.Kt = 3
        self.Ks = 1
        self.Lk = torch.rand(1, 1125, 1125)
        self.keep_prob = 0.5
        self.device = "cpu"
        self.enc_dim = 128
        self.out_dim = 128
        self.margin = 1
        self.beta = 0.5

if __name__ == "__main__":
    from utils.scaler import StandardScaler
    config = Config()
    scalar = StandardScaler(0, 1)
    from models.STGCN import STGCN
    model = STGCN(config)
    net = EncDec(model, config, scalar)
    batch = {
        'anchor': torch.rand(128, 1124, 12, 1),
        'positive': torch.rand(128, 1124, 12, 1),
        'negative': torch.rand(128, 1124, 12, 1),
        'y': torch.rand(128, 1124, 12)
    }
    out = net(batch)
    print(out.size())
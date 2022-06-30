from torch import nn
from models.AbstractTrafficStateModel import AbstractTrafficStateModel
from utils.evaluator import masked_mae
import torch
import torch.nn.functional as F
import torch.nn.init as init
import math
from layer import *

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

    def calculate_loss(self, batch, is_valid=False):
        pass

    def supervised_loss(self, y_predicted, y_true):
        loss = masked_mae(y_predicted, y_true, 0.0)
        if loss.item() > 1000:
            import pdb
            pdb.set_trace()
        return loss


class BaseLine(AbstractModel):
    def __init__(self, model: AbstractTrafficStateModel, config, scalar):
        super(BaseLine, self).__init__(model, config, scalar)
        self.fc = nn.Linear(config.enc_dim, config.n_pred)

    def forward(self, batch):
        enc = self.model.encoder(batch['anchor_x'])
        return self.fc(enc)

    def predict(self, batch):
        return self.forward(batch)

    def calculate_loss(self, batch, is_valid=False):
        y_predicted = self.scalar.inverse_transform(self.forward(batch))
        y_true = batch['anchor_y'].squeeze()
        return self.supervised_loss(y_predicted, y_true)


class Classification(BaseLine):
    def __init__(self, model: AbstractTrafficStateModel, config, scalar):
        super(Classification, self).__init__(model, config, scalar)
        self.fc = nn.Linear(config.enc_dim, config.n_pred)
        self.lin = nn.Linear(config.enc_dim, config.enc_dim)
        self.pool = nn.Linear(config.num_nodes, 1)
        self.classify = nn.Linear(config.enc_dim, 288)

    def forward(self, batch):
        enc = self.model.encoder(batch['anchor_x'])
        pred = self.fc(enc)

        classify = self.classify(self.pool(F.relu(self.lin(enc)).transpose(-1, -2)).squeeze())

        return pred, classify

    def predict(self, batch):
        return self.forward(batch)[0]

    def time_loss(self, pred, gt):
        pred = F.softmax(pred, dim=-1)
        pred = torch.clip(pred, 1e-6, 1 - 1e-6)
        return torch.mean(pred * gt)

    def calculate_loss(self, batch, is_valid=False):
        y_predicted, classify = self.forward(batch)
        y_predicted = self.scalar.inverse_transform(y_predicted)
        y_true = batch['anchor_y'].squeeze()

        if is_valid:
            return self.supervised_loss(y_predicted, y_true)
        else:
            return self.supervised_loss(y_predicted, y_true) \
                   + self.config.beta * self.time_loss(classify, batch['anchor_t'])


class Triplet(AbstractModel):
    def __init__(self, model: AbstractTrafficStateModel, config, scalar):
        super(Triplet, self).__init__(model, config, scalar)
        self.fc = nn.Linear(config.enc_dim, config.n_pred)

    def forward(self, batch):
        enc_anchor = self.model.encoder(batch['anchor_x'])
        enc_positive = self.model.encoder(batch['positive_x']).detach()
        enc_negative = self.model.encoder(batch['negative_x']).detach()
        pred = self.fc(enc_anchor)
        return enc_anchor, enc_positive, enc_negative, pred

    def predict(self, batch):
        return self.forward(batch)[3]

    def pointwise_triplet_loss(self, enc_anchor, enc_positive, enc_negative):
        enc_anchor = F.normalize(enc_anchor, dim=-1)
        enc_positive = F.normalize(enc_positive, dim=-1)
        enc_negative = F.normalize(enc_negative, dim=-1)
        sim_anchor_positive = torch.sum(enc_anchor * enc_positive, dim=-1)  # [batch_size, num_nodes]
        sim_anchor_negative = torch.sum(enc_anchor * enc_negative, dim=-1)  # [batch_size, num_nodes]
        zeros = torch.zeros_like(sim_anchor_positive).to(self.config.device)
        margin = torch.ones_like(sim_anchor_positive).to(self.config.device) * self.config.margin
        loss_triplet = torch.mean(torch.max(sim_anchor_negative + margin - sim_anchor_positive, zeros))
        return loss_triplet

    def calculate_loss(self, batch, is_valid=False):
        enc_anchor, enc_positive, enc_negative, y_predicted = self.forward(batch)
        y_predicted = self.scalar.inverse_transform(y_predicted)
        y_true = batch['anchor_y'].squeeze()
        loss_supervised = self.supervised_loss(y_predicted, y_true)

        loss_triplet = self.pointwise_triplet_loss(enc_anchor, enc_positive, enc_negative)

        if is_valid:
            return loss_supervised
        else:
            return loss_supervised + self.config.beta * loss_triplet

class EncDec(AbstractModel):
    def __init__(self, model: AbstractTrafficStateModel, config, scalar):
        super(EncDec, self).__init__(model, config, scalar)

        self.fc = nn.Linear(config.out_dim, 1)
        if 'AGCLSTM' not in config.gnn:
            self.gnn = get_gnn_model(config)
            self.lstm = nn.LSTM(hidden_size=config.out_dim, input_size=config.enc_dim, batch_first=True)
        else:
            self.lstm = AGCLSTM(input_sz=config.enc_dim, hidden_sz=config.out_dim, config=config)
        self.norm = nn.LayerNorm([config.num_nodes, config.out_dim])
        self.align = Align(config.out_dim, config.out_dim)
        self.relu = nn.ReLU()

    def decoder_gnn(self, enc):
        '''
        :param enc: (batch_size, num_nodes, enc_dim)
        :return: (batch_size, num_nodes, n_pred, out_dim)
        '''
        h = torch.zeros(1, enc.size()[0] * enc.size()[1], self.config.out_dim).to(self.config.device)
        c = torch.zeros(1, enc.size()[0] * enc.size()[1], self.config.out_dim).to(self.config.device)

        outputs = []
        input_embed = enc.reshape(enc.size()[0] * enc.size()[1], -1)
        input_embed = input_embed.unsqueeze(1)
        for di in range(self.config.n_pred):
            decoder_output, (h, c) = self.lstm(input_embed, (h, c))  # [batch_size * num_nodes, 1, out_dim]
            decoder_output = decoder_output.reshape(enc.size()[0], enc.size()[1], -1)
            outputs.append(decoder_output.unsqueeze(2))  # (batch_size, num_nodes, 1, out_dim)
            hn = h.reshape(enc.size()[0], enc.size()[1], -1)
            cn = c.reshape(enc.size()[0], enc.size()[1], -1)
            hn = self.gnn(hn)
            cn = self.gnn(cn)
            hn = self.norm(hn)
            cn = self.norm(cn)
            hn = hn.reshape(*h.size())
            cn = cn.reshape(*c.size())
            h = self.relu(h + hn)
            c = self.relu(c + cn)
        return torch.cat(outputs, dim=2)

    def decode_agclstm(self, enc):
        '''
                :param enc: (batch_size, num_nodes, enc_dim)
                :return: (batch_size, num_nodes, n_pred, out_dim)
                '''
        h = torch.zeros(1, enc.size()[0] * enc.size()[1], self.config.out_dim).to(self.config.device)
        c = torch.zeros(1, enc.size()[0] * enc.size()[1], self.config.out_dim).to(self.config.device)

        outputs = []
        input_embed = enc.reshape(enc.size()[0] * enc.size()[1], -1)
        input_embed = input_embed.unsqueeze(1)
        for di in range(self.config.n_pred):
            input_embed, (h, c) = self.lstm(input_embed, (h, c))  # [batch_size * num_nodes, 1, out_dim]
            outputs.append(input_embed.reshape(-1, self.config.num_nodes, 1, self.config.out_dim))
            # (batch_size, num_nodes, 1, out_dim)
        return torch.cat(outputs, dim=2)

    def decoder_gnn_v1(self, enc):
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
            decoder_output = self.gnn(decoder_output)
            decoder_output = self.norm(decoder_output)
            decoder_output = decoder_output.reshape(enc.size()[0] * enc.size()[1], -1)
            input_embed = input_embed.reshape(enc.size()[0], enc.size()[1], 1, -1).permute(0, 3, 2, 1)
            input_embed = self.align(input_embed).permute(0, 3, 2, 1).reshape(enc.size()[0] * enc.size()[1], -1)
            input_embed = self.relu(input_embed + decoder_output)
        return torch.cat(outputs, dim=2)

    def decoder(self, enc):
        if 'AGCLSTM' in self.config.gnn:
            dec = self.decode_agclstm(enc)
        elif self.config.gnn == 'GCN':
            dec = self.decoder_gnn(enc)
        elif self.config.gnn == 'GAT':
            dec = self.decoder_gnn(enc)
        else:
            raise NotImplementedError
        return dec

    def forward(self, batch):
        enc = self.model.encoder(batch['anchor_x'])
        dec = self.decoder(enc)
        return self.fc(dec).squeeze()

    def predict(self, batch):
        return self.forward(batch)

    def calculate_loss(self, batch, is_valid=False):
        y_predicted = self.scalar.inverse_transform(self.forward(batch))
        y_true = batch['anchor_y'].squeeze()
        return self.supervised_loss(y_predicted, y_true)


class DecTriplet(EncDec, Triplet):
    def __init__(self, model: AbstractTrafficStateModel, config, scalar):
        super(DecTriplet, self).__init__(model, config, scalar)
        self.lin = nn.Linear(self.config.n_pred * self.config.out_dim, self.config.enc_dim)

    def forward(self, batch):
        enc_anchor = self.model.encoder(batch['anchor_x'])
        enc_positive = self.model.encoder(batch['positive_x']).detach()
        enc_negative = self.model.encoder(batch['negative_x']).detach()
        enc_gt = self.model.encoder(batch['anchor_y']).detach()
        dec_anchor = self.decoder(enc_anchor)
        pred = self.fc(dec_anchor).squeeze()
        return enc_anchor, enc_positive, enc_negative, enc_gt, dec_anchor, pred

    def content_loss(self, enc_gt, dec_anchor):
        dec_anchor = dec_anchor.reshape(dec_anchor.size()[0], dec_anchor.size()[1], -1)
        dec_anchor = self.lin(dec_anchor)
        return torch.mean(torch.abs(enc_gt - dec_anchor))

    def calculate_loss(self, batch, is_valid=False):
        enc_anchor, enc_positive, enc_negative, enc_gt, dec_anchor, y_predicted = self.forward(batch)
        y_predicted = self.scalar.inverse_transform(y_predicted)
        y_true = batch['anchor_y'].squeeze()

        loss_supervised = self.supervised_loss(y_predicted, y_true)
        loss_triplet = self.pointwise_triplet_loss(enc_anchor, enc_positive, enc_negative)
        loss_content = self.content_loss(enc_gt, dec_anchor)

        if is_valid:
            return loss_supervised
        else:
            # return loss_supervised + self.config.beta * loss_triplet
            return loss_supervised + (self.config.beta * loss_triplet + self.config.beta * loss_content) / 2

    def predict(self, batch):
        return self.forward(batch)[5]

class Config:
    def __init__(self):
        self.blocks = [[1, 32, 64], [64, 32, 128]]
        self.n_pred = 12
        self.n_his = 12
        self.num_nodes = 1125
        self.Kt = 3
        self.Ks = 1
        self.Lk = torch.rand(1, 1125, 1125)
        self.adj_mx = torch.rand(1125, 1125)
        self.keep_prob = 0.5
        self.device = "cpu"
        self.enc_dim = 128
        self.out_dim = 128
        self.margin = 1
        self.beta = 0.5
        self.gnn = 'GCN_AGCLSTM'

if __name__ == "__main__":
    from utils.scaler import StandardScaler
    config = Config()
    scalar = StandardScaler(0, 1)
    from models.GWNET import GWNET
    from config.GWNET import model_config
    config = model_config(config)
    model = GWNET(config)
    net = EncDec(model, config, scalar)
    batch = {
        'anchor_x': torch.rand(102, 1125, 12, 1),
        'positive_x': torch.rand(102, 1125, 12, 1),
        'negative_x': torch.rand(102, 1125, 12, 1),
        'anchor_y': torch.rand(102, 1125, 12, 1)
    }
    out = net.calculate_loss(batch)
    print(out)
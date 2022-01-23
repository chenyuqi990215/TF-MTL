import os
import argparse
import tqdm
import pickle
from dataset import TFDataset
from torch.utils.data import DataLoader
from torch import optim
from utils.earlystopping import EarlyStopping
from model import *
from utils.scaler import StandardScaler
from utils.evaluator import metric
from utils.graph import *
import random

def get_model(args, scalar):
    try:
        base, model = args.model_type.split('_')
    except:
        raise NotImplementedError
    if base == 'STGCN':
        from models.STGCN import STGCN
        base_model = STGCN(args)
    else:
        raise NotImplementedError

    if model == 'BaseLine':
        net = BaseLine(base_model, config, scalar)
    elif model == 'Triplet':
        net = Triplet(base_model, config, scalar)
    elif model == 'EncDec':
        net = EncDec(base_model, config, scalar)
    elif model == 'DecTriplet':
        net = DecTriplet(base_model, config, scalar)
    else:
        raise NotImplementedError
    return net

def parse_arg():
    parser = argparse.ArgumentParser(description='TF-MTL')
    parser.add_argument('--enable-cuda', action='store_true', help='Enable CUDA')
    parser.add_argument('--device', '-device', type=str, default='cuda:0')
    parser.add_argument('--dataset', '-dataset', required=True, type=str)
    parser.add_argument('--data_root', '-data_root', required=True)
    parser.add_argument('--checkpoint', '-checkpoint', required=True)
    parser.add_argument('--n_his', '-n_his', required=False, default=12, type=int, choices=[12])
    parser.add_argument('--n_pred', '-n_pred', required=False, default=3, type=int, choices=[3, 6, 12])
    parser.add_argument('--epoch', '-epoch', required=False, default=500, type=int)
    parser.add_argument('--enable_earlystop', action='store_true', default=True)
    parser.add_argument('--batch_size', '-batch_size', required=False, default=64, type=int)
    parser.add_argument('--lr', '-lr', required=False, default=0.001, type=float)
    parser.add_argument('--decay', '-decay', required=False, default=0.0005, type=float)
    parser.add_argument('--step_size', '-step_size', required=False, default=10, type=int)
    parser.add_argument('--gamma', '-gamma', required=False, default=0.999, type=float)
    parser.add_argument('--optimizer', '-optimizer', required=False, default='Adam', type=str, choices=['Adam', 'AdamW', 'RMSProp'])
    parser.add_argument('--graph_type', '-graph_type', required=False, default='GCN', type=str, choices=['GCN', 'ChebNet'])
    parser.add_argument('--model_type', '-model_type', required=True, default='STGCN_BaseLine', type=str)
    parser.add_argument('--Ks', '-Ks', required=False, default=3, type=int)
    parser.add_argument('--Kt', '-Kt', required=False, default=3, type=int)
    parser.add_argument('--margin', '-margin', required=False, default=1, type=float)
    parser.add_argument('--beta', '-beta', required=False, default=0.5, type=float)
    parser.add_argument('--keep_prob', '-keep_prob', required=False, default=0.5, type=float)
    parser.add_argument('--load_pretrain', '-load_pretrain', required=False, default='', type=str)
    parser.add_argument('--verbose', '-verbose', action='store_true', default=False)
    args = parser.parse_args()
    args.root = os.path.join(args.data_root, args.dataset)
    if not args.enable_cuda or not torch.cuda.is_available():
        args.device = torch.device('cpu')
    torch.cuda.set_device(args.device)
    if args.graph_type == 'GCN':
        args.Ks = 1
    return args


def train(args, net, train_loader, val_loader, optimizer, scheduler, early_stopping):
    l_sum = 0
    n_sum = 0
    net.train()

    for i in range(args.epoch):
        for batch in (tqdm.tqdm(train_loader) if args.verbose else train_loader):
            optimizer.zero_grad()
            batch = {key: batch[key].to(args.device) for key in batch.keys()}
            loss = net.calculate_loss(batch)

            loss.backward()
            optimizer.step()

            l_sum += loss.item() * batch['anchor_y'].size()[0]
            n_sum += batch['anchor_y'].size()[0]

        val_loss = val(net, val_loader, args.device)
        if args.enable_earlystop:
            early_stopping(val_loss, net)
        print('Epoch: {:03d} | Lr: {:.20f} | Train loss: {:.6f} | Val loss: {:.6f}'. \
              format(i + 1, optimizer.param_groups[0]['lr'], l_sum / n_sum, val_loss))
        scheduler.step()

        if args.enable_earlystop and early_stopping.early_stop:
            print("Early stopping.")
            break

    print('\nTraining finished.\n')

def val(net, val_loader, device):
    l_sum = 0
    n_sum = 0
    net.eval()

    for batch in (tqdm.tqdm(val_loader) if args.verbose else val_loader):
        batch = {key: batch[key].to(device) for key in batch.keys()}
        loss = net.calculate_loss(batch)
        l_sum += loss.item() * batch['anchor_y'].size()[0]
        n_sum += batch['anchor_y'].size()[0]
    return l_sum / n_sum

def inverse_normalization(y, means, stds):
    data = y * stds.reshape((1, -1, 1))
    data = data + means.reshape((1, -1, 1))
    return data

def test(args, net, test_loader, device, scaler):
    net.load_state_dict(torch.load(args.checkpoint))
    net.eval()
    MAE, MAPE, RMSE = [], [], []

    for batch in (tqdm.tqdm(test_loader) if args.verbose else test_loader):
        batch = {key: batch[key].to(device) for key in batch.keys()}
        out = net.predict(batch).detach().cpu()
        y_pred = scaler.inverse_transform(out)
        y = batch['anchor_y'].cpu()
        mae, mape, rmse = metric(y_pred, y)
        MAE.append(mae)
        MAPE.append(mape)
        RMSE.append(rmse)

    MAE = np.array(MAE).mean()
    MAPE = np.array(MAPE).mean()
    RMSE = np.array(RMSE).mean()
    print(f'MAE {MAE:.6f} | MAPE {MAPE:.6f} |  RMSE {RMSE:.6f}')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    args = parse_arg()

    seed = 19971125
    setup_seed(seed)
    print('using seed: {}'.format(seed))

    adj_path = os.path.join(args.root, 'adj_mx.pkl')
    train_path = os.path.join(args.root, 'train_{}.npz'.format(args.n_pred))
    val_path = os.path.join(args.root, 'val_{}.npz'.format(args.n_pred))
    test_path = os.path.join(args.root, 'test_{}.npz'.format(args.n_pred))

    adj = pickle.load(open(adj_path, 'rb'))
    args.num_nodes = adj.shape[0]
    if args.graph_type == 'GCN':
        args.Lk = scaled_laplacian(adj).reshape(1, args.num_nodes, args.num_nodes)
    elif args.graph_type == 'ChebNet':
        W = first_approx(adj, args.num_nodes)
        args.Lk = cheb_poly_approx(W, args.Ks, args.num_nodes)
        args.Lk = args.Lk.reshape(args.num_nodes, args.Ks, args.num_nodes).transpose(1, 0)
        
    args.Lk = torch.from_numpy(args.Lk).to(args.device)
    args.output_dim = 1
    args.blocks = [[1, 32, 64], [64, 32, 128]]
    args.enc_dim = 128
    args.dec_dim = 128

    train_data = np.array(np.load(train_path)['x'])[:, :, :, 0]
    scaler = StandardScaler(mean=train_data[..., 0].mean(), std=train_data[..., 0].std())

    train_dataset = TFDataset(train_path, scaler)
    val_dataset = TFDataset(val_path, scaler)
    test_dataset = TFDataset(test_path, scaler)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = get_model(args, scaler)

    if args.load_pretrain != "" and os.path.isfile(args.load_pretrain):
        state_dict = torch.load(args.load_pretrain)
        model.load_state_dict(state_dict, strict=False)

    model = model.to(args.device)

    if args.optimizer == "RMSProp":
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.decay)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    elif args.optimizer == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.decay)
    else:
        raise ValueError(f'ERROR: optimizer {args.optimizer} is undefined.')

    early_stopping = None
    if args.enable_earlystop:
        early_stopping = EarlyStopping(patience=50, path=args.checkpoint, verbose=True)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    train(args, model, train_loader, val_loader, optimizer, scheduler, early_stopping)
    test(args, model, test_loader, args.device, scaler)
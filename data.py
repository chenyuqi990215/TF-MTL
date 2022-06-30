import pandas as pd
import numpy as np
import os
import argparse
import pickle

class TrafficStateDataset:
    def __init__(self, args):
        self.args = args
        self._load_geo(os.path.join(args.root_dir, '{}.geo'.format(args.dataset)))
        self._load_rel(os.path.join(args.root_dir, '{}.rel'.format(args.dataset)))
        self._load_dyna_3d(os.path.join(args.root_dir, '{}.dyna'.format(args.dataset)))
        self._generate_train_val_test(self.df)

    def _generate_graph_seq2seq_io_data(
        self, data, x_offsets, y_offsets
    ):
        """
        Generate samples from
        # x: (epoch_size, input_length, num_nodes, input_dim)
        # y: (epoch_size, output_length, num_nodes, output_dim)
        """

        num_samples, num_nodes, num_features = data.shape
        x, y = [], []
        min_t = abs(min(x_offsets))
        max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
        for t in range(min_t, max_t):  # t is the index of the last observation.
            x.append(data[t + x_offsets, ...])
            y.append(data[t + y_offsets, ...])
        x = np.stack(x, axis=0)
        y = np.stack(y, axis=0)
        return x, y


    def _generate_train_val_test(self, df):
        seq_length_x, seq_length_y = self.args.seq_length_x, self.args.seq_length_y
        # 0 is the latest observed sample.
        x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
        # Predict the next one hour
        y_offsets = np.sort(np.arange(self.args.y_start, (seq_length_y + 1), 1))

        # x: (num_samples, input_length, num_nodes, input_dim)
        # y: (num_samples, output_length, num_nodes, output_dim)
        x, y = self._generate_graph_seq2seq_io_data(
            df,
            x_offsets=x_offsets,
            y_offsets=y_offsets,
        )

        print("x shape: ", x.shape, ", y shape: ", y.shape)
        # Write the data into npz file.
        num_samples = x.shape[0]
        num_test = round(num_samples * 0.2)
        num_train = round(num_samples * 0.7)
        num_val = num_samples - num_test - num_train
        x_train, y_train = x[:num_train], y[:num_train]
        x_val, y_val = (
            x[num_train: num_train + num_val],
            y[num_train: num_train + num_val],
        )
        x_test, y_test = x[-num_test:], y[-num_test:]

        for cat in ["train", "val", "test"]:
            _x, _y = locals()["x_" + cat], locals()["y_" + cat]
            print(cat, "x: ", _x.shape, "y:", _y.shape)
            np.savez_compressed(
                os.path.join(self.args.output_dir, f"{cat}_{seq_length_y}.npz"),
                x=_x,
                y=_y,
            )


    def _load_geo(self, filename):
        """
        加载.geo文件，格式[geo_id, type, coordinates, properties(若干列)]
        """
        geofile = pd.read_csv(filename)
        self.geo_ids = list(geofile['geo_id'])
        self.geo_to_ind = {}
        for index, idx in enumerate(self.geo_ids):
            self.geo_to_ind[idx] = index
        print(len(self.geo_ids))


    def _calculate_adjacency_matrix(self):
        """
        使用带有阈值的高斯核计算邻接矩阵的权重，如果有其他的计算方法，可以覆盖这个函数,
        公式为：$ w_{ij} = \exp \left(- \\frac{d_{ij}^{2}}{\sigma^{2}} \\right) $, $\sigma$ 是方差,
        小于阈值`weight_adj_epsilon`的值设为0：$  w_{ij}[w_{ij}<\epsilon]=0 $
        Returns:
            np.ndarray: self.adj_mx, N*N的邻接矩阵
        """
        distances = self.adj_mx[~np.isinf(self.adj_mx)].flatten()
        std = distances.std()
        self.adj_mx = np.exp(-np.square(self.adj_mx / std))
        self.adj_mx[self.adj_mx < self.weight_adj_epsilon] = 0


    def _load_rel(self, filename):
        """
        加载.rel文件，格式[rel_id, type, origin_id, destination_id, properties(若干列)],
        生成N*N的矩阵，其中权重所在的列名用全局参数`weight_col`来指定,
        全局参数`calculate_weight_adj`表示是否需要对加载的.rel的默认权重进行进一步计算,
        如果需要，则调用函数self._calculate_adjacency_matrix()进行计算
        Returns:
            np.ndarray: self.adj_mx, N*N的邻接矩阵
        """
        relfile = pd.read_csv(filename)
        if len(relfile.columns) != 5:  # properties不只一列，且未指定weight_col，报错
            raise ValueError("Don't know which column to be loaded! Please set `weight_col` parameter!")
        else:  # properties只有一列，那就默认这一列是权重列
            self.weight_col = relfile.columns[-1]
            self.distance_df = relfile[~relfile[self.weight_col].isna()][[
                'origin_id', 'destination_id', self.weight_col]]

        self.adj_mx = np.zeros((len(self.geo_ids), len(self.geo_ids)), dtype=np.float32)
        for row in self.distance_df.values:
            if row[0] not in self.geo_to_ind or row[1] not in self.geo_to_ind:
                continue
            self.adj_mx[self.geo_to_ind[row[0]], self.geo_to_ind[row[1]]] = row[2]

        pickle.dump(self.adj_mx, open(os.path.join(self.args.output_dir, 'adj_mx.pkl'), 'wb+'))

        self.adj_idx = []
        for row in self.distance_df.values:
            if row[0] not in self.geo_to_ind or row[1] not in self.geo_to_ind:
                continue
            self.adj_idx.append([self.geo_to_ind[row[0]], self.geo_to_ind[row[1]]])
        self.adj_idx = np.array(self.adj_idx).transpose()
        pickle.dump(self.adj_idx, open(os.path.join(self.args.output_dir, 'adj_idx.pkl'), 'wb+'))



    def _load_dyna_3d(self, filename):
        """
        加载.dyna文件，格式[dyna_id, type, time, entity_id, properties(若干列)],
        .geo文件中的id顺序应该跟.dyna中一致,
        其中全局参数`data_col`用于指定需要加载的数据的列，不设置则默认全部加载
        Args:
            filename(str): 数据文件名，不包含后缀
        Returns:
            np.ndarray: 数据数组, 3d-array: (len_time, num_nodes, feature_dim)
        """
        # 加载数据集
        dynafile = pd.read_csv(filename)
        dynafile = dynafile[dynafile.columns[2:]]  # 从time列开始所有列
        # 求时间序列
        self.timesolts = list(dynafile['time'][:int(dynafile.shape[0] / len(self.geo_ids))])
        self.idx_of_timesolts = np.zeros((len(self.timesolts)), dtype=np.int32)
        if not dynafile['time'].isna().any():  # 时间没有空值
            self.timesolts = list(map(lambda x: x.replace('T', ' ').replace('Z', ''), self.timesolts))
            self.timesolts = np.array(self.timesolts, dtype='datetime64[s]')
            for idx, _ts in enumerate(self.timesolts):
                self.idx_of_timesolts[idx] = int(((_ts - _ts.astype('datetime64[D]')) / np.timedelta64(1, "D")) * 288)
        # 转3-d数组
        column_idx = dynafile.columns.values
        idx = -1
        for i in range(len(column_idx)):
            if column_idx[i] == args.task:
                idx = i
        if idx == -1:
            raise NotImplementedError
        df = dynafile[dynafile.columns[idx:idx+1]]

        len_time = len(self.timesolts)
        data = []
        time = []
        for i in range(0, df.shape[0], len_time):
            data.append(df[i:i + len_time].values)
            time.append(self.idx_of_timesolts.reshape(-1, 1))

        self.df = np.array(data, dtype=np.float)  # (len(self.geo_ids), len_time, feature_dim)
        self.df = np.concatenate((data, time), axis=-1)
        self.df = self.df.swapaxes(0, 1)  # (len_time, len(self.geo_ids), feature_dim)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', '-root_dir', type=str)
    parser.add_argument('--dataset', '-dataset', type=str)
    parser.add_argument('--output_dir', '-output_dir', type=str, required=False)
    parser.add_argument('--seq_length_x', '-seq_length_x', type=int, default=12, help="Sequence Length.", required=False)
    parser.add_argument('--seq_length_y', type=int, default=12, help="Sequence Length.", required=False)
    parser.add_argument('--y_start', type=int, default=1, help="Y pred start", required=False)
    parser.add_argument('--task', type=str, default='traffic_flow', required=False)
    args = parser.parse_args()

    if args.output_dir == "":
        args.output_dir = args.root_dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    return args

if __name__ == "__main__":
    args = parse_args()
    TrafficStateDataset(args)

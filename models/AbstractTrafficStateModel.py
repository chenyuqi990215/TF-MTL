from torch import nn

class AbstractTrafficStateModel(nn.Module):
    def __init__(self, config):
        super(AbstractTrafficStateModel, self).__init__()
        self.config = config

    def encoder(self, x):
        '''
            :param x: (batch_size, num_nodes, input_length, feature_dim)
            :return: (batch_size, num_nodes, output_dim)
        '''
        pass
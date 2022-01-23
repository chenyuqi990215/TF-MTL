from torch import nn

class AbstractTrafficStateModel(nn.Module):
    def __init__(self, config):
        super(AbstractTrafficStateModel, self).__init__()
        self.config = config

    def encoder(self, x):
        pass
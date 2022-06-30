def model_config(config):
    config.feature_dim = 1
    config.dropout = 0.3
    config.blocks = 4
    config.layers = 2
    config.gcn_bool = True
    config.addaptadj = True
    config.randomadj = True
    config.aptonly = True
    config.kernel_size = 2
    config.nhid = 32
    config.residual_channels = config.nhid
    config.dilation_channels = config.nhid
    config.skip_channels = config.nhid * 8
    config.end_channels = config.nhid * 16
    config.input_window = 12
    config.output_window = 128
    config.output_dim = 1
    return config
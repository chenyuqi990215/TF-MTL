def model_config(config):
    config.output_dim = 1
    config.blocks = [[1, 32, 64], [64, 32, 128]]
    config.keep_prob = 0.5
    return config
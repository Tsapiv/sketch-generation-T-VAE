from typing import Literal, Dict


class HParams:
    def __init__(self):
        self.enc_rnn_size = 256
        self.dec_rnn_size = 512
        self.z_size = 128
        self.num_mixture = 20
        self.input_dropout_prob = 0.9
        self.output_dropout_prob = 0.9
        self.batch_size = 100
        self.kl_weight_start = 0.01
        self.kl_decay_rate = 0.99995
        self.kl_tolerance = 0.2
        self.kl_weight = 100
        self.learning_rate = 0.001
        self.decay_rate = 0.9999
        self.min_learning_rate = 0.00001
        self.grad_clip = 1.
        self.max_seq_len = 200
        self.random_scale_factor = 0.15
        self.augment_stroke_prob = 0.10
        self.dim_feedforward = 2048
        self.single_embedding = False
        self.dist_matching: Literal['MMD', 'KL'] = 'KL'

        self.encoder: Literal['lstm', 'trans'] = 'trans'
        self.data_set = 'cat.npz'
        self.model_folder = 'trans_model'
        self.output_folder = 'trans_samples'

    def update(self, config: Dict):
        self.__dict__.update(config)

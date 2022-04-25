import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch import optim
from torch.autograd import Variable

from transformer import PositionalEncoding

use_cuda = torch.cuda.is_available()


###################################### hyperparameters
class HParams():
    def __init__(self):
        self.data_location = 'cat.npz'
        self.enc_hidden_size = 256
        self.dec_hidden_size = 512
        self.Nz = 123
        self.M = 20
        self.dropout = 0.9
        self.batch_size = 100
        self.eta_min = 0.01
        self.R = 0.99995
        self.KL_min = 0.2
        self.wKL = 0.5
        self.lr = 0.001
        self.lr_decay = 0.9999
        self.min_lr = 0.00001
        self.grad_clip = 1.
        self.temperature = 0.4
        self.max_seq_length = 200


hp = HParams()


################################# load and prepare data
def max_size(data):
    """larger sequence length in the data set"""
    sizes = [len(seq) for seq in data]
    return max(sizes)


def purify(strokes):
    """removes to small or too long sequences + removes large gaps"""
    data = []
    for seq in strokes:
        if seq.shape[0] <= hp.max_seq_length and seq.shape[0] > 10:
            seq = np.minimum(seq, 1000)
            seq = np.maximum(seq, -1000)
            seq = np.array(seq, dtype=np.float32)
            data.append(seq)
    return data


def calculate_normalizing_scale_factor(strokes):
    """Calculate the normalizing factor explained in appendix of sketch-rnn."""
    data = []
    for i in range(len(strokes)):
        for j in range(len(strokes[i])):
            data.append(strokes[i][j, 0])
            data.append(strokes[i][j, 1])
    data = np.array(data)
    return np.std(data)


def normalize(strokes):
    """Normalize entire dataset (delta_x, delta_y) by the scaling factor."""
    data = []
    scale_factor = calculate_normalizing_scale_factor(strokes)
    for seq in strokes:
        seq[:, 0:2] /= scale_factor
        data.append(seq)
    return data


dataset = np.load(hp.data_location, encoding='latin1', allow_pickle=True)
data = dataset['train']
data = purify(data)
data = normalize(data)
Nmax = max_size(data)


############################## function to generate a batch:
def make_batch(batch_size):
    batch_idx = np.random.choice(len(data), batch_size)
    batch_sequences = [data[idx] for idx in batch_idx]
    strokes = []
    lengths = []
    indice = 0
    for seq in batch_sequences:
        len_seq = len(seq[:, 0])
        new_seq = np.zeros((Nmax, 5))
        new_seq[:len_seq, :2] = seq[:, :2]
        new_seq[:len_seq - 1, 2] = 1 - seq[:-1, 2]
        new_seq[:len_seq, 3] = seq[:, 2]
        new_seq[(len_seq - 1):, 4] = 1
        new_seq[len_seq - 1, 2:4] = 0
        lengths.append(len(seq[:, 0]))
        strokes.append(new_seq)
        indice += 1

    if use_cuda:
        batch = Variable(torch.from_numpy(np.stack(strokes, 1)).cuda().float())
    else:
        batch = Variable(torch.from_numpy(np.stack(strokes, 1)).float())
    return batch, lengths


################################ adaptive lr
def lr_decay(optimizer):
    """Decay learning rate by a factor of lr_decay"""
    for param_group in optimizer.param_groups:
        if param_group['lr'] > hp.min_lr:
            param_group['lr'] *= hp.lr_decay
    return optimizer


################################# encoder and decoder modules
class EncoderRNN(nn.Module):
    def __init__(self, embed_dim=5, latent_dim=hp.enc_hidden_size, dropout=hp.dropout, nhead=4, num_layers=2,
                 layer_norm_eps=1e-5):
        super(EncoderRNN, self).__init__()
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.nhead = nhead
        self.num_layers = num_layers
        self.layer_norm_eps = layer_norm_eps

        self.fc_up = nn.Linear(self.embed_dim, self.latent_dim)

        self.mu_query = nn.Parameter(torch.randn(1, self.latent_dim))
        self.sigma_query = nn.Parameter(torch.randn(1, self.latent_dim))
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.latent_dim, nhead=self.nhead,
                                                   dim_feedforward=self.latent_dim * 4, dropout=self.dropout,
                                                   activation="gelu")
        norm_layer = nn.LayerNorm(self.latent_dim, eps=self.layer_norm_eps)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers, norm=norm_layer)

    def forward(self, inputs, batch_size, *args):

        x = self.fc_up(inputs)
        y = torch.zeros(batch_size, dtype=torch.long)
        xseq = torch.cat((self.mu_query[y][None], self.sigma_query[y][None], x), dim=0)
        xseq = self.sequence_pos_encoder(xseq)
        final = self.encoder(xseq)
        mu = final[0]
        sigma_hat = final[1]

        sigma = torch.exp(sigma_hat / 2.)
        # N ~ N(0,1)
        z_size = mu.size()
        if use_cuda:
            N = torch.normal(torch.zeros(z_size), torch.ones(z_size)).cuda()
        else:
            N = torch.normal(torch.zeros(z_size), torch.ones(z_size))
        z = mu + sigma * N
        # # mu and sigma_hat are needed for LKL loss
        return z, mu, sigma_hat


class DecoderRNN(nn.Module):
    def __init__(self, embed_dim=5, latent_dim=hp.enc_hidden_size, dropout=hp.dropout, nhead=4, num_layers=2,
                 layer_norm_eps=1e-5):
        super(DecoderRNN, self).__init__()
        # to init hidden and cell from z:
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.nhead = nhead
        self.num_layers = num_layers
        self.layer_norm_eps = layer_norm_eps

        # self.fc_up = nn.Linear(self.embed_dim, self.latent_dim)

        self.fc_params = nn.Linear(self.latent_dim, 6 * hp.M + 3)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.latent_dim, nhead=self.nhead,
                                                   dim_feedforward=self.latent_dim * 4, dropout=self.dropout,
                                                   activation="gelu")
        norm_layer = nn.LayerNorm(self.latent_dim, eps=self.layer_norm_eps)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.num_layers, norm=norm_layer)

        self.finallayer = nn.Linear(self.latent_dim, 6 * hp.M + 3)

    def forward(self, inputs, z, *args):
        # if inputs.shape[0] < Nmax:
        #     inputs = torch.cat([inputs, torch.zeros(Nmax - inputs.shape[0], *inputs.shape[1:])])
        # z = z + self.actionBiases
        # z = self.fc_hc(z[None])  # sequence of size 1
        # timequeries = self.fc_up(inputs)#torch.zeros(*inputs.shape[:-1], z.shape[-1], device=z.device)
        # timequeries = torch.normal(0, 1, size=(*inputs.shape[:-1], z.shape[-1]), device=z.device)
        timequeries = torch.ones(Nmax + 1, inputs.shape[-2], z.shape[-1], device=z.device)
        timequeries = self.sequence_pos_encoder(timequeries)
        outputs = self.decoder(tgt=timequeries, memory=z[None])

        y = self.fc_params(outputs.view(-1, self.latent_dim))

        # separate pen and mixture params:
        params = torch.split(y, split_size_or_sections=6, dim=1)
        params_mixture = torch.stack(params[:-1])  # trajectory
        params_pen = params[-1]  # pen up/down
        # identify mixture params:
        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy = torch.split(params_mixture, split_size_or_sections=1, dim=2)
        # preprocess params::
        if self.training:
            len_out = Nmax + 1
        else:
            len_out = 1

        pi = F.softmax(pi.transpose(0, 1).squeeze(), dim=-1).view(len_out, -1, hp.M)
        sigma_x = torch.exp(sigma_x.transpose(0, 1).squeeze()).view(len_out, -1, hp.M)
        sigma_y = torch.exp(sigma_y.transpose(0, 1).squeeze()).view(len_out, -1, hp.M)
        rho_xy = torch.tanh(rho_xy.transpose(0, 1).squeeze()).view(len_out, -1, hp.M)
        mu_x = mu_x.transpose(0, 1).squeeze().contiguous().view(len_out, -1, hp.M)
        mu_y = mu_y.transpose(0, 1).squeeze().contiguous().view(len_out, -1, hp.M)
        q = F.softmax(params_pen, dim=-1).view(len_out, -1, 3)
        return pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q


class Model:
    def __init__(self):
        # self.fc_up = nn.Linear(5, hp.enc_hidden_size)
        if use_cuda:
            self.encoder = EncoderRNN().cuda()
            self.decoder = DecoderRNN().cuda()
        else:
            self.encoder = EncoderRNN()
            self.decoder = DecoderRNN()
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), hp.lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), hp.lr)
        self.eta_step = hp.eta_min

    def make_target(self, batch, lengths):
        if use_cuda:
            eos = torch.stack([torch.Tensor([0, 0, 0, 0, 1])] * batch.size()[1]).cuda().unsqueeze(0)
        else:
            eos = torch.stack([torch.Tensor([0, 0, 0, 0, 1])] * batch.size()[1]).unsqueeze(0)
        batch = torch.cat([batch, eos], 0)
        mask = torch.zeros(Nmax + 1, batch.size()[1])
        for indice, length in enumerate(lengths):
            mask[:length, indice] = 1
        if use_cuda:
            mask = mask.cuda()
        dx = torch.stack([batch.data[:, :, 0]] * hp.M, 2)
        dy = torch.stack([batch.data[:, :, 1]] * hp.M, 2)
        p1 = batch.data[:, :, 2]
        p2 = batch.data[:, :, 3]
        p3 = batch.data[:, :, 4]
        p = torch.stack([p1, p2, p3], 2)
        return mask, dx, dy, p

    def train(self, epoch):
        self.encoder.train()
        self.decoder.train()
        batch, lengths = make_batch(hp.batch_size)
        # encode:
        z, mu, sigma = self.encoder(batch, hp.batch_size)
        # create start of sequence:
        if use_cuda:
            sos = torch.stack([torch.Tensor([0, 0, 1, 0, 0])] * hp.batch_size).cuda().unsqueeze(0)
        else:
            sos = torch.stack([torch.Tensor([0, 0, 1, 0, 0])] * hp.batch_size).unsqueeze(0)
        # had sos at the begining of the batch:
        batch_init = torch.cat([sos, batch], 0)
        # expend z to be ready to concatenate with inputs:
        # z_stack = torch.stack([z] * (Nmax + 1))
        # # inputs is concatenation of z and batch_inputs
        # inputs = torch.cat([batch_init, z_stack], 2)
        # decode:
        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q = self.decoder(batch_init, z)
        # prepare targets:
        mask, dx, dy, p = self.make_target(batch, lengths)
        # prepare optimizers:
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        # update eta for LKL:
        self.eta_step = 1 - (1 - hp.eta_min) * hp.R
        # compute losses:
        LKL = self.kullback_leibler_loss(mu, sigma)
        LR = self.reconstruction_loss(mask, pi, dx, dy, mu_x, mu_y, sigma_x, sigma_y, rho_xy, p, q)
        loss = LR + LKL
        # gradient step
        loss.backward()
        # gradient cliping
        nn.utils.clip_grad_norm_(self.encoder.parameters(), hp.grad_clip)
        nn.utils.clip_grad_norm_(self.decoder.parameters(), hp.grad_clip)
        # optim step
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        # some print and save:
        if epoch % 1 == 0:
            print('epoch', epoch, 'loss', loss.item(), 'LR', LR.item(), 'LKL', LKL.item())
            self.encoder_optimizer = lr_decay(self.encoder_optimizer)
            self.decoder_optimizer = lr_decay(self.decoder_optimizer)
        if epoch % 100 == 0:
            # self.save(epoch)
            self.conditional_generation(epoch)

    def bivariate_normal_pdf(self, dx, dy, mu_x, mu_y, sigma_x, sigma_y, rho_xy):
        z_x = ((dx - mu_x) / sigma_x) ** 2
        z_y = ((dy - mu_y) / sigma_y) ** 2
        z_xy = (dx - mu_x) * (dy - mu_y) / (sigma_x * sigma_y)
        z = z_x + z_y - 2 * rho_xy * z_xy
        exp = torch.exp(-z / (2 * (1 - rho_xy ** 2)))
        norm = 2 * np.pi * sigma_x * sigma_y * torch.sqrt(1 - rho_xy ** 2)
        return exp / norm

    def reconstruction_loss(self, mask, pi, dx, dy, mu_x, mu_y, sigma_x, sigma_y, rho_xy, p, q):
        pdf = self.bivariate_normal_pdf(dx, dy, mu_x, mu_y, sigma_x, sigma_y, rho_xy)
        LS = -torch.sum(mask * torch.log(1e-5 + torch.sum(pi * pdf, 2))) / float(Nmax * hp.batch_size)
        LP = -torch.sum(p * torch.log(q)) / float(Nmax * hp.batch_size)
        return LS + LP

    def kullback_leibler_loss(self, mu, sigma):
        LKL = -0.5 * torch.sum(1 + sigma - mu ** 2 - torch.exp(sigma)) / float(hp.batch_size)
        # if use_cuda:
        #     KL_min = Variable(torch.Tensor([hp.KL_min]).cuda()).detach()
        # else:
        #     KL_min = Variable(torch.Tensor([hp.KL_min])).detach()
        return hp.wKL * self.eta_step * LKL

    def save(self, epoch):
        sel = np.random.random(1)
        torch.save(self.encoder.state_dict(), 'encoderRNN_sel_%3f_epoch_%d.pth' % (sel, epoch))
        torch.save(self.decoder.state_dict(), 'decoderRNN_sel_%3f_epoch_%d.pth' % (sel, epoch))

    def load(self, encoder_name, decoder_name):
        saved_encoder = torch.load(encoder_name)
        saved_decoder = torch.load(decoder_name)
        self.encoder.load_state_dict(saved_encoder)
        self.decoder.load_state_dict(saved_decoder)

    def conditional_generation(self, epoch):
        batch, lengths = make_batch(1)
        # should remove dropouts:
        self.encoder.train(False)
        self.decoder.train(False)
        # encode:
        z, _, _ = self.encoder(batch, 1)
        if use_cuda:
            sos = Variable(torch.Tensor([0, 0, 1, 0, 0]).view(1, 1, -1).cuda())
        else:
            sos = Variable(torch.Tensor([0, 0, 1, 0, 0]).view(1, 1, -1))
        s = sos
        # seq_x = []
        # seq_y = []
        # seq_z = []
        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q = self.decoder(s, z)
        seq = np.asarray(self.sample_next_state(pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q))
        # for i in range(Nmax):
        #     # input = torch.cat([s], 2)
        #     # decode:
        #     pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q = self.decoder(s, z)
        #     # sample from parameters:
        #     s, dx, dy, pen_down, eos = self.sample_next_state()
        #     # ------
        #     seq_x.append(dx)
        #     seq_y.append(dy)
        #     seq_z.append(pen_down)
        #     if eos:
        #         print(i)
        #         break
        # visualize result:
        x_sample = np.cumsum(seq[:, 0], 0)
        y_sample = np.cumsum(seq[:, 1], 0)
        z_sample = np.array(seq[:, 2])
        sequence = np.stack([x_sample, y_sample, z_sample]).T
        make_image(sequence, epoch)

    def sample_next_state(self, pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q):

        def adjust_temp(pi_pdf):
            pi_pdf = np.log(pi_pdf) / hp.temperature
            pi_pdf -= pi_pdf.max()
            pi_pdf = np.exp(pi_pdf)
            pi_pdf /= pi_pdf.sum()
            return pi_pdf

        # get mixture indice:
        sequence = []
        for idx in range(len(pi[0])):
            # pi = pi[idx, 0, :].cpu().numpy()
            # pi = adjust_temp(pi[idx, 0, :].cpu().numpy())
            pi_idx = np.random.choice(hp.M, p=adjust_temp(pi[0, idx, :].detach().cpu().numpy()))
            # get pen state:
            # q = q[idx, 0, :].cpu().numpy()
            # q = adjust_temp(q[idx, 0, :].cpu().numpy())
            q_idx = np.random.choice(3, p=adjust_temp(q[0, idx, :].detach().cpu().numpy()))
            # get mixture params:
            # mu_x = mu_x[idx, 0, pi_idx].cpu().numpy()
            # mu_y = mu_y[idx, 0, pi_idx].cpu().numpy()
            # sigma_x = sigma_x[idx, 0, pi_idx].cpu().numpy()
            # sigma_y = sigma_y[idx, 0, pi_idx].cpu().numpy()
            # rho_xy = rho_xy[idx, 0, pi_idx].cpu().numpy()
            x, y = sample_bivariate_normal(mu_x[0, idx, pi_idx].detach().cpu().numpy(),
                                           mu_y[0, idx, pi_idx].detach().cpu().numpy(),
                                           sigma_x[0, idx, pi_idx].detach().cpu().numpy(),
                                           sigma_y[0, idx, pi_idx].detach().cpu().numpy(),
                                           rho_xy[0, idx, pi_idx].detach().cpu().numpy(),
                                           greedy=False)
            # next_state = torch.zeros(5)
            # next_state[0] = x
            # next_state[1] = y
            # next_state[q_idx + 2] = 1
            sequence.append((x, y, q_idx == 1))
            if q_idx == 2:
                return sequence
        return sequence



def sample_bivariate_normal(mu_x, mu_y, sigma_x, sigma_y, rho_xy, greedy=False):
    # inputs must be floats
    if greedy:
        return mu_x, mu_y
    mean = [mu_x, mu_y]
    sigma_x *= np.sqrt(hp.temperature)
    sigma_y *= np.sqrt(hp.temperature)
    cov = [[sigma_x * sigma_x, rho_xy * sigma_x * sigma_y], [rho_xy * sigma_x * sigma_y, sigma_y * sigma_y]]
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]


def make_image(sequence, epoch, name='_output_'):
    """plot drawing with separated strokes"""
    strokes = np.split(sequence, np.where(sequence[:, 2] > 0)[0] + 1)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for s in strokes:
        plt.plot(s[:, 0], -s[:, 1])
    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    pil_image = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
    name = str(epoch) + name + '.jpg'
    pil_image.save(name, "JPEG")
    plt.close("all")


if __name__ == "__main__":
    model = Model()
    for epoch in range(50001):
        model.train(epoch)

    '''
    model.load('encoder.pth','decoder.pth')
    model.conditional_generation(0)
    #'''

import os
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from parameters import HParams
from utils import to_normal_strokes, make_grid_svg, draw_strokes, sample_bivariate_normal

device = torch.device("cpu")  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
from loss_functions import reconstruction_loss, mmd_penalty
from components import EncoderTrans, EncoderRNN, DecoderRNN


class SketchModel(nn.Module):
    def __init__(self, hp: HParams):
        super(SketchModel, self).__init__()
        if hp.encoder == 'trans':
            self.encoder = EncoderTrans(hp).to(device)
        elif hp.encoder == 'lstm':
            self.encoder = EncoderRNN(hp).to(device)
        else:
            raise TypeError('Unknown encoder type')
        self.decoder = DecoderRNN(hp).to(device)
        self.train_params = self.parameters()
        self.optimizer = optim.Adam(self.train_params, hp.learning_rate)
        self.hp = hp

    def train_model(self, batch, lengths, step):
        self.train()
        self.optimizer.zero_grad()

        curr_learning_rate = ((self.hp.learning_rate - self.hp.min_learning_rate) *
                              self.hp.decay_rate ** step + self.hp.min_learning_rate)
        curr_kl_weight = (self.hp.kl_weight - (self.hp.kl_weight - self.hp.kl_weight_start) *
                          self.hp.kl_decay_rate ** step)

        post_dist = self.encoder(batch, lengths)

        z_vector = post_dist.rsample()
        start_token = torch.stack([torch.Tensor([0, 0, 1, 0, 0])] * self.hp.batch_size).unsqueeze(0).to(device)
        batch_init = torch.cat([start_token, batch], 0)
        z_stack = torch.stack([z_vector] * (self.hp.max_seq_len + 1))
        inputs = torch.cat([batch_init, z_stack], 2)

        output, _ = self.decoder(inputs, z_vector, lengths + 1)

        end_token = torch.stack([torch.Tensor([0, 0, 0, 0, 1])] * batch.shape[1]).unsqueeze(0).to(device)
        batch = torch.cat([batch, end_token], 0)
        x_target = batch.permute(1, 0, 2)  # batch-> Seq_Len, Batch, Feature_dim

        recons_loss = reconstruction_loss(output, x_target)

        if self.hp.dist_matching == 'KL':
            prior_distribution = torch.distributions.Normal(torch.zeros_like(post_dist.mean),
                                                            torch.ones_like(post_dist.stddev))
            kl_cost = torch.max(torch.distributions.kl_divergence(post_dist, prior_distribution).sum(),
                                torch.tensor(self.hp.kl_tolerance).to(device))
            loss = recons_loss + curr_kl_weight * kl_cost
        elif self.hp.dist_matching == 'MMD':
            z_fake = torch.randn(z_vector.shape).to(device)
            kl_cost = mmd_penalty(z_vector, z_fake)
            loss = recons_loss + 100 * kl_cost
        else:
            raise TypeError('Unknown distribution matching type')

        self.set_learning_rate(curr_learning_rate)
        loss.backward()
        nn.utils.clip_grad_norm(self.train_params, self.hp.grad_clip)
        self.optimizer.step()

        return kl_cost.item(), recons_loss.item(), loss.item(), curr_learning_rate, curr_kl_weight

    def generate(self, dataloader=None):
        self.eval()
        batch = None
        if dataloader is not None:
            batch, lengths = dataloader
            batch = torch.as_tensor(batch, device=device, dtype=torch.float).view(-1, 1, 5)
            lengths = torch.as_tensor(lengths, device=device, dtype=torch.int64)
            post_dist = self.encoder(batch, lengths)
            z_vector = post_dist.sample()
        else:
            z_vector = torch.randn(1, self.hp.z_size).to(device)

        start_token = torch.Tensor([0, 0, 1, 0, 0]).view(1, 1, -1).to(device)
        state = start_token
        hidden_cell = None
        gen_strokes = []
        for i in range(self.hp.max_seq_len):
            input = torch.cat([state, z_vector.unsqueeze(0)], 2)
            output, hidden_cell = self.decoder(input, z_vector, hidden_cell=hidden_cell, isTrain=False)
            state, next_state = self.sample_next_state(output)
            gen_strokes.append(next_state)
            if next_state[-1] == 1:
                break

        gen_strokes = torch.stack(gen_strokes).cpu().numpy()
        return to_normal_strokes(gen_strokes), to_normal_strokes(
            batch[:, 0, :].cpu().numpy()) if batch is not None else None

    def complete(self, dataloader, ratio: float = 1):
        self.eval()

        batch, lengths = dataloader
        batch = torch.as_tensor(batch, device=device, dtype=torch.float).view(-1, 1, 5)
        lengths = torch.as_tensor(lengths, device=device, dtype=torch.int64)
        original = to_normal_strokes(batch[:, 0, :].cpu().numpy())
        if ratio != 1:
            lengths = torch.LongTensor((lengths.cpu().numpy() * ratio).astype(np.int64)).to(device)
            batch[lengths[0]:, :, :] = 0
        post_dist = self.encoder(batch, lengths)
        z_vector = post_dist.sample()

        state = batch[0, 0, :].view(1, 1, -1)
        hidden_cell = None
        gen_strokes = [torch.squeeze(state).cpu()]
        for i in range(self.hp.max_seq_len):
            input = torch.cat([state, z_vector.unsqueeze(0)], 2)
            output, hidden_cell = self.decoder(input, z_vector, hidden_cell=hidden_cell, isTrain=False)
            if i < lengths[0] - 1:
                state = batch[i + 1, 0, :].view(1, 1, -1)
                gen_strokes.append(torch.squeeze(state).cpu())
            else:
                # batch = torch.cat([batch.to(device), state.to(device)], 0)
                # lengths += 1
                # post_dist = self.encoder(batch, lengths)
                # z_vector = post_dist.sample()
                state, next_state = self.sample_next_state(output)
                gen_strokes.append(next_state)
                if next_state[-1] == 1:
                    break
        gen_strokes = torch.stack(gen_strokes).cpu().numpy()
        return to_normal_strokes(gen_strokes), original#, to_normal_strokes(batch[:, 0, :].cpu().numpy())

    def generate_many(self, dataloader=None, step=0, number_of_sample=100, condition=False, grid_width=10, save=True,
                      mode: Literal['generate', 'complete'] = 'generate', ratio=0.5):

        input_strokes = []
        # shorten_input_strokes = []
        reconstructed_strokes = []

        for i in range(number_of_sample):
            if mode == 'complete':
                gen_strokes, org_strokes = self.complete(dataloader.valid_batch(1), ratio=ratio)
            elif mode == 'generate':
                gen_strokes, org_strokes = self.generate(dataloader.valid_batch(1) if condition else None)
            else:
                raise TypeError('Unknown mode type')
            if org_strokes is not None:
                input_strokes.append(org_strokes)
                # shorten_input_strokes.append(shorten_org_strokes)
            reconstructed_strokes.append(gen_strokes)
            print(i)
        if save:
            if not os.path.exists(self.hp.output_folder):
                os.makedirs(self.hp.output_folder)
            if grid_width:
                if dataloader is not None and condition:
                    draw_strokes(make_grid_svg(input_strokes, grid_width=grid_width),
                                 svg_filename=os.path.join(self.hp.output_folder, f'org-{self.hp.encoder}-{step}.svg'))
                    # draw_strokes(make_grid_svg(shorten_input_strokes, grid_width=grid_width),
                    #              svg_filename=os.path.join(self.hp.output_folder,
                    #                                        f'short-{self.hp.encoder}-{step}.svg'))
                draw_strokes(make_grid_svg(reconstructed_strokes, grid_width=grid_width),
                             svg_filename=os.path.join(self.hp.output_folder, f'gen-{self.hp.encoder}-{step}.svg'))
            else:
                for idx in range(len(reconstructed_strokes)):
                    if dataloader is not None and condition:
                        draw_strokes(input_strokes[idx],
                                     svg_filename=os.path.join(self.hp.output_folder,
                                                               f'org-{self.hp.encoder}-{step}-{idx}.svg'))
                        # draw_strokes(shorten_input_strokes[idx],
                        #              svg_filename=os.path.join(self.hp.output_folder,
                        #                                        f'short-{self.hp.encoder}-{step}-{idx}.svg'))
                    draw_strokes(reconstructed_strokes[idx],
                                 svg_filename=os.path.join(self.hp.output_folder,
                                                           f'gen-{self.hp.encoder}-{step}-{idx}.svg'))
        return input_strokes if dataloader else None, reconstructed_strokes

    def sample_next_state(self, output, temperature=0.2):

        def adjust_temp(pi_pdf):
            pi_pdf = np.log(pi_pdf) / temperature
            pi_pdf -= pi_pdf.max()
            pi_pdf = np.exp(pi_pdf)
            pi_pdf /= pi_pdf.sum()
            return pi_pdf

        [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen_logits] = output
        # get mixture indices:
        o_pi = o_pi.data[0, :].cpu().numpy()
        o_pi = adjust_temp(o_pi)
        pi_idx = np.random.choice(self.hp.num_mixture, p=o_pi)
        # get pen state:
        o_pen = torch.softmax(o_pen_logits, dim=-1)
        o_pen = o_pen.data[0, :].cpu().numpy()
        pen = adjust_temp(o_pen)
        pen_idx = np.random.choice(3, p=pen)
        # get mixture params:
        o_mu1 = o_mu1.data[0, pi_idx].item()
        o_mu2 = o_mu2.data[0, pi_idx].item()
        o_sigma1 = o_sigma1.data[0, pi_idx].item()
        o_sigma2 = o_sigma2.data[0, pi_idx].item()
        o_corr = o_corr.data[0, pi_idx].item()
        x, y = sample_bivariate_normal(o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, temperature=temperature, greedy=False)
        next_state = torch.zeros(5)
        next_state[0] = x
        next_state[1] = y
        next_state[pen_idx + 2] = 1
        return next_state.to(device).view(1, 1, -1), next_state

    def set_learning_rate(self, curr_learning_rate):
        for g in self.optimizer.param_groups:
            g['lr'] = curr_learning_rate

    @staticmethod
    def draw_batch_input(dataloader, number_of_sample=100):

        Batch_Input = []
        row_count = 0
        col_count = 0

        for i_x in range(number_of_sample):
            batch, lengths = dataloader.valid_batch(1)
            batch_input = to_normal_strokes(batch[:, 0, :].cpu().numpy())

            if (i_x + 0) % 10 == 0:
                row_count = row_count + 1
                col_count = 0

            Batch_Input.append([batch_input, [row_count - 1, col_count]])
            col_count = col_count + 1
            print(i_x)

        Batch_Input_grid = make_grid_svg(Batch_Input)
        draw_strokes(Batch_Input_grid, svg_filename='sample.svg')

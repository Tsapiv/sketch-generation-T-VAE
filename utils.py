import torch

use_cuda = True
from IPython.display import SVG, display
import numpy as np
import svgwrite
from six.moves import xrange
import math
import torch.nn as nn
from rdp import rdp


# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def sample_bivariate_normal(mu_x, mu_y, sigma_x, sigma_y, rho_xy, temperature=0.2, greedy=False):
    # inputs must be floats
    if greedy:
        return mu_x, mu_y
    mean = [mu_x, mu_y]
    sigma_x *= np.sqrt(temperature)  # confusion
    sigma_y *= np.sqrt(temperature)  # confusion
    cov = [[sigma_x * sigma_x, rho_xy * sigma_x * sigma_y], [rho_xy * sigma_x * sigma_y, sigma_y * sigma_y]]
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]


def sample_gaussian_2d(mu1, mu2, s1, s2, rho, temp=1.0, greedy=False):
    if greedy:
        return mu1, mu2
    mean = [mu1, mu2]
    s1 *= temp * temp
    s2 *= temp * temp
    cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]]
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]


# little function that displays vector images and saves them to .svg
def draw_strokes(data, factor=0.2, svg_filename='./sample.svg'):
    min_x, max_x, min_y, max_y = get_bounds(data, factor)
    dims = (50 + max_x - min_x, 50 + max_y - min_y)
    dwg = svgwrite.Drawing(svg_filename, size=dims)
    dwg.add(dwg.rect(insert=(0, 0), size=dims, fill='white'))
    lift_pen = 1
    abs_x = 25 - min_x
    abs_y = 25 - min_y
    p = "M%s,%s " % (abs_x, abs_y)
    command = "m"
    for i in xrange(len(data)):
        if (lift_pen == 1):
            command = "m"
        elif (command != "l"):
            command = "l"
        else:
            command = ""
        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor
        lift_pen = data[i, 2]
        p += command + str(x) + "," + str(y) + " "
    the_color = "black"
    stroke_width = 1
    dwg.add(dwg.path(p).stroke(the_color, stroke_width).fill("none"))
    dwg.save()
    display(SVG(dwg.tostring()))


# generate a 2D grid of many vector drawings
def make_grid_svg(s_list, grid_space=10.0, grid_space_x=16.0, grid_width=10):
    def get_start_and_end(x):
        x = np.array(x)
        x = x[:, 0:2]
        x_start = x[0]
        x_end = x.sum(axis=0)
        x = x.cumsum(axis=0)
        x_max = x.max(axis=0)
        x_min = x.min(axis=0)
        center_loc = (x_max + x_min) * 0.5
        return x_start - center_loc, x_end

    x_pos = 0.0
    y_pos = 0.0
    result = [[x_pos, y_pos, 1]]
    for i, sample in enumerate(s_list):
        grid_loc = divmod(i, grid_width)
        grid_y = grid_loc[0] * grid_space + grid_space * 0.5
        grid_x = grid_loc[1] * grid_space_x + grid_space_x * 0.5
        start_loc, delta_pos = get_start_and_end(sample)

        loc_x = start_loc[0]
        loc_y = start_loc[1]
        new_x_pos = grid_x + loc_x
        new_y_pos = grid_y + loc_y
        result.append([new_x_pos - x_pos, new_y_pos - y_pos, 0])

        result += sample.tolist()
        result[-1][2] = 1
        x_pos = new_x_pos + delta_pos[0]
        y_pos = new_y_pos + delta_pos[1]
    return np.array(result)


def to_normal_strokes(big_stroke):
    """Convert from stroke-5 format (from sketch-rnn paper) back to stroke-3."""
    l = 0
    for i in range(len(big_stroke)):
        if big_stroke[i, 4] > 0:
            l = i
            break
    if l == 0:
        l = len(big_stroke)
    result = np.zeros((l, 3))
    result[:, 0:2] = big_stroke[0:l, 0:2]
    result[:, 2] = big_stroke[0:l, 3]
    return result


def get_bounds(data, factor: float = 10):
    """Return bounds of data."""
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0

    abs_x = 0
    abs_y = 0
    for i in range(len(data)):
        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor
        abs_x += x
        abs_y += y
        min_x = min(min_x, abs_x)
        min_y = min(min_y, abs_y)
        max_x = max(max_x, abs_x)
        max_y = max(max_y, abs_y)

    return (min_x, max_x, min_y, max_y)


def normalize(seq):
    seq = np.asarray(seq)
    x = np.zeros_like(seq)
    x[:, 0] = np.cumsum(seq[:, 0])
    x[:, 1] = np.cumsum(seq[:, 1])
    delimiters = np.nonzero(seq[:, -1])[0]
    if len(delimiters) == 0:
        new_seq = np.diff(rdp(x, epsilon=2), axis=0, prepend=0)
        new_seq = np.hstack([new_seq, np.zeros((len(new_seq), 1))])
        new_seq[-1][-1] = 1
        return new_seq
    else:
        prev = 0
        seq_chunks = []
        last = np.asarray([[0, 0]])
        for idx in np.append(delimiters, len(x) - 1):
            points = rdp(x[prev: idx + 1, :-1], epsilon=2)
            points = np.vstack([last, points])
            new_seq = np.diff(points, axis=0)
            new_seq = np.hstack([new_seq, np.zeros((len(new_seq), 1))])
            new_seq[-1][-1] = 1
            prev = idx + 1
            last = x[idx, :-1].reshape(1, -1)
            seq_chunks.append(new_seq)
    seq = np.vstack(seq_chunks)
    return seq


def convert3to5(seq, max_seq_len, complete=False):
    seq = normalize(seq)
    seq[:, :2] /= np.std(seq[:, :2])
    len_seq = len(seq)
    new_seq = np.zeros((int(max(max_seq_len, len(seq) * 1.1)), 5))
    new_seq[:len_seq, :2] = seq[:, :2]
    new_seq[:len_seq - 1, 2] = 1 - seq[:-1, 2]
    new_seq[:len_seq, 3] = seq[:, 2]
    if not complete:
        new_seq[(len_seq - 1):, 4] = 1
        new_seq[len_seq - 1, 2:4] = 0
    return new_seq, [len_seq]

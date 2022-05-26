import argparse
import json

import torch

from dataset import SketchDataset
from model import SketchModel
from parameters import HParams

device = torch.device("cpu")  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Sketch Generation Model')
    #
    # parser.add_argument('--config', type=str, required=True)
    # parser.add_argument('--model', type=str, required=True)
    # parser.add_argument('--n_samples', type=int, default=100)
    # parser.add_argument('--condition', action='store_true')

    # args = parser.parse_args()
    config = HParams()
    path = 'models/firetruck/trans'
    config_path = f'{path}/config.json'
    config.__dict__.update(json.load(open(config_path, 'r')))
    config.output_folder = 'samples'
    config.data_set = 'data/firetruck.npz'
    model_path = f'{path}/trans-5999.pth'
    print(config)

    dataloader = SketchDataset(config)


    model = SketchModel(config)
    model.load_state_dict(torch.load(model_path, map_location=device))

    generated_samples = model.generate_many(dataloader=dataloader, number_of_sample=16, step=5,
                                            condition=True, grid_width=4, mode='complete', ratio=0.5)

import argparse
import json

import torch

from dataset import SketchDataset
from model import SketchModel
from parameters import HParams

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sketch Generation Model')

    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--condition', action='store_true')

    args = parser.parse_args()
    config = HParams()
    config.__dict__.update(json.load(open(args.config, 'r')))
    config.output_folder = 'lstm_generation'
    print(config)

    dataloader = SketchDataset(config)
    model = SketchModel(config)
    model.load_state_dict(torch.load(args.model, map_location=device))

    generated_samples = model.generate_many(dataloader=dataloader, number_of_sample=args.n_samples,
                                            condition=args.condition, grid_width=0)

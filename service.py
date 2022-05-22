import json
from argparse import ArgumentParser
from enum import Enum

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from pydantic.typing import List, Optional

from dataset import SketchDataset
from model import SketchModel, device
from parameters import HParams

app = FastAPI()

parser = ArgumentParser()
parser.add_argument('--port', type=int, default=56000)
args = parser.parse_args()


class ModelType(str, Enum):
    lstm = "lstm"
    transformer = "transformer"


class ModelCompeteQuery(BaseModel):
    strokes: List[List[float]]
    type: ModelType
    version: str = 'latest'


class ModelGenerateQuery(BaseModel):
    latent_vector: Optional[List[float]]
    type: ModelType
    version: str = 'latest'


def generate_image(category: str, query: ModelGenerateQuery):
    model_path = "lstm_model/sketch_rnn-rnn_2999.pth"

    config = HParams()
    print(json.load(open(f"{model_path.split('/')[0]}/config.json", 'r')))
    config.__dict__.update(json.load(open(f"{model_path.split('/')[0]}/config.json", 'r')))
    model = SketchModel(config)
    model.load_state_dict(torch.load(model_path, map_location=device))

    _, strokes = model.generation(number_of_sample=1, condition=False, one_image=False, save=False)
    return strokes[0].tolist()


@app.post("/complete/{category}")
def complete(category: str, query: ModelCompeteQuery):
    pass


@app.post("/generate/{category}")
def generate(category: str, query: ModelGenerateQuery):
    return {'strokes': generate_image(category, query)}


if __name__ == '__main__':
    uvicorn.run("service:app", reload=True)

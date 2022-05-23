import json
import os.path
from argparse import ArgumentParser
from enum import Enum

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from pydantic.typing import List, Optional, Union

from model import SketchModel, device
from parameters import HParams

app = FastAPI()

parser = ArgumentParser()
parser.add_argument('--port', type=int, default=56000)
args = parser.parse_args()

MODELS_LOCATION = {
    'lstm': {
        'cat': 'lstm_model/sketch_rnn-rnn_2999.pth'
    }
}


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


def setup_model(category: str, query: Union[ModelGenerateQuery, ModelCompeteQuery]):
    model_path = MODELS_LOCATION[query.type][category]
    config_path = os.path.join(os.path.dirname(model_path), 'config.json')
    config = HParams()
    config.update(json.load(open(config_path, 'r')))
    model = SketchModel(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def generate_image(category: str, query: ModelGenerateQuery):
    model = setup_model(category, query)
    strokes, _ = model.generate()
    return strokes.tolist()


def complete_image(category: str, query: ModelCompeteQuery):
    model = setup_model(category, query)
    strokes = model.complete(np.asarray(query.strokes))
    return strokes.tolist()


@app.post("/complete/{category}")
def complete(category: str, query: ModelCompeteQuery):
    return {'strokes': complete_image(category, query)}


@app.post("/generate/{category}")
def generate(category: str, query: ModelGenerateQuery):
    return {'strokes': generate_image(category, query)}


if __name__ == '__main__':
    uvicorn.run("service:app", reload=True)

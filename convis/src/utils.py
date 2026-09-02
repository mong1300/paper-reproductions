import torch
import yaml
import random
import numpy as np
import json
from pathlib import Path
from pygments.formatters import img


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(model_seed, image_seed):
    random.seed(image_seed)
    np.random.seed(image_seed)
    torch.manual_seed(model_seed)


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

def save_json(path, record):
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
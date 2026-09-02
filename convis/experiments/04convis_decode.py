import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import time
import torch
from PIL import Image
from diffusers import AutoPipelineForText2Image
from src.model import load_model, generate_caption, convis_decode
from src.t2i import load_t2i, regenerate
from src.model import load_model, generate_caption
from src.utils import load_config, set_seed, sync, save_json, load_json
import random

DTYPES = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}

out = ROOT / "outputs"
out.mkdir(parents=True, exist_ok=True)
img_dir = ROOT / "data/coco/images"
img_dir.mkdir(parents=True, exist_ok=True)
# convis

def main():
    cfg = load_config(HERE / "configs" / os.environ.get("CONVIS_CONFIG", "local_debug.yaml"))
    set_seed(model_seed=cfg["model_seed"], image_seed=cfg["image_seed"])
    dtype = DTYPES[cfg["dtype"]]
    model_id = cfg["model_id"]
    device = cfg["device"]
    prompt = cfg["prompt"]

    model, processor = load_model(model_id, dtype, device, load_in_4bit=cfg.get("load_in_4bit", False))
    captions = load_json(out / "coco/captions.json")

    answers = []
    for c in captions:
        ori_img = Image.open(img_dir / c["file_name"])
        recons = []
        for i in range(cfg['caption_num']):
            img = Image.open(out / "coco" / f"{c['file_name'][:-4]}_{i}.png")
            recons.append(img)

        result = convis_decode(model, processor, ori_img, recons, prompt, cfg['max_new_tokens'], cfg['alpha'], cfg['lam'])
        answer_convis = {"file_name" : c['file_name'], "image_id": c['id'], "caption" : result, "method": cfg['method'],
                  "model_seed": cfg["model_seed"], "alpha": cfg['alpha'], "lambda": cfg['lam'], "n": cfg['num_images']}
        answers.append(answer_convis)

    save_json(out / "coco/answers.json", answers)

if __name__ == "__main__":
    main()

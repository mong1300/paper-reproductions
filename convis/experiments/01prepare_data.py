import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import torch
from src.utils import load_config, set_seed, save_json
import random
from pycocotools.coco import COCO

img_dir = ROOT / "data/coco/images"
img_dir.mkdir(parents=True, exist_ok=True)

def main():
    cfg = load_config(HERE / "configs" / os.environ.get("CONVIS_CONFIG", "local_debug.yaml"))
    set_seed(model_seed=cfg["model_seed"], image_seed=cfg["image_seed"])

    coco = COCO(ROOT / "data/coco/instances_val2014.json")
    img_ids = coco.getImgIds()
    sampled = random.sample(img_ids, cfg["num_images"])
    coco.download(tarDir=str(img_dir), imgIds=sampled)

    samples = coco.loadImgs(sampled)
    save_json(ROOT / "data/coco/images/samples.json", samples)




if __name__ == "__main__":
    main()

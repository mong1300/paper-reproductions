import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import torch
from PIL import Image
from src.t2i import load_t2i, regenerate
from src.utils import load_config, set_seed, load_json

DTYPES = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}

out = ROOT / "outputs"
out.mkdir(parents=True, exist_ok=True)
(out / "coco").mkdir(parents=True, exist_ok=True)
img_dir = ROOT / "data/coco/images"
img_dir.mkdir(parents=True, exist_ok=True)
# convis

def main():
    cfg = load_config(HERE / "configs" / os.environ.get("CONVIS_CONFIG", "local_debug.yaml"))
    set_seed(model_seed=cfg["model_seed"], image_seed=cfg["image_seed"])
    dtype = DTYPES[cfg["dtype"]]
    model_id = cfg["t2i_id"]
    device = cfg["device"]

    pipe = load_t2i(dtype, device, model_id)

    captions = load_json(out / "coco/captions.json")

    for i, c in enumerate(captions):
        print(f"\nProcessing {i}th {c['file_name']}")
        img = Image.open(img_dir / c["file_name"])
        for i, caption in enumerate(c['captions']):
            recon = regenerate(pipe, caption, cfg["num_steps"])
            recon.save(out / "coco" / f"{c['file_name'][:-4]}_{i}.png")

if __name__ == "__main__":
    main()

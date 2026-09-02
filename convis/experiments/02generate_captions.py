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

    samples = load_json(img_dir / "samples.json")

    model, processor = load_model(cfg["model_id"], DTYPES[cfg["dtype"]], cfg["device"],
                                     load_in_4bit=cfg.get("load_in_4bit", False))

    captions_out = []
    for sample in samples:
        sample_file_name = sample["file_name"]
        sample_id = sample["id"]
        img = Image.open(img_dir / sample_file_name).convert("RGB")

        captions = []
        for i in range(cfg["caption_num"]):
            caption = generate_caption(model, processor, img, cfg['prompt'], cfg['caption_max_tokens'],
                             do_sample=True,
                             temperature=0.7,
                             top_p=0.9,
                             top_k=0,
                            )

            captions.append(caption)

        record = {"file_name": sample_file_name, "captions": captions, "id": sample_id}
        captions_out.append(record)

    save_json(out / "coco/captions.json", captions_out)


if __name__ == "__main__":
    main()

"""Stage 0 - 이미지 1장이 파이프라인 끝까지 통과하는지만 확인.

성능도 지표도 보지 않는다. 확인할 것은 딱 두 가지:
  1) VLM 과 T2I 가 동시에 메모리에 올라가는가
  2) 이미지 1장에 몇 초 걸리는가  (x100 이 Stage 1 의 실행 시간)

실행:
    cd convis
    python experiments/00_smoke_test.py --config experiments/configs/local_debug.yaml
"""

import sys
from pathlib import Path
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import time

def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


DTYPES = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}


HERE = Path(__file__).resolve().parent      # convis/experiments
ROOT = HERE.parent                          # convis

sys.path.insert(0, str(ROOT))

from src.model import load_model, generate_caption
from src.t2i import load_t2i, regenerate
from src.utils import load_config, set_seed


def main():
    cfg = load_config(HERE / "configs/local_debug.yaml")
    sync(cfg["device"])
    set_seed(cfg["seed"])

    img = Image.open(ROOT / "data/test.jpg").convert("RGB")

    model = LlavaForConditionalGeneration.from_pretrained(cfg["model_id"], device_map=cfg["device"])
    processor = AutoProcessor.from_pretrained(cfg["model_id"])

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": cfg["prompt"]},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, DTYPES[cfg["dtype"]])

    # Generate

    t0 = time.perf_counter()

    generate_ids = model.generate(**inputs, max_new_tokens=cfg["max_new_tokens"])

    sync(cfg["device"])
    print(f"VLM 캡션 생성: {time.perf_counter() - t0:.1f}s")

    trimmed = generate_ids[:, inputs["input_ids"].shape[1]:]
    caption = processor.decode(trimmed[0], skip_special_tokens=True).strip()
    print(caption)


    # --- 2. 캡션을 다시 이미지로 ------------------------------
    # TODO: load_t2i -> regenerate
    # TODO: 소요 시간과 메모리 print

    # --- 3. 원본 | 재구성 나란히 저장 --------------------------
    # TODO: results/smoke/comparison.png

    # 이 스크립트의 진짜 산출물은 위 print 두 줄이다.
    # 시간과 메모리를 보고 Stage 1 의 이미지 개수를 정한다.


if __name__ == "__main__":
    main()

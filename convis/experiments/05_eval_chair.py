import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import json
import subprocess
from collections import defaultdict

PY = sys.executable

ANSWERS = ROOT / "outputs/coco/answers.json"
CHAIR_DIR = ROOT / "outputs/chair"
ANN_DIR = ROOT / "data/annotations"


def split_by_method():
    records = json.load(open(ANSWERS))
    CHAIR_DIR.mkdir(parents=True, exist_ok=True)

    by_method = defaultdict(list)
    for r in records:
        by_method[r.get("method", "convis")].append(r)

    paths = {}
    for method, rows in by_method.items():
        path = CHAIR_DIR / f"{method}_generated_captions.json"
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"image_id": r["image_id"], "caption": r["caption"]}) + "\n")
        print(f"[split] {method}: {len(rows)}")
        paths[method] = path
    return paths


def run(cmd):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)


def main():
    paths = split_by_method()

    for method, cap_path in paths.items():
        print(f"\n===== {method} =====")

        r = run([PY, "eval/caption_to_chair.py",
                 "-c", str(cap_path.relative_to(ROOT)),
                 "--annotation_path", str(ANN_DIR.relative_to(ROOT))])
        if r.returncode != 0:
            print(r.stdout[-1500:], r.stderr[-1500:])
            continue

        chair_input = CHAIR_DIR / f"{method}_chair.json"
        r = run([PY, "eval/eval_hallucination.py", "--metric", "chair",
                 "--chair_input_path", str(chair_input.relative_to(ROOT)),
                 "--data_dir", "data"])
        if r.returncode != 0:
            print(r.stdout[-1500:], r.stderr[-1500:])
            continue

        for line in r.stdout.replace("\r", "\n").splitlines():
            if "Getting annotations" in line or not line.strip():
                continue
            if any(k in line for k in ("SPICE", "chairs", "chairi", "meteor", "bleu", "hallucinate_sum", "\t")):
                print(line)


if __name__ == "__main__":
    main()

import subprocess
import sys

SCRIPTS = [
    "01prepare_data.py",
    "02generate_captions.py",
    "03generate_images.py",
    "04convis_decode.py",
    "05_eval_chair.py",
]

for script in SCRIPTS:
    print(f"\n===== {script} =====", flush=True)
    if subprocess.run([sys.executable, f"experiments/{script}"]).returncode != 0:
        sys.exit(f"{script} failed)

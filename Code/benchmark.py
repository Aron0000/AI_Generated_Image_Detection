"""Measure encoder latency on the actual hardware used for the experiments.

CLIP is evaluated in the fp16 precision used by features.py; DINOv2 is fp32,
as in dino_features.py. Timings include the full batch forward pass only
(no disk I/O, no probe), averaged over repeated batches after warm-up.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from paths import DATA, DINO_WEIGHTS

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "Results" / "final_v3" / "efficiency.json"
SAMPLE_DIR = DATA / "Nature"
DINO_DIR = DINO_WEIGHTS
BATCH = 32
WARMUP = 3
REPEATS = 10


def time_model(fn, batches: int) -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(batches):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / (batches * BATCH) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure encoder latency and parameter counts.")
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--sample-dir", type=Path, default=SAMPLE_DIR)
    parser.add_argument("--dino-dir", type=Path, default=DINO_DIR)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = Image.open(sorted(args.sample_dir.iterdir())[0]).convert("RGB")

    import open_clip
    clip, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", device=device, precision="fp16"
    )
    clip.eval()
    clip_batch = torch.stack([clip_preprocess(image)] * BATCH).to(device).half()

    from transformers import AutoModel
    dino = AutoModel.from_pretrained(str(args.dino_dir)).eval().to(device)
    dino_preprocess = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dino_batch = torch.stack([dino_preprocess(image)] * BATCH).to(device).float()

    with torch.inference_mode():
        for _ in range(WARMUP):
            clip.encode_image(clip_batch)
            dino(dino_batch).last_hidden_state[:, 0]
        clip_ms = time_model(lambda: clip.encode_image(clip_batch), REPEATS)
        dino_ms = time_model(lambda: dino(dino_batch).last_hidden_state[:, 0], REPEATS)
        dual_ms = time_model(
            lambda: (clip.encode_image(clip_batch), dino(dino_batch).last_hidden_state[:, 0]), REPEATS // 2
        )

    result = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "batch_size": BATCH,
        "clip": {
            "model": "ViT-L-14 openai, fp16",
            "ms_per_image": clip_ms,
            "parameters_millions": sum(p.numel() for p in clip.parameters()) / 1e6,
        },
        "dino": {
            "model": "DINOv2 ViT-L/14, fp32",
            "ms_per_image": dino_ms,
            "parameters_millions": sum(p.numel() for p in dino.parameters()) / 1e6,
        },
        "dual_sequential": {"ms_per_image": dual_ms},
        "probe_parameters": 1793,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

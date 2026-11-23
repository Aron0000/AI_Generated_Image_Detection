from __future__ import annotations

"""Cache DINOv2 ViT-L/14 CLS features (the stronger vision foundation model
used by our improved detector)."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from features import degrade
from paths import DINO_WEIGHTS

WEIGHTS_DIR = DINO_WEIGHTS

TRANSFORM = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class DinoDataset(Dataset):
    def __init__(self, manifest, variant, transform, indices=None):
        self.manifest = manifest
        self.variant = variant
        self.transform = transform
        self.indices = indices if indices is not None else list(range(len(manifest["paths"])))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        with Image.open(self.manifest["paths"][index]) as handle:
            image = handle.convert("RGB")
        image = degrade(image, self.variant)
        return {"image": self.transform(image), "index": index}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=["clean"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--splits", nargs="*", default=None)
    args = parser.parse_args()
    args.cache = args.cache.expanduser()

    manifest = json.loads((args.cache / "manifest.json").read_text(encoding="utf-8"))
    indices = [i for i, split in enumerate(manifest["splits"])
               if args.splits is None or split in args.splits]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import AutoModel
    model = AutoModel.from_pretrained(str(WEIGHTS_DIR)).eval().to(device)

    for variant in args.variants:
        out_path = args.cache / f"dino_{variant}.npz"
        if out_path.exists():
            print("skip", out_path.name)
            continue
        loader = DataLoader(DinoDataset(manifest, variant, TRANSFORM, indices), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers, pin_memory=True)
        features = np.zeros((len(manifest["paths"]), 1024), dtype=np.float16)
        for step, batch in enumerate(loader):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                out = model(pixel_values=batch["image"].to(device, non_blocking=True)).last_hidden_state
            features[batch["index"].numpy()] = out[:, 0, :].float().cpu().numpy().astype(np.float16)
            if step % 50 == 0:
                print(f"{variant}: {min((step+1)*args.batch_size, len(loader.dataset))}/{len(loader.dataset)}",
                      flush=True)
        np.savez_compressed(
            out_path, dino=features,
            labels=np.asarray(manifest["labels"], dtype=np.int8),
            generators=np.asarray(manifest["generators"]),
            splits=np.asarray(manifest["splits"]),
            paths=np.asarray(manifest["paths"]),
        )
        print("saved", out_path)


if __name__ == "__main__":
    main()

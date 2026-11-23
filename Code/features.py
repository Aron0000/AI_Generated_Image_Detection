from __future__ import annotations

"""Extract and cache frozen-encoder features for every image in the dataset.

The cache holds three views of each image:
  * CLIP ViT-L/14 image embedding (768-d, the encoder used by UnivFD)
  * ImageNet ResNet-18 pooled feature (512-d)
  * hand-crafted frequency/artifact descriptor (24-d)

Each image is also encoded under several deterministic degradations so that
training-time augmentation and test-time robustness use identical code.
"""

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.data import GENERATORS, image_files, split_files

SPLITS = ("train", "val", "test")
VARIANTS = ("clean", "jpeg90", "jpeg50", "resize50", "blur1", "combo")


def build_manifest(root: Path, seed: int = 5438) -> dict:
    paths, labels, generators, splits = [], [], [], []
    for generator in GENERATORS:
        files = image_files(root / generator)
        for split in SPLITS:
            for path in split_files(files, split, seed):
                paths.append(str(path))
                labels.append(1)
                generators.append(generator)
                splits.append(split)
    for split in SPLITS:
        for path in split_files(image_files(root / "Nature"), split, seed):
            paths.append(str(path))
            labels.append(0)
            generators.append("Nature")
            splits.append(split)
    return {"paths": paths, "labels": labels, "generators": generators, "splits": splits}


def degrade(image: Image.Image, variant: str) -> Image.Image:
    if variant == "clean":
        return image
    if variant.startswith("jpeg"):
        quality = int(variant[4:])
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB").copy()
    if variant.startswith("resize"):
        scale = int(variant[6:]) / 100.0
        width, height = image.size
        small = image.resize(
            (max(32, int(width * scale)), max(32, int(height * scale))),
            Image.Resampling.BILINEAR,
        )
        return small.resize((width, height), Image.Resampling.BILINEAR)
    if variant == "blur1":
        return image.filter(ImageFilter.GaussianBlur(radius=1.0))
    if variant == "combo":
        image = degrade(image, "jpeg70")
        image = degrade(image, "resize75")
        return image.filter(ImageFilter.GaussianBlur(radius=0.6))
    raise ValueError(variant)


def frequency_feature(image: Image.Image) -> np.ndarray:
    gray = np.asarray(image.convert("L").resize((128, 128), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))
    yy, xx = np.indices(spectrum.shape)
    cy = cx = (spectrum.shape[0] - 1) / 2.0
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    edges = np.linspace(0, rr.max(), 17)
    radial = [
        float(spectrum[(rr >= a) & (rr < b)].mean()) if ((rr >= a) & (rr < b)).any() else 0.0
        for a, b in zip(edges[:-1], edges[1:])
    ]
    laplacian = (
        -4 * gray
        + np.roll(gray, 1, 0) + np.roll(gray, -1, 0)
        + np.roll(gray, 1, 1) + np.roll(gray, -1, 1)
    )
    high = spectrum[rr > rr.max() * 0.6]
    left = gray[:, :-8] - gray[:, 8:]
    top = gray[:-8, :] - gray[8:, :]
    extra = [
        float(laplacian.mean()), float(laplacian.std()), float(np.abs(laplacian).mean()),
        float(np.abs(laplacian).max()), float(high.mean()), float(high.std()),
        float(np.abs(left).mean()), float(np.abs(top).mean()),
    ]
    return np.asarray(radial + extra, dtype=np.float32)


class VariantDataset(Dataset):
    def __init__(self, manifest: dict, variant: str, clip_transform, resnet_transform):
        self.manifest = manifest
        self.variant = variant
        self.clip_transform = clip_transform
        self.resnet_transform = resnet_transform

    def __len__(self):
        return len(self.manifest["paths"])

    def __getitem__(self, index):
        with Image.open(self.manifest["paths"][index]) as handle:
            image = handle.convert("RGB")
        image = degrade(image, self.variant)
        return {
            "clip": self.clip_transform(image),
            "resnet": self.resnet_transform(image),
            "freq": torch.from_numpy(frequency_feature(image)),
            "index": index,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=5438)
    args = parser.parse_args()
    args.data = args.data.expanduser()
    args.cache = args.cache.expanduser()

    args.cache.mkdir(parents=True, exist_ok=True)
    manifest_path = args.cache / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = build_manifest(args.data, args.seed)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    print(f"manifest: {len(manifest['paths'])} images")

    import open_clip

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", device=device, precision="fp16"
    )
    clip_model.eval()

    resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
    resnet.fc = torch.nn.Identity()
    resnet.eval().to(device)
    resnet_preprocess = ResNet18_Weights.DEFAULT.transforms()

    for variant in args.variants:
        out_path = args.cache / f"{variant}.npz"
        if out_path.exists():
            print(f"skip existing {out_path.name}")
            continue
        dataset = VariantDataset(manifest, variant, clip_preprocess, resnet_preprocess)
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        )
        clip_features = np.zeros((len(dataset), 768), dtype=np.float16)
        resnet_features = np.zeros((len(dataset), 512), dtype=np.float16)
        freq_features = np.zeros((len(dataset), 24), dtype=np.float32)

        for step, batch in enumerate(loader):
            with torch.no_grad():
                images = batch["clip"].to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    clip_out = clip_model.encode_image(images)
                images = batch["resnet"].to(device, non_blocking=True)
                resnet_out = resnet(images)
            index = batch["index"].numpy()
            clip_features[index] = clip_out.float().cpu().numpy().astype(np.float16)
            resnet_features[index] = resnet_out.float().cpu().numpy().astype(np.float16)
            freq_features[index] = batch["freq"].numpy()
            if step % 20 == 0:
                done = min((step + 1) * args.batch_size, len(dataset))
                print(f"{variant}: {done}/{len(dataset)}", flush=True)

        np.savez_compressed(
            out_path,
            clip=clip_features,
            resnet=resnet_features,
            freq=freq_features,
            labels=np.asarray(manifest["labels"], dtype=np.int8),
            generators=np.asarray(manifest["generators"]),
            splits=np.asarray(manifest["splits"]),
            paths=np.asarray(manifest["paths"]),
        )
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()

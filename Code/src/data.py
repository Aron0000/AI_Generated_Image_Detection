from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
GENERATORS = [
    "ADM",
    "BigGAN",
    "glide",
    "Midjourney",
    "stable_diffusion_v_1_5",
    "VQDM",
    "wukong",
]


def image_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def stable_bucket(path: Path, seed: int = 5438) -> int:
    token = f"{seed}:{path.parent.name}/{path.name}".encode()
    return int(hashlib.sha1(token).hexdigest()[:8], 16) % 100


def split_files(files: Iterable[Path], split: str, seed: int = 5438) -> list[Path]:
    """Deterministic 70/15/15 train/validation/test split."""
    ranges = {"train": (0, 70), "val": (70, 85), "test": (85, 100)}
    lo, hi = ranges[split]
    return [p for p in files if lo <= stable_bucket(p, seed) < hi]


def limited(files: list[Path], maximum: int | None, seed: int = 5438) -> list[Path]:
    if maximum is None or len(files) <= maximum:
        return files
    return sorted(files, key=lambda p: stable_bucket(p, seed + 1))[:maximum]


class BinaryImageDataset(Dataset):
    """Balanced real-vs-generated dataset. Label 0=real, 1=AI-generated."""

    def __init__(
        self,
        root: str | Path,
        generators: list[str],
        split: str,
        transform,
        max_per_class: int | None = None,
        seed: int = 5438,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        fake = []
        for generator in generators:
            fake.extend(split_files(image_files(self.root / generator), split, seed))
        real = split_files(image_files(self.root / "Nature"), split, seed)
        fake = limited(fake, max_per_class, seed)
        real = limited(real, min(len(fake), max_per_class or len(fake)), seed + 2)
        fake = limited(fake, len(real), seed + 3)
        self.samples = [(p, 0) for p in real] + [(p, 1) for p in fake]
        self.samples.sort(key=lambda item: stable_bucket(item[0], seed + 4))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            return self.transform(image), label, str(path)


class MultiGeneratorImageDataset(Dataset):
    """Balanced multi-generator dataset for the three-generator ResNet-18 run.

    Each training generator contributes at most ``max_per_generator`` images and
    the real class is limited to the total number of generated images, so the
    two classes stay balanced regardless of how many generators are used.
    """

    def __init__(
        self,
        root: str | Path,
        generators: list[str],
        split: str,
        transform,
        max_per_generator: int | None = None,
        seed: int = 5438,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.generator_counts = {}
        fake = []
        for index, generator in enumerate(generators):
            files = limited(
                split_files(image_files(self.root / generator), split, seed),
                max_per_generator,
                seed + 100 + index,
            )
            self.generator_counts[generator] = len(files)
            fake.extend(files)
        real = limited(
            split_files(image_files(self.root / "Nature"), split, seed),
            len(fake),
            seed + 500,
        )
        if len(real) < len(fake):
            fake = limited(fake, len(real), seed + 600)
        self.real_count = len(real)
        self.fake_count = len(fake)
        self.samples = [(p, 0) for p in real] + [(p, 1) for p in fake]
        self.samples.sort(key=lambda item: stable_bucket(item[0], seed + 700))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            return self.transform(image), label, str(path)


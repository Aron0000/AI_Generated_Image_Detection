
from __future__ import annotations

import io
import random

from PIL import Image
from torchvision import transforms


class RandomJPEGCompression:
    """Randomly apply JPEG compression to a PIL image."""

    def __init__(self, probability=0.5, quality_min=40, quality_max=95):
        self.probability = probability
        self.quality_min = quality_min
        self.quality_max = quality_max

    def __call__(self, image):
        if random.random() >= self.probability:
            return image

        quality = random.randint(self.quality_min, self.quality_max)

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)

        compressed = Image.open(buffer).convert("RGB")
        return compressed.copy()


class RandomResizeDegradation:
    """Downsample and restore an image to simulate resize degradation."""

    def __init__(self, probability=0.5, min_scale=0.5, max_scale=0.9):
        self.probability = probability
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, image):
        if random.random() >= self.probability:
            return image

        width, height = image.size
        scale = random.uniform(self.min_scale, self.max_scale)

        small_w = max(32, int(width * scale))
        small_h = max(32, int(height * scale))

        degraded = image.resize(
            (small_w, small_h),
            Image.Resampling.BILINEAR
        )

        degraded = degraded.resize(
            (width, height),
            Image.Resampling.BILINEAR
        )

        return degraded


def build_robust_train_transform(base_transform):
    """
    Apply realistic image degradations before the standard
    ImageNet preprocessing pipeline.
    """

    return transforms.Compose([
        RandomJPEGCompression(
            probability=0.5,
            quality_min=40,
            quality_max=95
        ),

        RandomResizeDegradation(
            probability=0.5,
            min_scale=0.5,
            max_scale=0.9
        ),

        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))],
            p=0.3
        ),

        base_transform
    ])

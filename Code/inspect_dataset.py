from PIL import Image

from paths import DATA
from src.data import GENERATORS, image_files


root = DATA
total = 0
for name in ["Nature", *GENERATORS]:
    files = image_files(root / name)
    total += len(files)
    formats = sorted({p.suffix.lower() for p in files})
    dimensions = set()
    for path in files[:50]:
        with Image.open(path) as image:
            dimensions.add(image.size)
    print(f"{name:24s}: {len(files):5d} images | {','.join(formats)} | "
          f"sample dimensions: {sorted(dimensions)[:5]}")
print(f"{'TOTAL':24s}: {total:5d} images")

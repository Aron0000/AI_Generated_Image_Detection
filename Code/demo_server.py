"""Live inference server for the DELP demo page.

Loads the two frozen encoders (CLIP ViT-L/14 and DINOv2 ViT-L/14), rebuilds
the logistic probe from the cached training features with exactly the code
used by probe.py, and scores photographs uploaded from the browser in real
time.  The page itself is served from Report/demo, so no upload ever leaves
the machine.

Usage:
    python Code/demo_server.py            # http://127.0.0.1:8000
    python Code/demo_server.py --port 8010 --no-open
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import mimetypes
import sys
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
CODE = Path(__file__).resolve().parent
DEMO = ROOT / "Report" / "demo"

sys.path.insert(0, str(CODE))

from features import frequency_feature  # noqa: E402
from paths import CACHE, DINO_WEIGHTS  # noqa: E402
from probe import fit_logistic, select_test, select_train  # noqa: E402

TRAIN_GENERATORS = ["stable_diffusion_v_1_5", "BigGAN", "glide"]
UNSEEN_COMMON = ["ADM", "Midjourney", "VQDM", "wukong"]
GALLERY = [
    ("sample", "ADM", 1),
    ("adm", "ADM", 1),
    ("midjourney", "Midjourney", 1),
    ("wukong", "wukong", 1),
    ("sd15", "stable_diffusion_v_1_5", 1),
    ("real", "Nature", 0),
]
MAX_UPLOAD_BYTES = 15 * 1024 * 1024


class Detector:
    """Frozen dual encoder + logistic probe, rebuilt from the shipped caches."""

    def __init__(self, device: torch.device, cache: Path = CACHE, dino_weights: Path = DINO_WEIGHTS):
        self.device = device
        self.cache = cache
        self.dino_weights = dino_weights
        self.lock = threading.Lock()
        self.manifest = json.loads((self.cache / "manifest.json").read_text(encoding="utf-8"))
        self.clip = dict(np.load(self.cache / "clean.npz", allow_pickle=True))
        self.dino = dict(np.load(self.cache / "dino_clean.npz", allow_pickle=True))
        self._train_probe(device)
        self.unseen_accuracy = self._check_unseen()
        self.samples = self._build_gallery()
        self._load_encoders(device)

    # ---------------------------------------------------------------- probe
    @staticmethod
    def _blocks(ids: list[int], clip_table: dict, dino_table: dict) -> np.ndarray:
        table = clip_table
        parts = []
        block = np.asarray(dino_table["dino"][ids], dtype=np.float32)
        block /= np.linalg.norm(block, axis=1, keepdims=True) + 1e-8
        parts.append(block)
        block = np.asarray(table["clip"][ids], dtype=np.float32)
        block /= np.linalg.norm(block, axis=1, keepdims=True) + 1e-8
        parts.append(block)
        return np.concatenate(parts, axis=1)

    def _train_probe(self, device: torch.device) -> None:
        from sklearn.preprocessing import StandardScaler

        started = time.perf_counter()
        train_ids = select_train(self.manifest, TRAIN_GENERATORS, "train", None, 5438)
        val_ids = select_train(self.manifest, TRAIN_GENERATORS, "val", 67, 5438)
        labels = np.asarray(self.manifest["labels"])
        x_train = self._blocks(train_ids, self.clip, self.dino)
        x_val = self._blocks(val_ids, self.clip, self.dino)
        self.scaler = StandardScaler().fit(x_train)
        x_train = self.scaler.transform(x_train).astype(np.float32)
        x_val = self.scaler.transform(x_val).astype(np.float32)
        self.probe, self.validation_f1 = fit_logistic(
            x_train, labels[train_ids], x_val, labels[val_ids]
        )
        self.probe_ms = (time.perf_counter() - started) * 1000
        print(
            f"probe: {len(train_ids)} train / {len(val_ids)} val images, "
            f"val macro-F1 {self.validation_f1:.4f}, rebuilt in {self.probe_ms / 1000:.1f} s",
            flush=True,
        )

    def _check_unseen(self) -> float:
        accs = []
        for generator in UNSEEN_COMMON:
            ids, y = select_test(self.manifest, generator, 200, 5438)
            x = self.scaler.transform(self._blocks(ids, self.clip, self.dino)).astype(np.float32)
            p = self.probe.predict_proba(x)[:, 1]
            accs.append(float(((p >= 0.5).astype(int) == y).mean()))
        mean = float(np.mean(accs))
        print(
            "P1 unseen check: " + " ".join(
                f"{g} {a * 100:.1f}" for g, a in zip(UNSEEN_COMMON, accs))
            + f" | mean {mean * 100:.1f}", flush=True,
        )
        return mean

    def _probabilities(self, ids: list[int]) -> np.ndarray:
        x = self.scaler.transform(self._blocks(ids, self.clip, self.dino)).astype(np.float32)
        return self.probe.predict_proba(x)[:, 1]

    def _build_gallery(self) -> dict:
        entries = {}
        real_ids, _ = select_test(self.manifest, "ADM", 200, 5438)
        real_probs = self._probabilities(real_ids)
        real_best = real_ids[int(np.argmin(real_probs))]
        for key, generator, truth in GALLERY:
            if key == "sample":
                path = DEMO / "sample.jpg"
            elif generator == "Nature":
                path = (CODE / self.manifest["paths"][real_best]).resolve()
            else:
                ids, labels = select_test(self.manifest, generator, 200, 5438)
                probs = self._probabilities(ids)
                fakes = [(i, y, pr) for i, y, pr in zip(ids, labels, probs) if y == 1]
                best = max(fakes, key=lambda item: item[2])[0]
                path = (CODE / self.manifest["paths"][best]).resolve()
            entries[key] = {
                "key": key,
                "file": path.name,
                "generator": generator,
                "truth": truth,
                "path": str(path),
            }
        return entries

    # ------------------------------------------------------------- encoders
    def _load_encoders(self, device: torch.device) -> None:
        import open_clip
        from transformers import AutoModel

        started = time.perf_counter()
        precision = "fp16" if device.type == "cuda" else "fp32"
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai", device=device, precision=precision
        )
        self.clip_model.eval()
        self.dino_model = AutoModel.from_pretrained(str(self.dino_weights)).eval().to(device)
        from dino_features import TRANSFORM as dino_transform

        self.dino_transform = dino_transform
        self.load_seconds = time.perf_counter() - started
        print(f"encoders loaded in {self.load_seconds:.1f} s on {device}", flush=True)

    def _autocast(self):
        if self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.float16)
        from contextlib import nullcontext

        return nullcontext()

    # ------------------------------------------------------------ inference
    def predict(self, image: Image.Image) -> dict:
        started = time.perf_counter()
        image = ImageOps.exif_transpose(image).convert("RGB")
        with self.lock, torch.no_grad():
            clip_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            t0 = time.perf_counter()
            with self._autocast():
                clip_out = self.clip_model.encode_image(clip_input)
            clip_ms = (time.perf_counter() - t0) * 1000

            dino_input = self.dino_transform(image).unsqueeze(0).to(self.device)
            t0 = time.perf_counter()
            with self._autocast():
                dino_out = self.dino_model(pixel_values=dino_input).last_hidden_state
            dino_ms = (time.perf_counter() - t0) * 1000

            dino = dino_out[:, 0, :].float().cpu().numpy().astype(np.float32)
            clip = clip_out.float().cpu().numpy().astype(np.float32)
            dino /= np.linalg.norm(dino, axis=1, keepdims=True) + 1e-8
            clip /= np.linalg.norm(clip, axis=1, keepdims=True) + 1e-8
            x = np.concatenate([dino, clip], axis=1).astype(np.float32)
            x = self.scaler.transform(x).astype(np.float32)
            t0 = time.perf_counter()
            prob = float(self.probe.predict_proba(x)[0, 1])
            probe_ms = (time.perf_counter() - t0) * 1000

        freq = frequency_feature(image).tolist()
        return {
            "p": prob,
            "features": freq,
            "image": {"width": image.width, "height": image.height},
            "ms": {
                "clip": round(clip_ms, 1),
                "dino": round(dino_ms, 1),
                "probe": round(probe_ms, 2),
                "total": round((time.perf_counter() - started) * 1000, 1),
            },
        }

    def predict_path(self, path: Path) -> dict:
        with Image.open(path) as handle:
            result = self.predict(handle)
        result["image"]["bytes"] = path.stat().st_size
        return result

    def health(self) -> dict:
        return {
            "ok": True,
            "device": str(self.device),
            "models": {"clip": "ViT-L/14 (openai)", "dino": "ViT-L/14 (dinov2)"},
            "probe_parameters": 1793,
            "validation_macro_f1": round(self.validation_f1, 4),
            "unseen4_accuracy": round(self.unseen_accuracy * 100, 2),
            "load_seconds": round(self.load_seconds, 1),
            "threshold": 0.5,
            "gallery": [
                {"key": key, "file": s["file"], "generator": s["generator"], "truth": s["truth"]}
                for key, s in self.samples.items()
            ],
        }


def decode_image(data_url: str) -> Image.Image:
    if "," in data_url and data_url.strip().lower().startswith("data:"):
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("file too large")
    return Image.open(io.BytesIO(raw))


class Handler(SimpleHTTPRequestHandler):
    detector: Detector | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DEMO), **kwargs)

    def log_message(self, fmt, *args):
        print("  " + (fmt % args), flush=True)

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self._json(self.detector.health())
            return
        if parsed.path == "/api/gallery":
            self._json({"samples": [
                {"key": key, "file": s["file"], "generator": s["generator"], "truth": s["truth"]}
                for key, s in self.detector.samples.items()
            ]})
            return
        if parsed.path == "/api/sample":
            key = (parse_qs(parsed.query).get("key") or [""])[0]
            entry = self.detector.samples.get(key)
            if not entry:
                self._json({"ok": False, "error": "unknown sample"}, 404)
                return
            try:
                path = Path(entry["path"])
                body = path.read_bytes()
            except OSError as exc:
                self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 500)
                return
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def do_POST(self):
        if urlparse(self.path).path != "/api/predict":
            self._json({"ok": False, "error": "not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > 20 * 1024 * 1024:
                raise ValueError("bad payload size")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if "sample" in payload:
                entry = self.detector.samples.get(payload["sample"])
                if not entry:
                    raise ValueError("unknown sample")
                result = self.detector.predict_path(Path(entry["path"]))
                result.update({"source": "gallery", "generator": entry["generator"],
                               "truth": entry["truth"], "file": entry["file"]})
            else:
                image = decode_image(str(payload.get("data") or ""))
                result = self.detector.predict(image)
                result.update({"source": "upload", "generator": None, "truth": None,
                               "file": str(payload.get("name") or "upload")})
            result["ok"] = True
            self._json(result)
        except Exception as exc:  # report the failure, never invent a score
            self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 400)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--dino-dir", type=Path, default=DINO_WEIGHTS)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading detector on {device} ...", flush=True)
    detector = Detector(device, cache=args.cache, dino_weights=args.dino_dir)
    Handler.detector = detector

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"serving {DEMO} at {url}  (Ctrl+C to stop)", flush=True)
    if not args.no_open:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

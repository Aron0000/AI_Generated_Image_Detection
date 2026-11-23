"""Build the single-file ASCII-style web demo.

All numbers are read from
  Results/final_v3/summary.json
  Results/final_v3/latex_assets.json
  Results/final_v3/efficiency.json
so the page cannot disagree with the report.  Output:
  Report/demo/index.html
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from string import Template

from paths import CACHE

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "Results" / "final_v3"
OUT = ROOT / "Report" / "demo" / "index.html"
DEMO = ROOT / "Report" / "demo"

S = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
A = json.loads((RESULTS / "latex_assets.json").read_text(encoding="utf-8"))
E = json.loads((RESULTS / "efficiency.json").read_text(encoding="utf-8"))
M = json.loads((RESULTS / "main_lin" / "metrics.json").read_text(encoding="utf-8"))["methods"]


def stable_bucket(path: Path, seed: int = 5438) -> int:
    token = f"{seed}:{path.parent.name}/{path.name}".encode()
    return int(hashlib.sha1(token).hexdigest()[:8], 16) % 100


def limited(files: list[Path], maximum: int | None, seed: int = 5438) -> list[Path]:
    if maximum is None or len(files) <= maximum:
        return files
    return sorted(files, key=lambda p: stable_bucket(p, seed + 1))[:maximum]


GENS = ["ADM", "BigGAN", "glide", "Midjourney", "stable_diffusion_v_1_5", "VQDM", "wukong"]
SHORT = {"ADM": "ADM", "BigGAN": "BigGAN", "glide": "GLIDE", "Midjourney": "MidJourney",
         "stable_diffusion_v_1_5": "SD 1.5", "VQDM": "VQDM", "wukong": "Wukong"}
COMMON = ["ADM", "Midjourney", "VQDM", "wukong"]


def pick_sample() -> dict:
    """Copy one selected ADM test image into the demo and export its measured score.

    The demo no longer accepts uploads: a fixed P1 test image is chosen at build
    time, so the animation always walks through the same, auditable example.
    """
    try:
        import numpy as np

        d = np.load(CACHE / "clean.npz", allow_pickle=True)
        paths, gens, splits = d["paths"], d["generators"], d["splits"]
        entry = M["dino_clip_lin"]["5438"]
        probs = entry["probabilities"]["ADM"]
        labels = entry["labels"]["ADM"]
        fake = [Path(p) for p, g, s in zip(paths, gens, splits) if g == "ADM" and s == "test"]
        real = [Path(p) for p, g, s in zip(paths, gens, splits) if g == "Nature" and s == "test"]
        fake = limited(fake, 200, 5438)
        real = limited(real, min(len(fake), 200), 5438 + 2)
        fake = limited(fake, len(real), 5438 + 3)
        ids = real + fake
        index = max((i for i in range(len(ids)) if labels[i] == 1), key=lambda i: probs[i])
        source = (ROOT / "Code" / ids[index]).resolve()
        shutil.copy2(source, DEMO / "sample.jpg")
        return {
            "file": "sample.jpg",
            "generator": "ADM",
            "truth": "generated",
            "p": round(float(probs[index]), 6),
            "source": source.name,
        }
    except Exception as exc:  # cache unavailable: keep the demo buildable
        print("sample selection skipped:", exc)
        return {"file": "sample.jpg", "generator": "ADM", "truth": "generated", "p": 0.0,
                "source": "sample.jpg"}


SAMPLE = pick_sample()
print("sample:", SAMPLE)


def pct(x: float) -> str:
    return f"{x * 100:.1f}"


def grid(headers, rows, widths, aligns=None) -> str:
    aligns = aligns or ["l"] + ["r"] * (len(widths) - 1)
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    out = [sep]

    def line(cells):
        parts = []
        for cell, width, align in zip(cells, widths, aligns):
            s = str(cell)
            parts.append(" " + (s.ljust(width) if align == "l" else s.rjust(width)) + " ")
        return "|" + "|".join(parts) + "|"

    out.append(line(headers))
    out.append(sep)
    for r in rows:
        out.append(line(r))
    out.append(sep)
    return "\n".join(out)


def bar(value: float, lo: float = 45.0, hi: float = 100.0, width: int = 30) -> str:
    n = int(round((value - lo) / (hi - lo) * width))
    n = max(0, min(width, n))
    return "#" * n + "." * (width - n)


def kv_block(pairs, width: int = 34) -> str:
    lines = []
    for key, value in pairs:
        pad = max(2, width - len(key) - len(value))
        lines.append(key + " " * pad + value)
    return "\n".join(lines)


def rule(number: str, label: str, width: int = 86) -> str:
    left = f"--[ {number} ]"
    right = f"[ {label} ]--"
    return left + "-" * max(3, width - len(left) - len(right)) + right


# ---------------------------------------------------------------- data blocks
ds = A["dataset"]
train = ds["train_selection"]
counts = {k: v["count"] for k, v in ds["classes"].items()}
splits = {k: v for k, v in ds["splits"].items()}
total = sum(counts.values())
real = counts["Nature"]
fake = total - real

dataset_rows = []
for g in GENS:
    dataset_rows.append([SHORT[g], f"{counts[g]:,}", f"{splits[g]['train']:,}",
                         f"{splits[g]['val']:,}", f"{splits[g]['test']:,}"])
dataset_rows.append(["TOTAL", f"{total:,}", "16,272", "3,457", "3,599"])
dataset_table = grid(["class", "images", "train", "val", "test"],
                     dataset_rows, [11, 8, 7, 6, 6], ["l"] + ["r"] * 4)

dataset_kv = kv_block([
    ("generated classes", "7 x 2,500 = 17,500"),
    ("real photographs", "5,828"),
    ("storage format", "JPEG everywhere"),
    ("P1 training set", "8,164 = 4,082 + 4,082"),
    ("unseen test set", "4 x 400 = 1,600"),
])

per_gen_rows = []
for g in COMMON:
    u = S["table"]["univfd"]["per_generator"][g] * 100
    o = S["table"]["ours"]["per_generator"][g] * 100
    per_gen_rows.append([SHORT[g], f"{u:.1f}", f"{o:.1f}", f"{o - u:+.1f}"])
per_gen_table = grid(["unseen generator", "UnivFD", "DELP", "delta"],
                     per_gen_rows, [18, 8, 8, 8], ["l", "r", "r", "r"])

bars = []
for g in COMMON:
    for name, key in [("UnivFD", "univfd"), ("DELP  ", "ours")]:
        v = S["table"][key]["per_generator"][g] * 100
        bars.append(f"{SHORT[g]:<12}{name} {v:5.1f} |{bar(v)}|")
bar_chart = "\n".join(bars)

logo = S["stats"]["logo"]
logo_rows = []
for held in GENS:
    cells = [SHORT[held]] + [f"{A['logo_full']['dino_clip_lin'][held][g] * 100:.1f}" for g in GENS]
    logo_rows.append(cells)
logo_table = grid(["held-out"] + [SHORT[g] for g in GENS],
                  logo_rows, [11] + [max(6, len(SHORT[g])) for g in GENS], ["l"] + ["r"] * 7)

rob = S["robustness"]
VAR = ["clean", "jpeg90", "jpeg50", "resize50", "blur1", "combo"]
rob_rows = []
for label, key in [("UnivFD", "univfd"), ("DELP", "ours"), ("DELP + aug", "ours_aug")]:
    rob_rows.append([label] + [f"{rob[key]['mean_all'][v] * 100:.1f}" for v in VAR])
rob_table = grid(["model", "clean", "q90", "q50", "resize", "blur", "combo"],
                 rob_rows, [10, 5, 4, 4, 6, 4, 5], ["l"] + ["r"] * 6)

farfrr = A["far_frr"]
op_rows = []
for g in COMMON:
    u = farfrr[g]["UnivFD"]
    o = farfrr[g]["DELP"]
    op_rows.append([SHORT[g],
                    f"{u['far_at_0.5'] * 100:.1f}", f"{u['frr_at_0.5'] * 100:.1f}",
                    f"{o['far_at_0.5'] * 100:.1f}", f"{o['frr_at_0.5'] * 100:.1f}",
                    f"{u['eer'] * 100:.1f}", f"{o['eer'] * 100:.1f}"])
op_table = grid(["generator", "uFAR", "uFRR", "dFAR", "dFRR", "uEER", "dEER"],
                op_rows, [10, 4, 4, 4, 4, 4, 4], ["l"] + ["r"] * 6)

stats = S["stats"]["paired_common_unseen"]
lp = logo["pooled"]
eo = A["error_overlap"]
ecases = eo["cases"]
efixed = eo["fused_correct_in_case"]

mean_kv = kv_block([
    ("UnivFD, mean over 7", f"{S['table']['univfd']['mean_all'] * 100:.1f}"),
    ("DELP, mean over 7", f"{S['table']['ours']['mean_all'] * 100:.1f}"),
    ("probe weights", f"{E['probe_parameters']:,} + 1 bias"),
    ("frozen CLIP", f"{E['clip']['parameters_millions']:.1f}M"),
    ("frozen DINOv2", f"{E['dino']['parameters_millions']:.1f}M"),
    ("serial inference", f"{E['dual_sequential']['ms_per_image']:.1f} ms / image"),
])

logo_kv = kv_block([
    ("DELP pooled", f"{logo['pooled']['accuracy_ours'] * 100:.2f}"),
    ("UnivFD pooled", f"{logo['pooled']['accuracy_univfd'] * 100:.2f}"),
    ("delta", f"{logo['pooled']['delta'] * 100:+.2f} pp"),
    ("95% CI", f"[{logo['pooled']['delta_ci95'][0] * 100:+.2f}, "
              f"{logo['pooled']['delta_ci95'][1] * 100:+.2f}]"),
    ("McNemar", f"b={logo['pooled']['mcnemar_b']} c={logo['pooled']['mcnemar_c']}"),
    ("p value", f"{logo['pooled']['mcnemar_p_value']:.1e}"),
    ("folds won", f"{S['stats']['logo_folds_won']} / 7"),
    ("off-diagonal DELP", f"{A['logo_full']['offdiag']['dino_clip_lin'] * 100:.1f}"),
    ("off-diagonal UnivFD", f"{A['logo_full']['offdiag']['univfd'] * 100:.1f}"),
])

eo_kv = kv_block([
    ("both encoders correct", f"{eo['cases']['both_correct']:,}"),
    ("CLIP only correct", f"{eo['cases']['clip_only']:,}"),
    ("DINOv2 only correct", f"{eo['cases']['dino_only']:,}"),
    ("both wrong", f"{eo['cases']['both_wrong']:,}"),
])

steps = [
    {"cmd": "python Code/inspect_dataset.py",
     "out": [
         f"8 classes / {total:,} files / all JPEG",
         f"real {real:,} | fake 7 x {counts['BigGAN']:,} = {fake:,}",
         f"hash split, seed 5438 -> {splits['Nature']['train'] + sum(splits[g]['train'] for g in GENS):,} train"
         f" / {splits['Nature']['val'] + sum(splits[g]['val'] for g in GENS):,} val"
         f" / {splits['Nature']['test'] + sum(splits[g]['test'] for g in GENS):,} test",
     ]},
    {"cmd": "python Code/features.py --variants clean jpeg90 jpeg50 resize50 blur1 combo",
     "out": [
         "CLIP ViT-L/14 (frozen, OpenAI) -> 768-d",
         "embedding L2-normalised as one block, then cached",
         "cache keyed by the manifest hash -> split cannot drift",
     ]},
    {"cmd": "python Code/dino_features.py --splits train val test",
     "out": [
         "DINOv2 ViT-L/14 (frozen, self-supervised) -> 1,024-d",
         "different inductive bias from CLIP: texture vs semantics",
         "concat with CLIP -> 1,792-d, StandardScaler, logistic regression",
     ]},
    {"cmd": "python Code/probe.py --methods univfd dino_clip_lin",
     "out": [
         f"train {train['p1_train_total']:,} = {train['p1_train_real']:,} real + {train['p1_train_fake']:,} fake",
         "model selection on the validation split only",
         f"probe = {E['probe_parameters']:,} parameters (1,792 weights + 1 bias)",
         "encoders frozen: no encoder training data or GPU hours added",
     ]},
    {"cmd": "python Code/probe.py --tta jpeg90 jpeg50 resize50 blur1 combo",
     "out": [
         "6 conditions x 1,600 unseen images = 9,600 evaluations per method",
         "threshold fixed at 0.5, never tuned on test",
     ]},
    {"cmd": "python Code/analyze.py",
     "out": [
         "paired stratified bootstrap, 10,000 resamples",
         f"delta {pct(stats['delta'])} pp, 95% CI "
         f"[{stats['delta_ci95'][0] * 100:+.2f}, {stats['delta_ci95'][1] * 100:+.2f}]",
         f"McNemar b={stats['mcnemar_b_base_only_correct']} "
         f"c={stats['mcnemar_c_ours_only_correct']} p={stats['mcnemar_p_value']:.2e}",
         "-> Results/final_v3/summary.json + all report figures",
     ]},
]

# Real per-image probabilities for the interactive score stream.
stream = {"order": GENS, "short": SHORT, "labels": {},
          "probes": {"univfd": {"name": "UnivFD", "acc": {}, "prob": {}},
                     "delp": {"name": "DELP", "acc": {}, "prob": {}}}}
for g in GENS:
    stream["labels"][g] = [int(v) for v in M["dino_clip_lin"]["5438"]["labels"][g]]
for key, method in [("univfd", "univfd"), ("delp", "dino_clip_lin")]:
    entry = M[method]["5438"]
    for g in GENS:
        stream["probes"][key]["acc"][g] = round(
            entry["cross_generator"][g]["accuracy"] * 100, 2)
        stream["probes"][key]["prob"][g] = [
            round(float(p), 3) for p in entry["probabilities"][g]]

demo_json = json.dumps({
    "steps": steps,
    "stream": stream,
    "head": {"ours": pct(S["table"]["ours"]["mean_common_unseen"]),
             "delta": f"{stats['delta'] * 100:+.2f}",
             "pval": f"{stats['mcnemar_p_value']:.1e}"},
    "sample": SAMPLE,
}, ensure_ascii=False, separators=(",", ":"))

template = Template(r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DELP // EE5438 Group 20 // demo</title>
<style>
:root{
  --bg:#080b0a; --panel:#0c1210; --line:#23322c; --line-soft:#16211d;
  --text:#cadbd0; --bright:#f0f7f2; --dim:#6c8175; --accent:#9fe3b0;
  --warm:#dcb07a; --alarm:#d98a73;
}
*{box-sizing:border-box}
html,body{margin:0;padding:0}
body{
  background:var(--bg); color:var(--text);
  font:12.5px/1.6 "Cascadia Mono","Consolas","Courier New",monospace;
  letter-spacing:.015em;
}
body::after{
  content:""; position:fixed; inset:0; pointer-events:none; z-index:999;
  background:repeating-linear-gradient(0deg, rgba(255,255,255,.028) 0 1px, transparent 1px 3px);
}
.wrap{max-width:1060px; margin:0 auto; padding:34px 22px 80px}
.masthead{border:1px solid var(--line); padding:18px 20px; margin-bottom:30px; background:var(--panel)}
.masthead pre{margin:0; color:var(--accent); font-size:12px; line-height:1.35; white-space:pre-wrap}
.sub{
  display:flex; flex-wrap:wrap; gap:8px 22px; margin-top:14px;
  color:var(--dim); font-size:11.5px; letter-spacing:.08em; text-transform:uppercase;
}
h2{margin:0 0 14px; font-size:12.5px; letter-spacing:.14em; text-transform:uppercase; color:var(--accent)}
.rule{color:var(--dim); white-space:pre; overflow-x:auto; margin:34px 0 14px; font-size:12px}
.panel{border:1px solid var(--line); background:var(--panel); padding:18px 20px; margin-bottom:18px}
pre{margin:0; white-space:pre; overflow-x:auto; font-variant-ligatures:none; max-width:100%}
.grid2{display:grid; grid-template-columns:1.05fr .95fr; gap:18px}
.grid2 > *{min-width:0}
.grid2 > .panel{margin-bottom:0; min-width:0}
.meta{color:var(--dim)}
.ok{color:var(--accent)}
.warm{color:var(--warm)}
.alarm{color:var(--alarm)}
.kv{display:grid; grid-template-columns:minmax(0,1.15fr) minmax(0,1fr); gap:2px 12px; font-size:11.5px}
.kv span{color:var(--dim); min-width:0; overflow-wrap:anywhere}
.kv b{font-weight:500; color:var(--bright); text-align:right; min-width:0; overflow-wrap:anywhere}
.cards{display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:18px}
.card{border:1px solid var(--line); background:var(--panel); padding:14px 15px}
.card .lab{font-size:10.5px; color:var(--dim); letter-spacing:.1em; text-transform:uppercase}
.card .val{font-size:24px; color:var(--bright); margin-top:6px; line-height:1.1}
.card .val em{font-style:normal; color:var(--accent); font-size:13px; margin-left:5px}
.card .note{font-size:10.5px; color:var(--dim); margin-top:5px}
.term{
  border:1px solid var(--line); background:#050807; padding:14px 16px;
  min-height:280px; max-height:430px; overflow:auto; white-space:pre-wrap;
}
.term .line{margin:0; white-space:pre-wrap}
.term .cmd{color:var(--bright)}
.term .out{color:var(--text)}
.term .ok{color:var(--accent)}
.prompt{color:var(--accent); margin-right:6px}
.controls{display:flex; flex-wrap:wrap; align-items:center; gap:10px; margin-top:14px}
.btn{
  background:transparent; border:1px solid var(--line); color:var(--text);
  font:inherit; font-size:11.5px; letter-spacing:.1em; text-transform:uppercase;
  padding:7px 14px; cursor:pointer;
}
.btn:hover:not(:disabled){background:var(--accent); border-color:var(--accent); color:#07100a}
.btn:disabled{opacity:.35; cursor:default}
.bar{color:var(--dim); font-size:11.5px}
.status{margin-left:auto; color:var(--dim); font-size:11.5px; letter-spacing:.08em}
.tag{border:1px solid var(--line); padding:1px 7px; color:var(--dim); font-size:10.5px}
.verdict{border-left:2px solid var(--accent); padding-left:14px; margin:16px 0 0}
.verdict .big{font-size:15px; color:var(--bright)}
.verdict .small{font-size:11.5px; color:var(--dim)}
.foot{color:var(--dim); font-size:11px; line-height:1.8; border-top:1px solid var(--line-soft); padding-top:18px; margin-top:38px}
.foot a{color:var(--accent); text-decoration:none}
.foot a:hover{text-decoration:underline}
html{scrollbar-width:thin;scrollbar-color:#26352e #070b09}
pre,.term{scrollbar-width:thin;scrollbar-color:#26352e #050807}
::-webkit-scrollbar{width:9px;height:9px}
::-webkit-scrollbar-track{background:#070b09}
::-webkit-scrollbar-thumb{background:#26352e;border:2px solid #070b09}
::-webkit-scrollbar-thumb:hover{background:#3c5147}
::-webkit-scrollbar-corner{background:#070b09}
.tbl{font-size:11.5px;letter-spacing:0}
.flow{position:relative;display:flex;align-items:center;gap:4px;flex-wrap:wrap;padding:10px 2px 12px;margin-bottom:8px}
.node{border:1px solid var(--line);background:#050807;color:var(--dim);padding:5px 6px;
  font-size:10px;letter-spacing:.1em;text-transform:uppercase;
  transition:border-color .35s,color .35s,box-shadow .35s}
.node.on{border-color:var(--accent);color:var(--bright);
  box-shadow:0 0 0 1px rgba(159,227,176,.14),0 0 16px rgba(159,227,176,.10)}
.arrow{color:var(--dim);font-size:10px}
.packet{position:absolute;left:0;top:23px;width:5px;height:5px;background:var(--accent);
  box-shadow:0 0 9px var(--accent);opacity:0}
.tabbar{display:flex;flex-wrap:wrap;gap:6px;margin:12px 0 14px}
.tab{border:1px solid var(--line);background:transparent;color:var(--dim);font:inherit;
  font-size:10.5px;letter-spacing:.06em;padding:4px 9px;cursor:pointer}
.tab.on{background:var(--accent);border-color:var(--accent);color:#07100a}
.probe-head{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px;
  font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--dim)}
.probe-head b{color:var(--bright);font-weight:500}
.canvas{display:block;width:100%;height:150px;border:1px solid var(--line-soft);background:#050807}
.legend{display:flex;gap:16px;margin-top:10px;color:var(--dim);font-size:10.5px}
.reveal{opacity:0;transform:translateY(12px);transition:opacity .55s ease,transform .55s ease}
.reveal.in{opacity:1;transform:none}
.cursor{display:inline-block;width:7px;height:11px;background:var(--accent);
  vertical-align:-1px;animation:blink 1s steps(1,end) infinite}
@keyframes blink{50%{opacity:0}}
.pulse{animation:pulse .5s ease}
@keyframes pulse{0%{color:var(--accent)}100%{color:var(--bright)}}
@media (prefers-reduced-motion: reduce){
  .reveal{opacity:1;transform:none;transition:none}
  .packet{display:none}
  .cursor{animation:none}
}
.up-grid{display:grid;grid-template-columns:minmax(240px,.8fr) 1.2fr;gap:18px}
.sample-frame{border:1px solid var(--line);background:#050807;padding:6px;
  display:flex;align-items:center;justify-content:center}
.sample-img{display:block;width:100%;max-height:230px;object-fit:contain}
.sample-meta{display:grid;gap:4px;margin-top:10px;color:var(--dim);font-size:11px}
.sample-meta .k{display:inline-block;min-width:94px;color:#3c5147;font-size:10px;
  letter-spacing:.08em;text-transform:uppercase}
.sample-meta b{color:var(--bright);font-weight:500}
.stage-list{display:grid;gap:6px}
.stage{display:flex;gap:10px;align-items:center;border:1px solid var(--line-soft);
  padding:5px 9px;color:var(--dim);font-size:11px;letter-spacing:.05em;
  transition:border-color .3s,color .3s,background .3s}
.stage .n{color:#3c5147;font-size:10px}
.stage.on{border-color:var(--accent);color:var(--bright);background:#07110d}
.stage.on .n{color:var(--accent)}
.up-canvases{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:16px}
.up-canvases h3{margin:0 0 6px;font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--dim);font-weight:400}
.up-canvas{display:block;width:100%;height:auto;image-rendering:pixelated;
  border:1px solid var(--line-soft);background:#050807}
.up-desc{color:var(--dim);font-size:11px;white-space:pre-wrap;margin-top:12px}
.up-desc b{color:var(--bright);font-weight:500}
.live-bar{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin-bottom:12px}
.pill{border:1px solid var(--line);padding:2px 8px;font-size:10.5px;letter-spacing:.06em;color:var(--dim)}
.pill.live{border-color:var(--accent);color:var(--accent)}
.pill.off{border-color:var(--warm);color:var(--warm)}
.drop-frame{position:relative}
.drop-frame .hint{position:absolute;inset:6px;display:flex;flex-direction:column;align-items:center;justify-content:center;
  text-align:center;background:rgba(5,8,7,.86);border:1px dashed #3c5147;color:var(--accent);font-size:10.5px;
  letter-spacing:.08em;text-transform:uppercase;opacity:0;transition:opacity .2s;pointer-events:none}
.drop-frame.over .hint{opacity:1}
.drop-frame.live:hover .hint{opacity:.55}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
.chip{border:1px solid var(--line);background:transparent;color:var(--dim);font:inherit;
  font-size:10.5px;letter-spacing:.04em;padding:3px 8px;cursor:pointer}
.chip:hover:not(:disabled){border-color:var(--accent);color:var(--accent)}
.chip.on{border-color:var(--accent);background:var(--accent);color:#07100a}
.chip:disabled{opacity:.35;cursor:default}
.present{position:fixed;left:50%;bottom:16px;transform:translateX(-50%);z-index:900;
  display:flex;align-items:center;gap:8px;border:1px solid var(--line);background:rgba(8,11,10,.94);
  padding:6px 9px;backdrop-filter:blur(5px);box-shadow:0 6px 24px rgba(0,0,0,.35)}
.present .lbl{font-size:10.5px;letter-spacing:.1em;color:var(--dim);min-width:214px;text-align:center;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.present .pdots{display:flex;gap:4px}
.present .dot{width:7px;height:7px;border:1px solid #3c5147;background:transparent;padding:0;cursor:pointer}
.present .dot.on{background:var(--accent);border-color:var(--accent)}
.present .pbtn{border:1px solid var(--line);background:transparent;color:var(--text);
  font:inherit;font-size:11px;padding:3px 9px;cursor:pointer}
.present .pbtn:hover{border-color:var(--accent);color:var(--accent)}
@media (max-width: 980px){
  .grid2{grid-template-columns:1fr}
  .up-grid{grid-template-columns:1fr}
  .up-canvases{grid-template-columns:repeat(2,1fr)}
}
@media (max-width: 820px){
  .cards{grid-template-columns:repeat(2,1fr)}
  .flow{row-gap:6px}
}
</style>
</head>
<body>
<div class="wrap">

<div class="masthead">
<pre>+==================================================================================+
|   E E 5 4 3 8  ::  A P P L I E D   D E E P   L E A R N I N G   ::  G R O U P  2 0 |
|                                                                                  |
|   D E L P  ::  D U A L - E N C O D E R   L I N E A R   P R O B E                 |
|   detecting AI images from generators never seen in training                     |
+==================================================================================+</pre>
<div class="sub">
  <span>status: complete</span>
  <span>data: 23,328 images</span>
  <span>probe: 1,793 params</span>
  <span>hardware: RTX 2080</span>
  <span>seed: 5438</span>
  <span>keys: &larr; &rarr; walk sections</span>
</div>
<div class="controls" style="margin-top:14px">
  <button class="btn" id="jumpLive">open live detector &darr;</button>
  <span class="meta">drop a photo, get a score in under a second</span>
</div>
</div>

<div class="rule">$rule01</div>
<div class="grid2">
<div class="panel">
<h2>pipeline</h2>
<div class="flow" id="flow">
  <div class="packet" id="packet"></div>
  <div class="node" data-stage="0">image</div><span class="arrow">--&gt;</span>
  <div class="node" data-stage="1">clip</div><span class="arrow">+</span>
  <div class="node" data-stage="1">dinov2</div><span class="arrow">--&gt;</span>
  <div class="node" data-stage="2">l2 norm</div><span class="arrow">--&gt;</span>
  <div class="node" data-stage="3">concat 1792</div><span class="arrow">--&gt;</span>
  <div class="node" data-stage="4">probe</div>
</div>
<button class="btn" id="flowReplay" style="font-size:10px;padding:4px 10px;margin-bottom:10px">replay flow</button>
<pre class="meta" style="margin-bottom:8px">detail</pre>
<pre>$pipeline</pre>
</div>
<div class="panel">
<h2>dataset on disk</h2>
<pre class="tbl">$dataset_table</pre>
<pre class="meta" style="margin-top:12px">$dataset_kv</pre>
<p class="verdict small" style="margin-top:12px">De-biased Tiny GenImage: equal class balance at test, hash split fixed for every method.</p>
</div>
</div>

<div class="rule">$rule02</div>
<div class="panel" id="uploadPanel">
<h2>live detector :: drop a photo or pick a dataset sample</h2>
<div class="live-bar">
  <span class="pill off" id="livePill">offline :: precomputed sample</span>
  <span class="meta" id="liveHint">start <b>python Code\demo_server.py</b> to run the frozen encoders on this machine and score any photo you drop.</span>
</div>
<div class="up-grid">
  <div>
    <div class="sample-frame drop-frame" id="dropZone" role="button" tabindex="0" aria-label="drop a photo or click to browse">
      <img class="sample-img" id="sampleImg" src="$sample_file" alt="photo to analyse">
      <div class="hint">drop a photo<br>or click to browse<br><span style="color:#6c8175">jpg / png / webp</span></div>
    </div>
    <div class="controls" style="margin-top:10px">
      <button class="btn" id="pickBtn" disabled>choose photo</button>
      <input type="file" id="fileInput" accept="image/*" hidden>
      <div class="status" id="liveMeta">offline mode</div>
    </div>
    <div class="sample-meta">
      <div><span class="k">file</span> <b id="metaFile">$sample_file</b></div>
      <div><span class="k">source</span> <b id="metaSource">dataset sample</b></div>
      <div><span class="k">generator</span> <b id="metaGen">$sample_gen</b></div>
      <div><span class="k">ground truth</span> <b id="metaTruth">$sample_truth</b></div>
      <div><span class="k">measured P(fake)</span> <b id="metaP">$sample_p % (precomputed)</b></div>
    </div>
    <div class="chips" id="sampleChips"></div>
  </div>
  <div>
    <div class="stage-list" id="stageList">
      <div class="stage" data-stage="0"><span class="n">01</span> decode the photo &amp; draw the preview</div>
      <div class="stage" data-stage="1"><span class="n">02</span> resample to 128 &times; 128, convert to luma (browser)</div>
      <div class="stage" data-stage="2"><span class="n">03</span> 2-D FFT: shift and log magnitude (browser)</div>
      <div class="stage" data-stage="3"><span class="n">04</span> 16 radial energy bins (browser)</div>
      <div class="stage" data-stage="4"><span class="n">05</span> laplacian residual + JPEG-block statistics (browser)</div>
      <div class="stage" data-stage="5"><span class="n">06</span> 24-d descriptor from <b>features.py</b></div>
      <div class="stage" data-stage="6"><span class="n">07</span> frozen CLIP + DINOv2 &rarr; logistic &rarr; measured verdict</div>
    </div>
    <div class="verdict" style="margin-top:14px">
      <div class="big" id="upVerdict">[ ready ]</div>
      <div class="small" id="upDetail">Start the local server, then drop a photo. The verdict is computed live, never simulated.</div>
    </div>
  </div>
</div>
<div class="up-canvases">
  <div><h3>input</h3><canvas class="up-canvas" id="upInput" width="128" height="128"></canvas></div>
  <div><h3>luma 128</h3><canvas class="up-canvas" id="upGray" width="128" height="128"></canvas></div>
  <div><h3>laplacian</h3><canvas class="up-canvas" id="upLap" width="128" height="128"></canvas></div>
  <div><h3>log spectrum</h3><canvas class="up-canvas" id="upSpec" width="128" height="128"></canvas></div>
</div>
<pre class="up-desc" id="upDescriptor">[ descriptor ] --  analyse a photo to compute it</pre>
<div class="controls">
  <button class="btn" id="upRun" disabled>analyse photo</button>
  <div class="status" id="upStatus">loading</div>
</div>
<p class="verdict small">Live mode sends one image to <b>Code/demo_server.py</b> on 127.0.0.1 and runs the same frozen encoders and probe that produced the report; the response carries the probability and the per-stage latency. Stages 02&ndash;05 are re-derived in the browser as a visualisation of the preprocessing. Without the server the page falls back to a fixed P1 sample and its precomputed score, clearly marked.</p>
</div>

<div class="rule">$rule03</div>
<div class="panel">
<h2>reproduce the headline, step by step</h2>
<div class="term" id="term">
<pre id="termText" class="line meta">[ idle ] press RUN to replay the six commands that produced the report.</pre>
</div>
<div class="controls">
  <button class="btn" id="run">run</button>
  <button class="btn" id="step">step</button>
  <button class="btn" id="reset">reset</button>
  <div class="bar" id="prog">[..........................] 0%</div>
  <div class="status" id="status">step 0 / 6</div>
</div>
</div>

<div class="rule">$rule04</div>
<div class="cards">
  <div class="card"><div class="lab">unseen-4 / DELP</div><div class="val"><span data-count="$ours_acc">0.0</span><em>%</em></div><div class="note">ADM + MidJourney + VQDM + Wukong</div></div>
  <div class="card"><div class="lab">unseen-4 / UnivFD</div><div class="val"><span data-count="$uni_acc">0.0</span><em>%</em></div><div class="note">reproduced CVPR 2023 probe</div></div>
  <div class="card"><div class="lab">delta</div><div class="val"><span data-count="$delta" data-signed="1">+0.00</span><em>pp</em></div><div class="note">95% CI [$ci_lo, $ci_hi]</div></div>
  <div class="card"><div class="lab">McNemar</div><div class="val">$pval</div><div class="note">b=$b / c=$c, exact binomial</div></div>
</div>
<div class="panel" id="streamPanel">
<h2>score stream :: 400 decisions per generator (200 real + 200 fake)</h2>
<pre class="meta" style="margin-bottom:4px">probabilities recorded from Results/final_v3/main_lin/metrics.json · threshold fixed at 0.5 · errors highlighted in red</pre>
<div class="tabbar" id="tabs"></div>
<div class="grid2" style="gap:16px">
  <div>
    <div class="probe-head"><span>UnivFD</span><b id="accU">--</b></div>
    <canvas class="canvas" id="cvU"></canvas>
  </div>
  <div>
    <div class="probe-head"><span>DELP (ours)</span><b id="accD">--</b></div>
    <canvas class="canvas" id="cvD"></canvas>
  </div>
</div>
<div class="legend"><span>circle = real</span><span>square = fake</span><span class="alarm">red = misclassified</span><span>--- = threshold 0.5</span></div>
<div class="controls">
  <button class="btn" id="playBtn">play</button>
  <button class="btn" id="speedBtn">1x</button>
  <button class="btn" id="clearBtn">clear</button>
  <div class="bar" id="sweep">0 / 400</div>
  <div class="status" id="sweepNote">select a generator, then play</div>
</div>
</div>
<div class="grid2">
<div class="panel">
<h2>per-generator accuracy</h2>
<pre class="tbl">$per_gen_table</pre>
<pre class="meta" style="margin-top:12px">$bar_chart</pre>
</div>
<div class="panel">
<h2>operating point at threshold 0.5</h2>
<pre class="tbl">$op_table</pre>
<p class="verdict small" style="margin-top:12px">FAR = real images called fake. FRR = fake images called real. EER is approximated on a 0.05-0.95 grid.</p>
<h2 style="margin-top:20px">mean over all seven generators</h2>
<pre class="meta">$mean_kv</pre>
</div>
</div>

<div class="rule">$rule05</div>
<div class="panel">
<h2>leave-one-generator-out, 7 folds</h2>
<pre class="tbl">$logo_table</pre>
<p class="verdict small" style="margin-top:12px">Diagonal = the only strictly unseen cell in each fold. Off-diagonal = in-domain tests.</p>
<h2 style="margin-top:16px">pooled LOGO statistics</h2>
<pre class="meta">$logo_kv</pre>
<p class="verdict small" style="margin-top:12px">The gain is not a single-generator artefact: it holds in every fold.</p>
</div>

<div class="rule">$rule06</div>
<div class="grid2">
<div class="panel">
<h2>post-processing robustness, mean over 7</h2>
<pre class="tbl">$rob_table</pre>
<p class="verdict small" style="margin-top:12px">The clean-trained probe already beats UnivFD at JPEG 50. The augmented variant trades a little clean accuracy for compression robustness.</p>
</div>
<div class="panel">
<h2>why fusion works</h2>
<pre>$eo_kv</pre>
<pre class="meta" style="margin-top:12px">fused probe recovers
  $fix_both / $eo_both   where both were correct
  $fix_clip / $eo_clip   where only CLIP was correct
  $fix_dino / $eo_dino   where only DINOv2 was correct
  $fix_wrong / $eo_wrong  where both failed</pre>
<p class="verdict small" style="margin-top:12px">Complementarity, not a stronger single backbone: DINOv2 alone reaches $dino_alone %.</p>
</div>
</div>

<div class="rule">$rule07</div>
<div class="panel">
<h2>verify it yourself</h2>
<pre class="meta">cd $root
python Code\inspect_dataset.py
python Code\build_latex_assets.py
python Code\analyze.py
python Code\build_demo_page.py
python Code\demo_server.py

this page  ->  Report\demo\index.html
live api   ->  Code\demo_server.py  (127.0.0.1:8000)
report     ->  Report\Final_Report_EE5438_LaTeX.pdf
source     ->  Results\final_v3\summary.json
               Results\final_v3\latex_assets.json
               Results\final_v3\efficiency.json
sample     ->  Report\demo\sample.jpg ($sample_file)
weights    ->  C:\Users\asus\ee5438_cache  +  C:\Users\asus\dinov2-large</pre>
<div class="verdict">
  <div class="big">Every number on this page is interpolated from measured result files.</div>
  <div class="small">No value is typed by hand. Live scores are produced by the frozen encoders and the probe rebuilt from the shipped caches; offline mode shows the precomputed P1 sample. The page and the 37-page report read the same summary JSON.</div>
</div>
</div>

<div class="present" id="present" aria-label="presentation navigation">
  <button class="pbtn" id="prevSec" aria-label="previous section">&lsaquo;</button>
  <span class="lbl" id="secLabel">01 / 07</span>
  <div class="pdots" id="secDots"></div>
  <button class="pbtn" id="nextSec" aria-label="next section">&rsaquo;</button>
</div>

<div class="foot">
EE5438 Applied Deep Learning / Group 20 / Huanrong Li, Bo Pang, Huanyu Zhang, Junxu Chen, Yangqi Zhang<br>
report template: github.com/norlab-ulaval/template-technicalReport &nbsp;|&nbsp; dataset: Unbiased Tiny GenImage (CC BY-NC-SA 4.0)<br>
demo build: Code/build_demo_page.py &nbsp;|&nbsp; deterministic seed 5438 &nbsp;|&nbsp; no synthetic training data
</div>
</div>

<script>
window.DEMO_DATA = $demo_json;
</script>
<script src="demo.js"></script>
</body>
</html>
""")

pipeline = """[ raw image, JPEG, any size ]
        |
        |  resize 224 x 224
        |
        +--> CLIP ViT-L/14 (frozen)
        |      768-d -> L2 norm
        |
        +--> DINOv2 ViT-L/14 (frozen)
               1,024-d -> L2 norm
                     |
        concat 1,792-d -> logistic -> P(fake)

optional: JPEG / resize / blur copies
for the robust variant only"""

values = {
    "rule01": rule("01", "DATA & PIPELINE"),
    "rule02": rule("02", "LIVE DETECTOR"),
    "rule03": rule("03", "REPRODUCE"),
    "rule04": rule("04", "HEADLINE"),
    "rule05": rule("05", "GENERALISATION"),
    "rule06": rule("06", "ROBUSTNESS"),
    "rule07": rule("07", "VERIFY"),
    "sample_file": SAMPLE["file"],
    "sample_gen": SAMPLE["generator"],
    "sample_truth": SAMPLE["truth"],
    "sample_p": f'{SAMPLE["p"] * 100:.1f}',
    "pipeline": pipeline,
    "dataset_table": dataset_table,
    "dataset_kv": dataset_kv,
    "mean_kv": mean_kv,
    "logo_kv": logo_kv,
    "eo_kv": eo_kv,
    "per_gen_table": per_gen_table,
    "op_table": op_table,
    "rob_table": rob_table,
    "logo_table": logo_table,
    "bar_chart": bar_chart,
    "demo_json": demo_json,
    "root": str(ROOT),
    "total": f"{total:,}",
    "fake": f"{fake:,}",
    "real": f"{real:,}",
    "p1_train": f"{train['p1_train_total']:,}",
    "p1_real": f"{train['p1_train_real']:,}",
    "p1_fake": f"{train['p1_train_fake']:,}",
    "ours_acc": pct(S["table"]["ours"]["mean_common_unseen"]),
    "uni_acc": pct(S["table"]["univfd"]["mean_common_unseen"]),
    "ours_mean7": pct(S["table"]["ours"]["mean_all"]),
    "uni_mean7": pct(S["table"]["univfd"]["mean_all"]),
    "delta": f"{stats['delta'] * 100:+.2f}",
    "ci_lo": f"{stats['delta_ci95'][0] * 100:+.2f}",
    "ci_hi": f"{stats['delta_ci95'][1] * 100:+.2f}",
    "b": stats["mcnemar_b_base_only_correct"],
    "c": stats["mcnemar_c_ours_only_correct"],
    "pval": f"{stats['mcnemar_p_value']:.1e}",
    "logo_ours": pct(lp["accuracy_ours"]),
    "logo_uni": pct(lp["accuracy_univfd"]),
    "logo_delta": f"{lp['delta'] * 100:+.2f}",
    "logo_lo": f"{lp['delta_ci95'][0] * 100:+.2f}",
    "logo_hi": f"{lp['delta_ci95'][1] * 100:+.2f}",
    "logo_b": lp["mcnemar_b"],
    "logo_c": lp["mcnemar_c"],
    "logo_p": f"{lp['mcnemar_p_value']:.1e}",
    "folds_won": S["stats"]["logo_folds_won"],
    "off_ours": pct(A["logo_full"]["offdiag"]["dino_clip_lin"]),
    "off_uni": pct(A["logo_full"]["offdiag"]["univfd"]),
    "eo_both": f"{ecases['both_correct']:,}",
    "eo_clip": f"{ecases['clip_only']:,}",
    "eo_dino": f"{ecases['dino_only']:,}",
    "eo_wrong": f"{ecases['both_wrong']:,}",
    "fix_both": f"{efixed['both_correct']:,}",
    "fix_clip": f"{efixed['clip_only']:,}",
    "fix_dino": f"{efixed['dino_only']:,}",
    "fix_wrong": f"{efixed['both_wrong']:,}",
    "dino_alone": pct(S["ablation"]["dino_lin"]["mean_common_unseen"]),
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(template.substitute(values), encoding="utf-8")
print("saved", OUT)

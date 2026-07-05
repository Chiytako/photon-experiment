"""Generate the figures for the Zenn article from the actual experiment
artifacts (training logs, eval JSONs, benchmark JSONs) in checkpoints/.
Outputs PNGs to zenn/images/."""
import json
import os

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "checkpoints")
OUT = os.path.join(ROOT, "zenn", "images")
os.makedirs(OUT, exist_ok=True)

# Japanese font
for f in font_manager.fontManager.ttflist:
    if "Noto Sans CJK JP" in f.name:
        plt.rcParams["font.family"] = "Noto Sans CJK JP"
        break
else:
    for f in font_manager.fontManager.ttflist:
        if "Noto Serif CJK JP" in f.name:
            plt.rcParams["font.family"] = "Noto Serif CJK JP"
            break
plt.rcParams.update({"figure.dpi": 150, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})

C_BASE = "#4C72B0"   # baseline blue
C_PHOT = "#DD8452"   # photon orange
C_ACC = "#55A868"    # accent green


def read_log(name):
    steps_v, val = [], []
    with open(os.path.join(CKPT, name)) as f:
        for line in f:
            d = json.loads(line)
            if "val_loss" in d:
                steps_v.append(d["step"])
                val.append(d["val_loss"])
    return steps_v, val


def read_eval(name):
    with open(os.path.join(CKPT, name)) as f:
        return json.load(f)


# ---------------------------------------------------------------- fig 1
# Scale-up validation loss curves (medium pair, 1.5B tokens)
TOK_PER_STEP = 16 * 1024
fig, ax = plt.subplots(figsize=(7, 4.2))
for log, label, color in [("baseline_med_log.jsonl", "Baseline 187M", C_BASE),
                          ("photon_med_log.jsonl", "PHOTON 158M", C_PHOT)]:
    s, v = read_log(log)
    ax.plot([x * TOK_PER_STEP / 1e9 for x in s], v, label=label, color=color, lw=1.8)
ax.set_xlabel("学習トークン数 [B]")
ax.set_ylabel("検証 loss (FineWeb-Edu val)")
ax.set_title("スケールアップ学習曲線（medium ペア, 1.5B トークン）")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "photon-loss-curves.png"))
plt.close(fig)

# ---------------------------------------------------------------- fig 2
# Ablation perplexities. Values are the README ablation table (all measured
# on the ORIGINAL data/val.bin). We don't read the eval JSONs here because
# photon_eval.json was later re-run on data2b/val.bin for the scale table,
# so mixing JSONs would silently mix val sets.
runs = [("photon", "基準\nα=0, C=(4,4), R=(4,4)"),
        ("ph_a01", "α=0.1"), ("ph_a02", "α=0.2"), ("ph_a03", "α=0.3"),
        ("ph_c22", "C=(2,2)"), ("ph_r88", "R=(8,8)")]
fw = [63.6, 66.4, 69.5, 72.5, 56.6, 63.4]
wt = [261.8, 291.5, 291.4, 329.7, 174.0, 269.3]
labels = [l for _, l in runs]
x = range(len(runs))
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, vals, title in [(axes[0], fw, "FineWeb-Edu val ppl"),
                        (axes[1], wt, "WikiText-103 ppl")]:
    colors = [C_ACC if labels[i].startswith("C=(2,2)") else
              ("#999999" if i == 0 else C_PHOT) for i in x]
    bars = ax.bar(x, vals, color=colors)
    ax.bar_label(bars, fmt="%.1f", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_title(title)
    ax.set_ylim(0, max(vals) * 1.15)
fig.suptitle("アブレーション（小型 PHOTON, 各 200M トークン）— 低いほど良い", y=1.0)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "photon-ablation.png"))
plt.close(fig)

# ---------------------------------------------------------------- fig 3
# Quality gap vs scale (values = README scale table, data2b val set)
fig, ax = plt.subplots(figsize=(6.5, 4))
scales = ["small\n(44M/36M, 200M tok)", "medium\n(187M/158M, 1.5B tok)"]
fineweb = [1.98, 1.71]
wikitext = [3.73, 2.80]
ax.plot(scales, fineweb, "o-", color=C_BASE, lw=2, ms=8, label="FineWeb-Edu val（分布内）")
ax.plot(scales, wikitext, "s-", color=C_PHOT, lw=2, ms=8, label="WikiText-103（分布外）")
for xs, ys in [(scales, fineweb), (scales, wikitext)]:
    for xx, yy in zip(xs, ys):
        ax.annotate(f"{yy:.2f}x", (xx, yy), textcoords="offset points",
                    xytext=(10, 4), fontsize=10)
ax.axhline(1.0, color="#555555", ls="--", lw=1)
ax.annotate("1.0x = baseline と同品質", (0.02, 1.0), xycoords=("axes fraction", "data"),
            textcoords="offset points", xytext=(0, 4), fontsize=8.5, color="#555555")
ax.set_ylabel("品質ギャップ（PHOTON ppl ÷ baseline ppl）")
ax.set_title("品質ギャップはスケールで縮む")
ax.set_ylim(0.8, 4.2)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "photon-scale-gap.png"))
plt.close(fig)

# ---------------------------------------------------------------- fig 4
# Inference benchmark (medium pair). 2K regimes from benchmark_results.json,
# 8K decode from the re-run recorded in README (273 / 1733 tok/s).
bench = json.load(open(os.path.join(CKPT, "benchmark_results.json")))
regimes = ["プレフィル重視\n2048 in / 128 out", "デコード重視\n128 in / 2048 out",
           "長文デコード\n128 in / 8192 out"]
base_tps = [bench["baseline"]["prefill_heavy"]["throughput_tok_per_s"],
            bench["baseline"]["decode_heavy"]["throughput_tok_per_s"], 273]
phot_tps = [bench["photon"]["prefill_heavy"]["throughput_tok_per_s"],
            bench["photon"]["decode_heavy"]["throughput_tok_per_s"], 1733]
x = range(len(regimes))
w = 0.36
fig, ax = plt.subplots(figsize=(7.5, 4.2))
b1 = ax.bar([i - w / 2 for i in x], base_tps, w, color=C_BASE, label="Baseline 187M")
b2 = ax.bar([i + w / 2 for i in x], phot_tps, w, color=C_PHOT, label="PHOTON 158M")
ax.bar_label(b1, fmt="%.0f", fontsize=9)
ax.bar_label(b2, fmt="%.0f", fontsize=9)
# 8K の倍率は README 記載の実測比 6.36x（棒の値は丸め後なので割ると 6.35 になる）
ratios = [phot_tps[0] / base_tps[0], phot_tps[1] / base_tps[1], 6.36]
for i in x:
    ax.annotate(f"{ratios[i]:.2f}x", (i + w / 2, phot_tps[i]),
                textcoords="offset points", xytext=(0, 16), ha="center",
                fontsize=11, fontweight="bold", color=C_ACC)
ax.set_xticks(list(x))
ax.set_xticklabels(regimes, fontsize=9)
ax.set_ylabel("スループット [tok/s]（batch 4）")
ax.set_title("推論スループット（medium ペア）— 系列が長いほど PHOTON が有利")
ax.set_ylim(0, max(phot_tps) * 1.22)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "photon-inference-bench.png"))
plt.close(fig)

# ---------------------------------------------------------------- fig 5
# RecGen: alpha vs X̂1 fidelity and forced-decoding ppl (recgen_diag results)
alphas = ["α=0", "α=0.1", "α=0.2", "α=0.3"]
cosine = [0.64, 0.94, 0.94, 0.94]
rec_ppl = [1679, 315, 309, 304]
hier_ppl = [60, 61, 60, 60]
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ax = axes[0]
bars = ax.bar(alphas, cosine, color=[C_PHOT if a == "α=0" else C_ACC for a in alphas])
ax.bar_label(bars, fmt="%.2f", fontsize=10)
ax.set_ylim(0, 1.05)
ax.set_title("潜在再構成の忠実度（$\\hat{X}^1$ vs $X^1$ の cosine）")
ax.set_ylabel("cosine 類似度（高いほど良い）")
ax = axes[1]
xr = range(len(alphas))
b1 = ax.bar([i - 0.18 for i in xr], hier_ppl, 0.36, color=C_BASE, label="HierGen")
b2 = ax.bar([i + 0.18 for i in xr], rec_ppl, 0.36, color=C_PHOT, label="RecGen")
ax.bar_label(b1, fmt="%.0f", fontsize=9)
ax.bar_label(b2, fmt="%.0f", fontsize=9)
ax.set_yscale("log")
ax.set_xticks(list(xr))
ax.set_xticklabels(alphas)
ax.set_title("強制デコード ppl（対数軸, 低いほど良い）")
ax.set_ylabel("続き生成 ppl")
ax.legend()
fig.suptitle("RecGen には α>0 が必須 — ただしこの規模ではまだ HierGen に届かない", y=1.0)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "photon-recgen.png"))
plt.close(fig)

print("figures written to", OUT)
for f in sorted(os.listdir(OUT)):
    print(" -", f)

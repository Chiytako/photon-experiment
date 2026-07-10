"""Generate the figures for the (rewritten) Zenn article from the actual
experiment artifacts in checkpoints/. v2 figures compare the faithful
reimplementation against the retracted v1 numbers; each figure is skipped
gracefully if its v2 artifacts don't exist yet (pipeline still running).
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
C_V1 = "#B0B0B0"     # v1 (unfaithful) grey
C_PHOT = "#DD8452"   # photon v2 orange
C_ACC = "#55A868"    # accent green

# v1 reference numbers (from the retracted implementation's committed eval
# JSONs; the architecture differed, see README "What v1 got wrong").
V1_WIKITEXT = {"baseline": 70.2, "photon": 261.8}
TOK_PER_STEP = 16 * 1024


def read_log(name, key="val_token_loss", fallback_key="val_loss"):
    steps_v, val = [], []
    path = os.path.join(CKPT, name)
    if not os.path.exists(path):
        return steps_v, val
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if key in d:
                steps_v.append(d["step"]); val.append(d[key])
            elif fallback_key in d and key not in d:
                steps_v.append(d["step"]); val.append(d[fallback_key])
    return steps_v, val


def read_json(name):
    path = os.path.join(CKPT, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def skip(fig_name, missing):
    print(f"SKIP {fig_name}: missing {missing}")


# ---------------------------------------------------------------- fig 1
# v1 vs v2 loss curves (small pair, 200M tokens, same data/): the honest cost
# of implementing the paper's actual conditioning structure.
s_b, v_b = read_log("baseline_log.jsonl")
s_p1, v_p1 = read_log("photon_log.jsonl")
s_p2, v_p2 = read_log("photon2_log.jsonl")
if s_p2:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot([x * TOK_PER_STEP / 1e9 for x in s_b], v_b, label="Baseline 44M", color=C_BASE, lw=1.8)
    ax.plot([x * TOK_PER_STEP / 1e9 for x in s_p1], v_p1,
            label="PHOTON v1 36M（誤実装・撤回）", color=C_V1, lw=1.8, ls="--")
    ax.plot([x * TOK_PER_STEP / 1e9 for x in s_p2], v_p2,
            label="PHOTON v2 48M（論文忠実）", color=C_PHOT, lw=1.8)
    ax.set_xlabel("学習トークン数 [B]")
    ax.set_ylabel("検証 token loss (FineWeb-Edu val)")
    ax.set_title("v1（誤実装）vs v2（論文忠実）学習曲線 — small, 200M トークン")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "photon-v2-loss-curves.png"))
    plt.close(fig)
else:
    skip("photon-v2-loss-curves", "photon2_log.jsonl")

# ---------------------------------------------------------------- fig 2
# WikiText PPL: baseline vs v1 vs v2 (WikiText is externally fixed data, so
# it is the stable axis for the before/after comparison).
e_p2 = read_json("photon2_eval.json")
e_p2a = read_json("photon2_a03_eval.json")
if e_p2:
    labels = ["Baseline\n44M", "PHOTON v1\n36M（誤実装）", "PHOTON v2\n48M α=0"]
    vals = [V1_WIKITEXT["baseline"], V1_WIKITEXT["photon"], e_p2["wikitext_ppl"]]
    colors = [C_BASE, C_V1, C_PHOT]
    if e_p2a:
        labels.append("PHOTON v2\n48M α=0.3")
        vals.append(e_p2a["wikitext_ppl"])
        colors.append(C_ACC)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(labels, vals, color=colors)
    ax.bar_label(bars, fmt="%.0f", fontsize=10)
    ax.set_ylabel("WikiText-103 perplexity（低いほど良い）")
    ax.set_title("v1 の品質は誤実装によるゲタ — 忠実実装の実力はこちら")
    ax.set_ylim(0, max(vals) * 1.15)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "photon-v2-wikitext.png"))
    plt.close(fig)
else:
    skip("photon-v2-wikitext", "photon2_eval.json")

# ---------------------------------------------------------------- fig 3
# v2 benchmark, paper TPM protocol (throughput / per-sample KV-cache GiB).
bench = read_json("benchmark_results_v2.json")
if bench:
    regimes = [("prefill_heavy", "プレフィル重視\n2048 in / 128 out"),
               ("decode_heavy", "デコード重視\n128 in / 2048 out")]
    x = range(len(regimes))
    w = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    b_tpm = [bench["baseline"][k]["tpm_k_tok_per_s_per_gib"] for k, _ in regimes]
    p_tpm = [bench["photon"][k]["tpm_k_tok_per_s_per_gib"] for k, _ in regimes]
    b1 = ax.bar([i - w / 2 for i in x], b_tpm, w, color=C_BASE, label="Baseline")
    b2 = ax.bar([i + w / 2 for i in x], p_tpm, w, color=C_PHOT, label="PHOTON v2 (HierGen)")
    ax.bar_label(b1, fmt="%.1f", fontsize=9)
    ax.bar_label(b2, fmt="%.1f", fontsize=9)
    for i in x:
        ax.annotate(f"{p_tpm[i] / b_tpm[i]:.1f}x", (i + w / 2, p_tpm[i]),
                    textcoords="offset points", xytext=(0, 14), ha="center",
                    fontsize=11, fontweight="bold", color=C_ACC)
    ax.set_xticks(list(x)); ax.set_xticklabels([l for _, l in regimes], fontsize=9)
    ax.set_ylabel("TPM [K tok/s/GiB]（論文プロトコル）")
    ax.set_title("スループット / KVキャッシュメモリ")
    ax.set_ylim(0, max(p_tpm) * 1.25)
    ax.legend()
    ax = axes[1]
    b_kv = [bench["baseline"][k]["kv_cache_gib"] * 1024 for k, _ in regimes]
    p_kv = [bench["photon"][k]["kv_cache_gib"] * 1024 for k, _ in regimes]
    b1 = ax.bar([i - w / 2 for i in x], b_kv, w, color=C_BASE, label="Baseline")
    b2 = ax.bar([i + w / 2 for i in x], p_kv, w, color=C_PHOT, label="PHOTON v2")
    ax.bar_label(b1, fmt="%.0f", fontsize=9)
    ax.bar_label(b2, fmt="%.0f", fontsize=9)
    for i in x:
        ax.annotate(f"1/{b_kv[i] / p_kv[i]:.0f}", (i + w / 2, p_kv[i]),
                    textcoords="offset points", xytext=(0, 14), ha="center",
                    fontsize=11, fontweight="bold", color=C_ACC)
    ax.set_xticks(list(x)); ax.set_xticklabels([l for _, l in regimes], fontsize=9)
    ax.set_ylabel("KVキャッシュ [MiB]（batch 4, 系列末時点）")
    ax.set_title("KVキャッシュメモリ（低いほど良い）")
    ax.set_ylim(0, max(b_kv) * 1.25)
    ax.legend()
    fig.suptitle("v2 推論ベンチマーク（small ペア, 論文の TPM プロトコル）", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "photon-v2-bench.png"))
    plt.close(fig)
else:
    skip("photon-v2-bench", "benchmark_results_v2.json")

# ---------------------------------------------------------------- fig 4
# RecGen diagnostics (v2): recon fidelity per level + forced-decoding gap.
diag = read_json("recgen_diag_photon2_a03.json")
diag0 = read_json("recgen_diag_photon2.json")  # optional alpha=0 reference
if diag:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    entries = []
    if diag0:
        entries.append(("α=0", diag0))
    entries.append(("α=0.3", diag))
    levels = sorted(diag["recon_cosine"].keys())
    w = 0.8 / len(entries)
    for j, (label, d) in enumerate(entries):
        xs = [i + (j - (len(entries) - 1) / 2) * w for i in range(len(levels))]
        bars = ax.bar(xs, [d["recon_cosine"][l] for l in levels], w * 0.95,
                      label=label, color=C_PHOT if label == "α=0" else C_ACC)
        ax.bar_label(bars, fmt="%.2f", fontsize=9)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([l.replace("level", "レベル ") for l in levels])
    ax.set_ylim(0, 1.05)
    ax.set_title("潜在再構成の忠実度（$\\hat{X}$ vs $X$ の cosine）")
    ax.legend()
    ax = axes[1]
    pairs = [(lab, d["hiergen_ppl"], d["recgen_ppl"]) for lab, d in entries]
    xr = range(len(pairs))
    b1 = ax.bar([i - 0.18 for i in xr], [p[1] for p in pairs], 0.36, color=C_BASE, label="HierGen")
    b2 = ax.bar([i + 0.18 for i in xr], [p[2] for p in pairs], 0.36, color=C_PHOT, label="RecGen")
    ax.bar_label(b1, fmt="%.0f", fontsize=9)
    ax.bar_label(b2, fmt="%.0f", fontsize=9)
    ax.set_yscale("log")
    ax.set_xticks(list(xr)); ax.set_xticklabels([p[0] for p in pairs])
    ax.set_title("強制デコード ppl（対数軸, 低いほど良い）")
    ax.legend()
    fig.suptitle("RecGen 診断（v2, 忠実実装）", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "photon-v2-recgen.png"))
    plt.close(fig)
else:
    skip("photon-v2-recgen", "recgen_diag_photon2_a03.json")

# ---------------------------------------------------------------- fig 5
# Medium scale-up loss curves: baseline_med (v1-era, arch unchanged) vs
# photon2_med (v2), same 1.5B data2b tokens.
s_bm, v_bm = read_log("baseline_med_log.jsonl")
s_pm2, v_pm2 = read_log("photon2_med_log.jsonl")
if s_pm2:
    s_pm1, v_pm1 = read_log("photon_med_log.jsonl")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot([x * TOK_PER_STEP / 1e9 for x in s_bm], v_bm, label="Baseline 187M", color=C_BASE, lw=1.8)
    if s_pm1:
        ax.plot([x * TOK_PER_STEP / 1e9 for x in s_pm1], v_pm1,
                label="PHOTON v1 158M（誤実装・撤回）", color=C_V1, lw=1.8, ls="--")
    ax.plot([x * TOK_PER_STEP / 1e9 for x in s_pm2], v_pm2,
            label="PHOTON v2 204M（論文忠実）", color=C_PHOT, lw=1.8)
    ax.set_xlabel("学習トークン数 [B]")
    ax.set_ylabel("検証 token loss (data2b val)")
    ax.set_title("スケールアップ学習曲線（medium, 1.5B トークン）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "photon-v2-loss-curves-med.png"))
    plt.close(fig)
else:
    skip("photon-v2-loss-curves-med", "photon2_med_log.jsonl")

print("figures written to", OUT)
for f in sorted(os.listdir(OUT)):
    print(" -", f)

"""Figures and tables for the cross-family findings write-up.

    python -m rohan.report_families      # per-family summaries first
    python -m rohan.analyze_findings     # then this

Reads rohan/results/<Family>_Pure_Sweep/, results/Pure_Sweep/, and
results/task21_kv_workloads/; writes everything to rohan/results/findings/.
Missing runs (e.g. an unfinished sweep) are left blank and simply re-run later.
"""
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures without a display
import matplotlib.pyplot as plt
import pandas as pd

from rohan.run_family_sweep import BASE, FAMILIES, family_configs
from src.config import LLAMA_BUILD_DIR, PROJECT_ROOT


RESULTS = BASE.project_root / "results"
OUTPUT = RESULTS / "findings"
ORIGINAL_LLAMA = PROJECT_ROOT / "results" / "Pure_Sweep" / "summary.csv"
KV_SUMMARY = PROJECT_ROOT / "results" / "task21_kv_workloads" / "kv_cache_workload_summary.csv"
LEVELS = ["Q2", "Q3", "Q4", "Q5", "Q6", "Q8", "BF16"]
SAMPLES = {"gsm8k": 100, "ifeval": 50, "hellaswag": 200}
ORIGINAL_SAMPLES = {"gsm8k": 1000, "ifeval": 541, "hellaswag": 2000}
TASK_LABELS = {"gsm8k_first": "GSM8K, first answer (%)", "ifeval": "IFEval, strict prompt (%)",
               "hellaswag": "HellaSwag (%)"}

# Reference palette: the first three categorical slots pass all-pairs CVD checks.
COLORS = {"qwen": "#2a78d6", "gemma": "#eb6834", "llama": "#1baf7a"}
ORIGINAL = "llama original"
MUTED = "#8a8985"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"

# lm-eval's gsm8k flexible-extract filter and exact_match normalization.
NUMBER = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")
# Start of a self-invented follow-up question after the model's own answer.
NEW_QUESTION = re.compile(r"\[Question\]|\n\s*Question\b|\n\s*Q:|\bQ: ")


def normalize(text: str) -> str:
    text = re.sub(r"(?s).*#### ", "", text)
    for pattern in (",", r"\$", r"\.$"):
        text = re.sub(pattern, "", text)
    return text.strip().lower()


def last_number(text: str) -> str:
    matches = NUMBER.findall(text)
    if not matches:
        return "[invalid]"
    return matches[-1][0] or matches[-1][1]


def gsm8k_rescore(samples_file: Path) -> dict | None:
    """Official flexible-extract accuracy, and accuracy on the first answer only."""
    if not samples_file.is_file():
        return None
    rows = [row for row in json.loads(samples_file.read_text(encoding="utf-8"))["gsm8k"]
            if row["filter"] == "flexible-extract"]
    official = first = continued = 0
    for row in rows:
        response, target = row["resps"][0][0], normalize(row["target"])
        official += normalize(last_number(response)) == target
        first += normalize(last_number(NEW_QUESTION.split(response)[0])) == target
        continued += bool(NEW_QUESTION.search(response))
    # Guard: the re-implementation must agree with the harness score.
    if official != sum(row["exact_match"] for row in rows):
        raise RuntimeError(f"GSM8K re-implementation disagrees with lm-eval: {samples_file}")
    return {"gsm8k_first": 100 * first / len(rows),
            "gsm8k_continued": 100 * continued / len(rows)}


def level(run: str) -> str | None:
    if run == "BF16-pure-sweep":
        return "BF16"
    if run.startswith("pure-") and run.endswith("-sweep"):
        return run[len("pure-"):-len("-sweep")].upper()
    return None


def tensor_mix(path: Path) -> str:
    """Share of weights per tensor type, e.g. 'Q4_0 76%, Q2_K 24%'."""
    try:
        sys.path.insert(0, str(LLAMA_BUILD_DIR.parents[1] / "gguf-py"))
        from gguf import GGUFReader
    except ImportError:
        return ""
    counts = Counter()
    for tensor in GGUFReader(path).tensors:
        if tensor.tensor_type.name != "F32":  # norms and biases, <1% of weights
            counts[tensor.tensor_type.name] += int(tensor.n_elements)
    total = sum(counts.values())
    return ", ".join(f"{name} {count / total:.0%}" for name, count in counts.most_common())


def summary_rows(family: str, table: pd.DataFrame, experiment_dir: Path | None,
                 samples: dict) -> list[dict]:
    """One row per quantization level with bits per weight, scores, SE, and KL."""
    base_size = table.loc["BF16-pure-sweep", "size_mib"]
    kl_means = [col for col in table if col.startswith("kl_") and col.endswith("_mean")]
    models = {config.name: config.model_file for config in family_configs(family)} \
        if family in FAMILIES else {}
    rows = []
    for run, values in table.iterrows():
        quant = level(str(run))
        if quant is None:
            continue
        row = {"family": family, "level": quant, "size_mib": values["size_mib"],
               "bits_per_weight": 16 * values["size_mib"] / base_size,
               "gsm8k": values["gsm8k"], "ifeval": values["ifeval"],
               "hellaswag": values["hellaswag"],
               "kl_mean": values[kl_means].mean() if kl_means else math.nan}
        rescored = gsm8k_rescore(experiment_dir / str(run) / "samples.json") \
            if experiment_dir is not None else None
        # Without saved samples (original GPU sweep), the official score is the only one.
        row.update(rescored or {"gsm8k_first": row["gsm8k"], "gsm8k_continued": math.nan})
        for task in ("gsm8k_first", "ifeval", "hellaswag"):
            p, n = row[task] / 100, samples[task.split("_")[0]]
            row[f"{task}_se"] = 100 * math.sqrt(p * (1 - p) / n)
        model = models.get(str(run))
        row["tensor_types"] = tensor_mix(model) if model is not None and model.is_file() else ""
        rows.append(row)
    return rows


def load_summary() -> pd.DataFrame:
    rows = []
    for family, spec in FAMILIES.items():
        experiment = RESULTS / spec["experiment"]
        summary = experiment / "summary.csv"
        if summary.is_file():
            rows += summary_rows(family, pd.read_csv(summary, index_col=0), experiment, SAMPLES)
    if ORIGINAL_LLAMA.is_file():
        rows += summary_rows(ORIGINAL, pd.read_csv(ORIGINAL_LLAMA, index_col=0), None,
                             ORIGINAL_SAMPLES)
    table = pd.DataFrame(rows)
    table["level"] = pd.Categorical(table["level"], LEVELS, ordered=True)
    return table.sort_values(["family", "level"]).reset_index(drop=True)


# --- Figures ---------------------------------------------------------------
def style(ax, title: str) -> None:
    ax.set_title(title, fontsize=11, loc="left", color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def line_style(family: str) -> dict:
    if family == ORIGINAL:
        return dict(color=MUTED, linestyle="--", linewidth=2, marker="o", markersize=5, zorder=2)
    return dict(color=COLORS[family], linewidth=2, marker="o", markersize=6, zorder=3)


def legend_below(fig, ax, ncol: int) -> None:
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False,
               fontsize=9, labelcolor=INK_SECONDARY)


def plot_kl(table: pd.DataFrame, path: Path) -> None:
    """Mean KL against effective bits per weight; the most precise signal we have."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for family, rows in table[table["level"] != "BF16"].groupby("family", observed=True, sort=False):
        rows = rows.dropna(subset=["kl_mean"]).sort_values("bits_per_weight")
        if rows.empty:
            continue
        label = "llama (original GPU sweep)" if family == ORIGINAL else family
        ax.plot(rows["bits_per_weight"], rows["kl_mean"], label=label, **line_style(family))
        if family != ORIGINAL:
            # Llama and Qwen share bit widths; label one above and one below.
            offset = (-14, -13) if family == "qwen" else (5, 4)
            for _, row in rows.iterrows():
                ax.annotate(row["level"], (row["bits_per_weight"], row["kl_mean"]),
                            textcoords="offset points", xytext=offset, fontsize=7.5,
                            color=INK_SECONDARY)
    ax.set_yscale("log")
    ax.set_xlabel("effective bits per weight (file size relative to 16-bit source)",
                  color=INK_SECONDARY, fontsize=9)
    style(ax, "Mean KL divergence from the 16-bit model (nats, log scale; lower is closer)")
    legend_below(fig, ax, ncol=4)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_tasks(table: pd.DataFrame, path: Path) -> None:
    """Task scores per level with ±1 standard error, one panel per task."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=False)
    rows_by_family = table[table["family"] != ORIGINAL].groupby("family", observed=True, sort=False)
    for ax, (task, title) in zip(axes, TASK_LABELS.items()):
        for offset, (family, rows) in zip((-0.12, 0, 0.12), rows_by_family):
            rows = rows.set_index("level").reindex(LEVELS)
            x = [i + offset for i in range(len(LEVELS))]
            ax.errorbar(x, rows[task], yerr=rows[f"{task}_se"], capsize=2, elinewidth=1,
                        label=family, **line_style(family))
        ax.set_xticks(range(len(LEVELS)), LEVELS)
        ax.set_ylim(0, 100)
        style(ax, title)
    legend_below(fig, axes[0], ncol=3)
    fig.suptitle("Benchmark scores by quantization level (error bars: ±1 standard error)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_qwen_gsm8k(table: pd.DataFrame, path: Path) -> None:
    """Why Qwen's official GSM8K score rises with heavier quantization."""
    rows = table[table["family"] == "qwen"].set_index("level").reindex(LEVELS)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(LEVELS, rows["gsm8k"], color=MUTED, linestyle="--", linewidth=2, marker="o",
                 markersize=6, label="official score (last number in response)")
    axes[0].plot(LEVELS, rows["gsm8k_first"], color=COLORS["qwen"], linewidth=2, marker="o",
                 markersize=6, label="first answer only")
    axes[0].set_ylim(0, 100)
    style(axes[0], "Qwen GSM8K accuracy (%)")
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    axes[1].bar(LEVELS, rows["gsm8k_continued"], color=COLORS["qwen"], width=0.6)
    axes[1].set_ylim(0, 100)
    style(axes[1], "Responses that invent a new question after answering (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_reproducibility(table: pd.DataFrame, path: Path) -> None:
    """Llama rerun (CPU, small n) against the original sweep (GPU, large n)."""
    rerun = table[table["family"] == "llama"].set_index("level")
    original = table[table["family"] == ORIGINAL].set_index("level")
    shared = [lvl for lvl in LEVELS if lvl in rerun.index and lvl in original.index]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    kl_levels = [lvl for lvl in shared if lvl != "BF16"]
    # The runs nearly coincide, so the original is drawn as open rings around the rerun.
    axes[0].plot(kl_levels, original.loc[kl_levels, "kl_mean"], label="original (GPU)",
                 color=MUTED, linestyle="--", linewidth=2, marker="o", markersize=13,
                 markerfacecolor="none", markeredgewidth=2, zorder=2)
    axes[0].plot(kl_levels, rerun.loc[kl_levels, "kl_mean"], label="rerun (CPU)",
                 **line_style("llama"))
    axes[0].set_yscale("log")
    style(axes[0], "Mean KL vs 16-bit (nats, log)")
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    width = 0.38
    for k, (label, frame, family) in enumerate([("original (GPU)", original, ORIGINAL),
                                                ("rerun (CPU)", rerun, "llama")]):
        scores = [frame.loc[lvl, "ifeval"] for lvl in shared]
        axes[1].bar([j + (k - 0.5) * width for j in range(len(shared))], scores, width=width,
                    color=MUTED if family == ORIGINAL else COLORS["llama"], label=label)
    axes[1].set_xticks(range(len(shared)), shared)
    axes[1].set_ylim(0, 100)
    style(axes[1], "IFEval strict prompt accuracy (%)")
    axes[1].legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- KV-cache comparison -----------------------------------------------------
def kv_table() -> pd.DataFrame:
    """Quality spread and memory per KV-cache setting (results/task21_kv_workloads)."""
    data = pd.read_csv(KV_SUMMARY)
    data["kv"] = "K " + data["k_type"] + " / V " + data["v_type"]
    quality = [col for col in data if col.endswith("_quality")]
    data["task"] = data[quality].idxmax(axis=1, skipna=True).str.replace("_quality", "")
    data["quality"] = 100 * data[quality].max(axis=1)
    return data[["task", "kv", "sample_limit", "quality", "peak_rss_delta_mb",
                 "prompt_tokens_per_second"]]


def plot_kv(kv: pd.DataFrame, path: Path) -> None:
    order = sorted(kv["kv"].unique(), key=lambda name: ("f16" not in name, name))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    tasks = sorted(kv["task"].unique())
    for y, task in enumerate(tasks):
        rows = kv[kv["task"] == task]
        base = rows[rows["kv"] == "K f16 / V f16"].iloc[0]
        p, n = base["quality"] / 100, base["sample_limit"]
        se = 100 * math.sqrt(p * (1 - p) / n)
        axes[0].barh(y, 4 * se, left=base["quality"] - 2 * se, height=0.55, color=GRID,
                     label="±2 SE around f16 cache" if y == 0 else None)
        others = rows[rows["kv"] != "K f16 / V f16"]
        axes[0].scatter(others["quality"], [y] * len(others), s=36, color=COLORS["llama"],
                        alpha=0.75, zorder=3, label="quantized KV cache (8 settings)" if y == 0 else None)
        axes[0].scatter([base["quality"]], [y], s=70, marker="D", color=INK, zorder=4,
                        label="f16 cache" if y == 0 else None)
    axes[0].set_yticks(range(len(tasks)), [f"{task} (n={int(kv[kv['task'] == task]['sample_limit'].iloc[0])})"
                                           for task in tasks], fontsize=8.5)
    axes[0].set_xlim(-2, 100)
    style(axes[0], "Score (%) for each KV-cache setting")
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY,
                   loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    memory = kv.groupby("kv")["peak_rss_delta_mb"].mean().reindex(order)
    axes[1].bar(range(len(memory)), memory, color=COLORS["llama"], width=0.6)
    axes[1].set_xticks(range(len(memory)), memory.index, rotation=35, ha="right", fontsize=8)
    style(axes[1], "Peak memory above baseline (MB), mean over tasks")
    fig.suptitle("KV-cache quantization (Llama Q2_K, results/task21_kv_workloads)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table = load_summary()
    columns = ["family", "level", "size_mib", "bits_per_weight", "tensor_types",
               "gsm8k", "gsm8k_first", "gsm8k_first_se", "gsm8k_continued",
               "ifeval", "ifeval_se", "hellaswag", "hellaswag_se", "kl_mean"]
    table[columns].round(4).to_csv(OUTPUT / "summary_table.csv", index=False)
    plot_kl(table, OUTPUT / "kl_vs_bits.png")
    plot_tasks(table, OUTPUT / "task_scores.png")
    plot_qwen_gsm8k(table, OUTPUT / "qwen_gsm8k_artifact.png")
    plot_reproducibility(table, OUTPUT / "llama_reproducibility.png")
    if KV_SUMMARY.is_file():
        kv = kv_table()
        kv.round(2).to_csv(OUTPUT / "kv_cache_table.csv", index=False)
        plot_kv(kv, OUTPUT / "kv_cache_comparison.png")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table[columns].round(3).to_string(index=False))
    print(f"\nSaved figures and tables to {OUTPUT}")


if __name__ == "__main__":
    main()

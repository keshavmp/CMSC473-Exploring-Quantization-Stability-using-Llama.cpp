"""Tables and figures for the Llama context-length x weight-quantization runs.

    python -m rohan.analyze_long_context

Reads every finished run in rohan/results/Llama_Long_Context/ (and the pilot,
for memory only) and writes to rohan/results/Llama_Long_Context/findings/.
Unfinished runs are simply missing, so this can be re-run after every night.
"""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures without a display
import matplotlib.pyplot as plt
import pandas as pd

from rohan.analyze_findings import GRID, INK, INK_SECONDARY, style
from rohan.run_long_context import BASE, EXPERIMENT, PARTS

RESULTS = BASE.project_root / "results"
RUNS = RESULTS / EXPERIMENT
PILOT = RESULTS / f"{EXPERIMENT}-pilot"
OUTPUT = RUNS / "findings"

LEVELS = ["Q2", "Q3", "Q4", "Q5", "Q6", "Q8", "BF16", "Q2_K-mixed"]
# Sequential ramp: quant levels are ordered, so lighter = fewer bits.
COLORS = {"Q2": "#c6dbef", "Q3": "#9ecae1", "Q4": "#4292c6", "Q5": "#2171b5",
          "Q6": "#08519c", "Q8": "#08306b", "BF16": INK, "Q2_K-mixed": "#eb6834"}
TASK_LABELS = {"niah_multikey_2": "niah_multikey_2 (find 1 key among similar keys)",
               "ruler_vt": "ruler_vt (follow a chain of variable assignments)",
               "niah_single_1": "niah_single_1 (find 1 needle)"}
# Night 1 ran with llama-server's default 8 GB RAM prompt cache, which kept
# every past prompt's KV state and inflated peak memory by ~2 GB.
CACHE_ON_RUNS = {run.name for run in PARTS["night1"]}


def sample_scores(run_dir: Path, length: int) -> dict[str, float]:
    samples = json.loads((run_dir / "samples.json").read_text())
    return {str(s["doc_id"]): float(s[str(length)]) for s in samples}


def standard_error(scores) -> float:
    scores = list(scores)
    n = len(scores)
    if n < 2:
        return float("nan")
    mean = sum(scores) / n
    return math.sqrt(sum((x - mean) ** 2 for x in scores) / (n - 1) / n)


def load_runs(folder: Path) -> pd.DataFrame:
    rows = []
    for stats_file in sorted(folder.glob("*/stats.json")):
        stats = json.loads(stats_file.read_text())
        row = {k: v for k, v in stats.items() if not isinstance(v, (dict, list))}
        if row.get("kind") == "ruler":
            scores = sample_scores(stats_file.parent, row["length"])
            row["accuracy_se"] = standard_error(scores.values())
        row["cache_ram_on"] = stats.get("cache_ram_mib", 0) != 0 or (
            folder == RUNS and row["run"] in CACHE_ON_RUNS)
        rows.append(row)
    return pd.DataFrame(rows)


def paired_gaps(table: pd.DataFrame) -> pd.DataFrame:
    """Level minus BF16 on the same questions, with a paired standard error."""
    rows = []
    ruler = table[table.kind == "ruler"]
    for _, row in ruler[ruler.level != "BF16"].iterrows():
        base = ruler[(ruler.level == "BF16") & (ruler.task == row.task) & (ruler.length == row.length)]
        if base.empty:
            continue
        quant = sample_scores(RUNS / row.run, row.length)
        ref = sample_scores(RUNS / base.iloc[0].run, row.length)
        diffs = [quant[d] - ref[d] for d in quant if d in ref]
        rows.append({"task": row.task, "level": row.level, "length": row.length,
                     "gap": sum(diffs) / len(diffs), "gap_se": standard_error(diffs),
                     "questions": len(diffs)})
    return pd.DataFrame(rows)


def length_axis(ax, lengths) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(lengths)
    ax.set_xticklabels([f"{n // 1024}K" for n in lengths])
    ax.minorticks_off()
    ax.set_xlabel("Context length (tokens)", color=INK_SECONDARY, fontsize=9)


def level_lines(ax, table, y, err=None, levels=LEVELS):
    for level in levels:
        part = table[table.level == level].sort_values("length")
        if part.empty:
            continue
        kwargs = dict(color=COLORS[level], linewidth=2, marker="o", markersize=5, label=level,
                      linestyle="--" if level == "BF16" else "-")
        if err is not None:
            ax.errorbar(part.length, part[y], yerr=part[err], capsize=3, **kwargs)
        else:
            ax.plot(part.length, part[y], **kwargs)


def plot_accuracy(table: pd.DataFrame, path: Path) -> None:
    tasks = [t for t in ("niah_multikey_2", "ruler_vt") if t in set(table.task)]
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.5 * len(tasks), 4.4), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        part = table[(table.kind == "ruler") & (table.task == task)].copy()
        part["pct"], part["pct_se"] = part.accuracy * 100, part.accuracy_se * 100
        level_lines(ax, part, "pct", "pct_se")
        length_axis(ax, sorted(part.length.unique()))
        ax.set_ylim(-5, 105)
        ax.set_ylabel("Accuracy (%), ±1 SE", color=INK_SECONDARY, fontsize=9)
        style(ax, TASK_LABELS[task])
    axes[0][0].legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    fig.suptitle("RULER accuracy vs context length (Llama-3.2-1B, 20 questions per point)",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_gaps(gaps: pd.DataFrame, path: Path) -> None:
    tasks = [t for t in ("niah_multikey_2", "ruler_vt") if t in set(gaps.task)]
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.5 * len(tasks), 4.4), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        part = gaps[(gaps.task == task) & (gaps.level != "Q2")].copy()  # Q2 is 0 everywhere
        part["pct"], part["pct_se"] = part.gap * 100, part.gap_se * 100
        ax.axhline(0, color=INK, linewidth=1, linestyle="--")
        level_lines(ax, part, "pct", "pct_se")
        length_axis(ax, sorted(part.length.unique()))
        ax.set_ylabel("Accuracy − BF16 (points), ±1 paired SE", color=INK_SECONDARY, fontsize=9)
        style(ax, TASK_LABELS[task])
    axes[0][0].legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    fig.suptitle("Quantization gap vs context length (same questions at every level)",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cost(table: pd.DataFrame, pilot: pd.DataFrame, path: Path) -> None:
    ruler = table[table.kind == "ruler"]
    cost = ruler.groupby(["level", "length"], as_index=False)[
        ["seconds_per_question", "prompt_tokens_per_second", "generation_tokens_per_second"]].mean()
    # Memory only from runs without the RAM prompt cache; BF16/Q4 come from the pilot.
    clean = pd.concat([ruler[~ruler.cache_ram_on], pilot[pilot.kind == "ruler"]])
    memory = clean.groupby(["level", "length"], as_index=False)["peak_rss_mb"].max()
    memory["peak_gb"] = memory.peak_rss_mb / 1024
    panels = [(cost, "seconds_per_question", "Seconds per question"),
              (cost, "prompt_tokens_per_second", "Prompt reading (tok/s)"),
              (cost, "generation_tokens_per_second", "Generation (tok/s)"),
              (memory, "peak_gb", "Peak server memory (GB)")]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    for ax, (data, column, title) in zip(axes, panels):
        level_lines(ax, data, column)
        length_axis(ax, sorted(data.length.unique()))
        if column == "seconds_per_question":
            ax.set_yscale("log")
        style(ax, title)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    fig.suptitle("Cost vs context length on an Intel i5 MacBook Pro (CPU, 1 slot, f16 KV cache)",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def accuracy_table(table: pd.DataFrame) -> pd.DataFrame:
    ruler = table[table.kind == "ruler"].copy()
    ruler["cell"] = [f"{a * 100:.0f} ± {s * 100:.0f}" for a, s in zip(ruler.accuracy, ruler.accuracy_se)]
    wide = ruler.pivot_table(index=["task", "level"], columns="length", values="cell", aggfunc="first")
    wide.columns = [f"{c // 1024}K" for c in wide.columns]
    order = {level: i for i, level in enumerate(["BF16", "Q8", "Q6", "Q5", "Q4", "Q3", "Q2", "Q2_K-mixed"])}
    return wide.reset_index().sort_values(["task", "level"], key=lambda s: s.map(order) if s.name == "level" else s)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table = load_runs(RUNS)
    pilot = load_runs(PILOT) if PILOT.exists() else pd.DataFrame()
    if table.empty:
        raise SystemExit(f"No finished runs in {RUNS}")
    columns = ["run", "kind", "task", "level", "length", "accuracy", "accuracy_se", "kl_mean",
               "seconds_per_question", "mean_time_to_first_token_s", "prompt_tokens_per_second",
               "generation_tokens_per_second", "mean_prompt_tokens", "cache_hit_rate",
               "baseline_rss_mb", "peak_rss_mb", "kv_cache_mib", "model_file_mib",
               "cache_ram_on", "wall_seconds"]
    table.reindex(columns=columns).sort_values(["kind", "task", "length", "level"]).to_csv(
        OUTPUT / "summary_long_context.csv", index=False)
    accuracy_table(table).to_csv(OUTPUT / "ruler_accuracy_table.csv", index=False)
    gaps = paired_gaps(table)
    gaps.sort_values(["task", "level", "length"]).to_csv(OUTPUT / "quant_gap_vs_bf16.csv", index=False)
    plot_accuracy(table, OUTPUT / "ruler_accuracy_vs_length.png")
    plot_gaps(gaps, OUTPUT / "quant_gap_vs_length.png")
    plot_cost(table, pilot, OUTPUT / "cost_vs_length.png")
    print(f"{len(table)} runs -> {OUTPUT}")


if __name__ == "__main__":
    main()

"""Tables and figures for the Llama context-length x weight-quantization runs.

    python -m rohan.analyze_long_context

Reads every finished run in rohan/results/Llama_Long_Context/, the pure sweep
(GSM8K at padding 0) and task15's summary (KV-cache comparison), and writes to
rohan/results/Llama_Long_Context/findings/.
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
TASK15_SUMMARY = (BASE.project_root.parent / "results" / "task15_long_context_results"
                  / "statistics" / "task15_long_context_summary.csv")
OUTPUT = RUNS / "findings"
PURE_SWEEP = RESULTS / "Llama_Pure_Sweep"
GSM8K_LEVELS = ["BF16", "Q8", "Q6", "Q4", "Q3"]
GSM8K_PADDINGS = [0, 4096, 16384]

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
    fig.suptitle("RULER accuracy vs context length (Llama-3.2-1B, 20 questions per point, 10 at 32K)",
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


def clean_memory(table: pd.DataFrame) -> pd.DataFrame:
    """Peak server memory per (level, length), only from runs without the RAM
    prompt cache. Memory barely depends on the task, so the max over tasks is used."""
    ruler = table[(table.kind == "ruler") & ~table.cache_ram_on]
    memory = ruler.groupby(["level", "length"], as_index=False)[["peak_rss_mb", "baseline_rss_mb"]].max()
    memory["peak_gb"] = memory.peak_rss_mb / 1024
    return memory


def plot_cost(table: pd.DataFrame, path: Path) -> None:
    ruler = table[table.kind == "ruler"]
    cost = ruler.groupby(["level", "length"], as_index=False)[
        ["seconds_per_question", "prompt_tokens_per_second", "generation_tokens_per_second"]].mean()
    memory = clean_memory(table)
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


def kl_table(table: pd.DataFrame) -> pd.DataFrame:
    """KL from BF16 per window. Window W scores positions W/2..W of each chunk,
    over the same 32K tokens of essays, so a larger window means later positions."""
    rows = []
    for _, row in table[table.kind == "kl"].iterrows():
        essays = json.loads((RUNS / row.run / "kl.json").read_text())["domains"]["essays"]
        rows.append({"level": row.level, "window": row.length,
                     "scored_positions": f"{row.length // 2}-{row.length}",
                     "kl_mean": essays["mean"], "kl_median": essays["median"],
                     "kl_p95": essays["p95"], "kl_uncertainty": essays["uncertainty"],
                     "scored_tokens": essays["tokens"], "peak_rss_mb": row.peak_rss_mb,
                     "wall_seconds": row.wall_seconds})
    kl = pd.DataFrame(rows)
    if kl.empty:
        return kl
    first = kl[kl.window == kl.window.min()].set_index("level").kl_mean
    kl["change_from_smallest_window"] = [m / first[lv] - 1 if first.get(lv) else float("nan")
                                         for lv, m in zip(kl.level, kl.kl_mean)]
    return kl.sort_values(["level", "window"])


def plot_kl(kl: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    part = kl[kl.level != "BF16"].rename(columns={"window": "length"})
    level_lines(axes[0], part, "kl_mean", "kl_uncertainty")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Mean KL from BF16 (nats, log scale)", color=INK_SECONDARY, fontsize=9)
    style(axes[0], "KL by window size")
    part = part.assign(pct=part.change_from_smallest_window * 100)
    axes[1].axhline(0, color=INK, linewidth=1, linestyle="--")
    level_lines(axes[1], part, "pct")
    axes[1].set_ylabel("Change from the 4K window (%)", color=INK_SECONDARY, fontsize=9)
    style(axes[1], "Relative growth with window size")
    for ax in axes:
        length_axis(ax, sorted(part.length.unique()))
        ax.set_xlabel("KL window (scores positions W/2 to W)", color=INK_SECONDARY, fontsize=9)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, ncol=2)
    fig.suptitle("Does quantized output drift further from BF16 later in a long context? (32K tokens of essays)",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def needle_depths(table: pd.DataFrame) -> pd.DataFrame:
    """Per-question needle depth for niah_multikey_2: where the answer sits in
    the haystack, as a fraction of the text before the question."""
    rows = []
    runs = table[(table.kind == "ruler") & (table.task == "niah_multikey_2")]
    for _, row in runs.iterrows():
        samples = json.loads((RUNS / row.run / "samples.json").read_text())
        for s in samples:
            text, answer = s["doc"]["input"], s["doc"]["outputs"][0]
            end = text.rfind("What is the special magic number")
            position = text.find(answer)
            if position < 0 or end <= 0:
                continue
            rows.append({"level": row.level, "length": row.length, "doc_id": str(s["doc_id"]),
                         "depth": position / end, "score": float(s[str(row.length)])})
    return pd.DataFrame(rows)


def plot_needle_depth(depths: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Accuracy by depth quarter, pooled over lengths of 8K and above (4K is near 100%)."""
    part = depths[depths.length >= 8192].copy()
    part["quarter"] = (part.depth * 4).clip(upper=3.999).astype(int)
    summary = part.groupby(["level", "quarter"]).score.agg(["mean", "count"]).reset_index()
    summary["se"] = [math.sqrt(m * (1 - m) / n) if n > 1 else float("nan")
                     for m, n in zip(summary["mean"], summary["count"])]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    labels = ["0–25%\n(start)", "25–50%", "50–75%", "75–100%\n(end)"]
    for level in ["BF16", "Q6", "Q4", "Q3"]:
        rows = summary[summary.level == level].sort_values("quarter")
        if rows.empty:
            continue
        ax.errorbar(rows.quarter, rows["mean"] * 100, yerr=rows.se * 100, capsize=3,
                    color=COLORS[level], linewidth=2, marker="o", markersize=5, label=level,
                    linestyle="--" if level == "BF16" else "-")
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels)
    ax.set_ylim(-5, 105)
    ax.set_xlabel("Needle depth in the haystack", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("Accuracy (%), ±1 SE", color=INK_SECONDARY, fontsize=9)
    style(ax, "niah_multikey_2 accuracy by needle depth (8K–32K pooled)")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return summary


def task15_table() -> pd.DataFrame:
    if not TASK15_SUMMARY.is_file():
        return pd.DataFrame()
    t15 = pd.read_csv(TASK15_SUMMARY)
    return pd.DataFrame({
        "study": "task15 (KV cache varied, weights mixed Q2_K, Apple GPU, 50 Q)",
        "weights": "Q2_K-mixed", "kv_cache": t15.k_type, "length": t15.context_length,
        "accuracy": t15.quality, "questions": t15.sample_limit,
        "seconds_per_question": t15.evaluation_seconds / t15.sample_limit,
        "peak_rss_mb": t15.peak_server_rss_mb})


def weight_vs_kv(table: pd.DataFrame) -> pd.DataFrame:
    ours = table[(table.kind == "ruler") & (table.task == "niah_single_1")]
    mine = pd.DataFrame({
        "study": "this study (weights varied, f16 KV cache, Intel CPU, 20 Q)",
        "weights": ours.level, "kv_cache": "f16", "length": ours.length,
        "accuracy": ours.accuracy, "questions": ours.questions,
        "seconds_per_question": ours.seconds_per_question, "peak_rss_mb": ours.peak_rss_mb})
    return pd.concat([mine, task15_table()], ignore_index=True)


def plot_weight_vs_kv(both: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [8, 4]})
    lengths = [4096, 8192, 16384]
    shades = {4096: "#9ecae1", 8192: "#4292c6", 16384: "#08306b"}
    groups = [
        (axes[0], both[both.study.str.startswith("this")], "weights",
         ["BF16", "Q8", "Q6", "Q5", "Q4", "Q3", "Q2", "Q2_K-mixed"],
         "Weights varied, f16 KV cache (this study, CPU, 20 Q)"),
        (axes[1], both[both.study.str.startswith("task15")], "kv_cache", ["f16", "q8_0", "q4_0"],
         "KV cache varied, mixed Q2_K (task15, 50 Q)"),
    ]
    width = 0.27
    for ax, data, key, order, title in groups:
        for j, length in enumerate(lengths):
            rows = data[data.length == length].set_index(key)
            xs = [i + (j - 1) * width for i, k in enumerate(order) if k in rows.index]
            ys = [rows.loc[k, "accuracy"] * 100 for k in order if k in rows.index]
            if xs:
                ax.bar(xs, ys, width, color=shades[length], label=f"{length // 1024}K")
                for x, y in zip(xs, ys):
                    ax.text(x, y + 1.5, f"{y:.0f}", ha="center", fontsize=7, color=INK_SECONDARY)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, fontsize=9)
        ax.set_ylim(0, 110)
        style(ax, title)
    axes[0].set_ylabel("niah_single_1 accuracy (%)", color=INK_SECONDARY, fontsize=9)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="lower left")
    fig.suptitle("Simple retrieval: weight quantization vs KV-cache quantization",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def accuracy_cost(table: pd.DataFrame) -> pd.DataFrame:
    """Mean of niah_multikey_2 and ruler_vt per (level, length), with time and
    clean peak memory. 'frontier' marks points no other point beats on both
    accuracy and that cost."""
    hard = table[(table.kind == "ruler") & table.task.isin(["niah_multikey_2", "ruler_vt"])]
    both = hard.groupby(["level", "length"]).filter(lambda g: g.task.nunique() == 2)
    points = both.groupby(["level", "length"], as_index=False).agg(
        accuracy=("accuracy", "mean"), seconds_per_question=("seconds_per_question", "mean"))
    memory = clean_memory(table)[["level", "length", "peak_gb"]]
    points = points.merge(memory, on=["level", "length"], how="left")
    for cost in ("seconds_per_question", "peak_gb"):
        flags = []
        for _, p in points.iterrows():
            if pd.isna(p[cost]):
                flags.append(False)
                continue
            beaten = points[(points[cost] <= p[cost]) & (points.accuracy >= p.accuracy)
                            & ((points[cost] < p[cost]) | (points.accuracy > p.accuracy))]
            flags.append(beaten.empty)
        points[f"frontier_{cost}"] = flags
    return points


def plot_accuracy_cost(points: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    markers = {4096: "o", 8192: "s", 16384: "^"}
    for ax, cost, label in ((axes[0], "seconds_per_question", "Seconds per question (log)"),
                            (axes[1], "peak_gb", "Peak server memory (GB)")):
        data = points.dropna(subset=[cost])
        front = data[data[f"frontier_{cost}"]].sort_values(cost)
        ax.plot(front[cost], front.accuracy * 100, color=GRID, linewidth=6, zorder=1,
                solid_capstyle="round", label="best trade-off")
        for _, p in data.iterrows():
            ax.scatter(p[cost], p.accuracy * 100, color=COLORS[p.level], marker=markers[p.length],
                       s=60, zorder=3, edgecolor="white", linewidth=0.8)
            # BF16 and Q6 often coincide, so BF16's label goes below the point.
            ax.annotate(f"{p.level} {p.length // 1024}K", (p[cost], p.accuracy * 100),
                        textcoords="offset points", xytext=(5, -11 if p.level == "BF16" else 4),
                        fontsize=7, color=INK_SECONDARY)
        if cost == "seconds_per_question":
            ax.set_xscale("log")
        ax.set_xlabel(label, color=INK_SECONDARY, fontsize=9)
        ax.set_ylim(-5, 100)
        style(ax, f"Accuracy vs {label.split(' (')[0].lower()}")
    axes[0].set_ylabel("Mean accuracy, multikey_2 + vt (%)", color=INK_SECONDARY, fontsize=9)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="lower right")
    fig.suptitle("What each setting costs (circle 4K, square 8K, triangle 16K; Night-1 BF16/Q4 8K lack clean memory)",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def gsm8k_scores(level: str, padding: int) -> dict[int, float] | None:
    """Per-question exact match. Padding 0 is the pure sweep's flexible-extract score."""
    if padding == 0:
        name = "BF16-pure-sweep" if level == "BF16" else f"pure-{level.lower()}-sweep"
        path = PURE_SWEEP / name / "samples.json"
        if not path.is_file():
            return None
        rows = [r for r in json.loads(path.read_text())["gsm8k"] if r["filter"] == "flexible-extract"]
        return {int(r["doc_id"]): float(r["exact_match"]) for r in rows}
    path = RUNS / f"gsm8k-{level.lower()}-pad{padding // 1024}k" / "samples.json"
    if not path.is_file():
        return None
    return {int(r["doc_id"]): float(r["exact_match"]) for r in json.loads(path.read_text())}


def gsm8k_table() -> pd.DataFrame:
    """Accuracy at each padding, on the first 50 and first 100 questions, with
    the paired change from padding 0 on the same questions."""
    rows = []
    for level in GSM8K_LEVELS:
        base = gsm8k_scores(level, 0)
        for padding in GSM8K_PADDINGS:
            scores = gsm8k_scores(level, padding)
            if scores is None or base is None:
                continue
            for first in (50, 100):
                ids = [i for i in range(first) if i in scores and i in base]
                if len(ids) < first:
                    continue
                diffs = [scores[i] - base[i] for i in ids]
                rows.append({"level": level, "padding": padding, "questions": first,
                             "accuracy": sum(scores[i] for i in ids) / first,
                             "accuracy_se": standard_error(scores[i] for i in ids),
                             "change_from_pad0": sum(diffs) / first,
                             "change_se": standard_error(diffs)})
    return pd.DataFrame(rows)


def plot_gsm8k(gsm: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    positions = {p: i for i, p in enumerate(GSM8K_PADDINGS)}
    labels = ["0 (pure sweep)", "4K", "16K"]
    for ax, column, err, title in (
            (axes[0], "accuracy", "accuracy_se", "GSM8K accuracy (%), first 50 questions"),
            (axes[1], "change_from_pad0", "change_se", "Change from no padding (points), paired")):
        part = gsm[gsm.questions == 50]
        if column == "change_from_pad0":
            ax.axhline(0, color=INK, linewidth=1, linestyle="--")
        for level in GSM8K_LEVELS:
            rows = part[part.level == level].sort_values("padding")
            if rows.empty:
                continue
            ax.errorbar([positions[p] for p in rows.padding], rows[column] * 100, yerr=rows[err] * 100,
                        color=COLORS[level], linewidth=2, marker="o", markersize=5, capsize=3,
                        label=level, linestyle="--" if level == "BF16" else "-")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_xlabel("Essay padding before the question (tokens)", color=INK_SECONDARY, fontsize=9)
        style(ax, title)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    fig.suptitle("GSM8K behind irrelevant long text (zero-shot, ±1 SE)",
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
    plot_cost(table, OUTPUT / "cost_vs_length.png")
    gsm = gsm8k_table()
    if not gsm.empty:
        gsm.to_csv(OUTPUT / "gsm8k_padding.csv", index=False)
        plot_gsm8k(gsm, OUTPUT / "gsm8k_padding.png")
    kl = kl_table(table)
    if not kl.empty:
        kl.to_csv(OUTPUT / "kl_vs_window.csv", index=False)
        plot_kl(kl, OUTPUT / "kl_vs_window.png")
    depths = needle_depths(table)
    if not depths.empty:
        depths.to_csv(OUTPUT / "needle_depth_samples.csv", index=False)
        plot_needle_depth(depths, OUTPUT / "needle_depth.png").to_csv(
            OUTPUT / "needle_depth.csv", index=False)
    niah = weight_vs_kv(table)
    if not niah.empty:
        niah.sort_values(["study", "weights", "kv_cache", "length"]).to_csv(
            OUTPUT / "weight_vs_kv_niah.csv", index=False)
        plot_weight_vs_kv(niah, OUTPUT / "weight_vs_kv_niah.png")
    points = accuracy_cost(table)
    points.to_csv(OUTPUT / "accuracy_vs_cost.csv", index=False)
    plot_accuracy_cost(points, OUTPUT / "accuracy_vs_cost.png")
    print(f"{len(table)} runs -> {OUTPUT}")


if __name__ == "__main__":
    main()

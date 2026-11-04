"""Rank blocks and tensor types by quantization sensitivity and check whether the ranking holds across domains.

Reads the results of src/sensitivity.py. For each precision combination (background -> target) two separate sweeps:
    block    16 items: one whole block (blk.N) set to the target precision, the rest at the background
    tensor    7 items: one tensor type (attn_q ... ffn_down) set to the target precision in every block
An item's sensitivity is the KL it adds over the control, the pure background model:
    KL(one block or tensor type lowered to the target, rest background) - KL(pure background)
(BF16 background: the control is the BF16 source scored against its own reference, the KL floor.)
Prints the items from most to least sensitive with the most/least ratio, the Spearman rank
correlation between chat, math, and code (1 = same ranking), and a note for any result within
2x its uncertainty of zero.

Heatmap: combinations differ hugely in scale (a Q2 target adds about 1000x the KL of a Q6 target), so each
(combination, domain) row is divided by its own mean sensitivity (negative values count as 0). 1.0 is then an
average item in its row. Cells are averaged over the domains, and a last row averages all combinations.
Row labels show each row's raw peak KL; hatched cells are within 2x uncertainty of zero in most domains.

Combinations: BF16 -> Q2, Q4, Q6 | Q8 -> Q2, Q4, Q6 | Q6 -> Q2, Q4 | Q4 -> Q2 | Q3 -> Q2

    python -m src.report_blocks                                # both sweeps, results/Block_Sensitivity_Differences
    python -m src.report_blocks --sweep tensor                 # only the tensor-type sweep
    python -m src.report_blocks --lines                        # also the per-combination line plots (slower)
    python -m src.report_blocks --experiment Block_Sensitivity # another results folder
    python -m src.report_blocks --output-dir report            # write the CSVs and plots elsewhere

Reads block_sensitivity.csv, tensor_sensitivity.csv and pure_baselines.csv from results/<experiment>/.
Writes <sweep>_sensitivity_summary.csv, <sweep>_sensitivity_normalized.csv and <sweep>_sensitivity_heatmap.png
next to them (<sweep>_sensitivity.png with --lines).
Uncertainties are combined as independent, a simple approximation, since both runs score the same tokens.
"""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import PowerNorm
from matplotlib.patches import Rectangle

try:
    from src.config import Config
except ModuleNotFoundError:  # running directly: python src/report_blocks.py
    from config import Config

EXPERIMENT = "Block_Sensitivity_Differences"  # results folder written by src/sensitivity.py
CSV_NAMES = {"block": "block_sensitivity.csv", "tensor": "tensor_sensitivity.csv"}
CONTROL_CSV = "pure_baselines.csv"  # the pure background models to be compared against
CLEAR_FACTOR = 2      # number of uncertainties for difference to be clear
MIN_ITEMS = 5         # fewest blocks / tensor types for a meaningful rank correlation
MEAN_ROW = "mean of all"  # last heatmap row: the average over every combination
BACKGROUND_ORDER = {"bf16": 0, "q8_0": 1, "q6_k": 2, "q4_k": 3, "q3_k": 4}  # heatmap row order
TARGET_ORDER = {"q2_k": 0, "q4_k": 1, "q6_k": 2}
TENSOR_ORDER = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"]


def normalize_rows(table: pd.DataFrame) -> pd.Series:
    """Sensitivity divided by the mean of its (combination, domain) row, so every row has mean 1.

    Negative values (noise below the control) count as 0, so one noisy item can't distort the row.
    """
    positive = table["sensitivity"].clip(lower=0)
    return positive / positive.groupby([table["pair"], table["domain"]]).transform("mean")


def load_sweep(path: Path, controls: pd.DataFrame, sweep: str) -> pd.DataFrame:
    """One row per variant and domain, with the KL it adds over its background's control."""
    table = pd.read_csv(path)
    table["pair"] = table["background"] + "->" + table["target"]
    if sweep == "block":
        table["changed"] = table["changed"].astype(int)

    base = controls.set_index(["background", "domain"])
    keys = list(zip(table["background"], table["domain"]))
    missing = sorted({background for background, domain in keys if (background, domain) not in base.index})
    if missing:
        raise ValueError(f"no pure control for {', '.join(missing)} in {CONTROL_CSV}; run the baselines first")

    table["sensitivity"] = table["kl_mean"].to_numpy() - base["kl_mean"].reindex(keys).to_numpy()
    base_uncertainty = base["kl_uncertainty"].reindex(keys).to_numpy()
    table["sens_uncertainty"] = (table["kl_uncertainty"] ** 2 + base_uncertainty ** 2) ** 0.5
    table["clear"] = table["sensitivity"] > CLEAR_FACTOR * table["sens_uncertainty"]
    table["normalized"] = normalize_rows(table)
    return table


def ordered_pairs(table: pd.DataFrame) -> list[str]:
    """Combinations from the highest background to the lowest, targets Q2, Q4, Q6."""
    def key(pair: str) -> tuple[int, int]:
        background, target = pair.split("->")
        return BACKGROUND_ORDER.get(background, 9), TARGET_ORDER.get(target, 9)
    return sorted(table["pair"].unique(), key=key)


def build_ranking(table: pd.DataFrame, pair: str) -> pd.DataFrame:
    """Items from most to least sensitive: one column per domain plus their mean."""
    ranking = table[table["pair"] == pair].pivot(index="changed", columns="domain", values="sensitivity")
    ranking["mean"] = ranking.mean(axis=1)
    return ranking.sort_values("mean", ascending=False)


def build_correlation(ranking: pd.DataFrame) -> pd.DataFrame:
    """Spearman rank correlation between domains; empty until enough items are measured."""
    domains = ranking.drop(columns="mean")
    if len(domains) < MIN_ITEMS:
        return pd.DataFrame()
    return domains.corr(method="spearman")


def spread(ranking: pd.DataFrame) -> float | None:
    """Most/least sensitive ratio; undefined when the least sensitive item is at or below the control."""
    least = ranking["mean"].min()
    return ranking["mean"].max() / least if least > 0 else None


def build_heatmap(table: pd.DataFrame, sweep: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Rows are combinations plus a mean row, columns are blocks or tensor types.

    Each cell averages the normalized sensitivity over domains. Also returns which cells are clear in most
    domains and each combination's raw peak sensitivity in nats, so the normalization doesn't hide magnitude.
    """
    by_item = table.groupby(["pair", "changed"])
    cells = by_item["normalized"].mean().unstack("changed")
    clear = by_item["clear"].mean().unstack("changed") >= 0.5
    peak = by_item["sensitivity"].mean().unstack("changed").max(axis=1)

    if sweep == "tensor":
        columns = [c for c in TENSOR_ORDER if c in cells.columns] + sorted(set(cells.columns) - set(TENSOR_ORDER))
    else:
        columns = sorted(cells.columns)
    pairs = ordered_pairs(table)
    cells, clear, peak = cells.loc[pairs, columns], clear.loc[pairs, columns], peak.loc[pairs]
    cells.loc[MEAN_ROW] = cells.mean()
    return cells, clear, peak


def row_agreement(cells: pd.DataFrame) -> tuple[float, float] | None:
    """Mean and lowest Spearman correlation between the combinations' rankings (mean row excluded)."""
    rows = cells.iloc[:-1]
    if rows.shape[1] < MIN_ITEMS or len(rows) < 2:
        return None
    correlation = rows.T.corr(method="spearman").to_numpy()
    pairwise = correlation[np.triu_indices(len(correlation), k=1)]
    return float(np.nanmean(pairwise)), float(np.nanmin(pairwise))


def save_csv(table: pd.DataFrame, path: Path) -> None:
    ordered = table.sort_values(["pair", "domain", "sensitivity"], ascending=[True, True, False])
    ordered.to_csv(path, index=False)


def save_heatmap(cells: pd.DataFrame, clear: pd.DataFrame, peak: pd.Series, path: Path, sweep: str) -> None:
    """One heatmap per sweep: a row per combination, normalized within the row, plus the average row."""
    values = cells.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(max(7, 0.6 * values.shape[1] + 5), 0.55 * values.shape[0] + 2.5))
    image = ax.imshow(values, aspect="auto", cmap="viridis", norm=PowerNorm(0.5, vmin=0, vmax=np.nanmax(values)))
    for i, pair in enumerate(cells.index[:-1]):
        for j, item in enumerate(cells.columns):
            if not clear.loc[pair, item]:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="///",
                                       edgecolor="white", linewidth=0))
    ax.axhline(len(cells) - 1.5, color="white", linewidth=3)  # set the average row apart
    ax.set_xticks(range(len(cells.columns)), cells.columns, rotation=0 if sweep == "block" else 45)
    ax.set_yticks(range(len(cells)), [f"{pair}  (peak {peak[pair]:.2g})" for pair in cells.index[:-1]] + [MEAN_ROW])
    ax.set_xlabel("block (blk.N)" if sweep == "block" else "tensor type")
    ax.set_title(f"{sweep.capitalize()} sensitivity, normalized per row, averaged over domains")
    fig.colorbar(image, ax=ax, label="normalized sensitivity (1 = average item in its row)")
    fig.text(0.01, 0.01, "peak = largest raw KL added (nats); hatched = within "
             f"{CLEAR_FACTOR}x uncertainty of zero in most domains", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_plot(table: pd.DataFrame, path: Path, sweep: str) -> None:
    """One panel per background -> target pair, one line per domain, error bars show uncertainty."""
    pairs = ordered_pairs(table)
    columns = min(3, len(pairs))
    fig, axes = plt.subplots(math.ceil(len(pairs) / columns), columns,
                             figsize=(5.5 * columns, 4 * math.ceil(len(pairs) / columns)), squeeze=False)
    for ax, pair in zip(axes.flat, pairs):
        background, target = pair.split("->")
        for domain, rows in table[table["pair"] == pair].groupby("domain"):
            rows = rows.sort_values("changed")
            ax.errorbar(range(len(rows)), rows["sensitivity"], yerr=rows["sens_uncertainty"],
                        marker="o", capsize=2, label=domain)
            labels = rows["changed"]
        ax.set_xticks(range(len(labels)), labels, rotation=0 if sweep == "block" else 45)
        ax.set_title(f"One {sweep} at {target}, rest {background}")
        ax.set_xlabel("block (blk.N)" if sweep == "block" else "tensor type")
        ax.set_ylabel(f"KL added over pure {background} (nats)")
        ax.grid(alpha=0.2)
        ax.legend(title="domain")
    for ax in list(axes.flat)[len(pairs):]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", choices=["block", "tensor", "both"], default="both",
                        help="which sweep to analyze (default: both)")
    parser.add_argument("--experiment", default=EXPERIMENT, metavar="NAME",
                        help=f"results folder to analyze (default: {EXPERIMENT})")
    parser.add_argument("--output-dir", type=Path,
                        help="report destination (default: the experiment's results folder)")
    parser.add_argument("--lines", action="store_true",
                        help="also draw the per-combination line plots (slower)")
    args = parser.parse_args()

    try:
        scope = Config().with_overrides(experiment=args.experiment)
    except ValueError as error:
        parser.error(str(error))

    control_file = scope.results_dir / CONTROL_CSV
    if not control_file.exists():
        parser.error(f"no {CONTROL_CSV} in {scope.results_dir}; run python -m src.sensitivity first")
    controls = pd.read_csv(control_file)
    output_dir = args.output_dir if args.output_dir is not None else scope.results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for sweep in (["block", "tensor"] if args.sweep == "both" else [args.sweep]):
        source = scope.results_dir / CSV_NAMES[sweep]
        if not source.exists():
            if args.sweep != "both":
                parser.error(f"no {CSV_NAMES[sweep]} in {scope.results_dir}; run python -m src.sensitivity first")
            print(f"Skipping the {sweep} sweep: no {CSV_NAMES[sweep]} yet.")
            continue
        try:
            table = load_sweep(source, controls, sweep)
        except ValueError as error:
            parser.error(str(error))
        if table.empty:
            parser.error(f"no rows in {source}")

        print(f"\n===== {sweep} sweep =====")
        for pair in ordered_pairs(table):
            ranking = build_ranking(table, pair)
            print(f"\n{pair}: {len(ranking)} items, most to least sensitive (KL added over pure {pair.split('->')[0]})")
            print(ranking.round(5).to_string(na_rep="-"))

            ratio = spread(ranking)
            if ratio is not None:
                print(f"Most/least sensitive ratio: {ratio:.1f}x")

            correlation = build_correlation(ranking)
            if correlation.empty:
                print(f"Only {len(ranking)} items measured; need {MIN_ITEMS}+ for a rank correlation.")
            else:
                print("Spearman rank correlation between domains (1 = same ranking):")
                print(correlation.round(2).to_string())

            unclear = int((~table.loc[table["pair"] == pair, "clear"]).sum())
            if unclear:
                print(f"Note: {unclear} {sweep}/domain results are within {CLEAR_FACTOR}x uncertainty of zero.")

        cells, clear, peak = build_heatmap(table, sweep)
        print(f"\n{sweep} sweep, average of all {len(cells) - 1} combinations (normalized, 1 = average item):")
        print(cells.loc[MEAN_ROW].sort_values(ascending=False).round(2).to_string())
        agreement = row_agreement(cells)
        if agreement is not None:
            print(f"Rank agreement between combinations (Spearman): mean {agreement[0]:.2f}, lowest {agreement[1]:.2f}")

        csv_file = output_dir / f"{sweep}_sensitivity_summary.csv"
        normalized_file = output_dir / f"{sweep}_sensitivity_normalized.csv"
        heatmap_file = output_dir / f"{sweep}_sensitivity_heatmap.png"
        save_csv(table, csv_file)
        cells.round(4).to_csv(normalized_file, index_label="pair")
        save_heatmap(cells, clear, peak, heatmap_file, sweep)
        print(f"\nSaved: {csv_file}")
        print(f"Saved: {normalized_file}")
        print(f"Saved: {heatmap_file}")
        if args.lines:
            plot_file = output_dir / f"{sweep}_sensitivity.png"
            save_plot(table, plot_file, sweep)
            print(f"Saved: {plot_file}")


if __name__ == "__main__":
    main()
"""Rank the 16 transformer blocks by quantization sensitivity and check whether the ranking holds across domains.
 
Reads block_sensitivity.csv (written by src/sensitivity.py). For each precision level and domain,
a block's sensitivity is the KL it adds over the pure-q8 control:
    KL(one block lowered, rest Q8_0) - KL(pure-q8)
Prints the blocks from most to least sensitive with the most/least ratio, the Spearman rank
correlation between chat, math, and code (1 = same ranking), and a note for any result within
2x its uncertainty of zero.
 
    python -m src.analyze_blocks                                # results/Block_Sensitivity_Differences
    python -m src.analyze_blocks --experiment Block_Sensitivity # another results folder
    python -m src.analyze_blocks --output-dir report            # write the CSV and plot elsewhere
 
Writes block_sensitivity_summary.csv and block_sensitivity.png next to block_sensitivity.csv.
Uncertainties are combined as independent, a simple approximation, since both runs score the same tokens.
"""
 
import argparse
from pathlib import Path
 
import matplotlib
 
matplotlib.use("Agg")  # save figures without a display
import matplotlib.pyplot as plt
import pandas as pd
 
try:
    from src.config import Config
except ModuleNotFoundError:  # running directly: python src/analyze_blocks.py
    from config import Config
 
EXPERIMENT = "Block_Sensitivity_Differences"  # results folder written by src/sensitivity.py
CSV_NAME = "block_sensitivity.csv"
CONTROL = "pure-q8"   # every block model is compared against this run
CLEAR_FACTOR = 2      # a difference counts as clear above this many uncertainties
MIN_BLOCKS = 5        # fewest blocks for a meaningful rank correlation
 
 
def load_blocks(path: Path) -> pd.DataFrame:
    """One row per block model and domain, with the KL it adds over the control."""
    table = pd.read_csv(path)
    control = table[table["run"] == CONTROL].set_index("domain")
    if control.empty:
        raise ValueError(f"no '{CONTROL}' rows in {path}; run the baselines first")
 
    # Baseline rows have block == "all"; keep only the single-block models.
    blocks = table[pd.to_numeric(table["block"], errors="coerce").notna()].copy()
    blocks["block"] = blocks["block"].astype(int)
 
    base = blocks["domain"].map(control["kl_mean"])
    base_uncertainty = blocks["domain"].map(control["kl_uncertainty"])
    blocks["sensitivity"] = blocks["kl_mean"] - base
    blocks["sens_uncertainty"] = (blocks["kl_uncertainty"] ** 2 + base_uncertainty ** 2) ** 0.5
    blocks["clear"] = blocks["sensitivity"] > CLEAR_FACTOR * blocks["sens_uncertainty"]
    return blocks
 
 
def build_ranking(blocks: pd.DataFrame, level: str) -> pd.DataFrame:
    """Blocks from most to least sensitive: one column per domain plus their mean."""
    ranking = blocks[blocks["level"] == level].pivot(index="block", columns="domain", values="sensitivity")
    ranking["mean"] = ranking.mean(axis=1)
    return ranking.sort_values("mean", ascending=False)
 
 
def build_correlation(ranking: pd.DataFrame) -> pd.DataFrame:
    """Spearman rank correlation between domains; empty until enough blocks are measured."""
    domains = ranking.drop(columns="mean")
    if len(domains) < MIN_BLOCKS:
        return pd.DataFrame()
    return domains.corr(method="spearman")
 
 
def spread(ranking: pd.DataFrame) -> float | None:
    """Most/least sensitive ratio; undefined when the least sensitive block is at or below the control."""
    least = ranking["mean"].min()
    return ranking["mean"].max() / least if least > 0 else None
 
 
def save_csv(blocks: pd.DataFrame, path: Path) -> None:
    ordered = blocks.sort_values(["level", "domain", "sensitivity"], ascending=[True, True, False])
    ordered.to_csv(path, index=False)
 
 
def save_plot(blocks: pd.DataFrame, path: Path) -> None:
    """One panel per precision level, one line per domain, error bars show uncertainty."""
    levels = sorted(blocks["level"].unique())
    fig, axes = plt.subplots(1, len(levels), figsize=(5.5 * len(levels), 4), squeeze=False)
    for ax, level in zip(axes[0], levels):
        for domain, rows in blocks[blocks["level"] == level].groupby("domain"):
            rows = rows.sort_values("block")
            ax.errorbar(rows["block"], rows["sensitivity"], yerr=rows["sens_uncertainty"],
                        marker="o", capsize=2, label=domain)
        ax.set_title(f"One block at {level}, rest Q8_0")
        ax.set_xlabel("block (blk.N)")
        ax.set_ylabel(f"KL added over {CONTROL} (nats, lower is better)")
        ax.set_xticks(range(0, 16))
        ax.grid(alpha=0.2)
        ax.legend(title="domain")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
 
 
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", default=EXPERIMENT, metavar="NAME",
                        help=f"results folder to analyze (default: {EXPERIMENT})")
    parser.add_argument("--output-dir", type=Path,
                        help="report destination (default: the experiment's results folder)")
    args = parser.parse_args()
 
    try:
        scope = Config().with_overrides(experiment=args.experiment)
    except ValueError as error:
        parser.error(str(error))
 
    source = scope.results_dir / CSV_NAME
    if not source.exists():
        parser.error(f"no {CSV_NAME} in {scope.results_dir}; run python -m src.sensitivity first")
    try:
        blocks = load_blocks(source)
    except ValueError as error:
        parser.error(str(error))
    if blocks.empty:
        parser.error(f"no per-block rows in {source}")
    output_dir = args.output_dir if args.output_dir is not None else scope.results_dir
 
    print(f"Control in {scope.results_dir}: {CONTROL}")
    for level in sorted(blocks["level"].unique()):
        ranking = build_ranking(blocks, level)
        print(f"\n{level}: {len(ranking)} blocks, most to least sensitive (KL added over {CONTROL})")
        print(ranking.round(5).to_string(na_rep="-"))
 
        ratio = spread(ranking)
        if ratio is not None:
            print(f"Most/least sensitive ratio: {ratio:.1f}x")
 
        correlation = build_correlation(ranking)
        if correlation.empty:
            print(f"Only {len(ranking)} blocks measured; need {MIN_BLOCKS}+ for a rank correlation.")
        else:
            print("Spearman rank correlation between domains (1 = same ranking):")
            print(correlation.round(2).to_string())
 
        unclear = int((~blocks.loc[blocks["level"] == level, "clear"]).sum())
        if unclear:
            print(f"Note: {unclear} block/domain results are within {CLEAR_FACTOR}x uncertainty of zero.")
 
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_file = output_dir / "block_sensitivity_summary.csv"
    plot_file = output_dir / "block_sensitivity.png"
    save_csv(blocks, csv_file)
    save_plot(blocks, plot_file)
    print(f"\nSaved: {csv_file}")
    print(f"Saved: {plot_file}")
 
 
if __name__ == "__main__":
    main()
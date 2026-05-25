import multiprocessing
import re
from enum import StrEnum
from pathlib import Path

import pandas as pd
import seaborn as sns
import tqdm
from matplotlib import pyplot as plt

tight_layout_pad = 0.1

sns.set_theme(
    context="notebook",
    style="ticks",
    palette="deep",
    font_scale=1.6,
    rc={
        "lines.linewidth": 2.5,
        "axes.linewidth": 2.0,
        "axes.grid.axis": "y",
        "axes.grid": True,
        "grid.color": "0.8",
    },
)

print("Using Font:", plt.rcParams["font.family"])
print("Sans-serif fonts:", plt.rcParams["font.sans-serif"])


class PlotType(StrEnum):
    decode_logit = "decode_logit"
    decode_count = "decode_count"
    prefill_bucket_size = "prefill_bucket_size"
    all = "all"


def plot_decode_threshold_logit_vs_step(df: pd.DataFrame, output_img_file: Path):
    """
    Columns of df:
    - req_id (same for all rows)
    - layer_id (same for all rows)
    - step (int)
    - estimator (str)
    - num_logits_above_estimate (int)
    - offset_value (float) negated value of `topk_threshold_estimate`.
    - topk_threshold_estimate (float) the estimated threshold logit for top-k selection.
    """
    layer_id = df["layer_id"].iloc[0]

    smoothing_factors_to_keep = [
        0.5,
    ]

    def estimator_filter(estimator: str) -> bool:
        if estimator.startswith("Ground"):
            return True
        if estimator.startswith("Constant"):
            return False
        if estimator.startswith("EMA"):
            return any(f"{sf:.2f}" in estimator for sf in smoothing_factors_to_keep)
        return False

    def estimator_transform(estimator: str) -> str:
        if estimator.startswith("Ground"):
            return "Ground Truth Top-K Score"
        if estimator.startswith("EMA"):
            match = re.search(r"EMA\(smoothing_factor=([0-9.]+)", estimator)
            if not match:
                return estimator
            smoothing_factor = float(match.group(1))
            if smoothing_factor != 1.0:
                return f"EMA (α={smoothing_factor:.1f})"
            else:
                return "Last Step"

        return "?Unknown Estimator"

    color_mapping = {
        "Ground Truth Top-K Score": "C0",
        "EMA (α=0.5)": "C1",
    }

    df = df[df["estimator"].apply(estimator_filter)].copy()
    df["estimator"] = df["estimator"].apply(estimator_transform)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.lineplot(
        data=df,
        x="step",
        y="topk_threshold_estimate",
        hue="estimator",
        ax=ax,
        palette=color_mapping,
    )
    ax.set_title(f"Layer {layer_id}")
    ax.set_xlabel("Decoding Step")
    ax.set_ylabel(None)
    ax.margins(x=0.06)
    legend_loc = "lower right" if layer_id == 3 else "upper right"
    ax.legend(loc=legend_loc)
    plt.tight_layout(pad=tight_layout_pad)

    fig.savefig(output_img_file)
    plt.close(fig)


def batch_decode_threshold_logit_vs_step_worker(args):
    file, output_dir, format = args
    df = pd.read_csv(file)
    output_filename = output_dir / f"threshold_logit_vs_step_{file.stem}.{format}"
    plot_decode_threshold_logit_vs_step(df, output_filename)


def batch_decode_threshold_logit_vs_step(files: list[Path], output_dir: Path, format: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    args_list = [(file, output_dir, format) for file in files]
    with multiprocessing.Pool() as pool:
        _ = list(
            tqdm.tqdm(pool.imap_unordered(batch_decode_threshold_logit_vs_step_worker, args_list), total=len(files))
        )


def plot_decode_above_threshold_count_vs_step(df: pd.DataFrame, output_img_file: Path):
    """
    Columns of df:
    - req_id (same for all rows)
    - layer_id (same for all rows)
    - step (int)
    - estimator (str)
    - num_logits_above_estimate (int)
    - offset_value (float) negated value of `topk_threshold_estimate`.
    - topk_threshold_estimate (float) the estimated threshold logit for top-k selection.
    """
    layer_id = df["layer_id"].iloc[0]

    smoothing_factors_to_keep = [0.5]

    def estimator_filter(estimator: str) -> bool:
        if estimator.startswith("Ground"):
            return True
        if estimator.startswith("Constant"):
            return True
        if estimator.startswith("EMA"):
            return any(f"{sf:.2f}" in estimator for sf in smoothing_factors_to_keep)
        return False

    def estimator_transform(estimator: str) -> str:
        if estimator.startswith("Ground"):
            return "Ground Truth"
        if estimator.startswith("Constant"):
            return "Zero"
        if estimator.startswith("EMA"):
            match = re.search(r"EMA\(smoothing_factor=([0-9.]+)", estimator)
            if not match:
                return estimator
            smoothing_factor = float(match.group(1))
            return f"EMA (α={smoothing_factor:.1f})"

        return "?Unknown Estimator"

    color_mapping = {
        "Ground Truth": "#868686",
        "Zero": "C0",
        "EMA (α=0.5)": "C1",
    }

    df = df[df["estimator"].apply(estimator_filter)].copy()
    df = df.assign(estimator=df["estimator"].apply(estimator_transform))

    fig, ax = plt.subplots(figsize=(14, 4))
    sns.lineplot(
        data=df,
        x="step",
        y="num_logits_above_estimate",
        hue="estimator",
        marker="o",
        ax=ax,
        palette=color_mapping,
    )
    ax.set_title(f"Layer {layer_id}")
    ax.set_xlabel("Decoding Step")
    ax.set_ylabel(None)
    ax.legend(loc="upper left")
    plt.tight_layout(pad=tight_layout_pad)

    fig.savefig(output_img_file)
    plt.close(fig)


def batch_decode_above_threshold_count_vs_step_worker(args):
    file, output_dir, format = args
    df = pd.read_csv(file)
    output_filename = output_dir / f"above_threshold_count_vs_step_{file.stem}.{format}"
    plot_decode_above_threshold_count_vs_step(df, output_filename)


def batch_decode_above_threshold_count_vs_step(files: list[Path], output_dir: Path, format: str):
    args_list = [(file, output_dir, format) for file in files]
    with multiprocessing.Pool() as pool:
        _ = list(
            tqdm.tqdm(
                pool.imap_unordered(batch_decode_above_threshold_count_vs_step_worker, args_list), total=len(files)
            )
        )


def plot_prefill_threshold_bucket_size_vs_q(df: pd.DataFrame, output_img_file: Path):
    """
    Columns of df:
    - req_id (same for all rows)
    - layer_id (same for all rows)
    - step (int)
    - q_world (int)
    - estimator (str)
    - threshold_bucket_size (int)
    - offset_value (float)
    """
    layer_id = df["layer_id"].iloc[0]
    min_step = df["step"].min()

    smoothing_factors_to_keep = [0.5]

    def estimator_filter(estimator: str) -> bool:
        if estimator.startswith("Constant"):
            return True
        if estimator.startswith("TailOfPreviousChunk"):
            return any(f"{sf:.2f}" in estimator for sf in smoothing_factors_to_keep)
        return False

    def estimator_transform(estimator: str) -> str:
        if estimator.startswith("Constant"):
            return "No Shift"
        if estimator.startswith("TailOfPreviousChunk"):
            match = re.search(r"TailOfPreviousChunk\(EMA\(smoothing_factor=([0-9.]+)", estimator)
            if not match:
                return "?" + estimator
            smoothing_factor = float(match.group(1))
            return f"EMA (α={smoothing_factor:.1f})"

        return "?Unknown Estimator"

    color_mapping = {
        "No Shift": "C0",
        "EMA (α=0.5)": "C1",
    }

    df["step"] = df["step"] - min_step + 1

    df = df[df["estimator"].apply(estimator_filter)].copy()
    df = df.assign(estimator=df["estimator"].apply(estimator_transform))

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.boxplot(
        data=df,
        x="step",
        y="threshold_bucket_size",
        hue="estimator",
        ax=ax,
        showfliers=False,
        gap=0.1,
        palette=color_mapping,
        saturation=0.9,
        linewidth=2.0,
    )

    ax.set_title(f"Layer {layer_id}")
    ax.set_xlabel("Prefill Chunk ID")
    ax.set_ylabel("Threshold Bin Size")
    ax.legend(loc="upper left")
    plt.tight_layout(pad=tight_layout_pad)

    fig.savefig(output_img_file)
    plt.close(fig)


def batch_prefill_threshold_bucket_size_vs_q_worker(args):
    file, output_dir, format = args
    df = pd.read_csv(file)
    output_filename = output_dir / f"prefill_threshold_bucket_size_vs_q_{file.stem}.{format}"
    plot_prefill_threshold_bucket_size_vs_q(df, output_filename)


def batch_prefill_threshold_bucket_size_vs_q(files: list[Path], output_dir: Path, format: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    args_list = [(file, output_dir, format) for file in files]
    with multiprocessing.Pool() as pool:
        _ = list(
            tqdm.tqdm(pool.imap_unordered(batch_prefill_threshold_bucket_size_vs_q_worker, args_list), total=len(files))
        )


def main(args):
    _ = args.plot

    results_dir: Path = args.results_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    decode_csvs = list(results_dir.glob("decode*.csv"))
    prefill_csvs = list(results_dir.glob("prefill*.csv"))

    match PlotType(args.plot):
        case PlotType.all:
            batch_decode_threshold_logit_vs_step(decode_csvs, output_dir / "decode_logit", args.format)
            batch_decode_above_threshold_count_vs_step(decode_csvs, output_dir / "decode_count", args.format)
            batch_prefill_threshold_bucket_size_vs_q(prefill_csvs, output_dir / "prefill_bucket_size", args.format)

        case PlotType.decode_logit:
            batch_decode_threshold_logit_vs_step(decode_csvs, output_dir, args.format)
        case PlotType.decode_count:
            batch_decode_above_threshold_count_vs_step(decode_csvs, output_dir, args.format)
        case PlotType.prefill_bucket_size:
            batch_prefill_threshold_bucket_size_vs_q(prefill_csvs, output_dir, args.format)
        case _:
            raise ValueError(f"Unknown plot type: {args.plot}")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze full request performance data from a result directory.")
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Path to the results directory containing `decode_req{}_layer{}.csv` files.",
    )
    parser.add_argument(
        "plot",
        type=str,
        help="which plot to generate",
    )
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("./"))
    parser.add_argument("--format", "-f", type=str, default="png", help="Output image format", choices=["png", "pdf"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)

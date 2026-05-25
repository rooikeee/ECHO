import concurrent.futures
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from top_k.data import DataFile, DataLoader, _load_tensor_from_file_cached


def chunked_prefill_top_right_mask(q_len: int, seq_len: int) -> torch.Tensor:
    """
    Creates a 2D mask of shape (q_len, seq_len) that masks out the top-right triangle corner.
    Masked (True) region is where j > i + (seq_len - q_len).
    """

    diag_offset = seq_len - q_len

    # Align the causal diagonal for chunked prefill, where q_len <= seq_len.
    lower_mask = torch.tril(
        torch.ones(q_len, seq_len, dtype=torch.bool, device="cuda"),
        diagonal=diag_offset,
    )
    mask = ~lower_mask

    return mask


def process_data_file(data_file: DataFile) -> dict:
    """
    Read file, compute mean and stddev, return metadata dict.
    """
    info = data_file.info
    file = data_file.path
    tensor = _load_tensor_from_file_cached(file)
    tensor = tensor.to("cuda")
    assert tensor.ndim == 2, f"Expected 2D tensor, got {tensor.ndim}D tensor for file {file}"

    if tensor.shape[0] > 64:
        q_len, seq_len = tensor.shape
        mask = chunked_prefill_top_right_mask(q_len, seq_len)
        tensor[mask] = -float("inf")
        bottom_mask = ~mask
        assert bottom_mask[-1, -1].item(), "Mask bottom-right corner should be True"
        assert not bottom_mask[0, -1].item(), "Mask top-right corner should be False"
        assert bottom_mask[0, seq_len - q_len + 1 :].sum() == 0, "Mask top-right corner row check failed"

        mean = tensor[bottom_mask].mean().item()
        stddev = tensor[bottom_mask].std().item()
    else:
        mean = tensor.mean().item()
        stddev = tensor.std().item()

    return {
        "req_id": info.req_id,
        "layer_id": info.layer_id,
        "step": info.step,
        "file_path": str(file),
        "dim0": tensor.shape[0],
        "dim1": tensor.shape[1],
        "mean": mean,
        "stddev": stddev,
    }


def analyze_stages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'stage' column to the dataframe indicating 'Ignore', 'Prefill', or 'Decode'.
    """

    def get_stage_series(group):
        stages = []
        state = "Ignore"
        group = group.sort_values("step")
        for _, row in group.iterrows():
            dim0 = row["dim0"]

            if state == "Ignore":
                if dim0 > 32:
                    state = "Prefill"
            elif state == "Prefill":
                if dim0 == 1:
                    state = "Decode"

            stages.append(state)
        return pd.Series(stages, index=group.index)

    if not df.empty:
        df["stage"] = df.groupby(["req_id", "layer_id"], group_keys=False).apply(get_stage_series, include_groups=False)
    else:
        df["stage"] = []

    return df


def make_dataframe(dataloader: DataLoader, args) -> pd.DataFrame:
    all_data_files = []
    for request in dataloader.get_requests():
        all_data_files.extend(request.data_files)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        data = list(
            tqdm(
                executor.map(process_data_file, all_data_files),
                total=len(all_data_files),
            )
        )

    df = pd.DataFrame(data)
    df = df.sort_values(by=["req_id", "layer_id", "step"]).reset_index(drop=True)

    df = analyze_stages(df)
    return df


def make_catalog(dataloader: DataLoader, output_path: Path, args) -> None:
    """
    Outputs a CSV catalog of all data files in the dataloader.

    Each row contains:
    - req_id
    - layer_id
    - step
    - file_path
    - shape of the tensor (in two columns: dim0, dim1). will assert all tensors have 2 dims
    """
    df = make_dataframe(dataloader, args)
    df.to_csv(output_path, index=False)
    print(f"Catalog saved to {output_path}")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Make a catalog CSV from a dataloader.")
    parser.add_argument(
        "data_root",
        type=Path,
        help="Root directory of the data files.",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        type=Path,
        required=True,
        help="Path to save the output CSV catalog.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker threads for processing data files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_root: Path = args.data_root
    output_path: Path = args.output_path

    if output_path.exists():
        print(f"Output path {output_path} already exists.")
        return

    assert args.workers >= 1, "Number of workers must be at least 1."

    dataloader = DataLoader(data_root)
    make_catalog(dataloader, output_path, args)


if __name__ == "__main__":
    main()

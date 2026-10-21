# OSDI'26 Artifact Evaluation for *ECHO: Efficient KV Cache Offloading with Lossless Prefetching for Serving Native Sparse Attention LLMs*

## Hardware Requirements

All tests are run on a server with:

- 8 x NVIDIA H20 GPUs (96 GB HBM each)
- 1.5 TB CPU memory
- GPUs connected to the host via PCIe Gen5

## Preparation

This docker image already contains:

- The current repo (ECHO) at `/workspace/ECHO`, including `sglang/` and `DeepGEMM/`
- A pre-built environment that can run the source code (both `sglang` and `DeepGEMM` have been built, installed and tested)
- The InfiniteBench test dataset:
  - All requests: `/workspace/infini-100K-110K.jsonl`
  - Per-task splits: `/workspace/infinibench_task/` (`code_debug.jsonl`, `code_run.jsonl`, `longbook_choice_eng.jsonl`, `longbook_qa_eng.jsonl`)

### Download the model

Before running tests, download the model (DeepSeek-V3.2 with W4 AWQ):

```bash
modelscope download --local_dir /data1/dsv32awq QuantTrio/DeepSeek-V3.2-AWQ
```

We use `/data1/dsv32awq` as the default directory. If you want to change it, export the environment variable `DSV32W4` to the desired directory:

```bash
export DSV32W4=/path/to/DeepSeek-V3.2-AWQ
```

### Activate the prepared environment

```bash
source /workspace/ECHO/.venv/bin/activate
```

Now you can run the following tests.

### Optional: re-build the project

1. Re-build `sglang`:
   ```bash
   cd /workspace/ECHO/sglang
   uv pip install -e "python"
   ```
2. Re-build `sgl-kernel`:
   ```bash
   cd /workspace/ECHO/sglang/sgl-kernel
   make build && uv pip install dist/*whl --force-reinstall --no-deps
   ```
3. Re-build `DeepGEMM`:
   ```bash
   cd /workspace/ECHO/DeepGEMM
   ./install.sh && uv pip install dist/*whl --force-reinstall --no-deps
   ```

### Optional: tune the fused MoE kernel of SGLang

On a new machine, `sglang` may need re-tuning of the fused MoE kernel:

```bash
cd /workspace/ECHO/sglang
python benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py \
    --model "$DSV32W4" --tp-size 8 --dtype auto --tune
```

## Baselines under evaluation

Throughout this README we use the following two labels to distinguish the systems being compared:

| Label | System | Description |
| --- | --- | --- |
| `no-offload` | **SGLang (baseline)** | Vanilla SGLang, all KV cache kept on GPU HBM. |
| `offload` | **ECHO (ours)** | SGLang + ECHO, KV cache offloaded to CPU (`NSA_KV_OFFLOAD=1`), with a bounded GPU-side NSA cache of size `NSA_DEV_CACHE_SIZE`. |


## End-to-End Generation Throughput

> All commands in this section should be executed under `/workspace/ECHO/sglang`.

For the generation throughput results in Figures 12–13, use `e2e_test_pd_infini.sh`.

The script boots a Prefill / Decode (PD) disaggregated SGLang deployment on TP=8, DP=8 (with DP attention) and runs `python/sglang/bench_serving.py` against the InfiniteBench 100K–110K dataset (`/workspace/infini-100K-110K.jsonl`, 318 prompts).

It runs two back-to-back configurations on the same deployment topology:

1. **SGLang baseline (`no-offload`)** — vanilla SGLang, KV cache kept on GPU.
2. **ECHO (`offload`)** — KV cache offloaded to CPU (`NSA_KV_OFFLOAD=1`), GPU-side NSA cache of size `NSA_DEV_CACHE_SIZE`.

### Figure 12 — all GPU HBM allocated to KV cache

```bash
bash e2e_test_pd_infini.sh
```

Results are saved in:

- SGLang baseline: `infini-318/no-offload-<MMDD>/`
- ECHO: `infini-318/offload-<MMDD>/`

Paths are relative to the current working directory, and each directory contains one `req_rate_<RATE>.jsonl` + `req_rate_<RATE>.log` per request rate.

### Figure 13 — constrained GPU pool size

The size of the GPU-side pool can be changed via `--no-offload-max-total-tokens` (SGLang baseline) and `--offload-nsa-dev-cache-size` (ECHO):

```bash
bash e2e_test_pd_infini.sh --no-offload-max-total-tokens 200_000 --offload-nsa-dev-cache-size 200_000
# or
bash e2e_test_pd_infini.sh --no-offload-max-total-tokens 110_000 --offload-nsa-dev-cache-size 110_000
```

### Reducing the tested request rates

By default the script sweeps the request rates `0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, inf`, which is time-consuming. For a quicker test:

```bash
bash e2e_test_pd_infini.sh --request-rates 0.8,0.9,inf
```

### All options of `e2e_test_pd_infini.sh`

| Option | Applies to | Description |
| --- | --- | --- |
| `--no-offload-max-total-tokens VALUE` | SGLang baseline | Pass `VALUE` to the baseline decode server as `--max-total-tokens`. |
| `--offload-nsa-dev-cache-size VALUE` | ECHO | `NSA_DEV_CACHE_SIZE` for the ECHO decode server (default `260_000`). |
| `--no-offload-res-dir VALUE` | SGLang baseline | Output directory (default `infini-318/no-offload-<MMDD>`). |
| `--offload-res-dir VALUE` | ECHO | Output directory (default `infini-318/offload-<MMDD>`). |
| `--request-rates VALUE` | both | Comma-separated request rates to sweep (e.g. `0.1,0.5,inf`). |

> Note: ECHO's prefetch is disabled by default. To enable it, export
> `SGLANG_NSA_FUSE_LOGITS_RECALL_DECODE=1` and
> `SGLANG_NSA_FUSE_LOGITS_RECALL_EXTEND=1` before launching the script.

### Per-task throughput (Figure 14)

To produce the per-task throughput results in Figure 14, use `e2e_test_pd_infini_task.sh`:

```bash
bash e2e_test_pd_infini_task.sh
```

This iterates over every `.jsonl` file in `$TASK_DATASET_DIR` (`/workspace/infinibench_task` by default) and runs the same SGLang-baseline-then-ECHO sweep for each task.

For a quicker test:

```bash
bash e2e_test_pd_infini_task.sh --request-rates 0.8,0.9,inf
```

Results are saved under:

- SGLang baseline: `infini-task/no-offload-<MMDD>/<task_name>/`
- ECHO: `infini-task/offload-<MMDD>/<task_name>/`

Options:

| Option | Applies to | Description |
| --- | --- | --- |
| `--request-rates VALUE` | both | Comma-separated request rates to sweep. |
| `--no-offload-res-dir VALUE` | SGLang baseline | Output directory. |
| `--offload-res-dir VALUE` | ECHO | Output directory. |

## End-to-End Latency

> All commands in this section should be executed under `/workspace/ECHO/sglang`.

For the ShareGPT latency results in Figure 15, use `e2e_test_mix_sharegpt.sh`:

```bash
bash e2e_test_mix_sharegpt.sh
```

The script launches a non-disaggregated SGLang server (TP=8, DP=8, DP attention, `moe_wna16` quant) and benchmarks the `sharegpt` dataset in the same two configurations (SGLang baseline, then ECHO).

For a quicker test with fewer request rates:

```bash
bash e2e_test_mix_sharegpt.sh --request-rates 0.8,0.9,inf
```

Results are saved in:

- SGLang baseline: `sharegpt-100/no-offload-<MMDD>/`
- ECHO: `sharegpt-100/offload-<MMDD>/`

Each directory contains one `req_rate_<RATE>.jsonl` + `req_rate_<RATE>.log` per request rate.

Options:

| Option | Applies to | Description |
| --- | --- | --- |
| `--request-rates VALUE` | both | Comma-separated request rates (default `inf,0.9,...,0.1`). |
| `--num-prompts VALUE` | both | Number of ShareGPT prompts per run (default `100`). |
| `--warmup-requests VALUE` | both | Number of warm-up requests per run (default `0`). |
| `--no-offload-res-dir VALUE` | SGLang baseline | Output directory. |
| `--offload-res-dir VALUE` | ECHO | Output directory. |
| `--offload-nsa-dev-cache-size VALUE` | ECHO | `NSA_DEV_CACHE_SIZE` (default `200_000`). |
| `--offload-max-total-tokens VALUE` | ECHO | `--max-total-tokens` (default `1_000_000`). |

## Fused Prefetch Kernels (ECHO only)

> All commands in this section should be executed under `/workspace/ECHO/DeepGEMM`.

These microbenchmarks evaluate ECHO's fused prefetch kernels in isolation; there is no corresponding SGLang baseline here.

The results for **intra-query** prefetch in Figure 18 can be produced with:

```bash
python test_fuse.py decode_prefetch
```

The results for **inter-query** prefetch in Figure 19 can be produced with:

```bash
python test_fuse.py prefill_prefetch
```

## Troubleshooting

- If a server fails to start, inspect the corresponding log in the working directory:
  - SGLang baseline: `prefill_no_offload.log`, `decode_no_offload.log`, `router_no_offload.log`, `mix_sharegpt_no_offload.log`.
  - ECHO: `prefill.log`, `decode.log`, `router.log`, `mix_sharegpt_offload.log`.
- The launch scripts call `scripts/killall_sglang.sh` between the SGLang baseline and ECHO phases. If a previous run left processes alive, run it manually before retrying:
  ```bash
  bash /workspace/ECHO/sglang/scripts/killall_sglang.sh
  ```
- If GPU memory is not released between runs, use the helper script:
  ```bash
  bash /workspace/ECHO/sglang/scripts/ensure_vram_clear.sh
  ```

## Figure 7,8,9 — Top-K Figure Artifacts

see `top-k-figures/README.md` for instructions to regenerate the artifacts in Figures 7–9.

# Top-K Figure Artifact Pipeline

This repository regenerates the top-k figure artifacts from saved logits tensors.

## Setup

Pull LFS data files:

```bash
git lfs pull
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

Run the full local pipeline:

```bash
./run_pipeline.sh
```

In `data/` we provide the logits tensors captured on deepseek v3.2 for layer 3 and layer 50 and used for analysis and plotting in Figures 7–9.

By default this script uses `data/`. To run on a different logits directory with the
same layout, override `DATA_DIR`:

```bash
DATA_DIR=/path/to/logits ./run_pipeline.sh
```

Expected input layout:

```text
<DATA_DIR>/
  <req_id>/layer<layer_id>_step<step>.pt
```

Each `.pt` file contains a 2D tensor shaped `(q_len, seq_len)`.

## Outputs

`run_pipeline.sh` removes existing generated artifacts under `echo/`, then
regenerates:

```text
# catalog and intermediate results
echo/catalog.csv
echo/results/decode_req*_layer*.csv
echo/results/prefill_req*_layer*.csv
# final figures
echo/pdfs/decode_logit/*.pdf
echo/pdfs/decode_count/*.pdf
echo/pdfs/prefill_bucket_size/*.pdf
# also regenerates PNGs for quick viewing
echo/figs/*/*.png
```


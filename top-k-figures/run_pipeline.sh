#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data}"

mkdir -p echo
rm -rf echo/catalog.csv echo/results echo/pdfs echo/figs

python make_catalog.py "$DATA_DIR" -o echo/catalog.csv
python full_request_pd.py --sequential --output-dir echo/results echo/catalog.csv

bash decode_logits.sh
bash decode_count.sh
bash prefill_bucket_size.sh

RESULTS=echo/results
OUTPUT_PDF=echo/pdfs/decode_logit
OUTPUT_PNG=echo/figs/decode_logit

if [ ! -d "$RESULTS" ]; then
    echo "Missing $RESULTS. Run full_request_pd.py first."
    exit 1
fi

echo "Plot pdf into $OUTPUT_PDF"
python plot_full_request_pd.py $RESULTS decode_logit -o $OUTPUT_PDF -f pdf
echo "Plot png into $OUTPUT_PNG"
python plot_full_request_pd.py $RESULTS decode_logit -o $OUTPUT_PNG -f png

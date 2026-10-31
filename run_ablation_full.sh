#!/bin/bash
# Full-DSL (201 primitives) ablation: full evaluation set, first 300 training tasks (compute budget).
cd "$(dirname "$0")"
export ARC_DSL=full
D=${ARC_DATA:-../ARC-AGI-2/data}
for cfg in guided24_d3 all_d2 random24_d3 guided32_d3; do
  python3 -I run_guided.py $D/evaluation models/proposer_full.pkl ablate_full/evaluation_${cfg}.json $cfg >> ablate_full/log.txt 2>&1
  python3 -I run_guided.py $D/training models/proposer_full.pkl ablate_full/training300_${cfg}.json $cfg 300 >> ablate_full/log.txt 2>&1
done
echo DONE >> ablate_full/log.txt

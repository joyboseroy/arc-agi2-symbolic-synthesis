#!/bin/bash
# Big-DSL ablation: full evaluation set, first 400 training tasks (compute budget).
cd "$(dirname "$0")"
export ARC_DSL=big
D=${ARC_DATA:-../ARC-AGI-2/data}
for cfg in guided24_d3 random24_d3 all_d2 guided32_d3; do
  python3 -I run_guided.py $D/evaluation models/proposer_big.pkl ablate_big/evaluation_${cfg}.json $cfg >> ablate_big/log.txt 2>&1
  python3 -I run_guided.py $D/training models/proposer_big.pkl ablate_big/training400_${cfg}.json $cfg 400 >> ablate_big/log.txt 2>&1
done
echo DONE >> ablate_big/log.txt

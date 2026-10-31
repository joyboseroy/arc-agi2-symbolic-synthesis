#!/bin/bash
# Proposer training-seed robustness: retrain the disjoint proposers with two more MLP seeds, rerun top-24 depth 3.
cd "$(dirname "$0")"
D=${ARC_DATA:-../ARC-AGI-2/data}
for tier in big full; do
  for sd in 1 2; do
    python3 -I train_proposer.py models/synth_${tier}_disjoint.npz models/proposer_${tier}_disjoint_s${sd}.pkl $sd >> v2/log_seeds.txt 2>&1
    ARC_DSL=$tier python3 -I run_v2.py $D/training models/proposer_${tier}_disjoint_s${sd}.pkl v2/${tier}_guided24_d3_disjointseed${sd}_s0.json guided24_d3 subset300.json >> v2/log_seeds.txt 2>&1
  done
done
echo SEEDS_DONE >> v2/log_seeds.txt

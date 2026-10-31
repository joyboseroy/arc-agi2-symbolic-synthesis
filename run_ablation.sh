#!/bin/bash
# Runs every config on the training and evaluation sets, one after another.
cd "$(dirname "$0")"
D=${ARC_DATA:-../ARC-AGI-2/data}
for cfg in all_d2 guided16_d2 guided16_d3 random16_d3 guided24_d3; do
  for split in training evaluation; do
    python3 -I run_guided.py $D/$split models/proposer.pkl ablate/${split}_${cfg}.json $cfg >> ablate/log.txt 2>&1
  done
done
echo DONE >> ablate/log.txt

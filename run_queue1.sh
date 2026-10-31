#!/bin/bash
# Stage 1: task-disjoint synthetic data and proposers (inputs from the 700 training tasks outside the 300-task test subset).
cd "$(dirname "$0")"
D=${ARC_DATA:-../ARC-AGI-2/data}
for tier in big full; do
  export ARC_DSL=$tier
  n=40000; [ $tier = full ] && n=50000
  python3 -I gen_synth.py $D/training models/synth_${tier}_disjoint.npz $n 777 exclude_ids.json >> v2/log1.txt 2>&1
  python3 -I train_proposer.py models/synth_${tier}_disjoint.npz models/proposer_${tier}_disjoint.pkl >> v2/log1.txt 2>&1
done
echo STAGE1_DONE >> v2/log1.txt

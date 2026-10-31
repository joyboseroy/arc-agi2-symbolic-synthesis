#!/bin/bash
# Stage 2: all instrumented runs on the fixed random 300-task training subset.
cd "$(dirname "$0")"
while ! grep -q STAGE1_DONE v2/log1.txt 2>/dev/null; do sleep 30; done
D=${ARC_DATA:-../ARC-AGI-2/data}
run() { # tier model name cfg seed
  tier=$1; model=$2; name=$3; cfg=$4; seed=${5:-0}
  out=v2/${tier}_${cfg}_${name}_s${seed}.json
  [ -f $out ] && return
  ARC_DSL=$tier python3 -I run_v2.py $D/training $model $out $cfg subset300.json $seed >> v2/log2.txt 2>&1
}
# 201-primitive tier first (headline), then 135, then 44
for tier in full big; do
  run $tier none none all_d2
  for k in 24 32; do
    run $tier models/proposer_${tier}.pkl leaky guided${k}_d3
    run $tier models/proposer_${tier}_disjoint.pkl disjoint guided${k}_d3
  done
  for s in 0 1 2 3 4; do run $tier none none random24_d3 $s; done
done
run small none none all_d2
run small none none all_d3
for k in 16 24; do
  run small models/proposer.pkl leaky guided${k}_d3
done
for s in 0 1 2 3 4; do run small none none random16_d3 $s; done
echo STAGE2_DONE >> v2/log2.txt

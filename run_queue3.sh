#!/bin/bash
# Stage 3: diagnostics with the foreground metric and exhaustive depth-3 spot checks on evaluation tasks.
cd "$(dirname "$0")"
while ! grep -q STAGE2_DONE v2/log2.txt 2>/dev/null; do sleep 30; done
D=${ARC_DATA:-../ARC-AGI-2/data}
export ARC_DSL=full
python3 -I diagnose.py $D/training v2/diag_train300.json subset300.json >> v2/log3.txt 2>&1
python3 -I diagnose.py $D/evaluation v2/diag_eval.json >> v2/log3.txt 2>&1
ARC_DSL=small python3 -I run_v2.py $D/evaluation none v2/eval_small_all_d3.json all_d3 eval_sample12.json >> v2/log3.txt 2>&1
ARC_DSL=full python3 -I run_v2.py $D/evaluation none v2/eval_full_all_d3.json all_d3 eval_sample2.json >> v2/log3.txt 2>&1
echo STAGE3_DONE >> v2/log3.txt

#!/bin/bash
cd "$(dirname "$0")"
export ARC_DSL=full
D=${ARC_DATA:-../ARC-AGI-2/data}
python3 -I depthsearch.py $D/evaluation v2/d3_eval16.json eval_d3_16.json 3 > v2/d3_eval16.log 2>&1
python3 -I depthsearch.py $D/training v2/d3_train16.json train_d3_16.json 3 > v2/d3_train16.log 2>&1
echo DONE > v2/d3_done.flag

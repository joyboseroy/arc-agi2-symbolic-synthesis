#!/bin/bash
cd "$(dirname "$0")"
D=${ARC_DATA:-../ARC-AGI-2/data}
python3 -I run_orel.py $D/training v2/orel_sample300.json subset300.json > v2/orel_final.log 2>&1
python3 -I run_orel.py $D/evaluation v2/orel_eval120.json all >> v2/orel_final.log 2>&1
echo DONE >> v2/orel_final.log

D=${ARC_DATA:-../ARC-AGI-2/data}
for m in canon aug; do
  MODES=$m CAP=30 python3 -I run_celltree5.py $D/training v2/cell5${m}_sample300.json subset300.json > v2/cell5${m}_sample.log 2>&1
  MODES=$m CAP=30 python3 -I run_celltree5.py $D/evaluation v2/cell5${m}_eval120.json all > v2/cell5${m}_eval.log 2>&1
done
echo done > v2/cell5_final.done

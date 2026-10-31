"""Resumable wrapper around depthsearch.work: appends one JSON line per finished task."""
import json, os, sys
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import depthsearch as ds
T, OUT, IDS, DEPTH = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
done = set()
if os.path.exists(OUT):
    done = {json.loads(l)["id"] for l in open(OUT) if l.strip()}
ids = [i for i in json.load(open(IDS)) if i not in done]
rest = len(ids)
ids = ids[:int(os.environ.get("CHUNK", "100000"))]
print("done before:", len(done), "this chunk:", len(ids), "left after:", rest - len(ids), flush=True)
if __name__ == "__main__":
    with Pool(2) as p, open(OUT, "a") as f:
        for r in p.imap_unordered(ds.work, [(os.path.join(T, i + ".json"), DEPTH) for i in ids]):
            f.write(json.dumps(r) + "\n"); f.flush()

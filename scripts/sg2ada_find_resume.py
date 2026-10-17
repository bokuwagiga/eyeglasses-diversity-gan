"""Print the train.py arguments needed to resume the baseline from its latest snapshot.

stylegan3's --resume restarts the kimg counter at 0, so the remaining budget is passed as
--kimg and the run description records the absolute kimg offset ("from<K>"). A snapshot
network-snapshot-NNNNNN.pkl inside a run dir whose name contains "fromK" is at absolute
kimg K+NNNNNN. Usage: python sg2ada_find_resume.py <runs_root> [total_kimg]
"""
import glob, os, re, sys
root = sys.argv[1]; total = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
best = None
for pkl in glob.glob(os.path.join(root, "*", "network-snapshot-*.pkl")):
    run = os.path.basename(os.path.dirname(pkl))
    m = re.search(r"from(\d+)", run); offset = int(m.group(1)) if m else 0
    n = int(re.search(r"network-snapshot-(\d+)\.pkl", pkl).group(1))
    key = (offset + n, os.path.getmtime(pkl))
    if best is None or key > best[0]:
        best = (key, pkl)
if best is None:
    sys.exit("no network-snapshot-*.pkl found under " + root)
reached, pkl = best[0][0], best[1]
remaining = total - reached
if remaining <= 0:
    sys.exit(f"latest snapshot {pkl} is already at {reached} kimg >= {total}; nothing to resume")
print(f"--resume={pkl.replace(os.sep, '/')} --kimg={remaining} --desc=baseline-from{reached}")

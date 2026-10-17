"""Collect the StyleGAN2-ADA baseline deliverables into to_copy/ with checksums.

Adds to the existing to_copy/ tree (repo layout under to_copy/results/ for both
baselines, the training-run folders under to_copy/sg2ada_runs/, the baseline scripts) and rewrites SHA256SUMS.txt
(sha256 + byte size) over everything currently in to_copy/. Generated image
sets are never collected, only reports, configs, logs and the visual-check
images. Every copy is verified against its source.

Usage: python scripts/collect_baseline_to_copy.py
"""
import hashlib, os, shutil, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(REPO, "to_copy")
RUNS = os.path.join(os.path.dirname(REPO), "sg2ada_runs")
BASES = [os.path.join("results", "baseline_sg2ada"), os.path.join("results", "baseline_sg2ada_gamma10")]
SCRIPTS = ["run_sg2ada_baseline.bat", "resume_sg2ada_baseline.bat", "sg2ada_find_resume.py", "prep_sg2ada_data.py",
           "crop_sg2ada_generated.py", "phase3_eval_sg2ada_snap001600.bat", "collect_baseline_to_copy.py",
           "run_sg2ada_gamma10.ps1", "phase3_eval_sg2ada_gamma10_snap001800.bat", "sg2ada_contact_sheet.py",
           "build_baseline_results_table.py"]

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

def want(rel):
    parts = rel.split(os.sep)
    if "generated" in parts:                      # never the 10k image set
        return False
    if "visual_check" in parts:                   # contact sheets + the 3x24 images
        return rel.endswith(".png")
    if "novel_examples" in parts:                 # copies of nearest-neighbour pairs: keep, small
        return rel.endswith(".png")
    return rel.endswith((".json", ".txt", ".jsonl", ".log", ".csv", ".diff", ".png"))

items = []  # (absolute source, destination relative to to_copy)
for base in BASES:
    for root, _, files in os.walk(os.path.join(REPO, base)):
        for f in files:
            src = os.path.join(root, f); rel = os.path.relpath(src, REPO)
            if want(rel):
                items.append((src, rel))
for f in SCRIPTS:
    items.append((os.path.join(REPO, "scripts", f), os.path.join("scripts", f)))
for run in sorted(os.listdir(RUNS)) if os.path.isdir(RUNS) else []:
    rd = os.path.join(RUNS, run)
    if not os.path.isdir(rd):
        continue
    for f in ("training_options.json", "log.txt", "metric-fid50k_full.jsonl", "stats.jsonl", "stop_summary.txt"):
        p = os.path.join(rd, f)
        if os.path.isfile(p):
            items.append((p, os.path.join("sg2ada_runs", run, f)))
for f in sorted(os.listdir(RUNS)) if os.path.isdir(RUNS) else []:
    if f.endswith(".log"):
        items.append((os.path.join(RUNS, f), os.path.join("sg2ada_runs", f)))

copied, total = 0, 0
for src, rel in items:
    dst = os.path.join(DEST, rel); os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    if sha256(src) != sha256(dst):
        sys.exit("copy verification failed: " + rel)
    copied += 1; total += os.path.getsize(dst)
lines = []
for root, _, files in os.walk(DEST):
    for f in sorted(files):
        if f == "SHA256SUMS.txt":
            continue
        p = os.path.join(root, f); rel = os.path.relpath(p, DEST).replace(os.sep, "/")
        lines.append(f"{sha256(p)}  {os.path.getsize(p)}  {rel}")
with open(os.path.join(DEST, "SHA256SUMS.txt"), "w", newline="\n") as fh:
    fh.write("\n".join(sorted(lines, key=lambda l: l.split('  ', 2)[2])) + "\n")
print(f"collected {copied} baseline files ({total/1e6:.1f} MB); SHA256SUMS.txt now lists {len(lines)} files")

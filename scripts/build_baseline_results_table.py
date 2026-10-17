"""Print the metric table comparing the StyleGAN2-ADA baselines with sharp1 and mirror2.

Reads diversity_report.json, artifact_report.json and novelty_report.json of each
evaluation folder and prints the table used in RESULTS_SUMMARY.txt (one column per set).

Usage: python scripts/build_baseline_results_table.py
"""
import json, os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETS = [  # (column header, evaluation folder)
    ("gamma3.28",  "results/baseline_sg2ada/snap001600"),
    ("gamma10",    "results/baseline_sg2ada_gamma10/snap001800"),
    ("sharp1",     "results/sharp1/eval1600"),
    ("mirror2",    "results/mirror2/eval"),
]

def load(folder):
    g = lambda *p: json.load(open(os.path.join(REPO, folder, *p)))
    return g("diversity", "diversity_report.json"), g("artifacts_p2p98", "artifact_report.json"), g("novelty", "novelty_report.json")

reports = [load(f) for _, f in SETS]
real = reports[0][0]

def fmt(v):
    if isinstance(v, int):
        return str(v)
    return f"{v:.4g}"

rows = [  # (label, function(div, art, nov) -> value, note)
    ("FID (our suite, 256x512 crops, 5000 vs 5000)", lambda d, a, n: d["quality"]["fid"], "lower better"),
    ("KID mean", lambda d, a, n: d["quality"]["kid_mean"], "lower better"),
    ("PRDC precision (k=5)", lambda d, a, n: d["feature_space"]["prdc"]["precision"], "higher better"),
    ("PRDC recall", lambda d, a, n: d["feature_space"]["prdc"]["recall"], "higher better"),
    ("PRDC density", lambda d, a, n: d["feature_space"]["prdc"]["density"], "~1"),
    ("PRDC coverage", lambda d, a, n: d["feature_space"]["prdc"]["coverage"], "higher better"),
    (f"Vendi (2000, real {real['feature_space']['vendi_real']['vendi']:.4g})", lambda d, a, n: d["feature_space"]["vendi_gen"]["vendi"], "closer to real"),
    (f"LPIPS pairwise mean (real {real['lpips']['real']['mean']:.4g})", lambda d, a, n: d["lpips"]["gen"]["mean"], "closer to real"),
    (f"colour ab_coverage (real {real['color']['real']['ab_coverage']:.4g})", lambda d, a, n: d["color"]["gen"]["ab_coverage"], "closer to real"),
    (f"hue entropy (real {real['color']['real']['hue_entropy']:.4g})", lambda d, a, n: d["color"]["gen"]["hue_entropy"], "closer to real"),
    ("silhouette failures (of 5000)", lambda d, a, n: int(d["geometry"]["failed_gen"]), "lower better"),
    ("geometry JSD area_ratio", lambda d, a, n: d["geometry"]["jsd"]["area_ratio"], "lower better"),
    ("geometry JSD rim_thickness", lambda d, a, n: d["geometry"]["jsd"]["rim_thickness"], "lower better"),
    ("memorisation ratio gen/real NN dist", lambda d, a, n: d["feature_space"]["memorisation"]["ratio_gen_over_real"], "~1, <1 = copying"),
    (f"pose offcentre_frac (real {real['pose']['dispersion']['offcentre_frac_real']:.4g})", lambda d, a, n: d["pose"]["dispersion"]["offcentre_frac_gen"], "closer to real"),
    (f"artifacts: clean fraction (real {reports[0][1]['clean_fraction_real']:.4g})", lambda d, a, n: a["clean_fraction_gen"], "higher better"),
    ("artifacts: >=4 flags count (of 10000)", lambda d, a, n: sum(v for k, v in a["gen_flag_count_hist"].items() if int(k) >= 4), "lower better"),
    ("novelty rate", lambda d, a, n: n["novelty_rate"], "higher = more novel"),
    ("realistic novelty rate", lambda d, a, n: n["realistic_novelty_rate"], "higher better"),
    ("novel but unrealistic rate", lambda d, a, n: n["novel_but_unrealistic_rate"], "lower better"),
    ("coverage vs reference (novelty PRDC)", lambda d, a, n: n["prdc_vs_reference"]["coverage"], "higher better"),
    ("recall vs reference (novelty PRDC)", lambda d, a, n: n["prdc_vs_reference"]["recall"], "higher better"),
]

W = 46
print("metric".ljust(W) + "".join(h.rjust(11) for h, _ in SETS) + "   note")
for label, fn, note in rows:
    vals = [fmt(fn(*r)) for r in reports]
    print(label.ljust(W) + "".join(v.rjust(11) for v in vals) + "   " + note)

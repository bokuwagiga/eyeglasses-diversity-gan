"""Strict realistic novelty: novel, inside the real manifold, and artifact-clean.

Combines the per-image output of metrics/novelty_rate.py (novelty_scores.csv)
with the per-image artifact flags of metrics/diagnose_artifacts.py
(scores_gen.json) for the same generated set.

  python metrics/strict_novelty.py --set name=novelty_dir,artifact_scores.json ... \
      --out results/_probes/novelty_strict_rates.json
"""
import argparse
import csv
import json
import os


def strict_rates(novelty_dir, scores_path):
    art = json.load(open(scores_path))
    rows = list(csv.DictReader(open(os.path.join(novelty_dir,
                                                 'novelty_scores.csv'))))
    n = len(rows)
    novel = [r for r in rows if r['novel'] == '1']
    inside = [r for r in novel if r['inside_real_manifold'] == '1']
    matched = [r for r in inside if r['gen_file'] in art]
    clean = [r for r in matched if art[r['gen_file']]['n_flags'] == 0]
    all_m = [r for r in rows if r['gen_file'] in art]
    all_c = [r for r in all_m if art[r['gen_file']]['n_flags'] == 0]
    return {
        'n_scored': n,
        'artifact_scores_matched_all': len(all_m),
        'artifact_scores_matched_novel_inside': len(matched),
        'novel_inside_without_artifact_score': len(inside) - len(matched),
        'set_clean_rate': round(len(all_c) / max(len(all_m), 1), 4),
        'novelty_rate': round(len(novel) / n, 4),
        'novel_inside_manifold_rate': round(len(inside) / n, 4),
        'strict_novel_rate_zero_flags': round(len(clean) / n, 4),
        'clean_share_of_novel_inside': round(len(clean) / max(len(matched), 1), 4),
        'novelty_dir': novelty_dir,
        'artifact_source': scores_path,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', action='append', required=True,
                    help='name=novelty_dir,scores_gen.json (repeatable)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = {}
    for spec in args.set:
        name, paths = spec.split('=', 1)
        nov, scores = paths.split(',', 1)
        out[name] = strict_rates(nov, scores)
        r = out[name]
        print('%-12s novel %.4f  inside %.4f  strict %.4f  (matched %d/%d)'
              % (name, r['novelty_rate'], r['novel_inside_manifold_rate'],
                 r['strict_novel_rate_zero_flags'],
                 r['artifact_scores_matched_all'], r['n_scored']))
    note = ('Strict realistic novelty: novel (nearest-reference distance > tau) '
            'AND inside the k-NN manifold AND zero flags in the structural '
            'artifact suite (thresholds at real p2/p98). rebalance4 was scored '
            'under the 11-check suite; all other sets under the 12-check suite.')
    with open(args.out, 'w') as f:
        json.dump({'note': note, 'models': out}, f, indent=2)
    print('written', args.out)


if __name__ == '__main__':
    main()

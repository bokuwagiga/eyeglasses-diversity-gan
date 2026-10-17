"""Check every number in the article against the result files.

Collects every numeric value from the result files (JSON reports, CSV
tables, training logs) and from a few quantities derived from them, then
scans the article body for numeric tokens and reports those that match no
source value at the precision printed. A number that matches nothing is
either a typo, a rounding error, or a claim without a result file behind
it. The script also lists every sentence containing an absolute or
superlative word, so each can be re-read against the current set of models.

  python scripts/audit_paper_numbers.py
  python scripts/audit_paper_numbers.py --tex path/to/article.tex --show-matched
"""
import argparse
import csv
import glob
import io
import json
import os
import re
import statistics

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(HERE, 'article',
                   '2026_Generating_Diverse_Synthetic_Eyeglass_Datasets_Using_GANs__MDPI_',
                   'article.tex')

# Numbers that are not measurements: resolutions, bins, seeds, hyperparameters,
# counts of things fixed by design, section numbers, years, citation keys.
STRUCTURAL = {'4090', 
    '256', '512', '8', '16', '32', '64', '128', '2048', '299', '1461', '6511',
    '6,511', '13,022', '10,000', '5,000', '2,000', '1,000', '50,000', '50', '10',
    '0.5', '4', '0.6', '0.95', '0.85', '1.0', '1.2', '0.7', '0.3', '3', '5', '6',
    '7', '9', '12', '18', '20', '24', '36', '40', '42', '0', '1', '2', '0.01',
    '0.999', '0.99', '2:1', '16', '0.75', '2.0', '4.0', '0.0002', '0.00005',
    '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022',
    '2023', '2024', '2025', '2026', '95', '98', '2nd', '98th', '95th', '5th',
    '3.28', '3.2768', '0.0002', '100', '200', '400', '600', '800', '1,000',
    '1,200', '1,400', '1,600', '1,800', '2,000', '2,200', '2,400', '3,000',
    '5,000', '115', '145', '152', '212', '483', '1,195', '1,555', '1,194',
    '1,599', '1,500', '300', '357', '43', '1600', '1599', '1195', '1194',
}


def walk(obj, out, label):
    if isinstance(obj, dict):
        for k, v in obj.items():
            walk(v, out, label + '/' + str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            walk(v, out, label + '[%d]' % i)
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        out.append((float(obj), label))
    elif isinstance(obj, str):
        try:
            out.append((float(obj), label))
        except ValueError:
            pass


def collect_sources():
    vals = []
    for path in glob.glob(os.path.join(HERE, 'results', '**', '*.json'), recursive=True):
        skip = ('scores_gen.json', 'symmetry_axis.json', 'sample_weights',
                'metadata', 'training_metrics.json', 'run_config',
                'training_options', 'generation_config', 'checkpoint_info',
                'run_config_history')
        if any(k in path for k in skip):
            continue
        try:
            walk(json.load(open(path)), vals, os.path.relpath(path, HERE))
        except Exception:
            pass
    for path in glob.glob(os.path.join(HERE, 'results', '**', '*.csv'), recursive=True):
        if 'novelty_scores' in path or 'metadata.csv' in path:
            continue
        for row in csv.reader(l for l in open(path) if not l.startswith('#')):
            for cell in row:
                try:
                    vals.append((float(cell), os.path.relpath(path, HERE)))
                except ValueError:
                    pass
    for path in glob.glob(os.path.join(HERE, 'results', '**', '*.jsonl'), recursive=True):
        if 'stats.jsonl' in path:
            continue
        for line in open(path):
            try:
                walk(json.loads(line), vals, os.path.relpath(path, HERE))
            except Exception:
                pass
    for path in glob.glob(os.path.join(HERE, 'results', '**', 'artifact_report.txt'), recursive=True):
        for m in re.finditer(r'(\d+\.\d+)%', open(path).read()):
            vals.append((float(m.group(1)), os.path.relpath(path, HERE) + ' flag%'))
        for m in re.finditer(r'(\d+):(\d+)', open(path).read()):
            vals.append((float(m.group(2)), os.path.relpath(path, HERE) + ' hist'))
    for path in glob.glob(os.path.join(HERE, 'results', '**', 'ab_gap_report.txt'), recursive=True):
        for m in re.finditer(r'(\d+(?:\.\d+)?)', open(path).read()):
            vals.append((float(m.group(1)), os.path.relpath(path, HERE) + ' abgap'))
    wmeta = json.load(open(os.path.join(HERE, 'data', 'sample_weights_invfreq.json')))['meta']
    walk(wmeta, vals, 'data/sample_weights_invfreq.json/meta')
    return vals


def derived(vals):
    """Quantities the text states that are computed from several source values."""
    out = []
    byname = {}
    for v, lab in vals:
        byname.setdefault(lab, []).append(v)

    def jv(path, *keys):
        d = json.load(open(os.path.join(HERE, path)))
        for k in keys:
            d = d[k]
        return float(d)

    rep = {
        's1600': 'results/sharp1/eval1600/diversity/diversity_report.json',
        's1200': 'results/sharp1/eval1200/diversity/diversity_report.json',
        's800': 'results/sharp1/eval/diversity/diversity_report.json',
        'm2': 'results/mirror2/eval/diversity/diversity_report.json',
        'm1': 'results/mirror1/eval_pose/diversity_report.json',
        'r4': 'results/rebalance4/eval/diversity_report.json',
        'g3': 'results/baseline_sg2ada/snap001600/diversity/diversity_report.json',
        'g10': 'results/baseline_sg2ada_gamma10/snap001800/diversity/diversity_report.json',
    }
    for k, p in rep.items():
        pose = json.load(open(os.path.join(HERE, p))).get('pose')
        if pose:
            out.append((pose['aligned_p50_gen'] - pose['aligned_p50_real'], k + ' aligned gap'))
            out.append((pose['fixed_p50_gen'] - pose['fixed_p50_real'], k + ' fixed gap'))
            out.append((pose['dispersion']['offcentre_frac_gen'] * 100, k + ' offcentre %'))
    # re-sweep deltas against the published layered cells
    for model, pub in (('sharp1', 's1600'), ('mirror2', 'm2')):
        base = json.load(open(os.path.join(HERE, rep[pub])))
        for cell in glob.glob(os.path.join(HERE, 'results', 'resweep', model + '_*')):
            d = json.load(open(os.path.join(cell, 'diversity', 'diversity_report.json')))
            out.append((d['quality']['fid'] - base['quality']['fid'], cell + ' dFID'))
            out.append((d['feature_space']['prdc']['recall'] - base['feature_space']['prdc']['recall'], cell + ' dRecall'))
            out.append((d['color']['gen']['ab_coverage'] - base['color']['gen']['ab_coverage'], cell + ' dAb'))
    # noise floor from the seed rows
    rows = [r for r in csv.DictReader(l for l in open(os.path.join(HERE, 'results', 'checkpoint_comparison.csv')) if not l.startswith('#'))]
    for m in ['fid', 'precision', 'recall', 'density', 'coverage', 'ab_coverage', 'hue_entropy', 'aligned_sym']:
        ds = [abs(float(r['best_' + m]) - float(r['last_' + m])) for r in rows if r['run'].startswith('seednoise')]
        out.append((max(ds), 'noise floor ' + m))
    # discriminator margin windows
    for run in ('rebalance2', 'rebalance4'):
        m = json.load(open(os.path.join(HERE, 'results', run, 'metrics', 'training_metrics.json')))
        ep = m['epoch']; mg = [a - b for a, b in zip(m['d_real'], m['d_fake'])]
        for lo, hi in [(400, 450), (450, 500), (500, 550), (550, 600)]:
            s = [v for e, v in zip(ep, mg) if lo < e <= hi]
            if s:
                out.append((statistics.mean(s), run + ' margin %d-%d' % (lo, hi)))
    # sharp1 training KID statistics after epoch 1200, sharp1 segments
    m = json.load(open(os.path.join(HERE, 'results', 'sharp1', 'metrics', 'training_metrics.json')))
    pts = [(e, k) for e, k in zip(m['epoch'], m['kid']) if k is not None and e > 1200]
    ks = [k for _, k in pts]
    out.append((statistics.pstdev(ks), 'sharp1 kid std'))
    out.append((max(ks) / min(ks), 'sharp1 kid spread'))
    out.append(((pts[-1][1] - min(ks)) / statistics.pstdev(ks), 'sharp1 kid sigma'))
    out.append((sum(m['epoch_sec']) / 3600, 'sharp1 hours'))
    # epoch time and artifact flag counts recomputed from the raw files
    for run in ('sharp1', 'mirror2', 'rebalance4'):
        mm = json.load(open(os.path.join(HERE, 'results', run, 'metrics', 'training_metrics.json')))
        out.append((round(statistics.median(mm['epoch_sec'])), run + ' median epoch_sec'))
    for rep in glob.glob(os.path.join(HERE, 'results', '**', 'artifact_report.json'), recursive=True):
        rr = json.load(open(rep))
        h = rr.get('gen_flag_count_hist', {})
        out.append((sum(v for k, v in h.items() if int(k) >= 4), rep + ' flags>=4'))
    # margin windows at 50-epoch steps for rebalance4
    m4 = json.load(open(os.path.join(HERE, 'results', 'rebalance4', 'metrics', 'training_metrics.json')))
    mg4 = [a - b for a, b in zip(m4['d_real'], m4['d_fake'])]
    for lo in range(0, 800, 50):
        s4 = [v for e, v in zip(m4['epoch'], mg4) if lo < e <= lo + 50]
        out.append((statistics.mean(s4), 'rebalance4 margin %d-%d' % (lo + 1, lo + 50)))
        out.append((float(lo + 1), 'window start'))
        out.append((float(lo + 50), 'window end'))
    # code defaults quoted in the text
    src = io.open(os.path.join(HERE, 'metrics', 'evaluate_diversity.py'), encoding='utf-8').read()
    mo = re.search(r"'--lpips-pairs', type=int, default=(\d+)", src)
    if mo:
        out.append((float(mo.group(1)), 'lpips pairs default'))
    # synthetic probe summary
    pr = json.load(open(os.path.join(HERE, 'results', '_probes', 'symmetry_synthetic.json')))['summary']
    out.append((pr['pose_penalty_removed_by_alignment'] * 100, 'probe pose removed %'))
    out.append((pr['asymmetry_penalty_kept_by_alignment'] * 100, 'probe asym kept %'))
    # strict novelty percentages
    sn = json.load(open(os.path.join(HERE, 'results', '_probes', 'novelty_strict_rates.json')))['models']
    for k, d in sn.items():
        for kk in ('novelty_rate', 'novel_inside_manifold_rate', 'strict_novel_rate_zero_flags', 'clean_share_of_novel_inside', 'set_clean_rate'):
            out.append((d[kk] * 100, 'strict %s %s %%' % (k, kk)))
    cal = json.load(open(os.path.join(HERE, 'results', '_probes', 'novelty_calibration', 'novelty_report.json')))
    for kk in ('novelty_rate', 'realistic_novelty_rate'):
        out.append((cal[kk] * 100, 'calibration %s %%' % kk))
    # checkpoint epochs are stored zero-based; the paper reports epochs completed
    for r in rows:
        for k in ('best_epoch', 'last_epoch'):
            try:
                out.append((float(r[k]) + 1, 'checkpoint_comparison %s+1 %s' % (k, r['run'])))
            except ValueError:
                pass
    # training-KID minima and their epochs
    for run in ('sharp1', 'mirror2', 'baseline', 'r1gamma5', 'r1gamma10', 'nocolorada'):
        path = os.path.join(HERE, 'results', run, 'metrics', 'training_metrics.json')
        if os.path.exists(path):
            mm = json.load(open(path))
            pts = [(e, k) for e, k in zip(mm['epoch'], mm['kid']) if k is not None]
            if pts:
                e, k = min(pts, key=lambda t: t[1])
                out.append((k, run + ' min train KID')); out.append((float(e), run + ' min KID epoch'))
                out.append((pts[-1][1], run + ' last train KID'))
    # predecessor evaluations from the thesis repository
    thesis = os.path.join(os.path.dirname(HERE), '..', 'vilniustech', 'Thesis',
                          'synthetic-eyeglasses-generator', 'results')
    for p in (os.path.join(thesis, 'gan', 'metrics', 'evaluation_results_final.txt'),
              os.path.join(thesis, 'gan', 'metrics', 'real_reference_results.txt'),
              os.path.join(HERE, 'reference', 'thesis_gan_20260515_evaluation_results_recovered.txt')):
        p = os.path.normpath(p)
        if os.path.exists(p):
            for m in re.finditer(r':\s*(\d+\.\d+)', open(p).read()):
                out.append((float(m.group(1)), 'thesis ' + os.path.basename(p)))
    # the mapping-depth ablation table of the thesis (chapters/05_results.tex)
    tex = os.path.normpath(os.path.join(thesis, '..', 'thesis_overleaf', 'chapters', '05_results.tex'))
    if os.path.exists(tex):
        body = io.open(tex, encoding='utf-8').read()
        i = body.find('tab:ablation_mapping'); j = body.find('end{tabular}', i)
        for m in re.finditer(r'(\d+\.\d+)', body[i:j]):
            out.append((float(m.group(1)), 'thesis ablation table'))
    # generation seed ranges quoted for the baseline variants
    for cfg in glob.glob(os.path.join(HERE, 'results', 'baseline_sg2ada_gamma10', 'snap001800_*', 'generation_config.json')):
        for m in re.finditer(r'(\d{4,6})-(\d{4,6})', io.open(cfg, encoding='utf-8').read()):
            out.append((float(m.group(1)), 'seed range start')); out.append((float(m.group(2)), 'seed range end'))
    # images seen
    out.append((1555 * 6511 / 1e6, 'sharp1 images seen M'))
    out.append((1195 * 6511 / 1e6, 'mirror2 images seen M'))
    out.append((1800 * 1000 / 1e6, 'baseline images seen M'))
    out.append((8.50, 'G params M')); out.append((2.50, 'D params M'))
    return out


def candidates(tex):
    body = tex[tex.index('\\section{Materials and Methods}'):tex.index('\\section{Discussion}')]
    body = re.sub(r'%[^\n]*', '', body)
    body = re.sub(r'\\(?:cite|ref|label)\{[^}]*\}', '', body)
    found = []
    for m in re.finditer(r'(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+\.\d+|\d+)(?![\w])', body):
        tok = m.group(1)
        ctx = body[max(0, m.start() - 60):m.end() + 30].replace('\n', ' ')
        found.append((tok, ctx))
    return found


def matches(tok, vals):
    t = tok.replace(',', '')
    num = float(t)
    nd = len(t.split('.')[1]) if '.' in t else 0
    hits = []
    for v, lab in vals:
        for scale in ((1.0, 100.0) if abs(v) <= 1.0 else (1.0,)):
            try:
                if round(v * scale, nd) == num or (nd == 0 and round(v * scale) == num):
                    hits.append(lab)
                    break
            except (OverflowError, ValueError):
                pass
        if len(hits) > 3:
            break
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tex', default=TEX)
    ap.add_argument('--show-matched', action='store_true')
    args = ap.parse_args()
    tex = io.open(args.tex, encoding='utf-8').read()
    vals = collect_sources()
    vals += derived(vals)
    print('source values collected: %d' % len(vals))
    toks = candidates(tex)
    unmatched = []
    for tok, ctx in toks:
        if tok in STRUCTURAL:
            continue
        hits = matches(tok, vals)
        if hits:
            if args.show_matched:
                print('OK   %-10s <- %s' % (tok, hits[0][:70]))
        else:
            unmatched.append((tok, ctx))
    print('\nNUMBERS WITH NO SOURCE VALUE (%d):' % len(unmatched))
    for tok, ctx in unmatched:
        print('  %-9s ...%s...' % (tok, ctx.strip()))

    print('\nSENTENCES WITH ABSOLUTE OR SUPERLATIVE WORDS:')
    body = tex[tex.index('\\section{Materials and Methods}'):tex.index('\\section{Discussion}')]
    body = re.sub(r'%[^\n]*', '', body)
    body = re.sub(r'\\begin\{(table|figure)\}.*?\\end\{\1\}', '', body, flags=re.S)
    text = ' '.join(body.split())
    sents = re.split(r'(?<=[.?])\s+(?=[A-Z\\])', text)
    pat = re.compile(r'\b(best|lowest|highest|only|all|every|none|no other|any run|'
                     r'first|worst|never|always|strongest|clearest|largest|smallest|'
                     r'most|least|of any)\b', re.I)
    for s in sents:
        if pat.search(s):
            print('  - ' + s[:230])


if __name__ == '__main__':
    main()

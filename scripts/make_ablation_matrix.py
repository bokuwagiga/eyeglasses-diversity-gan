"""Model x generation-strategy matrix over every evaluation ever run.

Scans results/ for diversity_report.json files and asks, for each one:
which trained model produced the images, and under which generation
strategy. Both answers come from the run's own config file, not from the
experiment log, so the matrix cannot drift away from the measurements.

Provenance rules, in order of trust:
  1. generation_config.json      - written by --generate-only after the
                                   provenance fix; authoritative.
  2. run_config.json WITH cli_args.generate_only set - a pre-fix
     generate-only run, which overwrote run_config.json with its own
     generation arguments; authoritative for exactly that reason.
  3. anything else               - the file holds TRAINING defaults for
     the generation fields (psi 0.7 etc.), which are not what was run.
     Reported as unknown rather than guessed.

    python scripts/make_ablation_matrix.py --out article/figures
"""

import argparse
import csv
import json
from pathlib import Path

# Cells whose provenance file does not record the generation settings.
# Values are taken from article/experiment_log.md; keeping them here,
# separate from the machine-read cells, makes clear which numbers rest on
# the log rather than on a config file.
# psi 0.7 was the CLI default in the early era, before the psi sweep. The
# existence of a separate r1gamma5_psi10 directory is what pins this: psi
# 1.0 needed its own run, so the base directory is the 0.7 default, and
# its FID 21.29 matches the log's psi-0.7 number rather than the 15.2 the
# log reports for r1gamma5 at psi 1.0.
FROM_LOG = {
    'results/r1gamma5/diversity': ('r1gamma5/best', 'psi 0.7 mixed'),
    'results/r1gamma5/diversity_last': ('r1gamma5/last', 'psi 0.7 mixed'),
    'results/r1gamma5_psi10/diversity': ('r1gamma5/best', 'psi 1.0 mixed'),
    # Runs 3-5 were evaluated at psi 1.0: the log's FID 22.7 / 12.5 / 34
    # match these reports exactly, and psi 1.0 was the comparison protocol
    # of that era. Only r1gamma5 has a psi 0.7 pair, which is why it alone
    # needed a separate _psi10 directory.
    'results/r1gamma10/diversity': ('r1gamma10/best', 'psi 1.0 mixed'),
    'results/divcfg2/diversity': ('divcfg2/best', 'psi 1.0 mixed'),
    'results/divcfg_probe/diversity': ('divcfg_probe/last', 'psi 1.0 mixed'),
    # Run 15/16: the log states the headline protocol and checkpoint_best
    # for both; neither wrote a generation config.
    'results/rebalance3/eval': ('rebalance3/best', 'layered 1.0/1.2@3 pure'),
}

# One training run continued through several epochs is several models as
# far as this matrix is concerned: each was evaluated separately and, for
# sharp1, each produced a different point on the Pareto front. The epoch
# is not in any config file, so it comes from the log.
OPERATING_POINT = {
    'results/sharp1/eval': 'sharp1@ep800',
    'results/sharp1/eval1200': 'sharp1@ep1200',
    'results/sharp1/eval1600': 'sharp1@ep1600',
}

# Which checkpoint a train-then-generate run actually generated from.
# gan_train.py originally took the latest checkpoint; the preference for
# checkpoint_best was added *because* ppl4's latest was past a collapse,
# so runs before that fix generated from 'latest' and later ones from
# 'best'. Not recoverable from any config file.
CHECKPOINT_AT_GENERATE = {
    'ppl4': 'last (post-collapse)',
    'rebalance': 'best',
    'rebalance2': 'best',
}

# Smoke tests on the 60-image toy set: not experiments.
EXCLUDE = ('smoke_',)

# Reports the log declares void. Kept visible rather than deleted, because
# a blank cell reads as "never run" and these WERE run - they just measured
# something other than what the directory name says.
VOID = {
    'results/ppl4/diversity':
        'measured the post-collapse ep400 checkpoint, not ep310 '
        '(eval used find_latest_checkpoint); superseded by ppl4_best',
}

# Post-hoc filtering is a generation strategy too, but it is applied by a
# separate script and leaves no trace in the config, so it is read off the
# report directory name.
FILTER_SUFFIX = {
    'filtered_diversity': ' +precision filter',
    'diversity_clean': ' +zero-flag filter',
    'diversity_mf1': ' +max-1-flag filter',
}


def strategy_label(a):
    """Compact name for a generation configuration."""
    psi = a.get('truncation_psi')
    fine = a.get('truncation_psi_fine')
    cutoff = a.get('truncation_cutoff')
    pure = a.get('pure_frac')
    centers = a.get('truncation_centers')

    if fine is not None and fine != psi:
        label = f'layered {psi}/{fine}@{cutoff}'
    else:
        label = f'psi {psi}'
    label += ' pure' if pure == 1.0 else ' mixed'
    if centers and centers > 1:
        label += f' +centers{centers}'
    return label


def model_label(ckpt_path):
    """'results/nocolorada/checkpoints/checkpoint_best.pth' -> 'nocolorada/best'."""
    p = Path(ckpt_path)
    run = p.parent.parent.name
    kind = p.stem.replace('checkpoint_', '')
    return f'{run}/{kind}'


def provenance(report_path, results_root):
    """(model, strategy) for a report, or (None, None) if not recorded."""
    d = report_path.parent
    for _ in range(3):  # report may sit a level or two under the eval dir
        for name in ('generation_config.json', 'run_config.json'):
            cfg = d / name
            if not cfg.exists():
                continue
            cli = json.load(open(cfg)).get('cli_args', {})
            ckpt = cli.get('generate_only')
            if ckpt:
                return model_label(ckpt), strategy_label(cli)
            if name == 'run_config.json' and cli.get('generate'):
                # Trained and then generated in one invocation, so the
                # generation flags in this file are the ones that ran.
                # Which checkpoint it used is version-dependent (the
                # best-checkpoint preference was added after ppl4), so the
                # checkpoint is left to CHECKPOINT_AT_GENERATE.
                run = cfg.parent.name
                return (f'{run}/{CHECKPOINT_AT_GENERATE.get(run, "?")}',
                        strategy_label(cli))
            if name == 'run_config.json':
                continue  # training config: generation fields are defaults
        if d == results_root:
            break
        d = d.parent
    return None, None


def collect(results_root):
    rows = []
    for report in sorted(results_root.glob('**/diversity_report.json')):
        rel = report.parent.as_posix()
        if any(x in rel for x in EXCLUDE):
            continue
        # FROM_LOG first: it distinguishes cases a config file cannot,
        # such as r1gamma5's best-vs-last evaluations, which share one
        # run_config.json.
        model, strat = FROM_LOG.get(rel, (None, None))
        if model is None:
            model, strat = provenance(report, results_root)
        for eval_dir, name in OPERATING_POINT.items():
            if rel == eval_dir or rel.startswith(eval_dir + '/'):
                model = name + '/best'
        strat = (strat or 'UNKNOWN') + FILTER_SUFFIX.get(report.parent.name, '')

        d = json.load(open(report))
        prdc = d['feature_space']['prdc']
        rows.append({
            'dir': rel,
            'void': VOID.get(rel),
            'model': model or f'UNKNOWN ({rel})',
            'strategy': strat,
            'fid': d['quality']['fid'],
            'recall': prdc['recall'],
            'precision': prdc['precision'],
            'ab_coverage': d['color']['gen']['ab_coverage'],
        })
    return rows


def write_inventory(rows, out_dir):
    lines = ['%-34s %-30s %8s %8s %8s %9s'
             % ('model / checkpoint', 'generation strategy', 'FID',
                'Prec', 'Recall', 'ab cov')]
    lines.append('-' * len(lines[0]))
    for r in sorted(rows, key=lambda r: -r['fid']):
        lines.append('%-34s %-30s %8.3f %8.3f %8.3f %9.4f'
                     % (r['model'], r['strategy'], r['fid'], r['precision'],
                        r['recall'], r['ab_coverage']))
        if r['void']:
            lines.append('%-34s   VOID: %s' % ('', r['void']))
    txt = '\n'.join(lines)
    (out_dir / 'evaluation_inventory.txt').write_text(txt + '\n')
    print(txt)


def build_matrix(rows):
    """(models worst-to-best, strategies, {(model, strategy): fid})."""
    live = [r for r in rows if not r['void']]
    cell = {}
    for r in live:
        key = (r['model'], r['strategy'])
        if key in cell:
            # Two evaluations claiming the same cell means one of them is
            # mislabelled; silently keeping the last would hide a result.
            print(f'  WARNING: {r["dir"]} collides with an earlier report '
                  f'at {key}; the matrix shows only one of them')
        cell[key] = r['fid']

    # Rank models by their best result, worst first: alphabetical order
    # says nothing, and the progression is the story the table tells.
    best = {}
    for (m, _), fid in cell.items():
        best[m] = min(fid, best.get(m, fid))
    models = sorted(best, key=lambda m: -best[m])
    # Columns in the order the levers were added: the plain psi sweep, then
    # layered truncation, then anything with a post-hoc filter. Alphabetical
    # order interleaves them and hides that the sweep came first.
    strategies = sorted({r['strategy'] for r in live},
                        key=lambda s: ('filter' in s, s.startswith('layered'), s))
    return models, strategies, cell


def write_matrix(rows, out_dir):
    models, strategies, cell = build_matrix(rows)

    w = max(len(m) for m in models) + 1
    head = [' ' * w + ''.join('%8d' % (i + 1) for i in range(len(strategies)))]
    head.append('-' * len(head[0]))
    for m in models:
        row = '%-*s' % (w, m)
        for s in strategies:
            v = cell.get((m, s))
            row += '%8s' % (f'{v:.2f}' if v is not None else '.')
        head.append(row)
    head.append('')
    head.append('columns (generation strategies):')
    for i, s in enumerate(strategies):
        head.append('  %2d  %s' % (i + 1, s))
    head.append('')
    filled = len(cell)
    total = len(models) * len(strategies)
    head.append('%d of %d cells measured (%.0f%%); "." = never run'
                % (filled, total, 100 * filled / total))
    txt = '\n'.join(head)
    (out_dir / 'ablation_matrix.txt').write_text(txt + '\n')
    print('\n' + txt)


def write_csv(rows, out_dir):
    """Spreadsheet copies: the aligned text tables lose their columns the
    moment they are pasted into a proportional font (e.g. an email)."""
    models, strategies, cell = build_matrix(rows)

    with open(out_dir / 'ablation_matrix.csv', 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['model / checkpoint'] + strategies)
        for m in models:
            wr.writerow([m] + [('' if (v := cell.get((m, s))) is None
                                else f'{v:.2f}') for s in strategies])

    with open(out_dir / 'evaluation_inventory.csv', 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['model / checkpoint', 'generation strategy', 'FID',
                     'precision', 'recall', 'ab_coverage', 'void reason'])
        for r in sorted(rows, key=lambda r: -r['fid']):
            wr.writerow([r['model'], r['strategy'], f'{r["fid"]:.3f}',
                         f'{r["precision"]:.3f}', f'{r["recall"]:.3f}',
                         f'{r["ab_coverage"]:.4f}', r['void'] or ''])
    print(f'\n  wrote {out_dir / "ablation_matrix.csv"} '
          f'and {out_dir / "evaluation_inventory.csv"}')


def write_html(rows, out_dir):
    """A table that survives copy-paste into an email client."""
    models, strategies, cell = build_matrix(rows)
    filled = len(cell)
    total = len(models) * len(strategies)

    td = ('padding:3px 7px;border:1px solid #bbb;text-align:right;'
          'font-family:Arial,sans-serif;font-size:13px')
    th = td + ';background:#eee;font-weight:bold'
    left = td + ';text-align:left'

    h = ['<table style="border-collapse:collapse">', '<tr>',
         f'<th style="{th};text-align:left">model / checkpoint</th>']
    h += [f'<th style="{th}">S{i + 1}</th>' for i in range(len(strategies))]
    h.append('</tr>')
    for m in models:
        h.append('<tr>')
        h.append(f'<td style="{left}">{m}</td>')
        for s in strategies:
            v = cell.get((m, s))
            if v is None:
                h.append(f'<td style="{td};color:#bbb">.</td>')
            else:
                h.append(f'<td style="{td}"><b>{v:.2f}</b></td>')
        h.append('</tr>')
    h.append('</table>')

    h.append('<p style="font-family:Arial,sans-serif;font-size:13px">'
             f'Cell = FID (lower is better). "." = never run. '
             f'{filled} of {total} cells measured '
             f'({100 * filled / total:.0f}%).</p>')
    h.append('<p style="font-family:Arial,sans-serif;font-size:13px">')
    h += [f'<b>S{i + 1}</b> &nbsp;{s}<br>' for i, s in enumerate(strategies)]
    h.append('</p>')

    p = out_dir / 'ablation_matrix.html'
    p.write_text('\n'.join(h) + '\n')
    print(f'  wrote {p}  (open in a browser, select all, paste into Gmail)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results')
    ap.add_argument('--out', default='article/figures')
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = collect(Path(args.results))
    if not rows:
        raise SystemExit('No diversity_report.json found.')
    write_inventory(rows, out_dir)
    write_matrix(rows, out_dir)
    write_csv(rows, out_dir)
    write_html(rows, out_dir)


if __name__ == '__main__':
    main()

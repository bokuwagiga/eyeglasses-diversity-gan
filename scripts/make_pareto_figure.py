"""Quality-diversity Pareto figure and results table from diversity reports.

Reads the diversity_report.json of each run and plots fidelity against
diversity, which is the article's central claim: no single model is best,
and the choice of "best" is a choice of operating point.

Fidelity is FID (x, inverted so better is right). Diversity is plotted
twice because the two axes disagree and that disagreement is a result:
  - recall: fraction of the REAL manifold the generator reaches
  - ab_coverage: spread of frame colour in CIELAB, an eyeglass-specific
    measure from the thesis, shown against the real value

Runs are given as label=path pairs so the figure can be rebuilt for any
subset:

    python scripts/make_pareto_figure.py --out article/figures
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

# Default run set. rebalance3 is deliberately absent: at FID 7.17 /
# ab_coverage 0.176 it is so far from the rest that including it squashes
# every point of interest into the top of the colour panel. Add it back
# with --runs when a "where the project started" figure is wanted.
DEFAULT_RUNS = [
    'rebalance4=results/rebalance4/eval',
    'sharp1 ep800=results/sharp1/eval',
    'sharp1 ep1200=results/sharp1/eval1200',
    'sharp1 ep1600=results/sharp1/eval1600',
    'mirror1=results/mirror1/eval',
    'mirror2=results/mirror2/eval',
]

# Models the article puts forward, labelled in bold; the rest are context.
# sharp1 ep800 and ep1200 are deliberately not headline models even though
# ep1200 sits on both fronts: sharp1 was one continued run and gan_train.py
# overwrites checkpoint_best.pth, so their weights no longer exist. Their
# numbers stand as a training trajectory, but nobody can generate from
# them again, which is not something to put forward as a model.
HEADLINE = {'sharp1 ep1600', 'mirror2'}


def find_report(run_dir):
    """diversity_report.json sits either in the eval dir or a diversity/ subdir."""
    for cand in (Path(run_dir) / 'diversity_report.json',
                 Path(run_dir) / 'diversity' / 'diversity_report.json'):
        if cand.exists():
            return cand
    return None


def read_runs(specs):
    rows = []
    for spec in specs:
        label, _, run_dir = spec.partition('=')
        path = find_report(run_dir)
        if path is None:
            print(f'  skip {label}: no diversity_report.json under {run_dir}')
            continue
        d = json.load(open(path))
        prdc = d['feature_space']['prdc']
        pose = (d.get('pose') or {}).get('dispersion') or {}
        rows.append({
            'label': label,
            'fid': d['quality']['fid'],
            'kid': d['quality']['kid_mean'],
            'precision': prdc['precision'],
            'recall': prdc['recall'],
            'density': prdc['density'],
            'coverage': prdc['coverage'],
            'ab_coverage': d['color']['gen']['ab_coverage'],
            'ab_coverage_real': d['color']['real']['ab_coverage'],
            'lpips': d.get('lpips', {}).get('gen', {}).get('mean'),
            'lpips_real': d.get('lpips', {}).get('real', {}).get('mean'),
            'memorisation': d['feature_space']['memorisation']['ratio_gen_over_real'],
            'yaw_ratio': pose.get('offset_std_ratio'),
            'aligned_gap': ((d.get('pose') or {}).get('aligned_p50_gen', 0)
                            - (d.get('pose') or {}).get('aligned_p50_real', 0)
                            if d.get('pose') else None),
        })
    return rows


def pareto_front(rows, div_key):
    """Labels that no other run beats on BOTH fidelity and this diversity axis."""
    front = []
    for r in rows:
        if r[div_key] is None:
            continue
        dominated = any(o is not r
                        and o[div_key] is not None
                        and o['fid'] <= r['fid']
                        and o[div_key] >= r[div_key]
                        and (o['fid'] < r['fid'] or o[div_key] > r[div_key])
                        for o in rows)
        if not dominated:
            front.append(r['label'])
    return set(front)


def panel(ax, rows, div_key, div_name, real_value=None):
    front = pareto_front(rows, div_key)
    pts = [r for r in rows if r[div_key] is not None]

    # Pareto staircase, drawn first so markers sit on top.
    chain = sorted([r for r in pts if r['label'] in front], key=lambda r: r['fid'])
    if len(chain) > 1:
        ax.plot([r['fid'] for r in chain], [r[div_key] for r in chain],
                '-', color='0.6', lw=1.2, zorder=1,
                label='Pareto front')

    if real_value is not None:
        ax.axhline(real_value, ls='--', lw=1.0, color='crimson', zorder=0)
        ax.text(0.99, real_value, 'real dataset', color='crimson',
                fontsize=8, va='bottom', ha='right',
                transform=ax.get_yaxis_transform())

    # One visual channel per meaning:
    #   fill        -> on the Pareto front for this diversity axis
    #   label weight-> model the article puts forward
    best_fid = min(r['fid'] for r in pts)
    for r in pts:
        on_front = r['label'] in front
        ax.scatter(r['fid'], r[div_key], s=70,
                   facecolor='tab:blue' if on_front else 'white',
                   edgecolor='tab:blue', linewidth=1.4, zorder=3)
        headline = r['label'] in HEADLINE
        # The best-FID point sits on the right spine; label it inwards.
        left = r['fid'] == best_fid
        ax.annotate(r['label'], (r['fid'], r[div_key]),
                    textcoords='offset points',
                    xytext=(-7, 6) if left else (7, 5),
                    ha='right' if left else 'left',
                    fontsize=8.5,
                    fontweight='bold' if headline else 'normal',
                    color='black' if headline else '0.4')

    ax.set_xlabel('FID  (lower is better)')
    ax.set_ylabel(div_name)
    ax.invert_xaxis()          # better fidelity to the right
    ax.margins(x=0.16, y=0.12)  # room for the point labels
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)


def make_figure(rows, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    panel(axes[0], rows, 'recall', 'Recall  (coverage of the real manifold)')
    real_ab = rows[0]['ab_coverage_real'] if rows else None
    panel(axes[1], rows, 'ab_coverage', 'Colour coverage (CIELAB ab)',
          real_value=real_ab)
    axes[0].set_title('(a) Fidelity vs manifold coverage', fontsize=10)
    axes[1].set_title('(b) Fidelity vs colour diversity', fontsize=10)
    handles = [
        plt.Line2D([], [], ls='-', color='0.6', lw=1.2, label='Pareto front'),
        plt.Line2D([], [], ls='none', marker='o', mfc='tab:blue',
                   mec='tab:blue', ms=7, label='on the front'),
        plt.Line2D([], [], ls='none', marker='o', mfc='white',
                   mec='tab:blue', ms=7, label='dominated'),
    ]
    axes[0].legend(handles=handles, fontsize=8, loc='upper left',
                   framealpha=0.9)
    fig.suptitle('Quality-diversity trade-off across generator variants',
                 fontsize=12)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        p = out_dir / f'pareto.{ext}'
        fig.savefig(p, dpi=200 if ext == 'png' else None, bbox_inches='tight')
        print(f'  wrote {p}')
    plt.close(fig)


def fmt(v, nd=3):
    return '-' if v is None else f'{v:.{nd}f}'


def write_table(rows, out_dir):
    cols = [('label', 'model', 0), ('fid', 'FID', 3), ('kid', 'KID', 5),
            ('precision', 'Prec', 3), ('recall', 'Recall', 3),
            ('density', 'Dens', 3), ('coverage', 'Cov', 3),
            ('ab_coverage', 'ab cov', 4), ('lpips', 'LPIPS', 4),
            ('yaw_ratio', 'yaw ratio', 2), ('memorisation', 'mem', 2)]

    lines = ['%-16s' % 'model' + ''.join('%10s' % c[1] for c in cols[1:])]
    lines.append('-' * len(lines[0]))
    for r in rows:
        lines.append('%-16s' % r['label']
                     + ''.join('%10s' % fmt(r[k], nd) for k, _, nd in cols[1:]))
    txt = '\n'.join(lines)
    (out_dir / 'results_table.txt').write_text(txt + '\n')
    print('\n' + txt)

    tex = ['\\begin{tabular}{l' + 'r' * (len(cols) - 1) + '}', '\\hline',
           ' & '.join(c[1] for c in cols) + ' \\\\', '\\hline']
    for r in rows:
        cells = [r['label'].replace('_', '\\_')]
        cells += [fmt(r[k], nd) for k, _, nd in cols[1:]]
        tex.append(' & '.join(cells) + ' \\\\')
    tex += ['\\hline', '\\end{tabular}']
    (out_dir / 'results_table.tex').write_text('\n'.join(tex) + '\n')
    print(f'\n  wrote {out_dir / "results_table.tex"}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='*', default=DEFAULT_RUNS,
                    help='label=path/to/eval_dir pairs')
    ap.add_argument('--out', default='article/figures')
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_runs(args.runs)
    if not rows:
        raise SystemExit('No reports found.')
    make_figure(rows, out_dir)
    write_table(rows, out_dir)


if __name__ == '__main__':
    main()

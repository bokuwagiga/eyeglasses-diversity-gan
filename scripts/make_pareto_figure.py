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

# Default run set, with the labels used in the article. The run names of the
# experiment log are sharp1 (plain model), mirror2 (mirror-axis model),
# mirror1 (fixed-mirror variant) and rebalance4 (bilinear training resize).
DEFAULT_RUNS = [
    'Bilinear resize=results/rebalance4/eval',
    'Plain, 800 ep=results/sharp1/eval',
    'Plain, 1,200 ep=results/sharp1/eval1200',
    'Plain=results/sharp1/eval1600',
    'Fixed mirror=results/mirror1/eval',
    'Mirror axis=results/mirror2/eval',
]
DEFAULT_EXTRA = [
    'Plain, psi 1.2=results/resweep/sharp1_p12_pure',
    'Mirror axis, psi 1.2=results/resweep/mirror2_p12_pure',
]
DEFAULT_BASELINE = [
    'StyleGAN2-ADA=results/baseline_sg2ada_gamma10/snap001800',
]

# Models the article puts forward: filled markers, bold labels.
HEADLINE = {'Plain', 'Mirror axis'}
# One training run evaluated at three lengths, joined by a line.
TRAJECTORY = ['Plain, 800 ep', 'Plain, 1,200 ep', 'Plain']
# Protocol variants, joined to the model they were sampled from.
PARENT = {'Plain, psi 1.2': 'Plain', 'Mirror axis, psi 1.2': 'Mirror axis'}
# Text shown next to each point.
TEXT = {'Plain, 800 ep': '800 ep', 'Plain, 1,200 ep': '1,200 ep',
        'Plain': 'Plain (1,600 ep)', 'Plain, psi 1.2': r'$\psi = 1.2$',
        'Mirror axis, psi 1.2': r'$\psi = 1.2$'}
# Label offsets in points and horizontal alignment, per panel.
OFFSETS = {
    'recall': {
        'Plain': (-8, -3, 'right'), 'Plain, 1,200 ep': (8, -3, 'left'),
        'Plain, 800 ep': (8, -4, 'left'), 'Mirror axis': (-8, -9, 'right'),
        'Plain, psi 1.2': (8, 2, 'left'), 'Mirror axis, psi 1.2': (8, -3, 'left'),
        'Fixed mirror': (8, -3, 'left'), 'Bilinear resize': (-8, -3, 'right'),
        'StyleGAN2-ADA': (8, -3, 'left'),
    },
    'ab_coverage': {
        'Plain': (-2, -15, 'right'), 'Plain, 1,200 ep': (8, 3, 'left'),
        'Plain, 800 ep': (8, 3, 'left'), 'Mirror axis': (-8, 5, 'right'),
        'Plain, psi 1.2': (-8, -11, 'right'), 'Mirror axis, psi 1.2': (8, -12, 'left'),
        'Fixed mirror': (8, -3, 'left'), 'Bilinear resize': (-8, -3, 'right'),
        'StyleGAN2-ADA': (8, -3, 'left'),
    },
}
BLUE = '#1f5fa8'
ORANGE = '#d9730d'


def find_report(run_dir):
    """diversity_report.json sits either in the eval dir or a diversity/ subdir."""
    for cand in (Path(run_dir) / 'diversity_report.json',
                 Path(run_dir) / 'diversity' / 'diversity_report.json'):
        if cand.exists():
            return cand
    return None


def read_runs(specs, variant=False):
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
            'variant': variant,
        })
    return rows


def panel(ax, rows, div_key, div_name, tag, real_value=None):
    pts = {r['label']: r for r in rows if r[div_key] is not None}

    if real_value is not None:
        ax.axhline(real_value, ls='--', lw=0.9, color='0.45', zorder=0)
        ax.text(0.015, real_value, 'real catalogue', color='0.35', fontsize=8,
                va='bottom', ha='left', transform=ax.get_yaxis_transform())

    chain = [pts[k] for k in TRAJECTORY if k in pts]
    if len(chain) > 1:
        ax.plot([r['fid'] for r in chain], [r[div_key] for r in chain],
                '-', color=BLUE, lw=1.0, alpha=0.55, zorder=1)
    for child, parent in PARENT.items():
        if child in pts and parent in pts:
            ax.plot([pts[parent]['fid'], pts[child]['fid']],
                    [pts[parent][div_key], pts[child][div_key]],
                    ':', color=BLUE, lw=1.0, alpha=0.7, zorder=1)

    for label, r in pts.items():
        baseline = r.get('variant') == 'baseline'
        headline = label in HEADLINE
        colour = ORANGE if baseline else BLUE
        ax.scatter(r['fid'], r[div_key], s=62 if headline or baseline else 40,
                   marker={'protocol': 's', 'baseline': '^'}.get(r.get('variant'), 'o'),
                   facecolor=colour if headline or baseline else 'white',
                   edgecolor=colour, linewidth=1.3, zorder=3)
        dx, dy, ha = OFFSETS.get(div_key, {}).get(label, (8, 4, 'left'))
        ax.annotate(TEXT.get(label, label), (r['fid'], r[div_key]),
                    textcoords='offset points', xytext=(dx, dy), ha=ha,
                    fontsize=8.5, fontweight='bold' if headline else 'normal',
                    color=ORANGE if baseline else ('black' if headline else '0.3'))

    ax.set_xlabel('FID (better to the right)')
    ax.set_ylabel(div_name)
    ax.set_title(tag, fontsize=9.5, loc='left')
    ax.invert_xaxis()
    ax.margins(x=0.14, y=0.14)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def make_figure(rows, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    panel(axes[0], rows, 'recall', 'Recall', '(a)')
    real_ab = rows[0]['ab_coverage_real'] if rows else None
    panel(axes[1], rows, 'ab_coverage', 'Colour coverage', '(b)',
          real_value=real_ab)
    fig.tight_layout(w_pad=2.5)
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
    ap.add_argument('--extra-runs', nargs='*', default=DEFAULT_EXTRA,
                    help='label=path pairs plotted as protocol variants '
                         '(square markers); excluded from the table')
    ap.add_argument('--baseline-runs', nargs='*', default=DEFAULT_BASELINE,
                    help='label=path pairs of external baselines (triangle '
                         'markers); excluded from the table')
    ap.add_argument('--out', default='article/figures')
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_runs(args.runs)
    if not rows:
        raise SystemExit('No reports found.')
    extra = read_runs(args.extra_runs, variant='protocol')
    base = read_runs(args.baseline_runs, variant='baseline')
    make_figure(rows + extra + base, out_dir)
    write_table(rows, out_dir)


if __name__ == '__main__':
    main()

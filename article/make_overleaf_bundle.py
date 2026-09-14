"""Export an upload-ready Overleaf bundle from the MDPI template directory.

The template directory is the single source of truth and is where editing
happens; this script copies out of it the files Overleaf actually needs and
nothing else - no .aux/.log/.out/.bbl, no locally generated PDF, and none of
the _compiletest.* scaffolding used to work around the local LaTeX kernel.

Re-run it after any edit; it overwrites in place and deletes stale files, so
the bundle can never drift from the source.

  python article/make_overleaf_bundle.py

Then upload the contents of article/overleaf/ to Overleaf, or zip that folder
and use "New Project -> Upload Project". Set the main document to article.tex.
"""
import filecmp
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '2026_Generating_Diverse_Synthetic_Eyeglass_Datasets_Using_GANs__MDPI_')
DST = os.path.join(HERE, 'overleaf')

# Explicit allow-list. Anything not named here is not uploaded.
FILES = [
    'article.tex',       # the manuscript
    'refs.bib',          # bibliography database
    'results_table.tex', # \input by the results section
]
DIRS = [
    'Definitions',       # MDPI class, bst, journal names, logos
    'images',            # figures referenced by the manuscript
]

# Never copy these, wherever they appear.
SKIP_EXT = {'.aux', '.log', '.out', '.bbl', '.blg', '.synctex', '.gz',
            '.fls', '.fdb_latexmk', '.toc', '.lof', '.lot'}
SKIP_PREFIX = ('_compiletest',)


def keep(name):
    if name.startswith(SKIP_PREFIX):
        return False
    ext = os.path.splitext(name)[1].lower()
    if ext in SKIP_EXT:
        return False
    # The locally built PDF is a compile artefact, not an input.
    if name.lower() == 'article.pdf':
        return False
    return True


def copy_file(rel):
    s, d = os.path.join(SRC, rel), os.path.join(DST, rel)
    if not os.path.exists(s):
        print('  MISSING in source: %s' % rel)
        return None
    os.makedirs(os.path.dirname(d) or DST, exist_ok=True)
    unchanged = os.path.exists(d) and filecmp.cmp(s, d, shallow=False)
    if not unchanged:
        shutil.copy2(s, d)
    return (rel, os.path.getsize(s), 'unchanged' if unchanged else 'updated')


def main():
    if not os.path.isdir(SRC):
        sys.exit('source template directory not found: %s' % SRC)

    wanted, rows = set(), []
    for f in FILES:
        r = copy_file(f)
        if r:
            rows.append(r); wanted.add(f.replace('\\', '/'))
    for sub in DIRS:
        base = os.path.join(SRC, sub)
        if not os.path.isdir(base):
            print('  MISSING in source: %s/' % sub)
            continue
        for root, _, names in os.walk(base):
            for n in sorted(names):
                if not keep(n):
                    continue
                rel = os.path.relpath(os.path.join(root, n), SRC).replace('\\', '/')
                r = copy_file(rel)
                if r:
                    rows.append(r); wanted.add(rel)

    # Remove anything in the bundle that is no longer wanted.
    removed = []
    for root, _, names in os.walk(DST):
        for n in names:
            rel = os.path.relpath(os.path.join(root, n), DST).replace('\\', '/')
            if rel not in wanted:
                os.remove(os.path.join(root, n)); removed.append(rel)
    for root, dirs, names in os.walk(DST, topdown=False):
        for dd in dirs:
            p = os.path.join(root, dd)
            if not os.listdir(p):
                os.rmdir(p)

    total = sum(sz for _, sz, _ in rows)
    print('Overleaf bundle: %s' % DST)
    for rel, sz, state in rows:
        print('  %-9s %-46s %8.1f KB' % (state, rel, sz / 1024.0))
    for rel in removed:
        print('  removed   %s' % rel)
    print('  %d files, %.1f MB total' % (len(rows), total / 1048576.0))


if __name__ == '__main__':
    main()

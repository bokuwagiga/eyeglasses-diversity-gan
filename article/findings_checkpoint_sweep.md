# Best-vs-last checkpoint sweep - findings (2026-08-27)

Source: `results/checkpoint_comparison.csv`, 18 runs x 2 checkpoints,
frozen S7 protocol, 10k images, seed 42.

## Noise floor (measured, not assumed)

Same checkpoint generated at seed 42 vs seed 0 (mirror2, both checkpoints):

| metric | max abs delta |
|---|---|
| FID | 0.052 |
| recall | 0.019 |
| ab_coverage | 0.016 |
| precision | 0.008 |
| coverage | 0.003 |

Anything below these is not a result. This is itself worth reporting: it is
the first seed-noise measurement we have for the metric suite.

## Answer to the mentor: no, min-KID does not give the best metric set

Of 16 valid runs (sg2f excluded, see below):

**KID selection clearly right - late collapse or decay (5 runs)**
rebalance2 (FID 71.1 -> 7.9), wide2 (45.1 -> 14.2), ppl4 (27.2 -> 8.9),
divcfg (48.7 -> 33.7), wide_g2d15 (12.0 -> 9.0). Best wins on nearly every
metric. Here the checkpoint guard did exactly its job.

**KID selection clearly wrong (3 runs)**
nocolorada, rebalance, mirror1 - the final epoch is better. nocolorada is
the worst case and is discussed below.

**The two axes disagree (3 runs)**
divcfg, r1gamma5 (best FID by 0.85 but last has ab_coverage 0.434 vs 0.305),
rebalance4 (FID tied, last has recall 0.656 vs 0.623). This is precisely the
failure mode the mentor predicted.

**No meaningful difference (5 runs)**
sharp1, rebalance3, divcfg2, r1gamma10, baseline - all deltas at or below
the noise floor.

Conclusion for the paper: **KID checkpoint selection is a collapse guard,
not an optimiser.** It reliably prevents a catastrophic result and does not
reliably produce the best one.

## The nocolorada problem

nocolorada's LAST checkpoint (ep399) beats its min-KID checkpoint (ep279) on
every axis at once:

| | FID | recall | coverage | precision |
|---|---|---|---|---|
| best (ep279) | 10.20 | 0.212 | 0.746 | 0.879 |
| last (ep399) | **8.93** | **0.354** | **0.850** | 0.866 |

Recall is 67% higher. This matters beyond one row: **nocolorada is the model
the entire headline generation protocol (S7) was tuned on**, and it was tuned
on the weaker of its two checkpoints. This strengthens the case for the
12-cell protocol re-sweep on the current models.

## Two rows resolved - both were silent failures, confirmed on the GPU box

### sg2f: the code was changed out from under the checkpoint

Not a load failure. It loads with 0 missing and 0 unexpected keys and the
generate path uses a strict load. The images are not noise - they have
recognisable frame geometry with a blown-out yellow/red/black palette.

Cause, from git: commit c23dc0e (2026-08-01) added G-side blur and
equalized-lr; sg2f trained and was evaluated under that code (published FID
41.54); commit 894564f (2026-08-04 14:09) reverted both after sg2f mode
collapsed - 70 minutes AFTER the published evaluation. Equalized-lr is a
runtime per-layer weight scale, not a stored parameter, so removing it leaves
every key and shape intact. The checkpoint loads perfectly and then computes
something different.

**This is a methods finding, not just a broken row.** A checkpoint can be
invalidated by a code change that alters no key and no tensor shape, so
nothing raises. sg2f is our third unreproducible result, and the only one
whose cause is semantic rather than a deleted file.

Both sg2f cells are marked FAILED in the CSV with metrics blanked.

### edgeD: the generator did not train for its final 20 epochs

EMA weights are bit-identical between ep579 and ep599 (0 of 102 tensors
differ), which is why both cells produced literally the same images. G
differs in only 2 of 108 tensors, both spectral-norm power-iteration buffers.
D meanwhile updated normally (49 of 51). Every other run differs in 100% of G
and EMA tensors over its best-to-last window, so this is specific to edgeD.

Leading hypothesis: AMP's GradScaler silently skips the optimiser step when
gradients contain inf/NaN, so losses keep being computed and logged while the
weights never move. Not proven retroactively - the scaler state is not
checkpointed - but it is the one path that produces exactly this signature.

Guarded going forward in `gan/gan_train.py` (commit d085510): skipped
generator steps are now counted and reported per epoch, and the generate path
refuses a partial EMA load rather than silently mixing EMA and raw weights.

## Minor provenance note

`rebalance` (18.60 published vs 18.46) and `rebalance2` (7.80 vs 7.87) are
the only non-anomalous rows that miss their published value by more than the
noise floor. These are also two of the three rows whose generating checkpoint
was INFERRED rather than read from a config file (`CHECKPOINT_AT_GENERATE` in
`scripts/make_ablation_matrix.py`). Worth a look, but the gaps are small.

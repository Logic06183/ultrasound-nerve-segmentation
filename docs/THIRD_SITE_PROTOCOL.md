# Third-site external test protocol (BPSegData)

Written and committed 18 October 2026, before any BPSegData has been requested back or seen.

## Why

Version 3 was trained on two sites. Whether it works on a scanner it has never seen is untested.
BPSegData (Yao et al. 2022, Ningbo; 320 images, 3 annotating doctors) was requested from the authors
on 18 October 2026. This protocol fixes the test before the data arrives.

## Model and rule

- Frozen version-3 model: the 5-fold ensemble and fold 0 alone, exactly as in `results/metrics.json`.
- Frozen decision rule: presence ≥ 0.6 and largest component ≥ 1000 px (scaled by image area).
- Browser preprocessing: greyscale by luma, area-average resize to 224×288, rounded.
- No retraining, recalibration, intensity matching or threshold change before the primary result.

## Checks before scoring

1. Confirm the images do not overlap site B (near-duplicate thumbnail match, as in `analysis/profile_data.py`).
   Any overlapping image is removed and the count reported.
2. Build masks as the convex hull of all nerve polygons, matching sites A and B. If the label format
   cannot be mapped to the brachial plexus, stop and report that instead of scoring.
3. Patient IDs: if given, bootstrap by patient. If not, bootstrap by image and say so.

## Primary result

Sensitivity, specificity, AUROC of the presence score, and Dice on detected positives, with 95% CIs.
Reported whatever they are. If there are no nerve-free frames, specificity is reported as not estimable.

## Pass mark (set now)

The tool counts as generalising to a third scanner only if ensemble sensitivity's lower 95% bound is
≥ 0.40, the lowest site-level lower bound already seen (site A). Anything below is reported as a
failure to generalise, as version 2's external test was.

## After the primary result

Any later analysis (intensity matching, fine-tuning on part of the data) is labelled exploratory and
cannot replace the primary result.

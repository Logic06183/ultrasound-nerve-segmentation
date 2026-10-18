# External validation protocol

Written and committed 18 October 2026, before the shipped model was run on this data.

## Data

Kaggle dataset `alexcolmenero/ultrasonic-brachial-plexus`: 955 interscalene ultrasound frames from 88
patients, LabelMe polygons. Labels are pinyin: `shenjing` nerve, `dongmai` artery, `jingmai` vein,
`jirouzuzhi` muscle. 58 frames have no nerve polygon. Different site, scanner, image size and
annotation style from the training data. Licence listed as "unknown", so this is research use only
and no images from it are redistributed.

## What is frozen

The shipped model (fold 0, version 2), its decision rule (presence probability ≥ 0.6 and largest
region ≥ 4000 px at 420×580, scaled by image area), and the browser preprocessing: greyscale by
luma, area-average resize to 224×288, rounded to whole grey levels. Nothing is tuned on this data.

## Reference standard

- Nerve present: the frame has at least one `shenjing` polygon.
- Plexus region: the convex hull of all `shenjing` polygons. This is the closest analogue to the
  single plexus outline in the training labels, which enclose the roots as one region.

## Measures

All with 95% CIs from 2,000 bootstrap resamples of patients.

1. Sensitivity for presence. **Primary.**
2. Specificity for presence, on the 58 nerve-free frames. Imprecise by design.
3. Hit rate among nerve frames: predicted region overlaps any annotated nerve polygon.
4. Among detected nerve frames: centroid of the predicted region inside the plexus hull.
5. Among detected nerve frames: Dice against the plexus hull.

## Interpretation fixed in advance

A large drop from the internal test is expected and will be reported as a failure to generalise,
not explained away. The results will be added to the model card unchanged.

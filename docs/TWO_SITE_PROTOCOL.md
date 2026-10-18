# Two-site model protocol (version 3)

Written and committed 18 October 2026, before training.

## Why

The single-site model failed external validation (0 of 897 nerve frames detected on a second
scanner). Version 3 trains on both sites so that it has seen more than one machine.

## Data and splits

- Site A: Kaggle training set, 47 patients. Same split as before: 9 locked test patients, 38 in 5 folds.
- Site B: `alexcolmenero/ultrasonic-brachial-plexus`, 88 patients. 18 patients (20%, seed 0) become
  a locked site-B test set. The other 70 are spread over the same 5 folds.
- Site B masks: convex hull of all `shenjing` polygons, so both sites label the plexus as one region.
- Both sites use the browser preprocessing: greyscale by luma, area-average resize to 224×288, rounded.

## Training and tuning

Same architecture, loss, augmentation and 15-epoch schedule as version 2. The decision rule is tuned
on pooled out-of-fold predictions from both sites. Test patients of either site are never used.

## Reporting

Locked-test results per site, with patient-bootstrap 95% CIs, reported whatever they are.

Site A's 9 test patients are being scored for the third time (versions 1, 2 and 3). That makes the
site-A test estimate progressively less clean, and the model card will say so. Site B's 18 test
patients were scored once already, by the frozen version-2 model in the failed external validation;
version 3 is their first model trained with site-B data. There is no untouched external site left,
so version 3 makes no claim about generalisation to a third scanner.

"""Tune the decision rule on out-of-fold predictions, then score the locked test patients once.

Decision rule: nerve present iff presence probability >= t_cls and the largest connected
component of (mask probability >= 0.5) at full resolution has >= min_area pixels.
All metrics are computed at the original 420x580 resolution against the original masks.
Confidence intervals: 2000-replicate bootstrap over patients.
"""
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from skimage.io import imread
from sklearn.metrics import roc_auc_score

from nerveseg.data import ORIG_H, ORIG_W, list_pairs, load_cache, patient_split
from nerveseg.model import build_model
from nerveseg.train import DUP_THRESHOLD, predict

OUT = 'results'
N_BOOT = 2000


def upsample(prob):
    t = torch.from_numpy(prob.astype(np.float32)).unsqueeze(1)
    return F.interpolate(t, (ORIG_H, ORIG_W), mode='bilinear', align_corners=False)[:, 0].numpy()


def largest_components(prob_full):
    """Largest 4-connected region of (prob >= 0.5) per image, and its area."""
    out = np.zeros(prob_full.shape, bool)
    areas = np.zeros(len(prob_full), int)
    for i in range(len(prob_full)):
        lab, n = ndimage.label(prob_full[i] >= 0.5)
        if n:
            sizes = np.bincount(lab.ravel())[1:]
            k = int(np.argmax(sizes))
            out[i], areas[i] = lab == k + 1, sizes[k]
    return out, areas


def apply_rule(comp, areas, cls, t_cls, min_area):
    keep = (cls >= t_cls) & (areas >= min_area) & (areas > 0)
    return comp & keep[:, None, None]


def postprocess(prob_full, cls, t_cls, min_area):
    """Return binary full-res masks (largest component only) under the decision rule."""
    comp, areas = largest_components(prob_full)
    return apply_rule(comp, areas, cls, t_cls, min_area)


def per_image(pred, true):
    """Per-image arrays: Kaggle dice (empty/empty = 1), presence flags, centroid error."""
    inter = (pred & true).reshape(len(pred), -1).sum(1)
    ps = pred.reshape(len(pred), -1).sum(1)
    ts = true.reshape(len(true), -1).sum(1)
    dice = np.where((ps + ts) == 0, 1.0, 2 * inter / np.maximum(ps + ts, 1))
    cerr = np.full(len(pred), np.nan)
    for i in np.where((ps > 0) & (ts > 0))[0]:
        cerr[i] = np.linalg.norm(np.array(ndimage.center_of_mass(pred[i])) - ndimage.center_of_mass(true[i]))
    return dict(dice=dice, pred_pos=ps > 0, true_pos=ts > 0, cerr=cerr)


def metrics(r, cls):
    tp = r['pred_pos'] & r['true_pos']
    tn = ~r['pred_pos'] & ~r['true_pos']
    P, N = r['true_pos'].sum(), (~r['true_pos']).sum()
    out = {
        'kaggle_mean_dice': r['dice'].mean(),
        'sensitivity': tp.sum() / max(P, 1),
        'specificity': tn.sum() / max(N, 1),
        'ppv': tp.sum() / max(r['pred_pos'].sum(), 1),
        'npv': tn.sum() / max((~r['pred_pos']).sum(), 1),
        'dice_labelled_positives': r['dice'][r['true_pos']].mean(),
        'dice_true_detections': r['dice'][tp].mean() if tp.any() else np.nan,
        'centroid_error_px_median': np.nanmedian(r['cerr']) if tp.any() else np.nan,
        'brier_presence': np.mean((cls - r['true_pos']) ** 2),
    }
    if 0 < P < len(cls):
        out['auroc_presence'] = roc_auc_score(r['true_pos'], cls)
    return out


def bootstrap(r, cls, subj, seed=0):
    rng = np.random.default_rng(seed)
    subs = np.unique(subj)
    groups = {s: np.where(subj == s)[0] for s in subs}
    point = metrics(r, cls)
    reps = {k: [] for k in point}
    for _ in range(N_BOOT):
        idx = np.concatenate([groups[s] for s in rng.choice(subs, len(subs))])
        m = metrics({k: v[idx] for k, v in r.items()}, cls[idx])
        for k in reps:
            if k in m:
                reps[k].append(m[k])
    return {k: {'estimate': float(point[k]), 'ci95': [float(np.nanpercentile(reps[k], 2.5)),
                                                      float(np.nanpercentile(reps[k], 97.5))]}
            for k in point}


def full_res_masks(idx, meta):
    lookup = {(s, k): mp for s, k, _, mp in list_pairs('downloads')}
    return np.stack([imread(lookup[(int(meta[i, 0]), int(meta[i, 1]))]) > 0 for i in idx])


def contradictory_set(pos, keep):
    pairs = np.load('cache/dup_pairs.npy')
    sel = pairs[pairs[:, 2] < DUP_THRESHOLD]
    a, b = sel[:, 0].astype(int), sel[:, 1].astype(int)
    c = keep[a] & keep[b] & (pos[a] != pos[b])
    return set(a[c]) | set(b[c])


def main():
    os.makedirs(OUT, exist_ok=True)
    imgs, masks, meta, area = load_cache()
    subj = meta[:, 0]
    fold = np.array([patient_split(subj)[int(s)] for s in subj])
    pos = area > 0

    # 1. out-of-fold predictions on development patients -> tune decision rule
    oof = [np.load(f) for f in sorted(glob.glob('runs/fold*/val_pred.npz'))]
    assert len(oof) == 5, f'expected 5 folds, found {len(oof)}'
    d_idx = np.concatenate([o['idx'] for o in oof])
    d_prob = upsample(np.concatenate([o['mask'] for o in oof]))
    d_cls = np.concatenate([o['cls'] for o in oof])
    d_true = full_res_masks(d_idx, meta)
    d_comp, d_area = largest_components(d_prob)
    grid = []
    for t in np.round(np.arange(0.3, 0.81, 0.05), 2):
        for a in (0, 1000, 2000, 3000, 4000, 5000):
            r = per_image(apply_rule(d_comp, d_area, d_cls, t, a), d_true)
            grid.append((float(r['dice'].mean()), float(t), int(a)))
    best = max(grid)
    t_cls, min_area = best[1], best[2]
    rule = {'t_cls': t_cls, 'min_area_px_full_res': min_area, 'oof_kaggle_mean_dice': best[0]}
    oof_r = per_image(apply_rule(d_comp, d_area, d_cls, t_cls, min_area), d_true)
    dev = bootstrap(oof_r, d_cls, subj[d_idx])
    del d_prob

    # 2. locked test patients: 5-fold ensemble and the single fold-0 model
    t_idx = np.where(fold == -1)[0]
    t_true = full_res_masks(t_idx, meta)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    preds = []
    for f in range(5):
        net = build_model(weights=None).to(device)
        net.load_state_dict(torch.load(f'runs/fold{f}/best.pt', map_location=device))
        preds.append(predict(net, imgs[t_idx], device))
    res = {'test_patients': sorted(set(subj[t_idx].tolist())), 'n_test_images': int(len(t_idx)),
           'test_positive_rate': float(pos[t_idx].mean()), 'decision_rule': rule, 'development_oof': dev}
    contra = contradictory_set(pos, fold == -1)
    keep = np.array([i not in contra for i in t_idx])
    res['n_test_images_in_contradictory_pairs'] = int((~keep).sum())
    for name, (mp, cp) in {'ensemble5': (np.mean([p[0].astype(np.float32) for p in preds], 0),
                                          np.mean([p[1] for p in preds], 0)),
                           'single_fold0': (preds[0][0], preds[0][1])}.items():
        prob = upsample(mp)
        r = per_image(postprocess(prob, cp, t_cls, min_area), t_true)
        res[f'test_{name}'] = bootstrap(r, cp, subj[t_idx])
        res[f'test_{name}_consistent_labels_only'] = bootstrap({k: v[keep] for k, v in r.items()}, cp[keep],
                                                               subj[t_idx][keep])
        per_pat = {}
        for s in np.unique(subj[t_idx]):
            k = subj[t_idx] == s
            per_pat[int(s)] = {'n': int(k.sum()), 'positive_rate': float(r['true_pos'][k].mean()),
                               'kaggle_mean_dice': float(r['dice'][k].mean()),
                               'sensitivity': float((r['pred_pos'] & r['true_pos'])[k].sum() / max(r['true_pos'][k].sum(), 1)),
                               'specificity': float((~r['pred_pos'] & ~r['true_pos'])[k].sum() / max((~r['true_pos'][k]).sum(), 1))}
        res[f'test_{name}_per_patient'] = per_pat
        if name == 'ensemble5':
            np.savez_compressed(os.path.join(OUT, 'test_pred_ensemble.npz'), idx=t_idx, mask=mp, cls=cp,
                                dice=r['dice'])
    # trivial baseline: always predict "no nerve"
    empty = per_image(np.zeros_like(t_true), t_true)
    res['test_baseline_always_empty_kaggle_mean_dice'] = float(empty['dice'].mean())
    json.dump(res, open(os.path.join(OUT, 'metrics.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if 'per_patient' not in k}, indent=1))


if __name__ == '__main__':
    main()

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
from nerveseg.sites import load_all, native_truth
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


def frame_stats(prob_small, cls, refs):
    """Rule-independent per-frame quantities at each frame's native resolution.

    For every frame: area of the largest predicted region, Dice and centroid error if that region
    is kept, whether the truth is non-empty, and the area scale relative to 420x580.
    """
    n = len(refs)
    out = {k: np.zeros(n) for k in ('area', 'dice_kept', 'cerr_kept', 'scale')}
    out['true_pos'] = np.zeros(n, bool)
    for i, ref in enumerate(refs):
        truth = native_truth(ref)
        h, w = truth.shape
        t = torch.from_numpy(prob_small[i].astype(np.float32))[None, None]
        prob = F.interpolate(t, (h, w), mode='bilinear', align_corners=False)[0, 0].numpy()
        comp, area = largest_components(prob[None])
        comp, area = comp[0], area[0]
        ts = truth.sum()
        out['area'][i], out['scale'][i], out['true_pos'][i] = area, h * w / (ORIG_H * ORIG_W), ts > 0
        out['dice_kept'][i] = 2 * (comp & truth).sum() / max(comp.sum() + ts, 1) if area else (1.0 if ts == 0 else 0.0)
        out['cerr_kept'][i] = (np.linalg.norm(np.array(ndimage.center_of_mass(comp)) - ndimage.center_of_mass(truth))
                               if area and ts else np.nan)
    out['cls'] = np.asarray(cls, float)
    return out


def apply_rule_stats(st, t_cls, min_area):
    keep = (st['cls'] >= t_cls) & (st['area'] >= min_area * st['scale']) & (st['area'] > 0)
    dice = np.where(keep, st['dice_kept'], np.where(st['true_pos'], 0.0, 1.0))
    cerr = np.where(keep & st['true_pos'], st['cerr_kept'], np.nan)
    return dict(dice=dice, pred_pos=keep, true_pos=st['true_pos'], cerr=cerr)


def contradictory_set(pos, keep):
    pairs = np.load('cache/dup_pairs.npy')
    sel = pairs[pairs[:, 2] < DUP_THRESHOLD]
    a, b = sel[:, 0].astype(int), sel[:, 1].astype(int)
    c = keep[a] & keep[b] & (pos[a] != pos[b])
    return set(a[c]) | set(b[c])


def main():
    os.makedirs(OUT, exist_ok=True)
    d = load_all('AB')
    subj, site, fold, refs = d['subj'], d['site'], d['fold'], d['ref']
    pos = d['area'] > 0

    # 1. pooled out-of-fold predictions from both sites -> tune the decision rule
    oof = [np.load(f) for f in sorted(glob.glob('runs/fold*/val_pred.npz'))]
    assert len(oof) == 5, f'expected 5 folds, found {len(oof)}'
    d_idx = np.concatenate([o['idx'] for o in oof])
    dev = frame_stats(np.concatenate([o['mask'] for o in oof]), np.concatenate([o['cls'] for o in oof]),
                      [refs[i] for i in d_idx])
    grid = []
    for t in np.round(np.arange(0.3, 0.81, 0.05), 2):
        for a in (0, 1000, 2000, 3000, 4000, 5000):
            grid.append((float(apply_rule_stats(dev, t, a)['dice'].mean()), float(t), int(a)))
    best = max(grid)
    t_cls, min_area = best[1], best[2]
    res = {'decision_rule': {'t_cls': t_cls, 'min_area_px_full_res': min_area, 'oof_kaggle_mean_dice': best[0]}}
    for s_id, s_name in ((0, 'A'), (1, 'B')):
        k = site[d_idx] == s_id
        r = apply_rule_stats({kk: v[k] for kk, v in dev.items()}, t_cls, min_area)
        res[f'development_oof_site{s_name}'] = bootstrap(r, dev['cls'][k], subj[d_idx][k])

    # 2. locked test patients of each site: shipped fold-0 model and the 5-fold ensemble
    t_idx = np.where(fold == -1)[0]
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    preds = []
    for f in range(5):
        net = build_model(weights=None).to(device)
        net.load_state_dict(torch.load(f'runs/fold{f}/best.pt', map_location=device))
        preds.append(predict(net, d['imgs'][t_idx], device))
    variants = {'single_fold0': (preds[0][0], preds[0][1]),
                'ensemble5': (np.mean([p[0].astype(np.float32) for p in preds], 0), np.mean([p[1] for p in preds], 0))}
    contra = contradictory_set(pos[site == 0], (fold == -1)[site == 0])
    for name, (mp, cp) in variants.items():
        st = frame_stats(mp, cp, [refs[i] for i in t_idx])
        r_all = apply_rule_stats(st, t_cls, min_area)
        for s_id, s_name in ((0, 'A'), (1, 'B')):
            k = site[t_idx] == s_id
            r = {kk: v[k] for kk, v in r_all.items()}
            res[f'test_{name}_site{s_name}'] = bootstrap(r, cp[k], subj[t_idx][k])
            res[f'test_site{s_name}_patients'] = sorted(set(int(x) for x in subj[t_idx][k]))
            res[f'test_site{s_name}_n_images'] = int(k.sum())
            res[f'test_site{s_name}_positive_rate'] = float(pos[t_idx][k].mean())
            res[f'test_site{s_name}_baseline_always_empty'] = float((~r['true_pos']).mean())
            if s_id == 0:
                keep = np.array([i not in contra for i in t_idx[k]])
                res[f'test_{name}_siteA_consistent_labels_only'] = bootstrap(
                    {kk: v[keep] for kk, v in r.items()}, cp[k][keep], subj[t_idx][k][keep])
                res['n_siteA_test_images_in_contradictory_pairs'] = int((~keep).sum())
        if name == 'ensemble5':
            np.savez_compressed(os.path.join(OUT, 'test_pred_ensemble.npz'), idx=t_idx, mask=mp, cls=cp,
                                dice=r_all['dice'], site=site[t_idx])
    json.dump(res, open(os.path.join(OUT, 'metrics.json'), 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()

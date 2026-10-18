"""Run the frozen shipped model on the external dataset, per docs/EXTERNAL_VALIDATION_PROTOCOL.md."""
import glob
import json
import os

import numpy as np
import onnxruntime as ort
from PIL import Image
from skimage.draw import polygon as fill_poly
from skimage.morphology import convex_hull_image

from nerveseg.browser_preprocess import area_downscale
from nerveseg.data import H, ORIG_H, ORIG_W, W
from nerveseg.evaluate import N_BOOT, postprocess, upsample

ROOT = 'downloads_ext'


def load(p):
    b = open(p, 'rb').read()
    for enc in ('utf-8', 'gbk'):
        try:
            return json.loads(b.decode(enc))
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
    raise ValueError(p)


def main():
    cfg = json.load(open('app/config.json'))
    rule = cfg['decision_rule']
    sess = ort.InferenceSession(b''.join(open('app/' + p, 'rb').read() for p in cfg['model_parts']))
    rows = []
    for p in sorted(glob.glob(f'{ROOT}/train_mask/json_train/*.json')):
        j = load(p)
        name = os.path.basename(p)[:-5]
        rgb = np.asarray(Image.open(f'{ROOT}/train/JPEGImages/{name}.jpg').convert('RGB')).astype(np.float64)
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        h, w = gray.shape
        x = np.round(area_downscale(gray, H, W)).astype(np.float32)
        m, c = sess.run(None, {'image': x[None, None]})
        import torch, torch.nn.functional as F
        prob = F.interpolate(torch.from_numpy(m), (h, w), mode='bilinear', align_corners=False)[0, 0].numpy()
        min_area = rule['min_area_px_full_res'] * (w * h) / (ORIG_W * ORIG_H)
        pred = postprocess(prob[None], np.array([c[0, 0]]), rule['t_cls'], min_area)[0]
        nerve = np.zeros((h, w), bool)
        for s in j['shapes']:
            if s['label'] == 'shenjing':
                q = np.array(s['points'])
                rr, cc = fill_poly(q[:, 1], q[:, 0], (h, w))
                nerve[rr, cc] = True
        hull = convex_hull_image(nerve) if nerve.any() else nerve
        row = {'patient': name.split('_')[0], 'presence': float(c[0, 0]), 'pred_pos': bool(pred.any()),
               'true_pos': bool(nerve.any()), 'hit': bool((pred & nerve).any()),
               'centroid_in_hull': None, 'dice_hull': None}
        if pred.any() and nerve.any():
            yy, xx = np.nonzero(pred)
            row['centroid_in_hull'] = bool(hull[int(round(yy.mean())), int(round(xx.mean()))])
            row['dice_hull'] = float(2 * (pred & hull).sum() / (pred.sum() + hull.sum()))
        rows.append(row)
    json.dump(rows, open('results/external_rows.json', 'w'))

    pts = np.array([r['patient'] for r in rows])
    def stats(idx):
        R = [rows[i] for i in idx]
        pos = [r for r in R if r['true_pos']]
        neg = [r for r in R if not r['true_pos']]
        det = [r for r in pos if r['pred_pos']]
        f = lambda xs: float(np.mean(xs)) if xs else np.nan
        return {'sensitivity': f([r['pred_pos'] for r in pos]), 'specificity': f([not r['pred_pos'] for r in neg]),
                'hit_rate_nerve_frames': f([r['hit'] for r in pos]),
                'centroid_in_hull_detected': f([r['centroid_in_hull'] for r in det]),
                'dice_hull_detected': f([r['dice_hull'] for r in det])}
    point = stats(range(len(rows)))
    groups = {p: np.where(pts == p)[0] for p in np.unique(pts)}
    rng = np.random.default_rng(0)
    reps = [stats(np.concatenate([groups[p] for p in rng.choice(list(groups), len(groups))])) for _ in range(N_BOOT)]
    res = {'n_frames': len(rows), 'n_patients': len(groups), 'n_nerve_frames': sum(r['true_pos'] for r in rows),
           'protocol': 'docs/EXTERNAL_VALIDATION_PROTOCOL.md'}
    for k, v in point.items():
        res[k] = {'estimate': v, 'ci95': [float(np.nanpercentile([r[k] for r in reps], 2.5)),
                                          float(np.nanpercentile([r[k] for r in reps], 97.5))]}
    pres = np.array([r['presence'] for r in rows]); tp = np.array([r['true_pos'] for r in rows])
    res['presence_prob_median_nerve_frames'] = float(np.median(pres[tp]))
    res['presence_prob_median_nerve_free_frames'] = float(np.median(pres[~tp]))
    json.dump(res, open('results/external_metrics.json', 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()

"""Export the shipped model to ONNX for the browser app, with a parity check and app config.

The shipped model is the fold-0 model: a single network whose locked-test performance at each
site is reported in results/metrics.json (test_single_fold0_site*), so the app runs exactly
what was validated.
"""
import json
import os

import numpy as np
import onnxruntime as ort
import torch
from skimage.io import imsave

import shutil

from skimage.io import imread

from nerveseg.data import H, ORIG_H, ORIG_W, W, list_pairs
from nerveseg.sites import ext_gray, load_all, native_truth
from nerveseg.model import Exportable, build_model

APP = 'app'
PART_BYTES = 12_000_000  # split the model so each file stays under static-host size limits


def main():
    net = build_model(weights=None)
    net.load_state_dict(torch.load('runs/fold0/best.pt', map_location='cpu'))
    model = Exportable(net).eval()
    x = torch.rand(1, 1, H, W) * 255
    path = os.path.join(APP, 'model.onnx')
    torch.onnx.export(model, (x,), path, input_names=['image'], output_names=['mask', 'presence'],
                      opset_version=17, dynamo=False)
    # parity: ONNX Runtime vs PyTorch on real held-out images from both sites
    d = load_all('AB')
    t_idx = np.where(d['fold'] == -1)[0]
    sess = ort.InferenceSession(path)
    worst = 0.0
    for i in t_idx[::40]:
        xi = d['imgs'][i][None, None].astype(np.float32)
        om, oc = sess.run(None, {'image': xi})
        with torch.no_grad():
            tm, tc = model(torch.from_numpy(xi))
        worst = max(worst, float(np.abs(om - tm.numpy()).max()), float(np.abs(oc - tc.numpy()).max()))
    assert worst < 1e-3, worst
    data = open(path, 'rb').read()
    parts = []
    for k in range(0, len(data), PART_BYTES):
        name = f'model.part{len(parts)}.bin'
        open(os.path.join(APP, name), 'wb').write(data[k:k + PART_BYTES])
        parts.append(name)
    os.remove(path)

    # samples: locked test patients only, from both sites, a spread of positives and negatives
    res = json.load(open('results/metrics.json'))
    rng = np.random.default_rng(3)
    pos = d['area'] > 0
    chosen = []
    for s_id, n_pos, n_neg in ((0, 3, 3), (1, 3, 1)):
        k = t_idx[d['site'][t_idx] == s_id]
        chosen += rng.choice(k[pos[k]], n_pos, replace=False).tolist() + rng.choice(k[~pos[k]], n_neg, replace=False).tolist()
    sdir = os.path.join(APP, 'samples')
    shutil.rmtree(sdir, ignore_errors=True)
    os.makedirs(sdir)
    samples = []
    for i in chosen:
        ref = d['ref'][i]
        truth = native_truth(ref)
        if ref[0] == 'A':
            img = imread({(a, b): ip for a, b, ip, _ in list_pairs('downloads')}[(ref[1], ref[2])])
            sid, patient, frame, site = f'A_{ref[1]}_{ref[2]}', ref[1], str(ref[2]), 'Site A'
        else:
            img = np.round(ext_gray(ref[1])).astype(np.uint8)
            pt, fr = ref[1].split('_')
            sid, patient, frame, site = f'B_{ref[1]}', int(pt), fr, 'Site B'
        imsave(os.path.join(sdir, f'{sid}.png'), img, check_contrast=False)
        imsave(os.path.join(sdir, f'{sid}_mask.png'), (truth * 255).astype(np.uint8), check_contrast=False)
        samples.append({'id': sid, 'site': site, 'patient': patient, 'frame': frame, 'labelled_nerve': bool(pos[i])})
    keys = [k for k in res if k.startswith(('test_single_fold0', 'development_oof'))]
    cfg = {'model_parts': parts, 'input': {'height': H, 'width': W},
           'original': {'height': ORIG_H, 'width': ORIG_W},
           'decision_rule': res['decision_rule'], 'onnx_parity_max_abs_diff': worst,
           'validation': {k: res[k] for k in keys},
           'sites': {s: {'patients': len(res[f'test_site{s}_patients']), 'images': res[f'test_site{s}_n_images'],
                         'baseline_always_empty': res[f'test_site{s}_baseline_always_empty']} for s in 'AB'},
           'samples': samples}
    json.dump(cfg, open(os.path.join(APP, 'config.json'), 'w'), indent=1)
    print('parity max abs diff', worst, 'parts', parts, 'samples', [s['id'] for s in samples])


if __name__ == '__main__':
    main()

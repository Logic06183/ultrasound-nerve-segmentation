"""Export the shipped model to ONNX for the browser app, with a parity check and app config.

The shipped model is the fold-0 model: a single network whose locked-test performance is
reported separately in results/metrics.json (test_single_fold0), so the app runs exactly
what was validated.
"""
import json
import os

import numpy as np
import onnxruntime as ort
import torch
from skimage.io import imsave

from nerveseg.data import H, ORIG_H, ORIG_W, W, load_cache, patient_split
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
    # parity: ONNX Runtime vs PyTorch on real held-out images
    imgs, masks, meta, area = load_cache()
    fold = np.array([patient_split(meta[:, 0])[int(s)] for s in meta[:, 0]])
    t_idx = np.where(fold == -1)[0]
    sess = ort.InferenceSession(path)
    worst = 0.0
    for i in t_idx[::60]:
        xi = imgs[i][None, None].astype(np.float32)
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

    # samples: held-out test patients only; a spread of positives and negatives
    res = json.load(open('results/metrics.json'))
    pred = np.load('results/test_pred_ensemble.npz')
    dice = dict(zip(pred['idx'].tolist(), pred['dice'].tolist()))
    rng = np.random.default_rng(3)
    pos_i = [i for i in t_idx if area[i] > 0]
    neg_i = [i for i in t_idx if area[i] == 0]
    chosen = sorted(rng.choice(pos_i, 5, replace=False).tolist() + rng.choice(neg_i, 3, replace=False).tolist())
    os.makedirs(os.path.join(APP, 'samples'), exist_ok=True)
    from nerveseg.evaluate import full_res_masks
    from skimage.io import imread
    from nerveseg.data import list_pairs
    lookup = {(s, k): ip for s, k, ip, _ in list_pairs('downloads')}
    gt = full_res_masks(np.array(chosen), meta)
    samples = []
    for j, i in enumerate(chosen):
        s, k = int(meta[i, 0]), int(meta[i, 1])
        imsave(os.path.join(APP, 'samples', f'{s}_{k}.png'), imread(lookup[(s, k)]), check_contrast=False)
        imsave(os.path.join(APP, 'samples', f'{s}_{k}_mask.png'), (gt[j] * 255).astype(np.uint8), check_contrast=False)
        samples.append({'id': f'{s}_{k}', 'patient': s, 'frame': k, 'labelled_nerve': bool(area[i] > 0)})
    cfg = {'model_parts': parts, 'input': {'height': H, 'width': W},
           'original': {'height': ORIG_H, 'width': ORIG_W},
           'decision_rule': res['decision_rule'], 'onnx_parity_max_abs_diff': worst,
           'validation': {k: res[k] for k in ('test_single_fold0', 'test_single_fold0_consistent_labels_only',
                                              'test_ensemble5', 'development_oof')},
           'test_patients': res['test_patients'], 'n_test_images': res['n_test_images'],
           'baseline_always_empty': res['test_baseline_always_empty_kaggle_mean_dice'],
           'samples': samples}
    json.dump(cfg, open(os.path.join(APP, 'config.json'), 'w'), indent=1)
    print('parity max abs diff', worst, 'parts', parts)


if __name__ == '__main__':
    main()

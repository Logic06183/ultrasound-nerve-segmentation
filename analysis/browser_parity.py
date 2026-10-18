"""Does the browser's preprocessing change the shipped model's test results?"""
import json
import numpy as np
import onnxruntime as ort
from skimage.io import imread
from nerveseg.browser_preprocess import area_downscale
from nerveseg.data import H, W, list_pairs, load_cache, patient_split
from nerveseg.evaluate import full_res_masks, per_image, postprocess, upsample

imgs, masks, meta, area = load_cache()
fold = np.array([patient_split(meta[:, 0])[int(s)] for s in meta[:, 0]])
t_idx = np.where(fold == -1)[0]
rule = json.load(open('results/metrics.json'))['decision_rule']
data = b''.join(open(f'app/{p}', 'rb').read() for p in json.load(open('app/config.json'))['model_parts'])
sess = ort.InferenceSession(data)
lookup = {(s, k): ip for s, k, ip, _ in list_pairs('downloads')}
truth = full_res_masks(t_idx, meta)
out = {}
for name, prep in {'training_resize': lambda i: imgs[i].astype(np.float32),
                   'browser_area_average': lambda i: np.round(area_downscale(imread(lookup[tuple(meta[i])]), H, W)).astype(np.float32)}.items():
    ms, cs = [], []
    for i in t_idx:
        m, c = sess.run(None, {'image': prep(i)[None, None]})
        ms.append(m[0, 0]); cs.append(c[0, 0])
    ms, cs = np.stack(ms), np.array(cs)
    r = per_image(postprocess(upsample(ms), cs, rule['t_cls'], rule['min_area_px_full_res']), truth)
    out[name] = dict(cls=cs, r=r)
a, b = out['training_resize'], out['browser_area_average']
res = {'max_abs_presence_diff': float(np.abs(a['cls'] - b['cls']).max()),
       'decision_agreement': float((a['r']['pred_pos'] == b['r']['pred_pos']).mean()),
       'kaggle_dice_training_resize': float(a['r']['dice'].mean()),
       'kaggle_dice_browser': float(b['r']['dice'].mean()),
       'sensitivity_training_resize': float((a['r']['pred_pos'] & a['r']['true_pos']).sum() / a['r']['true_pos'].sum()),
       'sensitivity_browser': float((b['r']['pred_pos'] & b['r']['true_pos']).sum() / b['r']['true_pos'].sum())}
json.dump(res, open('results/browser_parity.json', 'w'), indent=1)
print(json.dumps(res, indent=1))

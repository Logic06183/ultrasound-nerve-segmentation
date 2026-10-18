"""Does the browser's preprocessing reproduce the validated inputs and decisions? Both sites' test frames."""
import json
import numpy as np
import onnxruntime as ort
from skimage.io import imread
from nerveseg.browser_preprocess import area_downscale
from nerveseg.data import H, W, list_pairs
from nerveseg.evaluate import apply_rule_stats, frame_stats
from nerveseg.sites import ext_gray, load_all

d = load_all('AB')
t_idx = np.where(d['fold'] == -1)[0]
rule = json.load(open('results/metrics.json'))['decision_rule']
cfg = json.load(open('app/config.json'))
sess = ort.InferenceSession(b''.join(open(f'app/{p}', 'rb').read() for p in cfg['model_parts']))
lookup = {(s, k): ip for s, k, ip, _ in list_pairs('downloads')}


def browser_input(ref):
    g = imread(lookup[(ref[1], ref[2])]).astype(float) if ref[0] == 'A' else ext_gray(ref[1])
    return np.round(area_downscale(g, H, W)).astype(np.float32)


out = {}
for name, prep in {'validated_cache': lambda i: d['imgs'][i].astype(np.float32),
                   'browser_pipeline': lambda i: browser_input(d['ref'][i])}.items():
    ms, cs = zip(*[sess.run(None, {'image': prep(i)[None, None]}) for i in t_idx])
    ms, cs = np.stack([m[0, 0] for m in ms]), np.array([c[0, 0] for c in cs])
    out[name] = (cs, apply_rule_stats(frame_stats(ms, cs, [d['ref'][i] for i in t_idx]), rule['t_cls'],
                                      rule['min_area_px_full_res']))
(a_c, a), (b_c, b) = out['validated_cache'], out['browser_pipeline']
res = {'n_test_frames': int(len(t_idx)), 'max_abs_presence_diff': float(np.abs(a_c - b_c).max()),
       'decision_agreement': float((a['pred_pos'] == b['pred_pos']).mean()),
       'max_abs_dice_diff': float(np.abs(a['dice'] - b['dice']).max()),
       'note': 'Site B frames are JPEG. This check decodes them with Pillow; a browser JPEG decoder may differ slightly.'}
json.dump(res, open('results/browser_parity.json', 'w'), indent=1)
print(json.dumps(res, indent=1))

"""Profile the dataset and quantify contradictory labels between near-duplicate frames."""
import json
import numpy as np
from skimage.transform import resize
from nerveseg.data import load_cache

imgs, masks, meta, area = load_cache()
subj = meta[:, 0]
pos = area > 0
per_subj = {int(s): (int((subj == s).sum()), float(pos[subj == s].mean())) for s in np.unique(subj)}
counts = np.array([v[0] for v in per_subj.values()])
rates = np.array([v[1] for v in per_subj.values()])

# near-duplicate search inside each patient on 28x36 thumbnails, z-scored per image
thumb = np.stack([resize(im, (28, 36), anti_aliasing=True) for im in imgs]).reshape(len(imgs), -1)
thumb = (thumb - thumb.mean(1, keepdims=True)) / (thumb.std(1, keepdims=True) + 1e-6)
pairs = []  # (i, j, dist)
for s in np.unique(subj):
    idx = np.where(subj == s)[0]
    t = thumb[idx]
    d = np.sqrt(np.maximum(((t[:, None] - t[None]) ** 2).mean(-1), 0))
    iu = np.triu_indices(len(idx), 1)
    for a, b, v in zip(idx[iu[0]], idx[iu[1]], d[iu]):
        pairs.append((a, b, v))
pairs = np.array(pairs)
out = {'n_images': int(len(imgs)), 'n_patients': int(len(per_subj)),
       'images_per_patient_min_median_max': [int(counts.min()), float(np.median(counts)), int(counts.max())],
       'positive_rate_overall': float(pos.mean()),
       'positive_rate_by_patient_min_max': [float(rates.min()), float(rates.max())],
       'mask_area_px_full_res_median_iqr': [float(np.median(area[pos])),
                                            float(np.percentile(area[pos], 25)), float(np.percentile(area[pos], 75))],
       'near_duplicates': {}}
for thr in (0.15, 0.2, 0.25, 0.3):
    sel = pairs[pairs[:, 2] < thr]
    a, b = sel[:, 0].astype(int), sel[:, 1].astype(int)
    contra = pos[a] != pos[b]
    out['near_duplicates'][str(thr)] = {'pairs': int(len(sel)), 'contradictory_pairs': int(contra.sum()),
                                        'images_in_contradictory_pairs': int(len(set(a[contra]) | set(b[contra])))}
np.save('cache/dup_pairs.npy', pairs)
json.dump(out, open('analysis/data_profile.json', 'w'), indent=1)
print(json.dumps(out, indent=1))

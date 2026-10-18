"""Load the Kaggle ultrasound nerve data, cache it at model resolution, split by patient."""
import os
import re

import numpy as np
from skimage.io import imread
from skimage.transform import resize

from nerveseg.browser_preprocess import area_downscale

ORIG_H, ORIG_W = 420, 580
H, W = 224, 288  # model resolution, both divisible by 32
CACHE = 'cache'


def list_pairs(root):
    """Return (subject, image_no, image_path, mask_path) sorted by subject then image."""
    out = []
    for name in os.listdir(os.path.join(root, 'image')):
        m = re.fullmatch(r'(\d+)_(\d+)\.tif', name)
        if m:
            subj, idx = int(m.group(1)), int(m.group(2))
            out.append((subj, idx, os.path.join(root, 'image', name),
                        os.path.join(root, 'mask', f'{subj}_{idx}_mask.tif')))
    return sorted(out)


def build_cache(root='downloads', cache=CACHE):
    """Read every TIFF once and cache model-resolution arrays.

    Images use the same area-average downscale as the browser app, so validated and shipped
    preprocessing are identical.
    """
    os.makedirs(cache, exist_ok=True)
    pairs = list_pairs(root)
    n = len(pairs)
    imgs = np.empty((n, H, W), np.uint8)
    masks = np.empty((n, H, W), np.uint8)
    full_area = np.empty(n, np.int32)
    for i, (_, _, ip, mp) in enumerate(pairs):
        img = imread(ip)
        msk = (imread(mp) > 0).astype(np.uint8)
        full_area[i] = msk.sum()
        imgs[i] = np.clip(area_downscale(img, H, W), 0, 255).astype(np.float32).round().astype(np.uint8)
        masks[i] = resize(msk.astype(float), (H, W)) > 0.5
    meta = np.array([(s, k) for s, k, _, _ in pairs], np.int32)
    np.save(os.path.join(cache, 'imgs.npy'), imgs)
    np.save(os.path.join(cache, 'masks.npy'), masks)
    np.save(os.path.join(cache, 'meta.npy'), meta)
    np.save(os.path.join(cache, 'full_area.npy'), full_area)
    return imgs, masks, meta, full_area


def load_cache(cache=CACHE):
    return tuple(np.load(os.path.join(cache, f + '.npy'))
                 for f in ('imgs', 'masks', 'meta', 'full_area'))


def patient_split(subjects, test_frac=0.2, n_folds=5, seed=0):
    """Hold out a locked test set of whole patients, then assign remaining patients to folds.

    Returns a dict subject -> fold, where fold == -1 means locked test.
    """
    rng = np.random.default_rng(seed)
    subs = np.array(sorted(set(int(s) for s in subjects)))
    rng.shuffle(subs)
    n_test = int(round(test_frac * len(subs)))
    assign = {int(s): -1 for s in subs[:n_test]}
    for j, s in enumerate(subs[n_test:]):
        assign[int(s)] = j % n_folds
    return assign

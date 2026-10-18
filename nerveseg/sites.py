"""Second site (alexcolmenero/ultrasonic-brachial-plexus) and the combined two-site dataset."""
import glob
import json
import os

import numpy as np
from PIL import Image
from skimage.draw import polygon as fill_poly
from skimage.io import imread
from skimage.morphology import convex_hull_image
from skimage.transform import resize

from nerveseg.browser_preprocess import area_downscale
from nerveseg.data import CACHE, H, W, list_pairs, load_cache, patient_split

EXT_ROOT = 'downloads_ext'
SITE_B_OFFSET = 1000  # site-B patient ids are stored as 1000 + original id


def _load_json(p):
    b = open(p, 'rb').read()
    for enc in ('utf-8', 'gbk'):
        try:
            return json.loads(b.decode(enc))
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
    raise ValueError(p)


def ext_frames(root=EXT_ROOT):
    return sorted(os.path.basename(p)[:-5] for p in glob.glob(f'{root}/train_mask/json_train/*.json'))


def ext_gray(name, root=EXT_ROOT):
    a = np.asarray(Image.open(f'{root}/train/JPEGImages/{name}.jpg').convert('RGB')).astype(np.float64)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def ext_mask(name, shape, root=EXT_ROOT):
    """Convex hull of all nerve polygons at native resolution (empty if no nerve)."""
    j = _load_json(f'{root}/train_mask/json_train/{name}.json')
    m = np.zeros(shape, bool)
    for s in j['shapes']:
        if s['label'] == 'shenjing':
            q = np.array(s['points'])
            rr, cc = fill_poly(q[:, 1], q[:, 0], shape)
            m[rr, cc] = True
    return convex_hull_image(m) if m.any() else m


def build_ext_cache(cache=CACHE):
    names = ext_frames()
    imgs = np.empty((len(names), H, W), np.uint8)
    masks = np.empty((len(names), H, W), np.uint8)
    area = np.empty(len(names), np.int32)
    subj = np.empty(len(names), np.int32)
    for i, n in enumerate(names):
        g = ext_gray(n)
        m = ext_mask(n, g.shape)
        imgs[i] = np.clip(np.round(area_downscale(g, H, W)), 0, 255)
        masks[i] = resize(m.astype(float), (H, W)) > 0.5
        area[i] = m.sum()
        subj[i] = SITE_B_OFFSET + int(n.split('_')[0])
    np.save(f'{cache}/ext_imgs.npy', imgs)
    np.save(f'{cache}/ext_masks.npy', masks)
    np.save(f'{cache}/ext_meta.npy', subj)
    np.save(f'{cache}/ext_area.npy', area)
    json.dump(names, open(f'{cache}/ext_names.json', 'w'))


def load_all(sites='AB'):
    """Concatenated arrays for the requested sites. Site A rows come first, so indices into site A
    (such as the near-duplicate pairs) stay valid. fold == -1 marks locked test patients."""
    imgs, masks, meta, area = load_cache()
    subj = meta[:, 0]
    fold = np.array([patient_split(subj)[int(s)] for s in subj])
    site = np.zeros(len(subj), np.int8)
    ref = [('A', int(s), int(k)) for s, k in meta]
    if 'B' in sites:
        bi, bm = np.load(f'{CACHE}/ext_imgs.npy'), np.load(f'{CACHE}/ext_masks.npy')
        bs, ba = np.load(f'{CACHE}/ext_meta.npy'), np.load(f'{CACHE}/ext_area.npy')
        bf = np.array([patient_split(bs)[int(s)] for s in bs])
        names = json.load(open(f'{CACHE}/ext_names.json'))
        imgs, masks = np.concatenate([imgs, bi]), np.concatenate([masks, bm])
        subj, area, fold = np.concatenate([subj, bs]), np.concatenate([area, ba]), np.concatenate([fold, bf])
        site = np.concatenate([site, np.ones(len(bs), np.int8)])
        ref += [('B', n) for n in names]
    return dict(imgs=imgs, masks=masks, subj=subj, area=area, fold=fold, site=site, ref=ref)


def native_truth(ref):
    """(native greyscale image shape, native-resolution truth mask) for one row reference."""
    if ref[0] == 'A':
        lookup = native_truth.lookup = getattr(native_truth, 'lookup', None) or \
            {(s, k): mp for s, k, _, mp in list_pairs('downloads')}
        m = imread(lookup[(ref[1], ref[2])]) > 0
        return m
    g = ext_gray(ref[1])
    return ext_mask(ref[1], g.shape)

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('segmentation_models_pytorch')

from nerveseg.data import patient_split
from nerveseg.evaluate import bootstrap, metrics, per_image, postprocess
from nerveseg.train import augment, contradictory_negatives


def test_patient_split_is_disjoint_and_stable():
    subs = np.repeat(np.arange(1, 48), 3)
    a, b = patient_split(subs), patient_split(subs)
    assert a == b
    assert sum(v == -1 for v in a.values()) == 9
    assert set(a.values()) == {-1, 0, 1, 2, 3, 4}


def test_postprocess_keeps_largest_component_and_applies_rule():
    p = np.zeros((3, 20, 20), np.float32)
    p[:, 2:4, 2:4] = 0.9      # small blob, 4 px
    p[:, 10:18, 10:18] = 0.9  # large blob, 64 px
    out = postprocess(p, np.array([0.9, 0.1, 0.9]), t_cls=0.5, min_area=50)
    assert out[0].sum() == 64 and not out[0, 3, 3]
    assert out[1].sum() == 0            # classifier says absent
    out = postprocess(p, np.array([0.9, 0.9, 0.9]), t_cls=0.5, min_area=100)
    assert out.sum() == 0               # largest blob below min area


def test_per_image_and_metrics_on_known_case():
    true = np.zeros((4, 10, 10), bool)
    pred = np.zeros_like(true)
    true[0, :5] = pred[0, :5] = True   # perfect detection
    true[1, :5] = True                 # miss
    pred[2, :5] = True                 # false alarm
    r = per_image(pred, true)          # image 3: both empty -> dice 1
    np.testing.assert_allclose(r['dice'], [1, 0, 0, 1])
    m = metrics(r, np.array([0.9, 0.2, 0.8, 0.1]))
    assert m['sensitivity'] == 0.5 and m['specificity'] == 0.5
    assert m['kaggle_mean_dice'] == 0.5 and m['auroc_presence'] == 0.75  # 3 of 4 pos-neg pairs ranked correctly


def test_bootstrap_ci_brackets_estimate():
    rng = np.random.default_rng(0)
    true = rng.random((60, 8, 8)) > 0.7
    pred = true.copy()
    pred[:10] = False
    out = bootstrap(per_image(pred, true), rng.random(60), np.repeat(np.arange(6), 10))
    lo, hi = out['kaggle_mean_dice']['ci95']
    assert lo <= out['kaggle_mean_dice']['estimate'] <= hi


def test_contradictory_negatives_only_drops_unlabelled_frame():
    pos = np.array([True, False, False])
    pairs = np.array([[0, 1, 0.01], [1, 2, 0.01], [0, 2, 0.5]])
    assert contradictory_negatives(pos, pairs, np.ones(3, bool)).tolist() == [1]


def test_augment_keeps_masks_binary_and_shapes():
    x = torch.rand(4, 1, 32, 48) * 255
    y = (torch.rand(4, 1, 32, 48) > 0.5).float()
    xa, ya = augment(x, y)
    assert xa.shape == x.shape and set(ya.unique().tolist()) <= {0.0, 1.0}
    assert float(xa.min()) >= 0 and float(xa.max()) <= 255


def test_frame_stats_matches_postprocess_path(monkeypatch):
    import nerveseg.evaluate as ev
    rng = np.random.default_rng(1)
    n = 12
    truths = [(rng.random((42, 58)) > 0.995) for _ in range(n)]
    for t in truths[:8]:
        t[10:25, 20:40] = True
    small = rng.random((n, 14, 18)).astype(np.float32) ** 3
    small[::2, 3:9, 5:12] = 0.9
    cls = rng.random(n)
    monkeypatch.setattr(ev, 'native_truth', lambda ref: truths[ref])
    monkeypatch.setattr(ev, 'ORIG_H', 42)
    monkeypatch.setattr(ev, 'ORIG_W', 58)
    st = ev.frame_stats(small, cls, list(range(n)))
    fast = ev.apply_rule_stats(st, 0.4, 30)
    t = torch.from_numpy(small)[:, None]
    full = torch.nn.functional.interpolate(t, (42, 58), mode='bilinear', align_corners=False)[:, 0].numpy()
    slow = per_image(postprocess(full, cls, 0.4, 30), np.stack(truths))
    np.testing.assert_allclose(fast['dice'], slow['dice'])
    np.testing.assert_array_equal(fast['pred_pos'], slow['pred_pos'])


def test_browser_downscale_js_matches_python(tmp_path):
    """Run the app's own JavaScript downscale under Node on awkward sizes and compare with Python."""
    import json
    import re
    import shutil
    import subprocess
    from nerveseg.browser_preprocess import area_downscale
    if not shutil.which('node'):
        pytest.skip('node not installed')
    html = open('app/index.html').read()
    fn = re.search(r'  function downscale\(.*?\n  }\n', html, re.S).group(0)
    rng = np.random.default_rng(0)
    for h, w in ((420, 580), (468, 604), (602, 563), (605, 563), (225, 289)):
        g = rng.integers(0, 256, (h, w)).astype(float)
        script = fn + f'const g = {json.dumps(g.ravel().tolist())};\n' \
            f'process.stdout.write(JSON.stringify(Array.from(downscale(g, {w}, {h}, 288, 224))));'
        (tmp_path / 'd.js').write_text(script)
        js = np.array(json.loads(subprocess.run(['node', str(tmp_path / 'd.js')], capture_output=True,
                                                text=True, check=True).stdout)).reshape(224, 288)
        np.testing.assert_array_equal(js, np.round(area_downscale(g, 224, 288)), err_msg=f'{h}x{w}')

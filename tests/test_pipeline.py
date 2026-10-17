import os

import numpy as np
import pytest
from skimage.io import imsave

import data
from submission import prep, run_length_enc


def rle_decode(rle, shape):
    """Inverse of run_length_enc: Kaggle format, 1-indexed, column-major."""
    flat = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    nums = [int(n) for n in rle.split()]
    for start, length in zip(nums[0::2], nums[1::2]):
        flat[start - 1:start - 1 + length] = 1
    return flat.reshape(shape[1], shape[0]).T


def test_rle_round_trip():
    mask = np.zeros((data.image_rows, data.image_cols), dtype=np.uint8)
    mask[100:150, 200:260] = 1
    mask[300:310, 10:20] = 1
    np.testing.assert_array_equal(rle_decode(run_length_enc(mask), mask.shape), mask)


def test_rle_treats_tiny_masks_as_empty():
    mask = np.zeros((data.image_rows, data.image_cols), dtype=np.uint8)
    mask[0, :5] = 1
    assert run_length_enc(mask) == ''


def test_prep_restores_original_resolution():
    pred = np.zeros((96, 96), dtype=np.float32)
    pred[20:40, 30:60] = 0.9
    out = prep(pred)
    assert out.shape == (data.image_rows, data.image_cols)
    assert set(np.unique(out)) <= {0, 1}
    assert out.sum() > 0


def test_create_train_and_test_data(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    shape = (data.image_rows, data.image_cols)
    (tmp_path / 'raw' / 'train').mkdir(parents=True)
    (tmp_path / 'raw' / 'test').mkdir(parents=True)
    for name in ('1_1', '1_2', '2_1'):
        imsave(str(tmp_path / 'raw' / 'train' / (name + '.tif')),
               rng.integers(0, 256, shape, dtype=np.uint8), check_contrast=False)
        imsave(str(tmp_path / 'raw' / 'train' / (name + '_mask.tif')),
               (rng.random(shape) > 0.5).astype(np.uint8) * 255, check_contrast=False)
    for img_id in (3, 1, 2):
        imsave(str(tmp_path / 'raw' / 'test' / ('%d.tif' % img_id)),
               rng.integers(0, 256, shape, dtype=np.uint8), check_contrast=False)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(data, 'data_path', 'raw/')
    data.create_train_data()
    data.create_test_data()

    imgs, masks = data.load_train_data()
    assert imgs.shape == masks.shape == (3,) + shape
    assert set(np.unique(masks)) <= {0, 255}

    imgs_test, ids = data.load_test_data()
    assert imgs_test.shape == (3,) + shape
    assert sorted(ids.tolist()) == [1, 2, 3]


def test_unet_shapes_and_dice():
    pytest.importorskip('keras')
    import train

    model = train.get_unet()
    x = np.zeros((2, train.img_rows, train.img_cols, 1), dtype=np.float32)
    assert model.predict(x, verbose=0).shape == x.shape

    y = np.ones((4, 4), dtype=np.float32)
    assert float(train.dice_coef(y, y)) == pytest.approx(1.0)
    assert float(train.dice_coef_loss(y, y)) == pytest.approx(-1.0)


def test_train_predict_and_submit_end_to_end(tmp_path, monkeypatch):
    pytest.importorskip('keras')
    import train
    import submission

    rng = np.random.default_rng(0)
    shape = (data.image_rows, data.image_cols)
    masks = np.zeros((5,) + shape, dtype=np.uint8)
    masks[:, 150:250, 200:350] = 255
    monkeypatch.chdir(tmp_path)
    np.save('imgs_train.npy', rng.integers(0, 256, (5,) + shape, dtype=np.uint8))
    np.save('imgs_mask_train.npy', masks)
    np.save('imgs_test.npy', rng.integers(0, 256, (3,) + shape, dtype=np.uint8))
    np.save('imgs_id_test.npy', np.array([3, 1, 2], dtype=np.int32))

    train.train_and_predict(epochs=1, batch_size=2)
    assert np.load('imgs_mask_test.npy').shape == (3, train.img_rows, train.img_cols, 1)
    assert sorted(os.listdir('preds')) == ['1_pred.png', '2_pred.png', '3_pred.png']

    submission.submission()
    lines = open('submission.csv').read().splitlines()
    assert lines[0] == 'img,pixels'
    assert [line.split(',')[0] for line in lines[1:]] == ['1', '2', '3']

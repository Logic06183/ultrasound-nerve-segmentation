"""Train one model on development patients, holding out one fold of patients for validation.

The locked test patients (fold -1) are never loaded here.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from nerveseg.sites import load_all
from nerveseg.model import PIX_MEAN, PIX_STD, build_model

DUP_THRESHOLD = 0.1  # thumbnail distance below which two frames count as the same view


def contradictory_negatives(pos, pairs, keep):
    """Indices of unlabelled frames that are near-duplicates of a labelled frame (within `keep`)."""
    sel = pairs[pairs[:, 2] < DUP_THRESHOLD]
    a, b = sel[:, 0].astype(int), sel[:, 1].astype(int)
    both = keep[a] & keep[b]
    a, b = a[both], b[both]
    drop = set(a[pos[b] & ~pos[a]]) | set(b[pos[a] & ~pos[b]])
    return np.array(sorted(drop), int)


def augment(x, y):
    """GPU augmentation: flip, small affine, gain/gamma. x uint8-range float, y in {0,1}."""
    n = x.shape[0]
    dev = x.device
    flip = (torch.rand(n, device=dev) < 0.5).float() * 2 - 1
    ang = (torch.rand(n, device=dev) - 0.5) * math.radians(20)
    sc = 1 + (torch.rand(n, device=dev) - 0.5) * 0.2
    tx, ty = [(torch.rand(n, device=dev) - 0.5) * 0.16 for _ in range(2)]
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos * flip, -sin, tx], 1), torch.stack([sin * flip, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    y = F.grid_sample(y, grid, mode='nearest', padding_mode='zeros', align_corners=False)
    gamma = torch.exp((torch.rand(n, 1, 1, 1, device=dev) - 0.5) * 0.5)
    gain = 1 + (torch.rand(n, 1, 1, 1, device=dev) - 0.5) * 0.3
    x = (255 * (x / 255).clamp(0, 1) ** gamma * gain).clamp(0, 255)
    return x, y


def loss_fn(m_logit, c_logit, y):
    present = (y.flatten(1).sum(1) > 0).float()
    bce = F.binary_cross_entropy_with_logits(m_logit, y)
    p = torch.sigmoid(m_logit).flatten(1)
    t = y.flatten(1)
    dice = 1 - (2 * (p * t).sum(1) + 1) / (p.sum(1) + t.sum(1) + 1)
    dice = (dice * present).sum() / present.sum().clamp(min=1)  # Dice only where a nerve is labelled
    cls = F.binary_cross_entropy_with_logits(c_logit[:, 0], present)
    return bce + dice + 0.5 * cls


@torch.no_grad()
def predict(model, imgs, dev, bs=64):
    model.eval()
    ms, cs = [], []
    for i in range(0, len(imgs), bs):
        x = torch.from_numpy(imgs[i:i + bs]).float().unsqueeze(1).to(dev)
        m, c = model((x - PIX_MEAN) / PIX_STD)
        ms.append(torch.sigmoid(m)[:, 0].cpu().numpy().astype(np.float16))
        cs.append(torch.sigmoid(c)[:, 0].cpu().numpy())
    return np.concatenate(ms), np.concatenate(cs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val-fold', type=int, default=0)
    ap.add_argument('--encoder', default='mobilenet_v2')
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--no-clean', action='store_true', help='keep contradictory negatives in training')
    ap.add_argument('--max-batches', type=int, default=0, help='debug: cap batches per epoch')
    ap.add_argument('--sites', default='AB', help='A = Kaggle only, AB = both sites')
    ap.add_argument('--out', default='runs/default')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'

    d = load_all(a.sites)
    imgs, masks, area, fold, subj = d['imgs'], d['masks'], d['area'], d['fold'], d['subj']
    pos = area > 0
    tr = (fold >= 0) & (fold != a.val_fold)
    va = fold == a.val_fold
    dropped = np.array([], int) if a.no_clean else contradictory_negatives(pos, np.load('cache/dup_pairs.npy'), tr)
    tr_idx = np.setdiff1d(np.where(tr)[0], dropped)
    va_idx = np.where(va)[0]
    print(f'sites {a.sites}: train {len(tr_idx)} imgs ({len(dropped)} contradictory negatives dropped), '
          f'val {len(va_idx)} imgs from {len(set(subj[va_idx].tolist()))} patients', flush=True)

    model = build_model(a.encoder).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * (len(tr_idx) // a.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)
    X = torch.from_numpy(imgs[tr_idx]).unsqueeze(1)
    Y = torch.from_numpy(masks[tr_idx]).unsqueeze(1)
    Xv = torch.from_numpy(imgs[va_idx]).float().unsqueeze(1).to(dev)
    Yv = torch.from_numpy(masks[va_idx]).float().unsqueeze(1).to(dev)
    best, hist = 1e9, []
    for ep in range(a.epochs):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        perm = torch.randperm(len(tr_idx))
        for i in range(0, len(perm) - a.batch_size + 1, a.batch_size):
            b = perm[i:i + a.batch_size]
            x, y = augment(X[b].float().to(dev), Y[b].float().to(dev))
            m, c = model((x - PIX_MEAN) / PIX_STD)
            loss = loss_fn(m, c, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
            nb += 1
            if a.max_batches and nb >= a.max_batches:
                break
        model.eval()
        with torch.no_grad():
            vl = []
            for i in range(0, len(Xv), 64):
                m, c = model((Xv[i:i + 64] - PIX_MEAN) / PIX_STD)
                vl.append(loss_fn(m, c, Yv[i:i + 64]).item() * len(m))
            vloss = sum(vl) / len(Xv)
        hist.append({'epoch': ep + 1, 'train_loss': tot / nb, 'val_loss': vloss, 'sec': time.time() - t0})
        print(json.dumps(hist[-1]), flush=True)
        if vloss < best:
            best = vloss
            torch.save(model.state_dict(), os.path.join(a.out, 'best.pt'))
    model.load_state_dict(torch.load(os.path.join(a.out, 'best.pt')))
    mprob, cprob = predict(model, imgs[va_idx], dev)
    np.savez_compressed(os.path.join(a.out, 'val_pred.npz'), idx=va_idx, mask=mprob, cls=cprob)
    json.dump({'args': vars(a), 'history': hist, 'best_val_loss': best, 'n_dropped': int(len(dropped))},
              open(os.path.join(a.out, 'train_log.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()

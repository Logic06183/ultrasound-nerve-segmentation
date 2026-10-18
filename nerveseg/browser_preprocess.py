"""NumPy port of the browser app's area-average downscale, for parity checks against training."""
import numpy as np


def area_downscale(g, H, W):
    """Exact area-weighted average of a (h, w) image onto an (H, W) grid, as in app/index.html."""
    h, w = g.shape
    def weights(n_in, n_out):
        m = np.zeros((n_out, n_in))
        s = n_in / n_out
        for o in range(n_out):
            a, b = o * s, (o + 1) * s
            for i in range(int(np.floor(a)), int(np.ceil(b))):
                m[o, i] = min(i + 1, b) - max(i, a)
        return m / m.sum(1, keepdims=True)
    return weights(h, H) @ g.astype(np.float64) @ weights(w, W).T

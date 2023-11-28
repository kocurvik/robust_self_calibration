import iterative_focal
import numpy as np


def kukelova_uncal(F, f1_prior, f2_prior,
                   p1_prior=(0.0, 0.0), p2_prior=(0.0, 0.0),
                   w1=5e-4, w2=1.0, w3=5e-4, w4=1.0,
                   return_err=False, iters=50, all_iters=False):

    try:
        f1, u1, v1, f2, u2, v2, iter, err = iterative_focal.focals_from_f(F, f1_prior, p1_prior[0], p1_prior[1], f2_prior, p2_prior[0], p2_prior[1], w1, w2, w3, w4, iters, all_iters)
    except Exception:
        if return_err:
            return f1_prior, f2_prior, p1_prior, p2_prior, np.array([]), np.inf
        return f1_prior, f2_prior, p1_prior, p2_prior

    p1 = np.array([u1, v1])
    p2 = np.array([u2, v2])

    if return_err:
        return f1, f2, p1, p2, np.array(err), iter
    return f1, f2, p1, p2
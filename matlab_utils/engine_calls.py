import numpy as np
import poselib
import matlab

# from methods.base import bougnoux_original


def ours_uncal(eng, F, f1_prior, f2_prior,
               p1_prior=(0.0, 0.0), p2_prior=(0.0, 0.0),
               w1=5e-4, w2=1.0, w3=5e-4, w4=1.0,
               return_err=False, iters=50, all_iters=False):

    # f1_sq, f2_sq = bougnoux_original(F, p1_prior, p2_prior)
    #
    # if f1_sq > 0 and f2_sq > 0:
    #     return np.sqrt(f1_sq), np.sqrt(f2_sq), p1_prior, p2_prior

    try:
        F = matlab.double(F.tolist())
        f1, u1, v1, f2, u2, v2, l1, l2, err, iter, time = eng.f1_f2_from_F(F, float(f1_prior), float(p1_prior[0]), float(p1_prior[1]), float(f2_prior), float(p2_prior[0]), float(p2_prior[1]), float(w1), float(w2), float(w3), float(w4), iters, all_iters, nargout=11)
    except Exception:
        if return_err:
            return f1_prior, f2_prior, p1_prior, p2_prior, np.array([]), np.inf
        return f1_prior, f2_prior, p1_prior, p2_prior

    p1 = np.array([u1, v1])
    p2 = np.array([u2, v2])

    # print("Err: ", err, "\t Iter: ", iter)
    # err = np.array(err)
    # print(iter)
    # print(time * 10**6, "us")

    if return_err:
        return f1, f2, p1, p2, np.array(err), iter
    return f1, f2, p1, p2


def ours_single_uncal(eng, F, f_prior, p_prior=[0.0, 0.0], w1=0.1, w2=1.0):
    f1, f2, p1, p2 = ours_uncal(eng, F, f_prior, f_prior, p1_prior=p_prior, p2_prior=p_prior, w1=w1, w2=w2, w3=w1, w4=w2)
    return (f1 + f2) / 2, (p1 + p2) / 2


def ours_single(eng, F, f_prior, p_prior=(0.0, 0.0), w1=0.01, w2=0.1,
                return_err=False, iters=50, all_iters=False):
    F = matlab.double(F.tolist())

    try:
        f, u, v, l1, l2, err, iter = eng.focal_from_F(F, float(f_prior), float(p_prior[0]), float(p_prior[1]), float(w1), float(w2), iters, all_iters, nargout=7)
    except Exception:
        print("Kukelova Single Exception")
        if return_err:
            return f_prior, p_prior, np.array([]), np.inf
        return f_prior, p_prior

    p = np.array([u, v])

    if return_err:
        return f, p, np.array(err), iter
    return f, p


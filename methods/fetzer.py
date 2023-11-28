import numpy as np
import torch
from scipy.optimize import least_squares
from torch.autograd.functional import jacobian


def get_d(ai, bi, aj, bj, u, v):
    return torch.stack([
        ai[u] * aj[v] - ai[v] * aj[u],
        ai[u] * bj[v] - ai[v] * bj[u],
        bi[u] * aj[v] - bi[v] * aj[u],
        bi[u] * bj[v] - bi[v] * bj[u]
    ])


def fetzer_cost(F):
    u, s, vt = np.linalg.svd(F)

    v_1 = torch.from_numpy(vt[0, :])
    v_2 = torch.from_numpy(vt[1, :])

    v_11, v_12, v_13 = v_1
    v_21, v_22, v_23 = v_2

    u_1 = torch.from_numpy(u[:, 0])
    u_2 = torch.from_numpy(u[:, 1])

    u_11, u_12, u_13 = u_1
    u_21, u_22, u_23 = u_2

    ai = torch.stack([
        s[0] ** 2 * (v_11 ** 2 + v_12 ** 2),
        s[0] * s[1] * (v_11 * v_21 + v_12 * v_22),
        s[1] ** 2 * (v_21 ** 2 + v_22 ** 2)])

    aj = torch.stack([
        u_21 ** 2 + u_22 ** 2,
        -(u_11 * u_21 + u_12 * u_22),
        u_11 ** 2 + u_12 ** 2])

    def _f(x):
        fi = x[0]
        fj = x[1]
        pi = torch.stack([x[2], x[3], torch.scalar_tensor(1.0)])
        pj = torch.stack([x[4], x[5], torch.scalar_tensor(1.0)])

        # pi = torch.stack([torch.scalar_tensor(0.0, dtype=torch.double), torch.scalar_tensor(0.0, dtype=torch.double), torch.scalar_tensor(1.0, dtype=torch.double)])
        # pj = torch.stack([torch.scalar_tensor(0.0, dtype=torch.double), torch.scalar_tensor(0.0, dtype=torch.double), torch.scalar_tensor(1.0, dtype=torch.double)])

        bi = torch.stack([
            s[0] ** 2 * torch.dot(pi, v_1) ** 2,
            s[0] * s[1] * torch.dot(pi, v_1) * torch.dot(pi, v_2),
            s[1] ** 2 * torch.dot(pi, v_2) ** 2])

        bj = torch.stack([
            torch.dot(pj, u_2) ** 2,
            -torch.dot(pj, u_1) * torch.dot(pj, u_2),
            torch.dot(pj, u_1) ** 2])

        d_12 = get_d(ai, bi, aj, bj, 1, 0)
        d_13 = get_d(ai, bi, aj, bj, 0, 2)
        d_23 = get_d(ai, bi, aj, bj, 2, 1)

        K1_12 = - (fj ** 2 * d_12[2] + d_12[3]) / (fj ** 2 * d_12[0] + d_12[1])
        K2_12 = - (fi ** 2 * d_12[1] + d_12[3]) / (fi ** 2 * d_12[0] + d_12[2])

        K1_13 = - (fj ** 2 * d_13[2] + d_13[3]) / (fj ** 2 * d_13[0] + d_13[1])
        K2_13 = - (fi ** 2 * d_13[1] + d_13[3]) / (fi ** 2 * d_13[0] + d_13[2])

        K1_23 = - (fj ** 2 * d_23[2] + d_23[3]) / (fj ** 2 * d_23[0] + d_23[1])
        K2_23 = - (fi ** 2 * d_23[1] + d_23[3]) / (fi ** 2 * d_23[0] + d_23[2])

        residuals = torch.stack([
            (fi ** 2 - K1_12) / (fi ** 2),
            (fj ** 2 - K2_12) / (fj ** 2),
            (fi ** 2 - K1_13) / (fi ** 2),
            (fj ** 2 - K2_13) / (fj ** 2),
            (fi ** 2 - K1_23) / (fi ** 2),
            (fj ** 2 - K2_23) / (fj ** 2),
        ])

        # print(residuals)

        return residuals

    return _f

def fetzer(F, fi_prior, fj_prior, pi=(0.0, 0.0), pj=(0.0, 0.0)):
    x_init = np.array([fi_prior, fj_prior, *pi, *pj])
    # x_init = np.array([fi_prior, fj_prior])

    cost_function = fetzer_cost(F)
    np_cost = lambda x: cost_function(torch.from_numpy(x)).numpy()
    np_jac = lambda x: jacobian(cost_function, torch.from_numpy(x), strict=False, vectorize=True).numpy()

    res = least_squares(np_cost, x_init, jac=np_jac, method='lm', max_nfev=100, verbose=0)

    fi, fj, pi_1, pi_2, pj_1, pj_2 = res.x

    return fi, fj, np.array([pi_1, pi_2]), np.array([pj_1, pj_2]), res.nfev



def fetzer_cost_focal_only(F):
    u, s, vt = np.linalg.svd(F)

    v_1 = torch.from_numpy(vt[0, :])
    v_2 = torch.from_numpy(vt[1, :])

    v_11, v_12, v_13 = v_1
    v_21, v_22, v_23 = v_2

    u_1 = torch.from_numpy(u[:, 0])
    u_2 = torch.from_numpy(u[:, 1])

    u_11, u_12, u_13 = u_1
    u_21, u_22, u_23 = u_2

    pi = torch.tensor([0.0, 0.0, 1.0], dtype=torch.double)
    pj = torch.tensor([0.0, 0.0, 1.0], dtype=torch.double)

    ai = torch.stack([
        s[0] ** 2 * (v_11 ** 2 + v_12 ** 2),
        s[0] * s[1] * (v_11 * v_21 + v_12 * v_22),
        s[1] ** 2 * (v_21 ** 2 + v_22 ** 2)])

    aj = torch.stack([
        u_21 ** 2 + u_22 ** 2,
        -(u_11 * u_21 + u_12 * u_22),
        u_11 ** 2 + u_12 ** 2])

    bi = torch.stack([
        s[0] ** 2 * torch.dot(pi, v_1) ** 2,
        s[0] * s[1] * torch.dot(pi, v_1) * torch.dot(pi, v_2),
        s[1] ** 2 * torch.dot(pi, v_2) ** 2])

    bj = torch.stack([
        torch.dot(pj, u_2) ** 2,
        -torch.dot(pj, u_1) * torch.dot(pj, u_2),
        torch.dot(pj, u_1) ** 2])

    d_12 = get_d(ai, bi, aj, bj, 1, 0)
    d_13 = get_d(ai, bi, aj, bj, 0, 2)
    d_23 = get_d(ai, bi, aj, bj, 2, 1)

    def _f(x):
        fi = x[0]
        fj = x[1]

        K1_12 = - (fj ** 2 * d_12[2] + d_12[3]) / (fj ** 2 * d_12[0] + d_12[1])
        K2_12 = - (fi ** 2 * d_12[1] + d_12[3]) / (fi ** 2 * d_12[0] + d_12[2])

        K1_13 = - (fj ** 2 * d_13[2] + d_13[3]) / (fj ** 2 * d_13[0] + d_13[1])
        K2_13 = - (fi ** 2 * d_13[1] + d_13[3]) / (fi ** 2 * d_13[0] + d_13[2])

        K1_23 = - (fj ** 2 * d_23[2] + d_23[3]) / (fj ** 2 * d_23[0] + d_23[1])
        K2_23 = - (fi ** 2 * d_23[1] + d_23[3]) / (fi ** 2 * d_23[0] + d_23[2])

        residuals = torch.stack([
            (fi ** 2 - K1_12) / (fi ** 2),
            (fj ** 2 - K2_12) / (fj ** 2),
            (fi ** 2 - K1_13) / (fi ** 2),
            (fj ** 2 - K2_13) / (fj ** 2),
            (fi ** 2 - K1_23) / (fi ** 2),
            (fj ** 2 - K2_23) / (fj ** 2),
        ])

        # print(residuals)

        return residuals

    return _f

def fetzer_focal_only(F, fi_prior, fj_prior, pi=(0.0, 0.0), pj=(0.0, 0.0)):
    try:
        x_init = np.array([fi_prior, fj_prior])

        cost_function = fetzer_cost_focal_only(F)
        np_cost = lambda x: cost_function(torch.from_numpy(x)).numpy()
        np_jac = lambda x: jacobian(cost_function, torch.from_numpy(x), strict=True).numpy()

        res = least_squares(np_cost, x_init, jac=np_jac, method='lm', max_nfev=100, verbose=0)

        fi, fj = res.x

        return fi, fj, res.nfev
    except Exception:
        return fi_prior, fj_prior, 0

def fetzer_cost_single(F):
    u, s, vt = np.linalg.svd(F)

    v_1 = torch.from_numpy(vt[0, :])
    v_2 = torch.from_numpy(vt[1, :])

    v_11, v_12, v_13 = v_1
    v_21, v_22, v_23 = v_2

    u_1 = torch.from_numpy(u[:, 0])
    u_2 = torch.from_numpy(u[:, 1])

    u_11, u_12, u_13 = u_1
    u_21, u_22, u_23 = u_2

    pi = torch.tensor([0.0, 0.0, 1.0], dtype=torch.double)
    pj = torch.tensor([0.0, 0.0, 1.0], dtype=torch.double)

    ai = torch.stack([
        s[0] ** 2 * (v_11 ** 2 + v_12 ** 2),
        s[0] * s[1] * (v_11 * v_21 + v_12 * v_22),
        s[1] ** 2 * (v_21 ** 2 + v_22 ** 2)])

    aj = torch.stack([
        u_21 ** 2 + u_22 ** 2,
        -(u_11 * u_21 + u_12 * u_22),
        u_11 ** 2 + u_12 ** 2])

    bi = torch.stack([
        s[0] ** 2 * torch.dot(pi, v_1) ** 2,
        s[0] * s[1] * torch.dot(pi, v_1) * torch.dot(pi, v_2),
        s[1] ** 2 * torch.dot(pi, v_2) ** 2])

    bj = torch.stack([
        torch.dot(pj, u_2) ** 2,
        -torch.dot(pj, u_1) * torch.dot(pj, u_2),
        torch.dot(pj, u_1) ** 2])

    d_12 = get_d(ai, bi, aj, bj, 1, 0)
    d_13 = get_d(ai, bi, aj, bj, 0, 2)
    d_23 = get_d(ai, bi, aj, bj, 2, 1)

    def _f(x):
        fi = x[0]

        K1_12 = - (fi ** 2 * d_12[2] + d_12[3]) / (fi ** 2 * d_12[0] + d_12[1])
        K2_12 = - (fi ** 2 * d_12[1] + d_12[3]) / (fi ** 2 * d_12[0] + d_12[2])

        K1_13 = - (fi ** 2 * d_13[2] + d_13[3]) / (fi ** 2 * d_13[0] + d_13[1])
        K2_13 = - (fi ** 2 * d_13[1] + d_13[3]) / (fi ** 2 * d_13[0] + d_13[2])

        K1_23 = - (fi ** 2 * d_23[2] + d_23[3]) / (fi ** 2 * d_23[0] + d_23[1])
        K2_23 = - (fi ** 2 * d_23[1] + d_23[3]) / (fi ** 2 * d_23[0] + d_23[2])

        residuals = torch.stack([
            (fi ** 2 - K1_12) / (fi ** 2),
            (fi ** 2 - K2_12) / (fi ** 2),
            (fi ** 2 - K1_13) / (fi ** 2),
            (fi ** 2 - K2_13) / (fi ** 2),
            (fi ** 2 - K1_23) / (fi ** 2),
            (fi ** 2 - K2_23) / (fi ** 2),
        ])

        # print(residuals)

        return residuals

    return _f

def fetzer_single(F, fi_prior, pi=(0.0, 0.0)):
    x_init = np.array([fi_prior])

    cost_function = fetzer_cost_single(F)
    np_cost = lambda x: cost_function(torch.from_numpy(x)).numpy()
    np_jac = lambda x: jacobian(cost_function, torch.from_numpy(x), strict=True).numpy()

    res = least_squares(np_cost, x_init, jac=np_jac, method='lm', max_nfev=100, verbose=0)

    fi = res.x[0]

    return fi, res.nfev







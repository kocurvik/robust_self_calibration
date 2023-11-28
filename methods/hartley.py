import numpy as np
import torch
from kornia.geometry import sampson_epipolar_distance
from torch.autograd.functional import jacobian

from scipy.optimize import least_squares


# https://github.com/orybkin/AlFocal/blob/master/deepag/bin/deepag/prior/uu2F_hartley.m

def F_param(x):
    f1, f2, f3, f4, f5, f6, f7 = x

    # F = torch.tensor([[f6 * (f1 * f4 + f2 * f5) + f7 * (f3 * f4 + f5), f1 * f6 + f3 * f7, f2 * f6 + f7],
    #                   [f1 * f4 + f2 * f5, f1, f2],
    #                   [f3 * f4 + f5, f3, 1.0]])

    F = torch.stack([
        torch.stack([f6 * (f1 * f4 + f2 * f5) + f7 * (f3 * f4 + f5), f1 * f6 + f3 * f7, f2 * f6 + f7]),
        torch.stack([f1 * f4 + f2 * f5, f1, f2]),
        torch.stack([f3 * f4 + f5, f3, torch.scalar_tensor(1.0)])])

    return F


def F_unparam(F):
    F = F / F[2, 2]

    f1 = F[1, 1]
    f2 = F[1, 2]
    f3 = F[2, 1]
    f4 = (F[1, 0] - f2 * F[2, 0]) / (f1 - f2 * f3)
    f5 = F[2, 0] - f3 * f4
    f6 = (F[0, 1] - f3 * F[0, 2]) / (f1 - f2 * f3)
    f7 = F[0, 2] - f2 * f6

    return torch.tensor([f1, f2, f3, f4, f5, f6, f7])


def correct_fundamental(F, f1_prior, f2_prior, p1_prior, p2_prior):
    K1 = np.array([
        [f1_prior, 0, p1_prior[0]],
        [0, f1_prior, p1_prior[1]],
        [0, 0, 1]
    ])

    K2 = np.array([
        [f2_prior, 0, p2_prior[0]],
        [0, f2_prior, p2_prior[1]],
        [0, 0, 1]
    ])

    E = K2.T @ F @ K1

    u, s, vt = np.linalg.svd(E)

    E_hat = u @ np.diag([1.0, 1.0, 0]) @ vt

    F_hat = np.linalg.inv(K2.T) @ E_hat @ np.linalg.inv(K1)

    return F_hat


def hartley(F_orig, kp1, kp2, f1_prior, f2_prior, p1_prior=(0.0, 0.0), p2_prior=(0.0, 0.0), w_f=1.0, w_p = 0.0001, w_focal=1e-6, w_ff = 0):
    ''' Calculates focal lengths using Hartley's method implemented based on:

    :param F_orig: 3 x 3 Fundamental matrix
    :param kp1: N x 2 array of detected keypoints in the first image - expected to be centered such that p ~ [0, 0]
    :param kp2: N x 2 array of detected keypoints in the second image
    :param f1_prior: Nominal (prior) value for the focal length of the first camera
    :param f2_prior: Nominal (prior) value for the focal length of the second camera
    :return: f1_sq, f2_sq - focal lenghts of the two cameras
    '''

    w_1 = w_2 = w_focal
    w_3 = w_ff  # 0.001
    w_4 = w_5 = .1  # 0.01
    f_min_sq = 100 ** 2

    f1_prior_sq = f1_prior ** 2
    f2_prior_sq = f2_prior ** 2

    p1_prior = torch.tensor([p1_prior[0], p1_prior[1]])
    p2_prior = torch.tensor([p2_prior[0], p2_prior[1]])

    kp1_h = torch.column_stack([torch.from_numpy(kp1), torch.ones([kp1.shape[0], 1])])
    kp2_h = torch.column_stack([torch.from_numpy(kp2), torch.ones([kp2.shape[0], 1])])

    try:
        F_hat = correct_fundamental(F_orig, f1_prior, f2_prior, p1_prior.numpy(), p2_prior.numpy())
    except Exception:
        return f1_prior, f2_prior, p1_prior, p2_prior, 0

    def cost_function(x):
        F = F_param(x[:7])

        p1 = x[-4:-2]
        p2 = x[-2:]

        f1_sq, f2_sq = bougnoux_torch(F, p1, p2)
        c_F = w_f * sampson(F, kp1_h, kp2_h)

        c_p2 = w_p * torch.norm(p1 - p1_prior)
        c_p1 = w_p * torch.norm(p2 - p2_prior)

        c_f = focal_loss(f1_sq, f2_sq, f1_prior_sq, f2_prior_sq, f_min_sq, [w_1, w_2, w_3, w_4, w_5])

        pad = torch.zeros(5)

        # print(c_F.item())

        # print("f1 sq: ", f1_sq.item())
        # print("f2 sq: ", f2_sq.item())
        # print("c_F: ", c_F.item(), " c_p1: ", c_p1.item(), " c_p2: ", c_p2.item(), " c_f:", *c_f)

        return torch.stack([c_F, c_p1, c_p2, *c_f, *pad])
        # return torch.stack([*c_F, c_p1, c_p2, *c_f])

    np_cost = lambda x: cost_function(torch.from_numpy(x)).numpy()
    np_jac = lambda x: jacobian(cost_function, torch.from_numpy(x), strict=False, vectorize=True).numpy()

    x_0 = np.concatenate([F_unparam(F_hat), p1_prior[:2], p2_prior[:2]])

    try:
        res = least_squares(np_cost, x_0, jac=np_jac, method='lm', max_nfev=100, verbose=0)
    except Exception:
        print("Hartley LM exception")
        return f1_prior, f2_prior, np.array([0, 0]), np.array([0, 0]), 0

    F_final = F_param(torch.from_numpy(res.x[:7])).numpy()
    p1_prior = res.x[-4:-2]
    p2_prior = res.x[-2:]

    f1_sq, f2_sq = bougnoux_torch(torch.from_numpy(F_final), torch.from_numpy(p1_prior), torch.from_numpy(p2_prior))

    return np.sqrt(f1_sq.numpy()), np.sqrt(f2_sq.numpy()), p1_prior, p2_prior, res.nfev


def hartley_single(F_orig, kp1, kp2, f_prior, p_prior=(0.0, 0.0), w_focal=1e-6):
    f1, f2, p1, p2, iter = hartley(F_orig, kp1, kp2, f_prior, f_prior, p1_prior=p_prior, p2_prior=p_prior, w_focal=w_focal)
    return (f1 + f2) / 2, (p1 + p2) / 2, iter


def hartley_sturm(F_orig, kp1, kp2, f_prior, p_prior=(0.0, 0.0)):
    ''' Calculates focal lengths using Hartley's method implemented based on:

    :param F_orig: 3 x 3 Fundamental matrix
    :param kp1: N x 2 array of detected keypoints in the first image - expected to be centered such that p ~ [0, 0]
    :param kp2: N x 2 array of detected keypoints in the second image
    :param f_prior: Nominal (prior) value for the focal length of the first camera
    :return: f_sq focal lenghts of the two cameras
    '''
    w_f = 1.0
    w_p = 0.0001
    w_f_prior = 1e-6  # 0.01
    w_min = .1  # 0.01
    f_min_sq = 100 ** 2

    f_prior_sq = f_prior ** 2

    p_prior = torch.tensor([p_prior[0], p_prior[1]])

    kp1_h = torch.column_stack([torch.from_numpy(kp1), torch.ones([kp1.shape[0], 1])])
    kp2_h = torch.column_stack([torch.from_numpy(kp2), torch.ones([kp2.shape[0], 1])])

    try:
        F_hat = correct_fundamental(F_orig, f_prior, f_prior, p_prior.numpy(), p_prior.numpy())
    except Exception:
        return f_prior, p_prior

    def cost_function(x):
        F = F_param(x[:7])

        p = x[-2:]

        c_F = w_f * sampson(F, kp1_h, kp2_h)
        f_sq = sturm_torch(F, p)

        c_p2 = w_p * torch.norm(p - p_prior)
        c_p1 = w_p * torch.norm(p - p_prior)

        c_f = focal_loss_single(f_sq, f_prior_sq, f_min_sq, [w_f_prior, w_min])

        pad = torch.zeros(4)

        # print(c_F.item())
        # print("c_F: ", c_F.item(), " c_p1: ", c_p1.item(), " c_p2: ", c_p2.item(), " c_f:", *c_f)

        return torch.stack([c_F, c_p1, c_p2, *c_f, *pad])

    np_cost = lambda x: cost_function(torch.from_numpy(x)).numpy()
    np_jac = lambda x: jacobian(cost_function, torch.from_numpy(x), strict=False, vectorize=True).numpy()

    x_0 = np.concatenate([F_unparam(F_hat), p_prior[:2]])

    # try:
    res = least_squares(np_cost, x_0, jac=np_jac, method='lm', max_nfev=100, verbose=0)
    # except Exception:
    #     print("Hartley LM exception")
        # return f1_prior, f2_prior, np.array([0, 0]), np.array([0, 0])

    F_final = F_param(torch.from_numpy(res.x[:7])).numpy()
    p_final = res.x[-2:]

    f_sq = sturm_torch(torch.from_numpy(F_final),torch.from_numpy(p_final))

    return np.sqrt(f_sq.numpy()), p_final


def sturm_torch(F, pp):
    # T = torch.tensor([[torch.scalar_tensor(1.0), torch.scalar_tensor(0), pp[0]],
    #                   [torch.scalar_tensor(0), torch.scalar_tensor(1.0), pp[1]],
    #                   [torch.scalar_tensor(0), torch.scalar_tensor(0), torch.scalar_tensor(1.0)]])
    # G = T.T @ F @ T

    u, s, vt = torch.linalg.svd(F, full_matrices=True)

    a = s[0]
    b = s[1]

    u13, u23 = u[2, 0], u[2, 1]
    v13, v23 = vt[0, 2], vt[1, 2]

    la1 = a * u13 * u23 * (1 - v13 ** 2) + b * v13 * v23 * (1 - u23 ** 2)
    lb1 = u23 * v13 * (a * u13 * v13 + b * u23 * v23)
    return -lb1/la1


def bougnoux_torch(F, p1, p2):
    p1 = torch.cat((p1, torch.tensor([1.0]))).reshape(3, 1)
    p2 = torch.cat((p2, torch.tensor([1.0]))).reshape(3, 1)
    e2, _, e1 = torch.linalg.svd(F, full_matrices=True)
    e1 = e1[2, :]
    e2 = e2[:, 2]

    # e1 = torch.cross(F[0, :], F[1, :])
    # e2 = torch.cross(F[:, 0], F[:, 1])

    e1 = e1 / e1[2]
    e2 = e2 / e2[2]

    s_e2 = torch.tensor([[0, -e2[2], e2[1]],
        [e2[2], 0, -e2[0]],
        [-e2[1], e2[0], 0]
    ])

    s_e1 = torch.tensor([
        [0, -e1[2], e1[1]],
        [e1[2], 0, -e1[0]],
        [-e1[1], e1[0], 0]
    ])

    II = torch.diag(torch.tensor([1.0, 1.0, 0.0], dtype=torch.double))

    f1 = (-p2.T @ s_e2 @ II @ F @ (p1 @ p1.T) @ F.T @ p2) / (p2.T @ s_e2 @ II @ F @ II @ F.T @ p2)
    f2 = (-p1.T @ s_e1 @ II @ F.T @ (p2 @ p2.T) @ F @ p1) / (p1.T @ s_e1 @ II @ F.T @ II @ F @ p1)

    return f1[0, 0], f2[0, 0]


# https://github.com/orybkin/AlFocal/blob/master/deepag/bin/deepag/low_level_geometry/sampson.m#L1
def sampson(F, kp1_h, kp2_h):
    # F1 = F @ kp1_h.t()
    # F2 = F.t() @ kp2_h.t()
    # num = torch.sum(kp2_h.T * F1, dim=0)**2
    # den = F1[0, :] ** 2 + F1[1, :] ** 2 + F2[0, :] ** 2 + F2[1, :] ** 2
    # # return torch.sqrt(num / den) / kp1_h.size(0)
    # return torch.sqrt(torch.mean(num / den))

    # return torch.sqrt(torch.mean(sampson_epipolar_distance(kp1_h.unsqueeze(0), kp2_h.unsqueeze(0), F.unsqueeze(0), squared=True)[0]))
    return torch.sqrt(torch.mean(sampson_epipolar_distance(kp1_h.unsqueeze(0), kp2_h.unsqueeze(0), F.unsqueeze(0), squared=True)[0]))



def focal_loss(f1_sq, f2_sq, f1_prior_sq, f2_prior_sq, f_min_sq, weights):
    ''' Calculates the focal loss

    :param f1_sq: estimated squared focal loss of the first camera
    :param f2_sq: estimated squared focal loss of the second camera
    :param f1_prior_sq: nominal (prior) focal loss of the first camera
    :param f2_prior_sq: nominal (prior) focal loss of the second camera
    :param f_min_sq: minimal focal loss to penalize negative and low focal losses during estimation
    :param weights: weights for the different terms
    :return: loss value
    '''

    c_f_1 = weights[0] * (f1_sq - f1_prior_sq)
    c_f_2 = weights[1] * (f2_sq - f2_prior_sq)
    c_d   =  weights[2] * torch.sqrt(torch.abs(f1_sq - f2_sq))
    if f1_sq < f_min_sq:
        c_z_1 = weights[3] * (f_min_sq - f1_sq)
    else:
        c_z_1 = torch.scalar_tensor(0.0)

    if f2_sq < f_min_sq:
        c_z_2 = weights[4] * (f_min_sq - f2_sq)
    else:
        c_z_2 = torch.scalar_tensor(0.0)

    return c_f_1, c_f_2, c_d, c_z_1, c_z_2

def focal_loss_single(f_sq, f1_prior_sq, f_min_sq, weights):
    ''' Calculates the focal loss

    :param f_sq: estimated squared focal loss of the first camera
    :param f_prior_sq: nominal (prior) focal loss of the first camera
    :param f_min_sq: minimal focal loss to penalize negative and low focal losses during estimation
    :param weights: weights for the different terms
    :return: loss value
    '''

    c_f_1 = weights[0] * (f_sq - f1_prior_sq)
    if f_sq < f_min_sq:
        c_z_1 = weights[1] * (f_min_sq - f_sq)
    else:
        c_z_1 = torch.scalar_tensor(0.0)

    return c_f_1, c_z_1





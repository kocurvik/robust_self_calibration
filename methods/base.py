import numpy as np
import poselib

from matlab_utils.engine_calls import ours_uncal


def bougnoux_rybkin(F):
    f_11, f_12, f_13 = F[0, 0], F[0, 1], F[0, 2]
    f_21, f_22, f_23 = F[1, 0], F[1, 1], F[1, 2]
    f_31, f_32, f_33 = F[2, 0], F[2, 1], F[2, 2]

    den_1 = f_11*f_12*f_31*f_33-f_11*f_13*f_31*f_32+f_12**2*f_32*f_33-f_12*f_13*f_32**2+f_21*f_22*f_31*f_33-f_21*f_23*f_31*f_32+f_22**2*f_32*f_33-f_22*f_23*f_32**2
    den_2 = f_11**2*f_31*f_33+f_11*f_12*f_32*f_33-f_11*f_13*f_31**2-f_12*f_13*f_31*f_32+f_21**2*f_31*f_33+f_21*f_22*f_32*f_33-f_21*f_23*f_31**2-f_22*f_23*f_31*f_32
    den_3 = f_11**2*f_31*f_32-f_11*f_12*f_31**2+f_11*f_12*f_32**2 - f_12**2*f_31*f_32 + f_21**2*f_31*f_32 - f_21*f_22*f_31**2 + f_21*f_22*f_32**2 - f_22**2*f_31*f_32

    i = np.argmax([den_1, den_2, den_3])

    if i == 0:
        num_1 = -f_33*(f_12*f_13*f_33-f_13**2*f_32+f_22*f_23*f_33-f_23**2*f_32)
        return num_1 / den_1

    if i == 1:
        num_2 = -f_33 * (f_11 * f_13 * f_33 - f_13 ** 2 * f_31 + f_21 * f_23 * f_33 - f_23 ** 2 * f_31)
        return num_2 / den_2

    num_3 = -f_33*(f_11*f_13*f_32 - f_12*f_13*f_31 + f_21*f_23*f_32 - f_22*f_23*f_31)
    return num_3 / den_3


def focal_svd(F, f_undo):
    if f_undo is not None:
        K_undo = np.diag([f_undo, f_undo, 1])
        F = K_undo.T @ F

    try:
        u, s, vt = np.linalg.svd(F)
    except Exception:
        return 0.0

    a, b = s[0], s[1]

    v_13, v_23 = vt[0, 2], vt[1, 2]

    if np.abs(v_23) > np.abs(v_13):
        a, b = b, a
        v_13, v_23 = v_23, v_13

    return a ** 2 * v_13 ** 2 / (a ** 2 * v_13 ** 2 - a ** 2 + b ** 2)


def ours_boug_uncal(eng, F, f1_prior, f2_prior, p1=(0.0, 0.0), p2=(0.0, 0.0), **kwargs):
    pp1 = np.append(p1, 1).reshape(3, 1)
    pp2 = np.append(p2, 1).reshape(3, 1)
    try:
        e2, _, e1 = np.linalg.svd(F)
    except Exception:
        return ours_uncal(eng, F, f1_prior, f2_prior, p1, p2, **kwargs)

    e1 = e1[2, :]
    e2 = e2[:, 2]

    s_e2 = np.array([
        [0, -e2[2], e2[1]],
        [e2[2], 0, -e2[0]],
        [-e2[1], e2[0], 0]
    ])
    s_e1 = np.array([
        [0, -e1[2], e1[1]],
        [e1[2], 0, -e1[0]],
        [-e1[1], e1[0], 0]
    ])

    II = np.diag([1, 1, 0])

    den_1 = (pp2.T @ s_e2 @ II @ F @ II @ F.T @ pp2)
    den_2 = (pp1.T @ s_e1 @ II @ F.T @ II @ F @ pp1)

    if np.abs(den_1)[0, 0] < 1e-8 or np.abs(den_2)[0, 0] < 1e-8:
        return ours_uncal(eng, F, f1_prior, f2_prior, p1, p2, **kwargs)

    f1 = (-pp2.T @ s_e2 @ II @ F @ (pp1 @ pp1.T) @ F.T @ pp2) / den_1
    f2 = (-pp1.T @ s_e1 @ II @ F.T @ (pp2 @ pp2.T) @ F @ pp1) / den_2

    if f1[0, 0] < 0.0 or f2[0, 0] < 0.0:
        return ours_uncal(eng, F, f1_prior, f2_prior, p1, p2, **kwargs)

    return np.sqrt(f1[0, 0]), np.sqrt(f2[0, 0]), p1, p2



def bougnoux_original(F, p1=np.array([0, 0]), p2=np.array([0, 0])):
    ''' Returns squared focal losses estimated using the Bougnoux formula with given principal points

    :param F: 3 x 3 Fundamental matrix
    :param p1: 2-dimensional coordinates of the principal point of the first camera
    :param p2: 2-dimensional coordinates of the principal point of the first camera
    :return: the estimated squared focal lengths for the two cameras
    '''
    p1 = np.append(p1, 1).reshape(3, 1)
    p2 = np.append(p2, 1).reshape(3, 1)
    try:
        e2, _, e1 = np.linalg.svd(F)
    except Exception:
        return np.nan, np.nan

    e1 = e1[2, :]
    e2 = e2[:, 2]


    s_e2 = np.array([
        [0, -e2[2], e2[1]],
        [e2[2], 0, -e2[0]],
        [-e2[1], e2[0], 0]
    ])
    s_e1 = np.array([
        [0, -e1[2], e1[1]],
        [e1[2], 0, -e1[0]],
        [-e1[1], e1[0], 0]
    ])

    II = np.diag([1, 1, 0])

    f1 = (-p2.T @ s_e2 @ II @ F @ (p1 @ p1.T) @ F.T @ p2) / (p2.T @ s_e2 @ II @ F @ II @ F.T @ p2)
    f2 = (-p1.T @ s_e1 @ II @ F.T @ (p2 @ p2.T) @ F @ p1) / (p1.T @ s_e1 @ II @ F.T @ II @ F @ p1)

    return f1[0, 0], f2[0, 0]


def single_6pt_minimal(x1, x2, args=None):
    if args is None:
        pose, info = poselib.estimate_singlefocal_relative_pose(x1, x2)
    else:
        pose, info = poselib.estimate_singlefocal_relative_pose(x1, x2, args)

    return pose.f, pose.R, pose.t


def get_focal_sturm(F, pp=(0.0, 0.0), f0=1.0):
    T = np.array([[1, 0, pp[0]], [0, 1, pp[1]], [0, 0, 1]])
    G = T.T @ F @ T

    F0 = np.diag([f0, f0, 1])
    G = F0 @ G @ F0
    # G = G / np.linalg.norm(G, ord='fro')

    u, s, vt = np.linalg.svd(F)

    a = s[0]
    b = s[1]

    u13, u23 = u[2, 0], u[2, 1]
    v13, v23 = vt[0, 2], vt[1, 2]
    # v13, v23 = vt[2, 0], vt[2, 1]

    la1 = a * u13 * u23 * (1 - v13 ** 2) + b * v13 * v23 * (1 - u23 ** 2)
    lb1 = u23 * v13 * (a * u13 * v13 + b * u23 * v23)

    # la2 = a * v13 * v23 * (1 - u13 ** 2) + b * u13 * u23 * (1 - v23 ** 2)
    # lb2 = u13 * v23 * (a * u13 * v13 + b * u23 * v23)

    # a * f**4  + b * f**2 + c = 0

    qa = a ** 2 * (1 - u13 ** 2) * (1 - v13 ** 2) \
         - b ** 2 * (1 - u23 ** 2) * (1 - v23 ** 2)

    qb = a ** 2 * (u13 ** 2 + v13 ** 2 - 2 * u13 ** 2 * v13 ** 2) \
         - b ** 2 * (u23 ** 2 + v23 ** 2 - 2 * u23 ** 2 * v23 ** 2)

    qc = a ** 2 * u13 ** 2 * v13 ** 2 \
         - b ** 2 * u23 ** 2 * v23 ** 2

    # f1, f2 = np.sqrt(np.roots([qa, qb, qc]))

    return np.sqrt(-lb1 / la1) * f0

    # lin_1 = np.abs(la1 * f1**2 + lb1)
    # lin_2 = np.abs(la1 * f2**2 + lb1)

    # print("focal 1: {}".format(f1 * f0))
    # print(qa * f1**4 + qb * f1**2 + qc)
    # print(la1 * f1**2 + lb1)
    # print(la2 * f1**2 + lb2)
    #
    # print("focal 2: {}".format(f2 * f0))
    # print(qa * f2**4  + qb * f2**2 + qc)
    # print(la1 * f2**2 + lb1)
    # print(la2 * f2**2 + lb2)

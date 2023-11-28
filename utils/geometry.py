import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def angle(x, y):
    x = x.ravel()
    y = y.ravel()
    return np.rad2deg(np.arccos(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))))

def angle_matrix(R):
    return np.rad2deg(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1.0)))

def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def project_to_plane(K, n, p):
    K_inv = np.linalg.inv(K)
    K_inv /= K_inv[0, 0]
    p = K_inv @ p
    P = - p / np.dot(p, n)
    return P


def line_distance(p1, dir1, p2, dir2):
    n = np.cross(dir1, dir2)
    dist = np.abs(np.dot(n, p1 - p2)) / np.linalg.norm(n)
    return dist

def get_K(f, p=np.array([0, 0])):
    return np.array([[f, 0, p[0]], [0, f, p[1]], [0, 0, 1]])

def pose_from_F(F, K1, K2, kp1, kp2):
    try:
        K1_inv = np.linalg.inv(K1)
        K2_inv = np.linalg.inv(K2)

        E = K2.T @(F @ K1)

        # print(np.linalg.svd(E)[1])

        kp1 = np.column_stack([kp1, np.ones(len(kp1))])
        kp2 = np.column_stack([kp2, np.ones(len(kp1))])

        kp1_unproj = (K1_inv @ kp1.T).T
        kp1_unproj = kp1_unproj[:, :2] / kp1_unproj[:, 2, np.newaxis]
        kp2_unproj = (K2_inv @ kp2.T).T
        kp2_unproj = kp2_unproj[:, :2] / kp2_unproj[:, 2, np.newaxis]

        _, R, t, mask = cv2.recoverPose(E, kp1_unproj, kp2_unproj)
    except:
        # print("Pose exception!")
        return np.eye(3), np.ones(3)

    return R, t

def reconstruct(F, kp1, kp2, f1, f2):
    K1_inv = np.diag([1/f1, 1/f1, 1])
    K1 = 1 / K1_inv
    K2_inv = np.diag([1/f2, 1/f2, 1])
    K2 = 1 / K2_inv

    E = K2 @(F @ K1)

    kp1 = np.column_stack([kp1, np.ones(len(kp1))])
    kp2 = np.column_stack([kp2, np.ones(len(kp1))])

    kp1_unproj = (K1_inv @ kp1.T).T
    kp1_unproj = kp1_unproj[:, :2] / kp1_unproj[:, 2, np.newaxis]
    kp2_unproj = (K2_inv @ kp2.T).T
    kp2_unproj = kp2_unproj[:, :2] / kp2_unproj[:, 2, np.newaxis]

    _, R, t, mask = cv2.recoverPose(E, kp1_unproj, kp2_unproj)

    P1 = K1 @ np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    P2 = K2 @ np.column_stack([R, t])

    X = cv2.triangulatePoints(P1, P2, kp1[mask, :2].T, kp2[mask, :2].T).T

    X = X[:, 0, :3] / X[:, 0, 3, np.newaxis]

    return X


def get_projection(P, X):
    x = P @ X.T
    x = x[:2, :] / x[2, np.newaxis, :]
    return x.T


def pose_from_estimated(F, prior_f_1, prior_f_2, f_1_est, f_2_est, info, kp_1, kp_2, pp1=(0, 0), pp2=(0, 0)):
    if f_1_est < 0 or f_2_est < 0 or np.isnan(f_1_est) or np.isnan(f_2_est):
        R, t = pose_from_F(F, get_K(prior_f_1, np.array([0, 0])), get_K(prior_f_2, np.array([0, 0])),
                           kp_1[info['inliers']],
                           kp_2[info['inliers']])
    else:
        R, t = pose_from_F(F, get_K(f_1_est, pp1), get_K(f_2_est, pp2), kp_1[info['inliers']],
                           kp_2[info['inliers']])

    return R, t


def pose_from_img_info(img_1, img_2):
    q_1 = (img_1['q'])
    R_1 = Rotation.from_quat([q_1[1], q_1[2], q_1[3], q_1[0]]).as_matrix()
    t_1 = np.array(img_1['t']).ravel()
    q_2 = (img_2['q'])
    R_2 = Rotation.from_quat([q_2[1], q_2[2], q_2[3], q_2[0]]).as_matrix()
    t_2 = np.array(img_2['t']).ravel()

    R = np.dot(R_2, R_1.T)
    t = t_2 - np.dot(R, t_1)

    return R, t / np.linalg.norm(t)

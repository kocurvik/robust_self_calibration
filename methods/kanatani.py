import numpy as np

from utils.geometry import skew


def kanatani(F, p1=(0.0, 0.0), p2=(0.0, 0.0)):
    F = F.T

    u, s, vt = np.linalg.svd(F)

    e1 = vt[2, :]
    e2 = u[:, 2]

    k = np.array([0.0, 0.0, 1.0])

    xi_num = np.linalg.norm(F @ k) ** 2 - np.dot(k, F @ F.T @ F @ k) * np.linalg.norm(np.cross(e1, k))**2 / np.dot(k, F @ k)
    xi_den = np.linalg.norm(np.cross(e1, k)) ** 2 * np.linalg.norm(F.T @ k) ** 2 - np.dot(k, F @ k) ** 2

    eta_num = np.linalg.norm(F.T @ k) ** 2 - np.dot(k, F @ F.T @ F @ k) * np.linalg.norm(np.cross(e2, k)) / np.dot(k, F @ k)
    eta_den = np.linalg.norm(np.cross(e2, k)) ** 2 * np.linalg.norm(F @ k) ** 2 - np.dot(k, F @ k) ** 2

    xi = xi_num / xi_den
    eta = eta_num / eta_den

    f1 = 1 / np.sqrt(1 + xi)
    f2 = 1 / np.sqrt(1 + eta)

    return f1, f2


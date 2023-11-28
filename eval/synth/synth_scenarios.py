import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from utils.geometry import get_projection
from methods.base import bougnoux_original
from methods.fetzer import fetzer_focal_only
from methods.kanatani import kanatani
from methods.melanitis import melanitis
from utils.plot import plot_scene


def visible_in_view(x, width=640, height=480):
    visible = np.logical_and(np.abs(x[:, 0]) <= width / 2, np.abs(x[:, 1]) <= height / 2)
    return visible


def set_scene(f1, f2, theta=0, y=0):
    R = Rotation.from_euler('xyz', (theta, 60, 0), degrees=True).as_matrix()
    c = np.array([2 * f1, y, f1])
    # R = Rotation.from_euler('xyz', (theta, 30, 0), degrees=True).as_matrix()
    # c = np.array([f1, y, 0])
    t = -R @ c
    return f1, f2, R, t


def get_pp_err(sigma_p):
    angle = np.random.rand() * 2 *np.pi
    dist = sigma_p * np.random.randn()

    return dist * np.array([[np.sin(angle), np.cos(angle)]])

def get_scene(f1, f2, R, t, num_pts, X=None, min_distance=1, depth=1, width=640, height=480, sigma_p=0.0, plot=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    K1 = np.diag([f1, f1, 1])
    K2 = np.diag([f2, f2, 1])

    P1 = K1 @ np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    P2 = K2 @ np.column_stack([R, t])

    if X is None:
        X = generate_points(3 * num_pts, f1, min_distance, depth, width=width, height=height)
    x1 = get_projection(P1, X)
    x2 = get_projection(P2, X)

    visible = visible_in_view(x2, width=width, height=height)

    x1, x2, X = x1[visible][:num_pts], x2[visible][:num_pts], X[visible]

    p_err_1 = get_pp_err(sigma_p)
    p_err_2 = get_pp_err(sigma_p)

    x1 += p_err_1
    x2 += p_err_2

    # run(f1, f2, x1, x2, scale=scale, name=name)
    if plot is not None:
        plot_scene(X, R, t, f1, f2, name=plot)
        # plt.savefig(f'{np.round(t)}.png')
        plt.show()

    return x1, x2, X


def generate_points(num_pts, f, distance, depth, width=640, height=480):
    zs = (1 + distance) * f + depth * np.random.rand(num_pts) * f
    xs = (np.random.rand(num_pts) - 0.5) * width * (1 + distance)
    ys = (np.random.rand(num_pts) - 0.5) * height * (1 + distance)
    return np.column_stack([xs, ys, zs, np.ones_like(xs)])


def noncoplanar_scene():
    f1 = 600
    f2 = 400

    R = Rotation.from_euler('xyz', (15, 30, 0), degrees=True).as_matrix()
    c = np.array([f1, 0, 0])
    t = -R @ c
    return f1, f2, R, t


def coplanar_scene():
    f1 = 600
    f2 = 400

    R = Rotation.from_euler('xyz', (0, 30, 0), degrees=True).as_matrix()
    c = np.array([f1, 0, 0])
    t = -R @ c
    return f1, f2, R, t

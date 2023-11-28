import argparse
import os.path
from multiprocessing import Pool
from time import perf_counter

import cv2
import joblib
import numpy as np
import pandas as pd
import pvsac
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from matplotlib import pyplot as plt
import pydegensac
import matlab.engine
import poselib
# import pymagsac

from datasets.definitions import get_subset_string
from eval.manager import focal_error
from matlab_utils.engine_calls import kukelova_uncal
from methods.base import bougnoux_original
from methods.hartley import hartley
from utils.geometry import pose_from_F, get_K, angle_matrix, angle, pose_from_estimated

from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name')
    parser.add_argument('-m', '--matcher', type=str, default='loftr')
    parser.add_argument('-l', '--load', action='store_true', default=False)
    parser.add_argument('-nw', '--num_workers', type=int, default=1)

    return parser.parse_args()


def pose_from_img_info(img_1, img_2):
    q_1 = (img_1['q'])
    R_1 = Rotation.from_quat([q_1[1], q_1[2], q_1[3], q_1[0]]).as_matrix()
    t_1 = img_1['t']
    q_2 = (img_2['q'])
    R_2 = Rotation.from_quat([q_2[1], q_2[2], q_2[3], q_2[0]]).as_matrix()
    t_2 = img_2['t']

    # R = R_1.T @ R_2
    # t = R_1.T @ (t_2 - t_1)
    R = np.dot(R_2, R_1.T)
    t = t_2 - np.dot(R, t_1)

    return R, t /np.linalg.norm(t)


# WEIGHTS = [10**e for e in range(-5, 5)]
# WEIGHTS = [1e-7, 1e-6]
WEIGHTS = [1e-2, 1e-1, 1.0, 10]


class WeightsManager:
    manager_str = 'uncal'

    def __init__(self, subset, eng=None, semi=True, **kwargs):
        # super().__init__(**kwargs)
        self.subset = subset
        self.semi = semi
        self.eng = eng

    def load_samples(self, path, extend=False):
        if extend:
            self.samples.extend(joblib.load(path))
        else:
            self.samples = joblib.load(path)

    def estimate_weights(self, sample):
        cam_1 = sample['cam_1']
        cam_2 = sample['cam_2']

        p_1 = np.array([cam_1['width'] / 2, cam_1['height'] / 2])
        p_2 = np.array([cam_2['width'] / 2, cam_2['height'] / 2])

        # p_1 = np.array([cam_1['cx'], cam_1['cy']])
        # p_2 = np.array([cam_2['cx'], cam_2['cy']])

        colmap_1 = 1.2 * max(cam_1['width'], cam_1['height'])
        colmap_2 = 1.2 * max(cam_2['width'], cam_2['height'])

        kp_1 = sample['kp_1'] - p_1[np.newaxis, :]
        kp_2 = sample['kp_2'] - p_2[np.newaxis, :]

        conf = sample['conf']

        kp_1 = kp_1[np.argsort(conf)[::-1]]
        kp_2 = kp_2[np.argsort(conf)[::-1]]
        conf = conf[np.argsort(conf)[::-1]]

        f_1, f_2 = cam_1['focal'], cam_2['focal']

        gt_R, gt_t = pose_from_img_info(sample['img_1'], sample['img_2'])

        for weight in WEIGHTS:
            F, mask = cv2.findFundamentalMat(kp_1, kp_2, cv2.USAC_MAGSAC_RFD, ransacReprojThreshold=3.0,
                                                  confidence=1.0,
                                                  maxIters=10000)
            info = {'inliers': mask.ravel().astype(bool), 'num_inliers': np.sum(mask)}

            # f_1_est, f_2_est, pp1, pp2 = kukelova_uncal(self.eng, F, colmap_1, colmap_2, w1=weight, w3=weight)

            f_1_est, f_2_est, pp1, pp2 = hartley(F, kp_1[info['inliers']], kp_2[info['inliers']], colmap_1, colmap_2,
                                                 w_f=weight)
            R, t = pose_from_estimated(F, colmap_1, colmap_2, f_1_est, f_2_est, info, kp_1, kp_2, pp1, pp2)
            entry = {'subset': self.subset, 'weight': weight, 'F': F,
                     'R_err': angle_matrix(R.T @ gt_R), 't_err': angle(t, gt_t),
                     'f_1_err': focal_error(f_1_est, f_1), 'f_2_err': focal_error(f_2_est, f_2)}

            self.df.loc[len(self.df)] = entry

    def evaluate(self):
        # self.estimates = {n: [[] for _ in self.iters] for n, m in METHODS.items()}
        self.df = pd.DataFrame(columns=['subset', 'weight', 'F', 'ratio', 'R_err', 't_err', 'f_1_err', 'f_2_err'])
        for sample in tqdm(self.samples):
            self.estimate_weights(sample)
        return self.df

    def save_results(self, path):
        self.df.to_pickle(path)

    def load_results(self, path):
        self.df = pd.read_pickle(path)


def draw_plot(df, dataset_name, matcher_string):
    fig = plt.figure(figsize=(10, 10))

    figure_pose, axes_pose = plt.subplots()
    figure_focal, axes_focal = plt.subplots()

    xs = []
    pose_ys = []
    focal_ys = []

    for weight in WEIGHTS:
        dff = df.loc[df['weight'] == weight]
        xs.append(weight)

        R_errs = dff['R_err'].to_numpy()
        t_errs = dff['t_err'].to_numpy()
        f_errs = np.concatenate([dff['f_1_err'].to_numpy(), dff['f_2_err']])

        max_errs = np.maximum(R_errs, t_errs)
        max_errs[np.isnan(max_errs)] = 360
        mAA_p, _ = np.histogram(max_errs, bins=np.arange(11))
        mAA_f, _ = np.histogram(f_errs, bins=np.arange(11) / 100)

        pose_ys.append(np.mean(np.cumsum(mAA_p) / len(dff)))
        focal_ys.append(np.mean(np.cumsum(mAA_f)) / len(f_errs))

        mAA, _ = np.histogram(max_errs, bins=np.arange(11))

    axes_pose.semilogx(xs, pose_ys)
    axes_focal.semilogx(xs, focal_ys)

    # axes_pose.set_title(f'RFC - {dataset_name} - Pose')
    axes_pose.set_xlabel('Mean Runtime (ms)')
    # plt.xlabel('Iters')
    axes_pose.set_ylabel('mAA$_p$(10°)')
    axes_pose.legend()
    if not os.path.exists(f'figs/weights/{matcher_string}'):
        os.makedirs(f'figs/weights/{matcher_string}')
    figure_pose.savefig(f'figs/weights/{matcher_string}/{dataset_name}_pose.pdf')


    # axes_focal.set_title(f'RFC - {dataset_name} - Pose')
    axes_focal.set_xlabel('Mean Runtime (ms)')
    # plt.xlabel('Iters')
    axes_focal.set_ylabel('mAA$_f$(0.1)')
    axes_focal.legend()
    if not os.path.exists(f'figs/weights/{matcher_string}'):
        os.makedirs(f'figs/weights/{matcher_string}')
    figure_focal.savefig(f'figs/weights/{matcher_string}/{dataset_name}_focal.pdf')


def eval_subset(subset, matcher_string, save_string):
    eng = matlab.engine.start_matlab()
    s = eng.genpath('matlab_utils')
    eng.addpath(s, nargout=0)

    eval_manager = WeightsManager(subset, eng=eng, verbosity=0)
    eval_manager.load_samples(f'saved/{matcher_string}/{save_string.format(subset)}.joblib', extend=False)
    ret = eval_manager.evaluate()
    if not os.path.exists(f'results/{matcher_string}/thresholds'):
        os.makedirs(f'results/{matcher_string}/thresholds')

    eval_manager.save_results(f'results/{matcher_string}/thresholds/{save_string.format(subset)}.pkl')
    return ret


if __name__ == '__main__':
    args = parse_args()
    save_string, subsets, rows, cols = get_subset_string(args.dataset_name)

    subset = subsets[0]

    print(subsets)

    if args.load:
        df = pd.read_pickle(f'results/{args.matcher}/weights/{save_string.format(subset)}.pkl')
    else:
        eng = matlab.engine.start_matlab()
        s = eng.genpath('matlab_utils')
        eng.addpath(s, nargout=0)

        eval_manager = WeightsManager(subset, eng=eng, verbosity=0)
        eval_manager.load_samples(f'saved/{args.matcher}/{save_string.format(subset)}.joblib', extend=False)
        ret = eval_manager.evaluate()
        if not os.path.exists(f'results/{args.matcher}/weights'):
            os.makedirs(f'results/{args.matcher}/weights')

        eval_manager.save_results(f'results/{args.matcher}/weights/{save_string.format(subset)}.pkl')
        df = eval_manager.df


    draw_plot(df, args.dataset_name, args.matcher)

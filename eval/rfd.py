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
from matlab_utils.engine_calls import ours_uncal
from methods.base import bougnoux_original
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


METHODS = {
    'usac_default': cv2.USAC_DEFAULT,
    'usac_default_rfc': cv2.USAC_DEFAULT_RFD,
    'magsac': cv2.USAC_MAGSAC,
    'magsac_rfc': cv2.USAC_MAGSAC_RFD,
    'usac_accurate': cv2.USAC_ACCURATE,
    'usac_accurate_rfc': cv2.USAC_ACCURATE_RFD,
    'prosac': cv2.USAC_PROSAC,
    'prosac_rfc': cv2.USAC_PROSAC_RFD,
    # 'pydegensac': 100,
    # 'pydegensac_rfc': 101,
    'vsac2_uni_mag': 200,
    'vsac2_uni_mag_rfc': 201,
    'vsac2_pro_mag': 202,
    'vsac2_pro_mag_rfc': 203,
    'poselib': 300,
    'poselib_rfc': 301,
    'poselib_pro': 302,
    'poselib_pro_rfc': 303
    }

COLORS = {'usac_default': 'tab:blue',
           'usac_default_rfc': 'tab:blue',
           'magsac': 'tab:brown',
           'magsac_rfc': 'tab:brown',
           'usac_accurate': 'tab:orange',
           'usac_accurate_rfc': 'tab:orange',
           'prosac': 'tab:pink',
           'prosac_rfc': 'tab:pink',
           'pydegensac': 'tab:green',
           'pydegensac_rfc': 'tab:green',
           'vsac2_uni_mag': 'tab:gray',
           'vsac2_uni_mag_rfc': 'tab:gray',
           'vsac2_pro_mag': 'tab:olive',
           'vsac2_pro_mag_rfc': 'tab:olive',
           'poselib': 'tab:purple',
           'poselib_rfc': 'tab:purple',
           'poselib_pro': 'tab:cyan',
           'poselib_pro_rfc': 'tab:cyan'}


METHOD_STRING = {
           'usac_default': 'OpenCV Default',
           'usac_default_rfc': 'OpenCV Default',
           'magsac': 'OpenCV MAGSAC++',
           'magsac_rfc': 'OpenCV MAGSAC++',
           'usac_accurate': 'OpenCV Accurate',
           'usac_accurate_rfc': 'OpenCV Accurate',
           'prosac': 'OpenCV PROSAC',
           'prosac_rfc': 'OPENCV PROSAC',
           'pydegensac': 'PyDEGENSAC',
           'pydegensac_rfc': 'PyDEGENSAC',
           'vsac2_uni_mag': 'VSAC2 MAGSAC++',
           'vsac2_uni_mag_rfc': 'VSAC2 MAGSAC++',
           'vsac2_pro_mag': 'VSAC2 PRO+MAGSAC++',
           'vsac2_pro_mag_rfc': 'VSAC2 PRO+MAGSAC++',
           'poselib': 'PoseLib',
           'poselib_rfc': 'PoseLib',
           'poselib_pro': 'PoseLib PROSAC',
           'poselib_pro_rfc': 'PoseLib PROSAC'
}



# ITERS = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
ITERS = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
# ITERS = [500, 1000, 2000, 5000, 10000]
# ITERS = list(2 * np.arange(1, 5))

class RFDManager:
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

    def run_ransac(self, kp_1, kp_2, method, conf, sample, iters=1000):
        start = perf_counter()
        if method < 100:
            F, mask = cv2.findFundamentalMat(kp_1, kp_2, method, ransacReprojThreshold=3.0, confidence=1.0, maxIters=iters)
        elif method == 100:
            F, mask = pydegensac.findFundamentalMatrix(kp_1, kp_2, 3.0, 1.0, iters, enable_real_focal_check=False)
        elif method == 101:
            F, mask = pydegensac.findFundamentalMatrix(kp_1, kp_2, 3.0, 1.0, iters, enable_real_focal_check=True)
        if method == 200:
            params = pvsac.Params(pvsac.EstimationMethod.Fundamental, 3.0, 1.0, iters,
                                  pvsac.SamplingMethod.SAMPLING_UNIFORM, pvsac.ScoreMethod.SCORE_METHOD_MAGSAC)
            F, mask = pvsac.estimate(params, kp_1, kp_2)

        if method == 201:
            params = pvsac.Params(pvsac.EstimationMethod.Fundamental, 3.0, 1.0, iters,
                                  pvsac.SamplingMethod.SAMPLING_UNIFORM, pvsac.ScoreMethod.SCORE_METHOD_MAGSAC)
            params.setRealFocalCheck(True)
            F, mask = pvsac.estimate(params, kp_1, kp_2)
        if method == 202:
            params = pvsac.Params(pvsac.EstimationMethod.Fundamental, 3.0, 1.0, iters,
                                  pvsac.SamplingMethod.SAMPLING_PROSAC, pvsac.ScoreMethod.SCORE_METHOD_MAGSAC)
            F, mask = pvsac.estimate(params, kp_1, kp_2)

        if method == 203:
            params = pvsac.Params(pvsac.EstimationMethod.Fundamental, 3.0, 1.0, iters,
                                  pvsac.SamplingMethod.SAMPLING_PROSAC, pvsac.ScoreMethod.SCORE_METHOD_MAGSAC)
            params.setRealFocalCheck(True)
            F, mask = pvsac.estimate(params, kp_1, kp_2)
        if method == 300:
            F, info = poselib.estimate_fundamental(kp_1, kp_2, {'max_iterations': iters,
                                                                'min_iterations': 10,
                                                                'success_prob': 1.0,
                                                                'max_epipolar_error': 3.0,
                                                                'progressive_sampling': False})
            mask = np.array(info['inliers'])
        if method == 301:
            F, info = poselib.estimate_fundamental_valid_only(kp_1, kp_2, {'max_iterations': iters,
                                                                'min_iterations': 10,
                                                                'success_prob': 1.0,
                                                                'max_epipolar_error': 3.0,
                                                                'progressive_sampling': False})
            mask = np.array(info['inliers'])
        if method == 302:
            F, info = poselib.estimate_fundamental(kp_1, kp_2, {'max_iterations': iters,
                                                                'min_iterations': 10,
                                                                'success_prob': 1.0,
                                                                'max_epipolar_error': 3.0,
                                                                'progressive_sampling': True})
            mask = np.array(info['inliers'])
        if method == 303:
            F, info = poselib.estimate_fundamental_valid_only(kp_1, kp_2, {'max_iterations': iters,
                                                                'min_iterations': 10,
                                                                'success_prob': 1.0,
                                                                'max_epipolar_error': 3.0,
                                                                'progressive_sampling': True})

            mask = np.array(info['inliers'])

        # F, mask = cv2.findFundamentalMat(kp_1, kp_2, method, ransacReprojThreshold=iters, confidence=1.0, maxIters=500)
        end = perf_counter()

        mask = mask.ravel()
        elapsed = end - start
        inlier_ratio = np.sum(mask)/len(mask)

        return F, mask, inlier_ratio, elapsed


    def estimate_rfd(self, sample):
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

        for name, method in METHODS.items():
            for i, iters in enumerate(ITERS):
                F, mask, ratio, elapsed = self.run_ransac(kp_1, kp_2, method, conf, sample, iters=iters)
                info = {'inliers': mask.ravel().astype(bool), 'num_inliers': np.sum(mask)}

                K_1 = np.array(
                    [[cam_1['fx'], 0, cam_1['cx'] - p_1[0]], [0, cam_1['fy'], cam_1['cy'] - p_1[1]], [0, 0, 1]])
                K_2 = np.array(
                    [[cam_2['fx'], 0, cam_2['cx'] - p_2[0]], [0, cam_2['fy'], cam_2['cy'] - p_2[1]], [0, 0, 1]])

                f_1_gt = cam_1['focal']
                f_2_gt = cam_2['focal']

                R, t = pose_from_F(F, K_1, K_2,
                                   kp_1[info['inliers']],
                                   kp_2[info['inliers']])
                entry = {'subset': self.subset, 'f_method': 'gt', 'f_elapsed': 0,
                         'f_1_est': cam_1['focal'], 'f_2_est': cam_2['focal'], 'f_1_err': 0.0, 'f_2_err': 0.0,
                         'method_name': name, 'method_num': method, 'iters': iters, 'F': F, 'ratio': ratio,
                         'elapsed': elapsed, 'R_err': angle_matrix(R.T @ gt_R), 't_err': angle(t, gt_t)}
                self.df.loc[len(self.df)] = entry

                R, t = pose_from_estimated(F, colmap_1, colmap_2, colmap_1, colmap_2, info, kp_1, kp_2)
                entry = {'subset': self.subset, 'f_method': 'prior',  'f_elapsed': 0,
                         'f_1_est': colmap_1, 'f_2_est': colmap_2, 'f_1_err': focal_error(f_1_gt, colmap_1), 'f_2_err': focal_error(f_2_gt, colmap_2),
                         'method_name': name, 'method_num': method, 'iters': iters, 'F': F, 'ratio': ratio,
                         'elapsed': elapsed, 'R_err': angle_matrix(R.T @ gt_R), 't_err': angle(t, gt_t)}
                self.df.loc[len(self.df)] = entry

                start = perf_counter()
                f_1_est, f_2_est = (np.sqrt(f) for f in bougnoux_original(F))
                end = perf_counter()
                R, t = pose_from_estimated(F, colmap_1, colmap_2, f_1_est, f_2_est, info, kp_1, kp_2)
                entry = {'subset': self.subset, 'f_method': 'bougnoux', 'f_elapsed': end - start,
                         'f_1_est': f_1_est, 'f_2_est': f_2_est, 'f_1_err': focal_error(f_1_gt, f_1_est),
                         'f_2_err': focal_error(f_2_gt, f_2_est),
                         'method_name': name, 'method_num': method, 'iters': iters, 'F': F, 'ratio': ratio,
                         'elapsed': elapsed, 'R_err': angle_matrix(R.T @ gt_R), 't_err': angle(t, gt_t)}
                self.df.loc[len(self.df)] = entry

                if iters < 500 and (method == cv2.USAC_DEFAULT or method == cv2.USAC_DEFAULT_RFD):
                    continue

                start = perf_counter()
                f_1_est, f_2_est, pp1, pp2 = ours_uncal(self.eng, F, colmap_1, colmap_2)
                end = perf_counter()
                R, t = pose_from_estimated(F, colmap_1, colmap_2, f_1_est, f_2_est, info, kp_1, kp_2, pp1, pp2)
                entry = {'subset': self.subset, 'f_method': 'kukelova', 'f_elapsed': end - start,
                         'f_1_est': f_1_est, 'f_2_est': f_2_est, 'f_1_err': focal_error(f_1_gt, f_1_est),
                         'f_2_err': focal_error(f_2_gt, f_2_est),
                         'method_name': name, 'method_num': method, 'iters': iters, 'F': F, 'ratio': ratio,
                         'elapsed': elapsed, 'R_err': angle_matrix(R.T @ gt_R), 't_err': angle(t, gt_t)}
                self.df.loc[len(self.df)] = entry

    def evaluate(self):
        # self.estimates = {n: [[] for _ in self.iters] for n, m in METHODS.items()}
        self.df = pd.DataFrame(columns=['subset', 'f_elapsed', 'f_method', 'method_name', 'method_num', 'iters', 'F', 'ratio', 'elapsed', 'R_err', 't_err', 'f_1_est', 'f_2_est', 'f_1_err', 'f_2_err'])
        # self.samples = [sample for i, sample in enumerate(self.samples) if i % 5 == 0]
        for sample in tqdm(self.samples):
            self.estimate_rfd(sample)
        return self.df

    def save_results(self, path):
        self.df.to_pickle(path)

    def load_results(self, path):
        self.df = pd.read_pickle(path)


def draw_plot(df, dataset_name, matcher_string):

    for f_method in ['gt', 'prior', 'bougnoux', 'kukelova']:
        figure_pose, axes_pose = plt.subplots()
        figure_focal, axes_focal = plt.subplots()

        dff = df.loc[df['f_method'] == f_method]
        for method_name, _ in METHODS.items():
            df_method = dff.loc[dff['method_name'] == method_name]
            xs = []
            pose_ys = []
            focal_ys = []

            for iters in ITERS:
                df_iters = df_method.loc[df_method['iters'] == iters]

                xs.append(1000 * np.nanmean(df_iters['elapsed'].to_numpy() + df_iters['f_elapsed'].to_numpy().astype(float)))
                # xs.append(iters)

                R_errs = df_iters['R_err'].to_numpy()
                t_errs = df_iters['t_err'].to_numpy()
                f_errs = np.concatenate([df_iters['f_1_err'].to_numpy(), df_iters['f_2_err']])

                max_errs = np.maximum(R_errs, t_errs)
                max_errs[np.isnan(max_errs)] = 360
                mAA_p, _ = np.histogram(max_errs, bins=np.arange(11))
                mAA_f, _ = np.histogram(f_errs, bins=np.arange(11)/100)


                pose_ys.append(np.mean(np.cumsum(mAA_p) / len(df_iters)))
                focal_ys.append(np.mean(np.cumsum(mAA_f)) / len(f_errs))

            axes_pose.semilogx(xs, pose_ys, label=METHOD_STRING[method_name], color=COLORS[method_name],
                     marker='*' if 'rfc' in method_name else 'd',
                     linestyle='-' if 'rfc' in method_name else ':',)

            axes_focal.semilogx(xs, focal_ys, label=METHOD_STRING[method_name], color=COLORS[method_name],
                          marker='*' if 'rfc' in method_name else 'd',
                          linestyle='-' if 'rfc' in method_name else ':', )

        # axes_pose.set_title(f'RFC - {dataset_name} - Pose')
        axes_pose.set_xlabel('Mean Runtime (ms)')
        # plt.xlabel('Iters')
        axes_pose.set_ylabel('mAA$_p$(10°)')
        axes_pose.legend()
        if not os.path.exists(f'figs/rfd/{matcher_string}'):
            os.makedirs(f'figs/rfd/{matcher_string}')
        figure_pose.savefig(f'figs/rfd/{matcher_string}/{dataset_name}_{f_method}_pose.pdf')


        # axes_focal.set_title(f'RFC - {dataset_name} - Pose')
        axes_focal.set_xlabel('Mean Runtime (ms)')
        # plt.xlabel('Iters')
        axes_focal.set_ylabel('mAA$_f$(0.1)')
        axes_focal.legend()
        if not os.path.exists(f'figs/rfd/{matcher_string}'):
            os.makedirs(f'figs/rfd/{matcher_string}')
        figure_focal.savefig(f'figs/rfd/{matcher_string}/{dataset_name}_{f_method}_focal.pdf')


def eval_subset(subset, matcher_string, save_string):
    eng = matlab.engine.start_matlab()
    s = eng.genpath('matlab_utils')
    eng.addpath(s, nargout=0)

    eval_manager = RFDManager(subset, eng=eng, verbosity=0)
    eval_manager.load_samples(f'saved/{matcher_string}/{save_string.format(subset)}.joblib', extend=False)
    ret = eval_manager.evaluate()
    if not os.path.exists(f'results/{matcher_string}/rfd'):
        os.makedirs(f'results/{matcher_string}/rfd')

    eval_manager.save_results(f'results/{matcher_string}/rfd/{save_string.format(subset)}.pkl')
    return ret


if __name__ == '__main__':
    args = parse_args()
    save_string, subsets, rows, cols = get_subset_string(args.dataset_name)

    subsets = subsets

    print(subsets)

    if args.load:
        # dfs = [pd.read_pickle(f'results/{args.matcher}/rfd/{save_string.format(subset)}.pkl') for subset in subsets if os.path.exists(f'results/{args.matcher}/rfd/{save_string.format(subset)}.pkl')]
        dfs = [pd.read_pickle(f'results/{args.matcher}/rfd/{save_string.format(subset)}.pkl') for subset in subsets]
    else:
        save_strings = [save_string for _ in subsets]
        matcher_strings = [args.matcher for _ in subsets]
        pool = Pool(args.num_workers)
        dfs = pool.starmap(eval_subset, zip(subsets, matcher_strings, save_strings))
    df = pd.concat(dfs, ignore_index=True)

    draw_plot(df, args.dataset_name, args.matcher)

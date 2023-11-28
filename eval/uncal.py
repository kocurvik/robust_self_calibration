import argparse
import pickle
import sys
from time import perf_counter

import cv2
# import matlab
import numpy as np
import pandas as pd
import poselib
from tqdm import tqdm
import pvsac
import seaborn as sns

from datasets.definitions import get_subset_string
from eval.manager import EvalManager, focal_error, run_eval
from methods.ba import bundle_adjust
from methods.fetzer import fetzer, fetzer_focal_only
from utils.geometry import pose_from_F, angle_matrix, angle, get_K, pose_from_estimated, pose_from_img_info
from methods.hartley import hartley

from methods.base import bougnoux_original
from matlab_utils.engine_calls import ours_uncal

from methods.ours import ours_uncal as ours_uncal_cxx


class UncalManager(EvalManager):
    manager_str = 'uncal'
    methods = ['kukelova', 'kukelova_rfc', 'hartley', 'hartley_rfc', 'fetzer', 'fetzer_focal_rfc', 'bougnoux', 'bougnoux_rfc', 'prior', 'prior_rfc', 'gt', 'gt_rfc']

    method_colors = {'kukelova': sns.color_palette()[0],
                     'hartley': sns.color_palette()[1],
                     'fetzer': sns.color_palette()[2],
                     'bougnoux': sns.color_palette()[3],
                     'prior': sns.color_palette()[4],
                     'gt': sns.color_palette()[5],
                     'kukelova-cxx': sns.color_palette()[6]}

    method_names = {'kukelova': 'Ours',
                    'hartley': 'Hartley',
                    'fetzer': 'Fetzer et al.',
                    'bougnoux': 'Bougnoux',
                    'prior': 'Prior',
                    'gt': 'GT focal',
                    'kukelova-cxx': 'Ours C++'}


    def __init__(self, subset, **kwargs):
        super().__init__(**kwargs)
        self.subset = subset

    def estimate_uncal(self, sample, exif=False):
        cam_1 = sample['cam_1']
        cam_2 = sample['cam_2']
        f_1_gt = cam_1['focal']
        f_2_gt = cam_2['focal']

        img_1, img_2 = sample['img_1'], sample['img_2']
        R_gt, t_gt = pose_from_img_info(img_1, img_2)

        p_1 = np.array([cam_1['width'] / 2, cam_1['height'] / 2])
        p_2 = np.array([cam_2['width'] / 2, cam_2['height'] / 2])

        if exif:
            colmap_1 = cam_1['exif_focal']
            colmap_2 = cam_2['exif_focal']
        else:
            colmap_1 = 1.2 * max(cam_1['width'], cam_1['height'])
            colmap_2 = 1.2 * max(cam_2['width'], cam_2['height'])

        kp_1 = sample['kp_1'] - p_1[np.newaxis, :]
        kp_2 = sample['kp_2'] - p_2[np.newaxis, :]

        try:
            F_best, mask = cv2.findFundamentalMat(kp_1, kp_2, cv2.USAC_MAGSAC, ransacReprojThreshold=3.0, confidence=1.0,
                                                  maxIters=10000)
            info = {'inliers': mask.ravel().astype(bool), 'num_inliers': np.sum(mask)}

            # F_best, info = poselib.estimate_fundamental(kp_1, kp_2, {'max_iterations': 10000,
            #                                                          'min_iterations': 10,
            #                                                          'success_prob': 1.0,
            #                                                          'max_epipolar_error': 3.0,
            #                                                          'progressive_sampling': False})
        except Exception:
            print("RANSAC did not produce any model")
            return




        if 'gt' in self.methods:
            K_1 = np.array([[cam_1['fx'], 0, cam_1['cx'] - p_1[0]], [0, cam_1['fy'], cam_1['cy'] - p_1[1]], [0, 0, 1]])
            K_2 = np.array([[cam_2['fx'], 0, cam_2['cx'] - p_2[0]], [0, cam_2['fy'], cam_2['cy'] - p_2[1]], [0, 0, 1]])

            R, t = pose_from_F(F_best, K_1, K_2,
                               kp_1[info['inliers']],
                               kp_2[info['inliers']])
            entry = {'subset': self.subset, 'method_name': 'gt', 'pp1': np.array([cam_1['cx'], cam_1['cy']])- p_1, 'pp2': np.array([cam_2['cx'], cam_2['cy']])- p_2,
                     'f_1_est': f_1_gt, 'f_2_est': f_2_gt,
                     'R_err': angle_matrix(R.T @ R_gt), 't_err': angle(t, t_gt), 'R': R, 't': t, 'F': F_best,
                     'f_1_err': 0.0, 'f_2_err': 0.0,
                     'f_elapsed': 0, 'iters': 0}
            self.df.loc[len(self.df)] = entry


        # prior
        if 'prior' in self.methods:
            R, t = pose_from_F(F_best, get_K(colmap_1), get_K(colmap_2), kp_1, kp_2)

            entry = {'subset': self.subset, 'method_name': 'prior', 'pp1': np.array([0, 0]), 'pp2': np.array([0, 0]), 'f_1_est': colmap_1, 'f_2_est': colmap_2,
                     'R_err': angle_matrix(R.T @ R_gt), 't_err': angle(t, t_gt), 'R': R, 't': t, 'F': F_best,
                     'f_1_err': focal_error(f_1_gt, colmap_1), 'f_2_err': focal_error(f_2_gt, colmap_2),
                     'f_elapsed': 0, 'iters': 0}
            self.df.loc[len(self.df)] = entry

        # bougnoux
        if 'bougnoux' in self.methods:
            start = perf_counter()
            f_1_est, f_2_est = (np.sqrt(f) for f in bougnoux_original(F_best))
            end = perf_counter()
            R, t = pose_from_estimated(F_best, colmap_1, colmap_2, f_1_est, f_2_est, info, kp_1, kp_2)
            entry = {'subset': self.subset, 'method_name': 'bougnoux', 'pp1': np.array([0, 0]), 'pp2': np.array([0, 0]), 'f_1_est': f_1_est, 'f_2_est': f_2_est,
                     'R_err': angle_matrix(R.T @ R_gt), 't_err': angle(t, t_gt), 'R': R, 't': t, 'F': F_best,
                     'f_1_err': focal_error(f_1_gt, f_1_est), 'f_2_err': focal_error(f_2_gt, f_2_est),
                     'f_elapsed': end - start, 'iters': 0}
            self.df.loc[len(self.df)] = entry

        # hartley
        if 'hartley' in self.methods:
            start = perf_counter()
            f_1_est, f_2_est, pp1, pp2, iters = hartley(F_best, kp_1[info['inliers']], kp_2[info['inliers']], colmap_1, colmap_2)
            end = perf_counter()
            R, t = pose_from_estimated(F_best, colmap_1, colmap_2, f_1_est, f_2_est, info, kp_1, kp_2, pp1, pp2)
            entry = {'subset': self.subset, 'method_name': 'hartley', 'pp1': pp1, 'pp2': pp2, 'f_1_est': f_1_est, 'f_2_est': f_2_est,
                     'R_err': angle_matrix(R.T @ R_gt), 't_err': angle(t, t_gt), 'R': R, 't': t, 'F': F_best,
                     'f_1_err': focal_error(f_1_gt, f_1_est), 'f_2_err': focal_error(f_2_gt, f_2_est),
                     'f_elapsed': end - start, 'iters': iters}
            self.df.loc[len(self.df)] = entry

        # kukelova
        if 'kukelova' in self.methods:
            start = perf_counter()
            f_1_est, f_2_est, pp1, pp2, _, iters = ours_uncal(self.eng, F_best, colmap_1, colmap_2, return_err=True)
            end = perf_counter()
            R, t = pose_from_estimated(F_best, colmap_1, colmap_2, f_1_est, f_2_est, info, kp_1, kp_2, pp1, pp2)
            entry = {'subset': self.subset, 'method_name': 'kukelova', 'pp1': pp1, 'pp2': pp2, 'f_1_est': f_1_est, 'f_2_est': f_2_est,
                     'R_err': angle_matrix(R.T @ R_gt), 't_err': angle(t, t_gt), 'R': R, 't': t, 'F': F_best,
                     'f_1_err': focal_error(f_1_gt, f_1_est), 'f_2_err': focal_error(f_2_gt, f_2_est),
                     'f_elapsed': end - start, 'iters': iters}
            self.df.loc[len(self.df)] = entry

        # kukelova
        if 'kukelova-cxx' in self.methods:
            start = perf_counter()
            f_1_est, f_2_est, pp1, pp2, _, iters = ours_uncal_cxx(F_best, colmap_1, colmap_2, iters=100, return_err=True)
            end = perf_counter()
            R, t = pose_from_estimated(F_best, colmap_1, colmap_2, f_1_est, f_2_est, info, kp_1, kp_2, pp1, pp2)
            entry = {'subset': self.subset, 'method_name': 'kukelova-cxx', 'pp1': pp1, 'pp2': pp2, 'f_1_est': f_1_est, 'f_2_est': f_2_est,
                     'R_err': angle_matrix(R.T @ R_gt), 't_err': angle(t, t_gt), 'R': R, 't': t, 'F': F_best,
                     'f_1_err': focal_error(f_1_gt, f_1_est), 'f_2_err': focal_error(f_2_gt, f_2_est),
                     'f_elapsed': end - start, 'iters': iters}
            self.df.loc[len(self.df)] = entry

        # fetzer
        if 'fetzer' in self.methods:
            start = perf_counter()
            f_1_est, f_2_est, iters = fetzer_focal_only(F_best, colmap_1, colmap_2)
            end = perf_counter()
            R, t = pose_from_estimated(F_best, colmap_1, colmap_2, f_1_est, f_2_est, info, kp_1, kp_2)
            entry = {'subset': self.subset, 'method_name': 'fetzer', 'pp1': np.array([0, 0]), 'pp2': np.array([0, 0]), 'f_1_est': f_1_est, 'f_2_est': f_2_est,
                     'R_err': angle_matrix(R.T @ R_gt), 't_err': angle(t, t_gt), 'R': R, 't': t, 'F': F_best,
                     'f_1_err': focal_error(f_1_gt, f_1_est), 'f_2_err': focal_error(f_2_gt, f_2_est),
                     'f_elapsed': end - start, 'iters': 0}
            self.df.loc[len(self.df)] = entry

        try:
            F_best, mask = cv2.findFundamentalMat(kp_1, kp_2, cv2.USAC_MAGSAC_RFD, ransacReprojThreshold=3.0,
                                                  confidence=1.0,
                                                  maxIters=10000)
            info = {'inliers': mask.ravel().astype(bool), 'num_inliers': np.sum(mask)}

            # F_best, info = poselib.estimate_fundamental_valid_only(kp_1, kp_2, {'max_iterations': 10000,
            #                                                          'min_iterations': 10,
            #                                                          'success_prob': 1.0,
            #                                                          'max_epipolar_error': 3.0,
            #                                                          'progressive_sampling': False})

        except Exception:
            print("RFC did not produce any model")




        if 'gt_rfc' in self.methods:
            K_1 = np.array([[cam_1['fx'], 0, cam_1['cx'] - p_1[0]], [0, cam_1['fy'], cam_1['cy'] - p_1[1]], [0, 0, 1]])
            K_2 = np.array([[cam_2['fx'], 0, cam_2['cx'] - p_2[0]], [0, cam_2['fy'], cam_2['cy'] - p_2[1]], [0, 0, 1]])

            R, t = pose_from_F(F_best, K_1, K_2,
                               kp_1[info['inliers']],
                               kp_2[info['inliers']])

            entry = {'subset': self.subset, 'method_name': 'gt_rfc', 'pp1': np.array([cam_1['cx'], cam_1['cy']])- p_1, 'pp2': np.array([cam_2['cx'], cam_2['cy']])- p_2,
                     'f_1_est': f_1_gt, 'f_2_est': f_2_gt,
                     'R_err': angle_matrix(R.T @ R_gt), 't_err': angle(t, t_gt), 'R': R, 't': t, 'F': F_best,
                     'f_1_err': 0.0, 'f_2_err': 0.0,
                     'f_elapsed': 0, 'iters': 0}
            self.df.loc[len(self.df)] = entry

        # prior
        if 'prior_rfc' in self.methods:
            R, t = pose_from_F(F_best, get_K(colmap_1), get_K(colmap_2), kp_1, kp_2)
            entry = {'subset': self.subset, 'method_name': 'prior_rfc', 'pp1': np.array([0, 0]), 'pp2': np.array([0, 0]), 'f_1_est': colmap_1, 'f_2_est': colmap_2,
                     'R_err': angle_matrix(R.T @ R_gt), 't_err': angle(t, t_gt), 'R': R, 't': t, 'F': F_best,
                     'f_1_err': focal_error(f_1_gt, colmap_1), 'f_2_err': focal_error(f_2_gt, colmap_2),
                     'f_elapsed': 0, 'iters': 0}
            self.df.loc[len(self.df)] = entry

        # bougnoux
        if 'bougnoux_rfc' in self.methods:
            start = perf_counter()
            f_1_est, f_2_est = (np.sqrt(f) for f in bougnoux_original(F_best))
            end = perf_counter()
            R, t = pose_from_estimated(F_best, colmap_1, colmap_2, f_1_est, f_2_est, info, kp_1, kp_2)
            entry = {'subset': self.subset, 'method_name': 'bougnoux_rfc', 'pp1': np.array([0, 0]), 'pp2': np.array([0, 0]), 'f_1_est': f_1_est, 'f_2_est': f_2_est,
                     'R_err': angle_matrix(R.T @ R_gt), 't_err': angle(t, t_gt), 'R': R, 't': t, 'F': F_best,
                     'f_1_err': focal_error(f_1_gt, f_1_est), 'f_2_err': focal_error(f_2_gt, f_2_est),
                     'f_elapsed': end - start, 'iters': 0}
            self.df.loc[len(self.df)] = entry


        # hartley
        if 'hartley_rfc' in self.methods:
            start = perf_counter()
            f_1_est, f_2_est, pp1, pp2, iters = hartley(F_best, kp_1[info['inliers']], kp_2[info['inliers']], colmap_1, colmap_2)
            end = perf_counter()
            R, t = pose_from_estimated(F_best, colmap_1, colmap_2, f_1_est, f_2_est, info, kp_1, kp_2, pp1, pp2)
            entry = {'subset': self.subset, 'method_name': 'hartley_rfc', 'pp1': pp1, 'pp2': pp2, 'f_1_est': f_1_est, 'f_2_est': f_2_est,
                     'R_err': angle_matrix(R.T @ R_gt), 't_err': angle(t, t_gt), 'R': R, 't': t, 'F': F_best,
                     'f_1_err': focal_error(f_1_gt, f_1_est), 'f_2_err': focal_error(f_2_gt, f_2_est),
                     'f_elapsed': end - start, 'iters': iters}
            self.df.loc[len(self.df)] = entry

        # kukelova
        if 'kukelova_rfc' in self.methods:
            start = perf_counter()
            f_1_est, f_2_est, pp1, pp2, _, iters = ours_uncal(self.eng, F_best, colmap_1, colmap_2, return_err=True)
            end = perf_counter()
            R, t = pose_from_estimated(F_best, colmap_1, colmap_2, f_1_est, f_2_est, info, kp_1, kp_2, pp1, pp2)
            entry = {'subset': self.subset, 'method_name': 'kukelova_rfc', 'pp1': pp1, 'pp2': pp2, 'f_1_est': f_1_est, 'f_2_est': f_2_est,
                     'R_err': angle_matrix(R.T @ R_gt), 't_err': angle(t, t_gt), 'R': R, 't': t, 'F': F_best,
                     'f_1_err': focal_error(f_1_gt, f_1_est), 'f_2_err': focal_error(f_2_gt, f_2_est),
                     'f_elapsed': end - start, 'iters': iters}
            self.df.loc[len(self.df)] = entry

        # kukelova
        if 'kukelova-cxx_rfc' in self.methods:
            start = perf_counter()
            f_1_est, f_2_est, pp1, pp2, _, iters = ours_uncal_cxx(F_best, colmap_1, colmap_2, iters=100, return_err=True)
            end = perf_counter()
            R, t = pose_from_estimated(F_best, colmap_1, colmap_2, f_1_est, f_2_est, info, kp_1, kp_2, pp1, pp2)
            entry = {'subset': self.subset, 'method_name': 'kukelova-cxx_rfc', 'pp1': pp1, 'pp2': pp2, 'f_1_est': f_1_est, 'f_2_est': f_2_est,
                     'R_err': angle_matrix(R.T @ R_gt), 't_err': angle(t, t_gt), 'R': R, 't': t, 'F': F_best,
                     'f_1_err': focal_error(f_1_gt, f_1_est), 'f_2_err': focal_error(f_2_gt, f_2_est),
                     'f_elapsed': end - start, 'iters': iters}
            self.df.loc[len(self.df)] = entry

        # fetzer
        if 'fetzer_focal_rfc' in self.methods:
            start = perf_counter()
            f_1_est, f_2_est, iters = fetzer_focal_only(F_best, colmap_1, colmap_2)
            end = perf_counter()
            R, t = pose_from_estimated(F_best, colmap_1, colmap_2, f_1_est, f_2_est, info, kp_1, kp_2)
            entry = {'subset': self.subset, 'method_name': 'fetzer_focal_rfc', 'pp1': np.array([0, 0]), 'pp2': np.array([0, 0]), 'f_1_est': f_1_est, 'f_2_est': f_2_est,
                     'R_err': angle_matrix(R.T @ R_gt), 't_err': angle(t, t_gt), 'R': R, 't': t, 'F': F_best,
                     'f_1_err': focal_error(f_1_gt, f_1_est), 'f_2_err': focal_error(f_2_gt, f_2_est),
                     'f_elapsed': end - start, 'iters': iters}
            self.df.loc[len(self.df)] = entry


    def evaluate(self, exif=False):
        self.df = pd.DataFrame(columns=['subset', 'method_name', 'pp1', 'pp2', 'R_err', 't_err', 'R', 't', 'F', 'f_1_est', 'f_2_est', 'f_1_err', 'f_2_err', 'f_elapsed', 'iters'])
        # self.samples = [s for i, s in enumerate(self.samples) if i % 10 == 0]
        for sample in tqdm(self.samples, disable=self.verbosity > 0):
            self.estimate_uncal(sample, exif=exif)

        return self.df

    @staticmethod
    def f_errs(df):
        errs_1 = df['f_1_err'].to_numpy()
        errs_2 = df['f_2_err'].to_numpy()
        errs = np.concatenate([errs_1, errs_2])
        return errs

if __name__ == '__main__':
    run_eval(UncalManager)
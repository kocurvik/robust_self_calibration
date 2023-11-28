import cv2
import numpy as np
import pandas as pd
import poselib
import pvsac
from tqdm import tqdm
import seaborn as sns

from eval.manager import EvalManager, focal_error, run_eval
from methods.base import get_focal_sturm, single_6pt_minimal
from methods.fetzer import fetzer_single
from methods.hartley import hartley_sturm, hartley_single
from utils.geometry import pose_from_F, angle_matrix, angle, get_K, pose_from_estimated, pose_from_img_info

from matlab_utils.engine_calls import ours_single, ours_uncal


class OneFocalManager(EvalManager):
    manager_str = 'single'
    methods = ['kukelova', 'kukelova_rfc', 'hartley', 'hartley_rfc', 'fetzer', 'fetzer_rfc','sturm', 'sturm_rfc',
               'minimal', 'prior', 'prior_rfc' , 'gt', 'gt_rfc']

    method_colors = {'kukelova': sns.color_palette()[0],
                     'kukelova-diff': sns.color_palette()[6],
                     'hartley': sns.color_palette()[1],
                     'minimal': sns.color_palette()[5],
                     'fetzer': sns.color_palette()[2],
                     'sturm': sns.color_palette()[6],
                     'prior': sns.color_palette()[4],
                     'gt': sns.color_palette()[7]}

    method_names = {'kukelova': 'Ours',
                    'fetzer': 'Fetzer',
                    'sturm': 'Sturm',
                    'minimal': 'Minimal',
                    'prior': 'Prior',
                    'gt': 'GT focal',
                    'hartley': 'Hartley'}

    def __init__(self, subset, semi=True, **kwargs):
        super().__init__(**kwargs)
        self.subset = subset
        self.semi = semi

    def estimate_single(self, sample, exif=False):
        cam = sample['cam_1']

        p = np.array([cam['width'] / 2, cam['height'] / 2])
        f_gt = cam['focal']

        if exif:
            colmap = cam['exif_focal']
        else:
            colmap = 1.2 * max(cam['width'], cam['height'])

        kp_1 = sample['kp_1'] - p[np.newaxis, :]
        kp_2 = sample['kp_2'] - p[np.newaxis, :]

        img_1, img_2 = sample['img_1'], sample['img_2']
        R_gt, t_gt = pose_from_img_info(img_1, img_2)

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

        if F_best is None:
            print("RANSAC did not produce any model")
            return


        if 'gt' in self.methods:
            K = np.array([[cam['fx'], 0, cam['cx'] - p[0]], [0, cam['fy'], cam['cy'] - p[1]], [0, 0, 1]])

            R, t = pose_from_F(F_best, K, K,
                               kp_1[info['inliers']],
                               kp_2[info['inliers']])
            entry = {'subset': self.subset, 'method_name': 'gt', 'pp1': np.array([cam['cx'], cam['cy']])- p,
                     'f_est': f_gt, 'f_err': 0.0,
                     'R_err': angle_matrix(R.T @ R_gt), 't_err': angle(t, t_gt), 'R': R, 't': t, 'F': F_best,
                     'f_1_err': 0.0, 'f_2_err': 0.0}
            self.df.loc[len(self.df)] = entry


        # prior
        if 'prior' in self.methods:
            R, t = pose_from_F(F_best, get_K(colmap), get_K(colmap), kp_1, kp_2)
            entry = {'subset': self.subset, 'method_name': 'prior', 'pp1': [0.0, 0.0], 'f_est': colmap,
                     'R_err': angle_matrix(R.T @ R_gt), 't_err': angle(t, t_gt), 'R': R, 't': t, 'F': F_best,
                     'f_err': focal_error(f_gt, colmap)}
            self.df.loc[len(self.df)] = entry

        # kukelova
        if 'kukelova' in self.methods:
            f_est, pp = ours_single(self.eng, F_best, colmap, w1=0.01)
            R, t = pose_from_estimated(F_best, colmap, colmap, f_est, f_est, info, kp_1, kp_2, pp, pp)
            entry = {'subset': self.subset, 'method_name': 'kukelova', 'pp': pp, 'f_est': f_est,
                     'R_err': angle_matrix(R.T @ R_gt), 't_err': angle(t, t_gt), 'R': R, 't': t, 'F': F_best,
                     'f_err': focal_error(f_gt, f_est)}
            self.df.loc[len(self.df)] = entry

        if 'kukelova-diff' in self.methods:
            f_1_est, f_2_est, pp_1, pp_2 = ours_uncal(self.eng, F_best, colmap, colmap, w1=0.005, w3=0.005)
            f_est = (f_1_est + f_2_est) / 2
            pp = (pp_1 + pp_2) / 2
            R, t = pose_from_estimated(F_best, colmap, colmap, f_est, f_est, info, kp_1, kp_2, pp, pp)
            entry = {'subset': self.subset, 'method_name': 'kukelova-diff', 'pp': pp, 'f_est': f_est,
                     'R_err': angle_matrix(R.T @ R_gt), 't_err': angle(t, t_gt), 'R': R, 't': t, 'F': F_best,
                     'f_err': focal_error(f_gt, f_est)}
            self.df.loc[len(self.df)] = entry

        if 'sturm' in self.methods:
            f_est = get_focal_sturm(F_best)
            R, t = pose_from_estimated(F_best, colmap, colmap, f_est, f_est, info, kp_1, kp_2, [0.0, 0.0], [0.0, 0.0])
            entry = {'subset': self.subset, 'method_name': 'sturm', 'pp': [0.0, 0.0], 'f_est': f_est,
                     'R_err': angle_matrix(R.T @ R_gt), 't_err': angle(t, t_gt), 'R': R, 't': t, 'F': F_best,
                     'f_err': focal_error(f_gt, f_est)}
            self.df.loc[len(self.df)] = entry


        if 'fetzer' in self.methods:
            f_est, _ = fetzer_single(F_best, colmap)
            R, t = pose_from_estimated(F_best, colmap, colmap, f_est, f_est, info, kp_1, kp_2)
            entry = {'subset': self.subset, 'method_name': 'fetzer', 'pp': [0.0, 0.0], 'f_est': f_est,
                     'R_err': angle_matrix(R.T @ R_gt), 't_err': angle(t, t_gt), 'R': R, 't': t, 'F': F_best,
                     'f_err': focal_error(f_gt, f_est)}
            self.df.loc[len(self.df)] = entry

        if 'minimal' in self.methods:
            f_est, R, t = single_6pt_minimal(kp_1, kp_2, {'max_iterations': 10000,
                                                          'min_iterations': 10,
                                                          'success_prob': 1.0,
                                                          'max_epipolar_error': 3.0,
                                                          'progressive_sampling': False})

            entry = {'subset': self.subset, 'method_name': 'minimal', 'pp': [0.0, 0.0], 'f_est': f_est,
                     'R_err': angle_matrix(R.T @ R_gt), 't_err': angle(t, t_gt), 'R': R, 't': t, 'F': F_best,
                     'f_err': focal_error(f_gt, f_est)}
            self.df.loc[len(self.df)] = entry

        if 'hartley' in self.methods:
            f_est, pp, _ = hartley_single(F_best, kp_1[info['inliers']], kp_2[info['inliers']], colmap)
            R, t = pose_from_estimated(F_best, colmap, colmap, f_est, f_est, info, kp_1, kp_2, pp, pp)
            entry = {'subset': self.subset, 'method_name': 'hartley', 'pp': pp, 'f_est': f_est,
                     'R_err': angle_matrix(R.T @ R_gt), 't_err': angle(t, t_gt), 'R': R, 't': t, 'F': F_best,
                     'f_err': focal_error(f_gt, f_est)}
            self.df.loc[len(self.df)] = entry

        try:
            F_best, mask = cv2.findFundamentalMat(kp_1, kp_2, cv2.USAC_MAGSAC_RFD, ransacReprojThreshold=3.0, confidence=1.0,
                                                  maxIters=10000)
            info = {'inliers': mask.ravel().astype(bool), 'num_inliers': np.sum(mask)}

            # F_best, info = poselib.estimate_fundamental_valid_only(kp_1, kp_2, {'max_iterations': 10000,
            #                                                          'min_iterations': 10,
            #                                                          'success_prob': 1.0,
            #                                                          'max_epipolar_error': 3.0,
            #                                                          'progressive_sampling': False})
        except Exception:
            print("RANSAC did not produce any model")
            return

        if F_best is None:
            print("RANSAC did not produce any model")
            return


        if 'gt_rfc' in self.methods:
            K = np.array([[cam['fx'], 0, cam['cx'] - p[0]], [0, cam['fy'], cam['cy'] - p[1]], [0, 0, 1]])

            R, t = pose_from_F(F_best, K, K,
                               kp_1[info['inliers']],
                               kp_2[info['inliers']])
            entry = {'subset': self.subset, 'method_name': 'gt_rfc', 'pp1': np.array([cam['cx'], cam['cy']])- p,
                     'f_est': f_gt, 'f_err': 0.0,
                     'R_err': angle_matrix(R.T @ R_gt), 't_err': angle(t, t_gt), 'R': R, 't': t, 'F': F_best,
                     'f_1_err': 0.0, 'f_2_err': 0.0}
            self.df.loc[len(self.df)] = entry


        # prior
        if 'prior_rfc' in self.methods:
            R, t = pose_from_F(F_best, get_K(colmap), get_K(colmap), kp_1, kp_2)
            entry = {'subset': self.subset, 'method_name': 'prior_rfc', 'pp1': [0.0, 0.0], 'f_est': colmap,
                     'R_err': angle_matrix(R.T @ R_gt), 't_err': angle(t, t_gt), 'R': R, 't': t, 'F': F_best,
                     'f_err': focal_error(f_gt, colmap)}
            self.df.loc[len(self.df)] = entry

        # kukelova
        if 'kukelova_rfc' in self.methods:
            f_est, pp = ours_single(self.eng, F_best, colmap, w1=0.01)
            R, t = pose_from_estimated(F_best, colmap, colmap, f_est, f_est, info, kp_1, kp_2, pp, pp)
            entry = {'subset': self.subset, 'method_name': 'kukelova_rfc', 'pp': pp, 'f_est': f_est,
                     'R_err': angle_matrix(R.T @ R_gt), 't_err': angle(t, t_gt), 'R': R, 't': t, 'F': F_best,
                     'f_err': focal_error(f_gt, f_est)}
            self.df.loc[len(self.df)] = entry

        if 'sturm_rfc' in self.methods:
            f_est = get_focal_sturm(F_best)
            R, t = pose_from_estimated(F_best, colmap, colmap, f_est, f_est, info, kp_1, kp_2, [0.0, 0.0], [0.0, 0.0])
            entry = {'subset': self.subset, 'method_name': 'sturm_rfc', 'pp': [0.0, 0.0], 'f_est': f_est,
                     'R_err': angle_matrix(R.T @ R_gt), 't_err': angle(t, t_gt), 'R': R, 't': t, 'F': F_best,
                     'f_err': focal_error(f_gt, f_est)}
            self.df.loc[len(self.df)] = entry

        if 'fetzer_rfc' in self.methods:
            f_est, _ = fetzer_single(F_best, colmap)
            R, t = pose_from_estimated(F_best, colmap, colmap, f_est, f_est, info, kp_1, kp_2)
            entry = {'subset': self.subset, 'method_name': 'fetzer_rfc', 'pp': [0.0, 0.0], 'f_est': f_est,
                     'R_err': angle_matrix(R.T @ R_gt), 't_err': angle(t, t_gt), 'R': R, 't': t, 'F': F_best,
                     'f_err': focal_error(f_gt, f_est)}
            self.df.loc[len(self.df)] = entry

        if 'hartley_rfc' in self.methods:
            f_est, pp, _ = hartley_single(F_best, kp_1[info['inliers']], kp_2[info['inliers']], colmap)
            R, t = pose_from_estimated(F_best, colmap, colmap, f_est, f_est, info, kp_1, kp_2, pp, pp)
            entry = {'subset': self.subset, 'method_name': 'hartley_rfc', 'pp': pp, 'f_est': f_est,
                     'R_err': angle_matrix(R.T @ R_gt), 't_err': angle(t, t_gt), 'R': R, 't': t, 'F': F_best,
                     'f_err': focal_error(f_gt, f_est)}
            self.df.loc[len(self.df)] = entry


    def evaluate(self, exif=False):
        self.df = pd.DataFrame(
            columns=['subset', 'method_name', 'pp', 'R_err', 't_err', 'R', 't', 'F', 'f_est', 'f_err'])
        for sample in tqdm(self.samples, disable=self.verbosity > 0):
            self.estimate_single(sample, exif=exif)

        return self.df

    @staticmethod
    def f_errs(df):
        return df['f_err'].to_numpy()


if __name__ == '__main__':
    run_eval(OneFocalManager)
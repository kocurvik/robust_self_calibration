import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from eval.manager import EvalManager, focal_error, run_eval
from utils.geometry import pose_from_F, angle_matrix, angle, get_K, pose_from_estimated, pose_from_img_info

from methods.base import focal_svd
from matlab_utils.engine_calls import ours_uncal


class OneFocalManager(EvalManager):
    manager_str = 'onefocal'

    def __init__(self, subset, semi=True, **kwargs):
        super().__init__(**kwargs)
        self.subset = subset
        self.semi = semi
        self.methods = ['svd', 'kukelova', 'prior']

    def estimate_uncal(self, sample, exif=False):
        cam_1 = sample['cam_1']
        cam_2 = sample['cam_2']

        p_1 = np.array([cam_1['width'] / 2, cam_1['height'] / 2])
        p_2 = np.array([cam_2['width'] / 2, cam_2['height'] / 2])
        f_1_gt = cam_1['focal']
        f_2_gt = cam_2['focal']

        if exif:
            colmap_1 = cam_1['exif_focal']
            colmap_2 = cam_2['exif_focal']
        else:
            colmap_1 = 1.2 * max(cam_1['width'], cam_1['height'])
            colmap_2 = 1.2 * max(cam_2['width'], cam_2['height'])

        kp_1 = sample['kp_1'] - p_1[np.newaxis, :]
        kp_2 = sample['kp_2'] - p_2[np.newaxis, :]

        img_1, img_2 = sample['img_1'], sample['img_2']
        R_gt, t_gt = pose_from_img_info(img_1, img_2)

        F_best, mask = cv2.findFundamentalMat(kp_1, kp_2, cv2.USAC_MAGSAC)
        info = {'inliers': mask.ravel().astype(np.bool), 'num_inliers': np.sum(mask)}

        # prior
        R, t = pose_from_F(F_best, get_K(colmap_1), get_K(cam_2['focal']), kp_1, kp_2)
        entry = {'subset': self.subset, 'method_name': 'prior', 'pp1': p_1, 'pp2': p_2, 'f_1_est': colmap_1,
                 'R_err': angle_matrix(R.T @ R_gt), 't_err': angle(t, t_gt), 'R': R, 't': t, 'F': F_best,
                 'f_err': focal_error(f_1_gt, colmap_1)}
        self.df.loc[len(self.df)] = entry

        # svd
        f_1_est = np.sqrt(focal_svd(F_best, cam_2['focal']))
        R, t = pose_from_estimated(F_best, colmap_1, colmap_2, f_1_est, cam_2['focal'], info, kp_1, kp_2)
        entry = {'subset': self.subset, 'method_name': 'svd', 'pp1': p_1, 'pp2': p_2, 'f_1_est': f_1_est,
                 'R_err': angle_matrix(R.T @ R_gt), 't_err': angle(t, t_gt), 'R': R, 't': t, 'F': F_best,
                 'f_err': focal_error(f_1_gt, f_1_est)}
        self.df.loc[len(self.df)] = entry

        # kukelova
        f_1_est, f_2_est, pp1, pp2 = ours_uncal(self.eng, F_best, colmap_1, cam_2['focal'], w3=100, w4=100)
        R, t = pose_from_estimated(F_best, colmap_1, colmap_2, f_1_est, cam_2['focal'], info, kp_1, kp_2, pp1)
        entry = {'subset': self.subset, 'method_name': 'kukelova', 'pp1': p_1, 'pp2': p_2, 'f_1_est': f_1_est,
                 'R_err': angle_matrix(R.T @ R_gt), 't_err': angle(t, t_gt), 'R': R, 't': t, 'F': F_best,
                 'f_err': focal_error(f_1_gt, f_1_est)}
        self.df.loc[len(self.df)] = entry

    def evaluate(self, exif=False):
        self.df = pd.DataFrame(columns=['subset', 'method_name', 'pp1', 'pp2', 'R_err', 't_err', 'R', 't', 'F', 'f_est', 'f_err'])
        for sample in tqdm(self.samples, disable=self.verbosity > 0):
            self.estimate_uncal(sample, exif=exif)

        return self.df

    @staticmethod
    def f_errs(df):
        return df['f_err'].to_numpy()


if __name__ == '__main__':
    run_eval(OneFocalManager)
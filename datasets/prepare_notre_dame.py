import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import pycolmap

from scipy.spatial.transform import Rotation

from eval.uncal import UncalManager
from utils.matching import LoFTRMatcher


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_samples', type=int, default=1000)
    parser.add_argument('-rs', '--reload_samples', action='store_true', default=False)
    parser.add_argument('dataset_path')

    return parser.parse_args()

def load_phototourism_data(dataset_path):

    list_path = dataset_path / 'list.txt'
    with open(list_path, 'r') as f:
        img_list = f.readlines()

    img_list = [s.strip() for s in img_list]
    data_path = dataset_path / 'notredame.out'

    num_views, num_points = np.loadtxt(data_path, max_rows=1)
    num_views, num_points = int(num_views), int(num_views)

    cam_data = np.loadtxt(data_path, skiprows=2, max_rows=num_views * 5)
    cam_list = []

    for i, img_path in enumerate(img_list):
        focal, k1, k2 = cam_data[5 * i]
        R = cam_data[5 * i + 1: 5 * i + 4, :]
        t = cam_data[5 * i + 4, :]

        qx, qy, qz, qw = Rotation.from_matrix(R).as_quat()

        img_path_undistorted = img_path.replace('.jpg', '.rd.jpg')

        exif_camera = pycolmap.infer_camera_from_image(str(dataset_path / img_path))
        default_focal = 1.2 * max(exif_camera.width, exif_camera.height)
        exif_focal = exif_camera.params[0] if default_focal != exif_camera.params[0] else None
        # exif_focal = exif_camera.params[0]

        cam = {'focal': focal, 'k1': k1, 'k2': k2, 't': t, 'R': R, 'q': np.array([qw, qx, qy, qz]), 'image_path': img_path,
               'image_path_undistorted': img_path_undistorted, 'visible_pts': [], 'width': exif_camera.width, 'height': exif_camera.height, 'exif_focal': exif_focal}
        cam_list.append(cam)

    pt_list = []
    with open(data_path, 'r') as f:
        pt_lines = f.readlines()
    pt_lines = pt_lines[num_views * 5 + 2:]
    for i in range(num_points):
        pos = np.fromstring(pt_lines[i * 3].strip(), sep=' ')
        color = np.fromstring(pt_lines[i * 3 + 1].strip(), sep=' ').astype(int)
        views = np.fromstring(pt_lines[i * 3 + 2].strip(), sep=' ')
        views = views[1::4].astype(int)

        pt = {'pos': pos, 'color': color, 'views': views}
        pt_list.append(pt)
        for j in views:
            cam_list[j]['visible_pts'].append(i)

    return cam_list, pt_list


def cam_pair_generator(dataset_path, cam_list, min_overlap=10, max_pairs=100, seed=42, verbose=0):
    np.random.seed(seed)
    output = 0

    matcher = LoFTRMatcher(max_dim=512, device='cuda')

    while output < max_pairs:
        cam_1, cam_2 = random.sample(cam_list, 2)

        if cam_1['exif_focal'] is None or cam_2['exif_focal'] is None:
            continue

        overlap = len(set(cam_1['visible_pts']).intersection(set(cam_2['visible_pts'])))

        if overlap >= min_overlap:
            img_1 = cv2.imread(str(dataset_path / cam_1['image_path']))
            img_2 = cv2.imread(str(dataset_path / cam_2['image_path']))


            if verbose:
                cv2.imshow('img_1', img_1)
                cv2.imshow('img_2', img_2)
                cv2.waitKey(1)

            conf, kp_1, kp_2 = matcher.match(img_1, img_2)

            if len(kp_1) > 6:
                output += 1
                yield {'conf': conf, 'kp_1': kp_1, 'kp_2': kp_2, 'cam_1': cam_1, 'cam_2': cam_2, 'img_1': cam_1,
                       'img_2': cam_2}


def run_im(args):
    dataset_path = Path(args.dataset_path)
    cam_list, pt_list = load_phototourism_data(dataset_path)

    gen = cam_pair_generator(dataset_path, cam_list, max_pairs=args.num_samples)
    eval_manager = UncalManager(verbosity=0)
    eval_manager.add_generator_samples(gen, args.num_samples)
    eval_manager.save_samples('saved/nd_exif.joblib')


if __name__ == '__main__':
    args = parse_args()
    run_im(args)
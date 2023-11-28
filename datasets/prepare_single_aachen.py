import argparse
import os
import random
from multiprocessing import Pool
from pathlib import Path

import cv2
import joblib
import numpy as np
from tqdm import tqdm

from datasets.definitions import img_to_dict, cam_to_dict
from utils.matching import LoFTRMatcher, SIFTMatcher, get_matcher

from utils.read_write_colmap import read_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--compress', type=int, default=3)
    parser.add_argument('-n', '--num_samples', type=int, default=200)
    parser.add_argument('-w', '--num_workers', type=int, default=1)
    parser.add_argument('-m', '--matcher', type=str, default='loftr')
    parser.add_argument('dataset_path')

    return parser.parse_args()


def get_area(pts):
    width = np.max(pts[:, 0]) - np.min(pts[:, 0])
    height = np.max(pts[:, 1]) - np.min(pts[:, 1])
    return width * height


def cam_pair_generator(matcher, img_path, cameras, images, pts, min_overlap=10, max_pairs=100, seed=42, verbose=0):
    np.random.seed(seed)
    random.seed(seed)
    output = 0

    cam_imgs = {cam_id: [] for cam_id in cameras.keys()}

    for img_id, img in images.items():
        cam_id = img.camera_id
        cam_imgs[cam_id].append(img_id)

    valid_cam_ids = [cam_id for cam_id in cameras.keys() if len(cam_imgs[cam_id]) > 20]

    for cam_id in valid_cam_ids:
        output = 0
        while output < max_pairs:
            try:
                cam_1 = cameras[cam_id]
                cam_2 = cameras[cam_id]

                img_id1, img_id2 = random.sample(cam_imgs[cam_id], 2)

                img1 = images[img_id1]
                img2 = images[img_id2]

                img1_point3D_ids = np.array(img1.point3D_ids)
                img1_point3D_ids = img1_point3D_ids[img1_point3D_ids != -1]
                img2_point3D_ids = np.array(img2.point3D_ids)
                img2_point3D_ids = img2_point3D_ids[img2_point3D_ids != -1]
                overlap = set(img1_point3D_ids).intersection(set(img2_point3D_ids))

                if len(overlap) < 8:
                    continue

                pts_img_1 = []
                pts_img_2 = []

                for pt_id in list(overlap):
                    pt = pts[pt_id]
                    if img_id1 in pt.image_ids and img_id2 in pt.image_ids:
                        idx1 = np.where(pt.image_ids == img_id1)[0][0]
                        idx2 = np.where(pt.image_ids == img_id2)[0][0]

                        im_idx1 = pt.point2D_idxs[idx1]
                        im_idx2 = pt.point2D_idxs[idx2]

                        pts_img_1.append(img1.xys[im_idx1])
                        pts_img_2.append(img2.xys[im_idx2])

                pts_img_1 = np.array(pts_img_1)
                pts_img_2 = np.array(pts_img_2)

                area_1 = get_area(pts_img_1) / (cam_1.width * cam_1.height)
                area_2 = get_area(pts_img_2) / (cam_2.width * cam_2.height)

                if area_1 > 0.1 and area_2 > 0.1:
                # if len(overlap) >= min_overlap:
                    img_1_path = os.path.join(img_path, img1.name)
                    img_2_path = os.path.join(img_path, img2.name)
                    if not os.path.exists(img_1_path) or not os.path.exists(img_2_path):
                        continue
                    img_1 = cv2.imread(img_1_path)
                    img_2 = cv2.imread(img_2_path)

                    if img_1 is None or img_2 is None:
                        continue

                    if cam_1.width != img_1.shape[1]:
                        if cam_1.width == img_1.shape[0]:
                            img_1 = np.swapaxes(img_1, 0, 1)
                        else:
                            continue

                    if cam_2.width != img_2.shape[1]:
                        if cam_2.width == img_2.shape[0]:
                            img_2 = np.swapaxes(img_2, 0, 1)
                        else:
                            continue

                    if verbose:
                        cv2.imshow('img_1', img_1)
                        cv2.imshow('img_2', img_2)
                        cv2.waitKey(1)

                    conf, kp_1, kp_2 = matcher.match(img_1, img_2)

                    if len(kp_1) > 6:
                        output += 1
                        yield {'conf': conf, 'kp_1': kp_1, 'kp_2': kp_2,
                               'cam_1': cam_to_dict(cam_1), 'cam_2': cam_to_dict(cam_2),
                               'img_1': img_to_dict(img1), 'img_2': img_to_dict(img2)}
            except Exception:
                continue


def run_im(args):
    dataset_path = Path(args.dataset_path)
    matcher = get_matcher(args.matcher)

    prepare_single(args.num_samples, dataset_path, matcher)


def prepare_single(num_samples, dataset_path, matcher):
    model_path = os.path.join(dataset_path, '3D-models/aachen_v_1_1')
    img_path = os.path.join(dataset_path, 'images_upright')

    cameras, images, points = read_model(model_path)

    gen = cam_pair_generator(matcher, img_path, cameras, images, points, max_pairs=num_samples)
    samples = [s for s in tqdm(gen, total=2 * num_samples)]
    if not os.path.exists('saved/{}'.format(args.matcher)):
        os.makedirs('saved/{}'.format(args.matcher))
    joblib.dump(samples, 'saved/{}/single_aachen.joblib'.format(args.matcher))


if __name__ == '__main__':
    args = parse_args()
    run_im(args)
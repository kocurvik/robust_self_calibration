import argparse
import itertools
import os
import random
from pathlib import Path

import cv2
import joblib
import numpy as np
from tqdm import tqdm

from datasets.definitions import img_to_dict, cam_to_dict
from utils.matching import LoFTRMatcher, get_matcher

from utils.read_write_colmap import read_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--skip', type=int, default=5)
    parser.add_argument('-c', '--compress', type=int, default=3)
    parser.add_argument('-n', '--num_samples', type=int, default=200)
    parser.add_argument('-o', '--original', action='store_true', default=False)
    parser.add_argument('-m', '--matcher', type=str, default='loftr')
    parser.add_argument('dataset_path')

    return parser.parse_args()


def cam_pair_generator(matcher, dataset_path, cameras, images, pts, single=False, min_overlap=500, max_pairs=100, seed=42, verbose=0):
    np.random.seed(seed)
    output = 0

    image_id_lists = {}
    for cam_id in cameras.keys():
        image_id_lists[cam_id] = sorted([im for im in images.values() if im.camera_id == cam_id], key= lambda x: x.id)


    while output < max_pairs:
        if single:
            id1 = random.choice(list(cameras.keys()))
            id2 = id1
        else:
            id1, id2 = random.sample(list(cameras.keys()), 2)
        cam_1 = cameras[id1]
        cam_2 = cameras[id2]

        idx1 = np.random.choice(len(image_id_lists[id1]))
        idx2 = np.random.choice(len(image_id_lists[id1]))
        img1 = image_id_lists[id1][idx1]
        img2 = image_id_lists[id2][idx2]

        set_1 = set(image_id_lists[0][idx1].point3D_ids)\
            .union(set(image_id_lists[1][idx1].point3D_ids))\
            .union(set(image_id_lists[2][idx1].point3D_ids))\
            .union(set(image_id_lists[3][idx1].point3D_ids))

        set_2 = set(image_id_lists[0][idx2].point3D_ids)\
            .union(set(image_id_lists[1][idx2].point3D_ids))\
            .union(set(image_id_lists[2][idx2].point3D_ids))\
            .union(set(image_id_lists[3][idx2].point3D_ids))

        overlap = set_1.intersection(set_2)

        # pts_img_1 = []
        # pts_img_2 = []
        #
        # for pt in pts.values():
        #     if img1.id in pt.image_ids and img2.id in pt.image_ids:
        #         idx1 = np.where(pt.image_ids == id1)[0][0]
        #         idx2 = np.where(pt.image_ids == id2)[0][0]
        #
        #         im_idx1 = pt.point2D_idxs[idx1]
        #         im_idx2 = pt.point2D_idxs[idx2]
        #
        #         pts_img_1.append(img1.xys[im_idx1])
        #         pts_img_2.append(img2.xys[im_idx2])

        if len(overlap) >= min_overlap:
            img_1 = cv2.imread(f'{dataset_path}/images/{img1.name}')
            img_2 = cv2.imread(f'{dataset_path}/images/{img2.name}')

            if verbose:
                # plt.imshow(img_1[:, :, ::-1])
                # plt.show()
                # plt.imshow(img_2[:, :, ::-1])
                # plt.show()
                cv2.imshow('img_1', img_1)
                cv2.imshow('img_2', img_2)
                cv2.waitKey(1)

            conf, kp_1, kp_2 = matcher.match(img_1, img_2)

            if len(kp_1) > min_overlap:
                output += 1
                yield {'conf': conf, 'kp_1': kp_1, 'kp_2': kp_2,
                       'cam_1': cam_to_dict(cam_1), 'cam_2': cam_to_dict(cam_2),
                       'img_1': img_to_dict(img1), 'img_2': img_to_dict(img2)}


def cam_pair_generator_rig(dataset_path, cameras, images, points, skip=1, verbose=0, min_overlap=10):
    matcher = LoFTRMatcher(max_dim=512, device='cuda')
    cam_pairs = list(itertools.combinations(cameras.keys(), 2))

    image_id_lists = {}
    for cam_id in cameras.keys():
        image_id_lists[cam_id] = sorted([im for im in images.values() if im.camera_id == cam_id], key= lambda x: x.id)

    for idx in tqdm(range(0, len(image_id_lists[list(cameras.keys())[0]]), skip)):
        for id1, id2 in cam_pairs:
            cam_1 = cameras[id1]
            cam_2 = cameras[id2]

            img1 = image_id_lists[id1][idx]
            img2 = image_id_lists[id2][idx]

            img_1 = cv2.imread(f'{dataset_path}/images/{img1.name}')
            img_2 = cv2.imread(f'{dataset_path}/images/{img2.name}')

            if verbose:
                cv2.imshow('img_1', img_1)
                cv2.imshow('img_2', img_2)
                cv2.waitKey(1)

            conf, kp_1, kp_2 = matcher.match(img_1, img_2)

            yield {'conf': conf, 'kp_1': kp_1, 'kp_2': kp_2,
                   'cam_1': cam_to_dict(cam_1), 'cam_2': cam_to_dict(cam_2),
                   'img_1': img_to_dict(img1), 'img_2': img_to_dict(img2)}


def prepare_eth3d(args):
    dataset_path = Path(args.dataset_path)

    subsets = ['delivery_area', 'electro', 'forest', 'playground', 'terrains']

    matcher = get_matcher(args.matcher)

    undistort = not args.original
    undistort_str = 'undistort' if undistort else 'original'

    if not os.path.exists(f'saved/{args.matcher}'):
        os.makedirs(f'saved/{args.matcher}')

    for subset in subsets:
        subset_path = os.path.join(dataset_path, subset)
        # cameras, pts = load_eth3D_data(subset_path, undistort=undistort)

        if args.original:
            cameras, images, points = read_model(os.path.join(subset_path, 'rig_calibration'))
        else:
            cameras, images, points = read_model(os.path.join(subset_path, 'rig_calibration_undistorted'))

        # gen = cam_pair_generator_rig(subset_path, cameras, images, points, skip=args.skip)
        # joblib.dump([s for s in tqdm(gen)], 'saved/eth3d_rig_{}_{}.joblib'.format(undistort_str, subset))

        # gen = cam_pair_generator(matcher, subset_path, cameras, images, points, max_pairs=args.num_samples)
        # joblib.dump([s for s in tqdm(gen, total=args.num_samples)], 'saved/{}/eth3d_multi_{}_{}.joblib'.format(args.matcher, undistort_str, subset))

        gen = cam_pair_generator(matcher, subset_path, cameras, images, points, max_pairs=args.num_samples, single=True)
        joblib.dump([s for s in tqdm(gen, total=args.num_samples)], 'saved/{}/eth3d_single_{}_{}.joblib'.format(args.matcher, undistort_str, subset))






if __name__ == '__main__':
    args = parse_args()
    prepare_eth3d(args)
import sys

import cv2
import numpy as np
import torch
from kornia.feature import LoFTR

sys.path.append('../SuperGluePretrainedNetwork')
from models.matching import Matching


def frame2tensor(frame, device):
    return torch.from_numpy(frame/255.).float()[None, None].to(device)


def match_keypoints_loftr(img_1, img_2, debug=False):
    img1_b = cv2.cvtColor(img_1, cv2.COLOR_RGB2GRAY)
    img2_b = cv2.cvtColor(img_2, cv2.COLOR_RGB2GRAY)
    data = {'image0': frame2tensor(img1_b, 'cuda'), 'image1': frame2tensor(img2_b, 'cuda')}
    # m = Matching({'superglue': {'weights': 'outdoor'}})
    # pred = m(data)
    # kp_1 = pred['keypoints0'][0].numpy()
    # kp_2 = pred['keypoints1'][0].numpy()
    # matches = pred['matches0'][0].numpy()
    loftr = LoFTR('outdoor')
    pred = loftr(data)
    kp_1 = pred['keypoints0'].numpy()
    kp_2 = pred['keypoints1'].numpy()
    confidence = pred['confidence']

    if debug:
        img1_vis = np.copy(img_1)
        img2_vis = np.copy(img_2)

        w = img1_vis.shape[1]

        for i in range(len(kp_1)):
            img1_vis = cv2.circle(img1_vis, (int(kp_1[i][0]), int(kp_1[i][1])), 3, (0, 255, 0))
            img2_vis = cv2.circle(img2_vis, (int(kp_2[i][0]), int(kp_2[i][1])), 3, (0, 255, 0))

        img_combined = np.concatenate([img1_vis, img2_vis], axis=1)

        for i in range(len(kp_1)):
            img_combined = cv2.line(img_combined, (int(kp_1[i][0]), int(kp_1[i][1])),
                                    (int(kp_2[i][0]) + w, int(kp_2[i][1])), (
                                        0, int(255 * confidence[i]), int(255 - 255 * confidence[i])), 1)
        cv2.imshow("matches", img_combined)
        cv2.waitKey(0)

    return confidence, kp_1, kp_2


def match_vis(matcher, img1, img2):
    max_height = max(img1.shape[0], img2.shape[0])
    offset = img1.shape[1]
    vis_image = np.zeros([max_height, img1.shape[1] + img2.shape[1], 3], dtype=np.uint8)
    vis_image[:img1.shape[0], :img1.shape[1], :] = img1[:, :]
    vis_image[:img2.shape[0], img1.shape[1]:, :] = img2[:, :]

    conf, kp_1, kp_2 = matcher.match(img1, img2)

    for i in range(len(kp_1)):
        cv2.line(vis_image, (int(kp_1[i, 0]), int(kp_1[i, 1])), (int(offset + kp_2[i, 0]), int(kp_2[i, 1])), (0, conf[i] * 255, 255 * (1 - conf[i])))

    cv2.imshow("Matches", vis_image)
    cv2.waitKey(0)

    return conf, kp_1, kp_2


def get_matcher(matcher_name):
    if matcher_name == 'loftr':
        matcher = LoFTRMatcher()
    elif matcher_name == 'loftr1024':
        matcher = LoFTRMatcher(max_dim=1024)
    elif matcher_name == 'loftr2048':
        matcher = LoFTRMatcher(max_dim=2048)
    elif matcher_name == 'loftr1600':
        matcher = LoFTRMatcher(max_dim=1600)
    elif matcher_name == 'loftr1800':
        matcher = LoFTRMatcher(max_dim=1800)
    elif matcher_name == 'rootsift':
        matcher = SIFTMatcher(upright=False)
    elif matcher_name == 'upright_rootsift':
        matcher = SIFTMatcher(upright=True)
    elif matcher_name == 'sift':
        matcher = SIFTMatcher(root=False, upright=False)
    elif matcher_name == 'upright_sift':
        matcher = SIFTMatcher(root=False, upright=True)
    elif matcher_name == 'sg':
        matcher = SGMatcher()
    elif matcher_name == 'sg1024':
        matcher = SGMatcher(max_dim=1024)
    elif matcher_name == 'sg2048':
        matcher = SGMatcher(max_dim=2048)
    elif matcher_name == 'sg1536':
        matcher = SGMatcher(max_dim=1536)
    else:
        raise NotImplementedError

    return matcher


class SIFTMatcher():
    def __init__(self, num_kp=8000, upright=False, root=True):
        self.upright = upright
        self.root = root
        self.num_kp = 8000
        self.matcher_str = 'upright_rootsift' if upright else 'rootsift'

    def kp_desc(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        feature = cv2.SIFT_create(100000000,
                                              contrastThreshold=-10000,
                                              edgeThreshold=-10000)

        kp = feature.detect(gray, None)

        if self.upright:
            unique_kp = []
            for i, x in enumerate(kp):
                if i > 0:
                    if x.response == kp[i - 1].response:
                        continue
                x.angle = 0
                unique_kp.append(x)
            unique_kp, unique_desc = feature.compute(gray, unique_kp, None)
            top_resps = np.array([x.response for x in unique_kp])
            idxs = np.argsort(top_resps)[::-1]
            kp = np.array(unique_kp)[idxs[:min(len(unique_kp), self.num_kp)]]
            desc = unique_desc[idxs[:min(len(unique_kp), self.num_kp)]]
        else:
            kp, desc = feature.compute(gray, kp, None)

        # Use root-SIFT
        if self.root:
            desc /= desc.sum(axis=1, keepdims=True) + 1e-8
            desc = np.sqrt(desc)

        responses = [x.response for x in kp]
        kp = np.array([x.pt for x in kp])

        return kp, desc


    def match(self, img1, img2):
        kp1, desc1 = self.kp_desc(img1)
        kp2, desc2 = self.kp_desc(img2)

        index_params = dict(algorithm=1, trees=4)
        search_params = dict(checks=128)
        bf = cv2.FlannBasedMatcher(index_params, search_params)
        matches = bf.knnMatch(desc1, desc2, k=2)

        valid_matches = []
        for cur_match in matches:
            tmp_valid_matches = [
                nn_1 for nn_1, nn_2 in zip(cur_match[:-1], cur_match[1:])
                if nn_1.distance <= 0.8 * nn_2.distance
            ]
            valid_matches.extend(tmp_valid_matches)

        matches_list = np.array([[m.queryIdx, m.trainIdx] for m in valid_matches])

        kp1 = kp1[matches_list[:, 0]]
        kp2 = kp2[matches_list[:, 1]]

        return np.ones(len(kp1)), kp1, kp2


class LoFTRMatcher():
    matcher_str = 'loftr'

    def __init__(self, weights='outdoor', device='cuda', max_dim=512):
        self.loftr = LoFTR(weights).to(device)
        self.device = device
        self.max_dim = max_dim

    def enforce_dim(self, img):
        if self.max_dim is None:
            return img, 1.0

        h, w = img.shape[:2]

        gr = max(h, w)

        if gr > self.max_dim:
            scale_factor = self.max_dim / gr
            img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
            return img, scale_factor
        else:
            return img, 1.0

    def match(self, img_1, img_2):
        img1_b = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
        img2_b = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

        img1_b, s1 = self.enforce_dim(img1_b)
        img2_b, s2 = self.enforce_dim(img2_b)

        data = {'image0': frame2tensor(img1_b, self.device), 'image1': frame2tensor(img2_b, self.device)}
        pred = self.loftr(data)
        kp_1 = pred['keypoints0'].detach().cpu().numpy() / s1
        kp_2 = pred['keypoints1'].detach().cpu().numpy() / s2
        conf = pred['confidence'].detach().cpu().numpy()

        return conf, kp_1, kp_2

class SGMatcher():
    matcher_str = 'sg'

    def __init__(self, weights='outdoor', device='cuda', max_dim=512):
        self.m = Matching({'superglue': {'weights': weights}}).eval().to(device)
        self.device = device
        self.max_dim = max_dim

    def enforce_dim(self, img):
        if self.max_dim is None:
            return img, 1.0
        h, w = img.shape[:2]

        gr = max(h, w)

        if gr > self.max_dim:
            scale_factor = self.max_dim / gr
            img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
            return img, scale_factor
        else:
            return img, 1.0

    def match(self, img_1, img_2):
        img1_b = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
        img2_b = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

        img1_b, s1 = self.enforce_dim(img1_b)
        img2_b, s2 = self.enforce_dim(img2_b)

        data = {'image0': frame2tensor(img1_b, self.device), 'image1': frame2tensor(img2_b, self.device)}
        pred = self.m(data)
        kp_1 = pred['keypoints0'][0].detach().cpu().numpy()
        kp_2 = pred['keypoints1'][0].detach().cpu().numpy()
        matches = pred['matches0'][0].detach().cpu().numpy()

        kp_1 = kp_1[matches != -1] / s1
        kp_2 = kp_2[matches[matches != -1]] / s2
        conf = pred['matching_scores0'][0].detach().cpu().numpy()[matches!=-1]

        return conf, kp_1, kp_2

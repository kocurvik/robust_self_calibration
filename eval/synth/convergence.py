import cv2
import joblib
import matlab.engine
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from datasets.definitions import get_subset_string
from eval.synth.synth_scenarios import get_scene, coplanar_scene, noncoplanar_scene, set_scene
from matlab_utils.engine_calls import kukelova_uncal


def percentile_under(rel_change, param):
    cmp = rel_change < param

    portions = np.zeros(rel_change.shape[1])
    for col in range(rel_change.shape[1]):
        portions[col] = np.mean(np.any(cmp[:, :col + 1], axis=1))

    return portions



def convergence_plot(f1, f2, R, t, max_iters=5000):
    sigma = 1.0

    eng = matlab.engine.start_matlab()
    s = eng.genpath('matlab_utils')
    eng.addpath(s, nargout=0)

    iters_list = []
    errs_list = []

    for _ in tqdm(range(1000)):
        x1, x2, _ = get_scene(f1, f2, R, t, 100, sigma_p=10.0)
        xx1 = x1 + sigma * np.random.randn(*(x1.shape))
        xx2 = x2 + sigma * np.random.randn(*(x1.shape))
        F, mask = cv2.findFundamentalMat(xx1, xx2, cv2.USAC_MAGSAC, ransacReprojThreshold=3.0, confidence=1.0,
                                              maxIters=10000)

        _, _, _, _, errs, iters = kukelova_uncal(eng, F, 660, 440, iters=max_iters, return_err=True, all_iters=True)
        iters_list.append(iters)
        errs_list.append(errs)

    errs = np.array(errs_list)[:, :, 0]

    print(errs.shape)

    rel_change = np.abs(errs[:, 1:] - errs[:, :-1]) / errs[:, 1:]

    # plt.semilogx(lims, cum_hist)
    # plt.semilogy(np.arange(max_iters + 1), np.median(errs, axis=0), label='Median')
    # plt.semilogy(np.arange(max_iters + 1), np.quantile(errs, 0.75, axis=0), label='75th Percentile')
    # plt.semilogy(np.arange(max_iters + 1), np.quantile(errs, 0.95, axis=0), label='95th Percentile')
    # plt.semilogy(np.arange(max_iters + 1), np.quantile(errs, 0.99, axis=0), label='99th Percentile')
    # plt.semilogy(np.arange(max_iters), np.median(rel_change, axis=0), label='Median')
    # plt.semilogy(np.arange(max_iters), np.quantile(rel_change, 0.75, axis=0), label='75th Percentile')
    # plt.semilogy(np.arange(max_iters), np.quantile(rel_change, 0.95, axis=0), label='95th Percentile')
    # plt.semilogy(np.arange(max_iters), np.quantile(rel_change, 0.99, axis=0), label='99th Percentile')

    for i in np.arange(2, 9, 2):
        val = 10.0 ** (-i)
        plt.plot(np.arange(max_iters + 1), [0, *percentile_under(rel_change, val)], label='$10^{-' + str(i) + '}$')

    plt.xlim([0.0, max_iters])
    plt.ylim([0.0, 1.0])
    plt.legend(title='$\epsilon$', loc='lower right')
    plt.xlabel('Iterations')
    plt.ylabel('Portion of converged samples')


def pt_convergence_plot(max_iters=5000):
    eng = matlab.engine.start_matlab()
    s = eng.genpath('matlab_utils')
    eng.addpath(s, nargout=0)

    iters_list = []
    errs_list = []

    save_string, subsets, _, _ = get_subset_string('phototourism')

    samples = []

    for subset in subsets:
        samples.extend(joblib.load('saved/loftr1024/' + save_string.format(subset) + '.joblib'))

    for sample in tqdm(samples):
        cam_1 = sample['cam_1']
        cam_2 = sample['cam_2']
        p_1 = np.array([cam_1['width'] / 2, cam_1['height'] / 2])
        p_2 = np.array([cam_2['width'] / 2, cam_2['height'] / 2])

        colmap_1 = 1.2 * max(cam_1['width'], cam_1['height'])
        colmap_2 = 1.2 * max(cam_2['width'], cam_2['height'])

        kp_1 = sample['kp_1'] - p_1[np.newaxis, :]
        kp_2 = sample['kp_2'] - p_2[np.newaxis, :]

        F, mask = cv2.findFundamentalMat(kp_1, kp_2, cv2.USAC_MAGSAC, ransacReprojThreshold=3.0, confidence=1.0,
                                         maxIters=10000)

        _, _, _, _, errs, iters = kukelova_uncal(eng, F, colmap_1, colmap_2, iters=max_iters, return_err=True, all_iters=True)
        s = np.array(errs).shape
        if s != (max_iters+1, 1):
            continue
            print("Warning: ", s)
        iters_list.append(iters)
        errs_list.append(errs)


    errs = np.array(errs_list)[:, :, 0]

    rel_change = np.abs(errs[:, 1:] - errs[:, :-1]) / (errs[:, 1:] + 1e-16)

    mins = np.minimum.accumulate(errs[:, :-1], axis=1)

    both = np.minimum(mins, rel_change)

    percentile_under(rel_change, 1e-4)

    # plt.semilogx(lims, cum_hist)
    # plt.semilogy(np.arange(max_iters + 1), np.median(errs, axis=0), label='Median')
    # plt.semilogy(np.arange(max_iters + 1), np.quantile(errs, 0.75, axis=0), label='75th Percentile')
    # plt.semilogy(np.arange(max_iters + 1), np.quantile(errs, 0.95, axis=0), label='95th Percentile')
    # plt.semilogy(np.arange(max_iters + 1), np.quantile(errs, 0.99, axis=0), label='99th Percentile')
    # plt.semilogy(np.arange(max_iters), np.median(rel_change, axis=0), label='Median')
    # plt.semilogy(np.arange(max_iters), np.quantile(rel_change, 0.75, axis=0), label='75th Percentile')
    # plt.semilogy(np.arange(max_iters), np.quantile(rel_change, 0.95, axis=0), label='95th Percentile')
    # plt.semilogy(np.arange(max_iters), np.quantile(rel_change, 0.99, axis=0), label='99th Percentile')

    for i in np.arange(2, 9, 2):
        val = 10.0 ** (-i)
        plt.plot(np.arange(max_iters), percentile_under(both, val), label='$10^{-' + str(i) + '}$')

    plt.xlim([0.0, max_iters - 1])
    plt.ylim([0.0, 1.0])
    plt.legend(title='$\epsilon$', loc='lower right')
    plt.xlabel('Iterations')
    plt.ylabel('Portion of converged samples')

if __name__ == '__main__':
    plt.figure(figsize=(0.8 * 6, 0.6 * 4))
    convergence_plot(*set_scene(600, 400, theta=0, y=300), max_iters=10)
    plt.savefig('figs/synth/convergence_noncoplanar.pdf')
    plt.figure()

    plt.figure(figsize=(0.8 * 6, 0.6 * 4))
    convergence_plot(*set_scene(600, 400, theta=0, y=0), max_iters=1000)
    plt.savefig('figs/synth/convergence_coplanar.pdf')
    plt.figure()

    plt.figure(figsize=(0.8 * 6, 0.6 * 4))
    pt_convergence_plot(max_iters=500)
    plt.savefig('figs/synth/convergence_pt.pdf')



import numpy as np
import poselib
from matplotlib import pyplot as plt
from sklearn.neighbors import KernelDensity

from matlab_utils.engine_calls import kukelova_uncal
from methods.base import bougnoux_original


def vote_7pt(eng, kp_1, kp_2, f_1_prior, f_2_prior, f_1_gt, n=1000, width=50):

    kp_1 = np.column_stack([kp_1, np.ones([len(kp_1), 1])])
    kp_2 = np.column_stack([kp_2, np.ones([len(kp_2), 1])])

    n_pts = len(kp_1)

    valid_f_1s = []
    valid_f_2s = []

    kukelova_valid_f_1s = []
    kukelova_valid_f_2s = []

    kukelova_all_f_1s = []
    kukelova_all_f_2s = []

    for _ in range(n):
        idxs = np.random.choice(range(n_pts), size=7, replace=False)
        Fs, _ = poselib.fundamental_matrix_7pt(kp_1[idxs], kp_2[idxs])

        for F in Fs:
            f_1_sq, f_2_sq = bougnoux_original(F)

            f_1_kuk, f_2_kuk, pp1, pp2 = kukelova_uncal(eng, F, f_1_prior, f_2_prior)

            kukelova_all_f_1s.append(f_1_kuk)
            kukelova_all_f_2s.append(f_2_kuk)

            if f_1_sq > 0 and f_2_sq > 0:
                f_1, f_2 = np.sqrt(f_1_sq), np.sqrt(f_2_sq)

                valid_f_1s.append(f_1)
                valid_f_2s.append(f_2)

                kukelova_valid_f_1s.append(f_1_kuk)
                kukelova_valid_f_2s.append(f_2_kuk)

            # else:
            #     kukelova_valid_f_1s.append(f_1_kuk)
            #     kukelova_valid_f_2s.append(f_2_kuk)

    x = np.linspace(0, 3000, 6000).reshape(-1, 1)

    y_valid_f1 = KernelDensity(kernel='gaussian', bandwidth=width).fit(np.array(valid_f_1s).reshape(-1, 1)).score_samples(x)
    y_valid_f2 = KernelDensity(kernel='gaussian', bandwidth=width).fit(np.array(valid_f_2s).reshape(-1, 1)).score_samples(x)

    y_kukelova_valid_f1 = KernelDensity(kernel='gaussian', bandwidth=width).fit(np.array(kukelova_valid_f_1s).reshape(-1, 1)).score_samples(x)
    y_kukelova_valid_f2 = KernelDensity(kernel='gaussian', bandwidth=width).fit(np.array(kukelova_valid_f_2s).reshape(-1, 1)).score_samples(x)

    y_kukelova_all_f1 = KernelDensity(kernel='gaussian', bandwidth=width).fit(np.array(kukelova_all_f_1s).reshape(-1, 1)).score_samples(x)
    y_kukelova_all_f2 = KernelDensity(kernel='gaussian', bandwidth=width).fit(np.array(kukelova_all_f_2s).reshape(-1, 1)).score_samples(x)

    plt.figure()
    plt.plot(x, np.power(np.e, y_valid_f1), label='valid')
    plt.plot(x, np.power(np.e, y_kukelova_valid_f1), label='kukelova_valid')
    plt.plot(x, np.power(np.e, y_kukelova_all_f1), label='kukelova_all')
    plt.plot([f_1_prior, f_1_prior], [0, np.e ** np.max(y_kukelova_all_f1)], label='prior')
    plt.plot([f_1_gt, f_1_gt], [0, np.e ** np.max(y_kukelova_all_f1)], label='gt')

    plt.legend()
    plt.show()








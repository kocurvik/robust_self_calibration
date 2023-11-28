import argparse
import os.path
import pickle
from multiprocessing import Pool

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

import matlab.engine

from datasets.definitions import get_subset_string


def focal_error(f_estimated, f_gt):
    return np.abs(f_estimated - f_gt) / np.maximum(f_estimated, f_gt)


def noisy_focal(f_gt, sigma):
    eps = sigma * np.random.randn()

    if eps < 0:
        return (1 + eps) * f_gt
    return f_gt / (1 - eps)


class EvalManager():
    def __init__(self, eng=None, verbosity=0, max_iters=1000, success_prob=0.99):
        self.verbosity = verbosity
        self.max_iters = max_iters
        self.eng = eng
        self.success_prob = success_prob

        self.samples = []

    def print_fs(self, method, f_1, f_2):
        self.print_method(method, '\t f_1: {:.3f} \t f_2: {:.3f}'.format(f_1, f_2))

    @staticmethod
    def print_method(method, str):
        num_spaces = 40 - len(method)
        print(method, num_spaces * ' ', str)

    def add_generator_samples(self, gen, expected=None):
        for sample in tqdm(gen, total=expected):
            self.samples.append(sample)

    def save_samples(self, path, compress=3):
        joblib.dump(self.samples, path, compress=compress)

    def load_samples(self, path, extend=False):
        if extend:
            self.samples.extend(joblib.load(path))
        else:
            self.samples = joblib.load(path)

    def save_results(self, path):
        self.df.to_pickle(path)

    def load_results(self, path):
        self.df = pd.read_pickle(path)


def draw_cumhist(df, manager_class, axes_f=None, axes_R=None, axes_t=None, num_bins=100, title=None):
    if axes_f is not None:
        # x = np.logspace(-2, 0, num_bins)
        x = np.linspace(0, 1, num_bins)
        for method in manager_class.methods:
            if 'gt' in method:
                continue
            df_method = df.loc[df['method_name'] == method]
            errs = manager_class.f_errs(df_method)
            total = len(errs)
            res = np.array([np.sum(errs < t) / total for t in x])
            # axes_f.semilogx(x, res, label=method)
            method_basename = method.split('_')[0]
            rfc = 'rfc' in method# or 'Uncal' not in str(manager_class)

            axes_f.plot(x, res,
                        label=manager_class.method_names[method_basename],
                        color=manager_class.method_colors[method_basename],
                        linestyle='solid' if rfc else 'dashed')
                        # linestyle='solid')

        axes_f.axis(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)
        axes_f.set_ylabel('Portion of samples')
        axes_f.set_xlabel('Focal length error $f_i^{err}$')
        if title is not None:
            axes_f.set_title(title)
        # axes_f.legend()

    if axes_R is not None:
        x = np.linspace(0, 180, num_bins)
        for method in manager_class.methods:
            df_method = df.loc[df['method_name'] == method]
            errs = df_method['R_err'].to_numpy()
            total = len(errs)
            res = np.array([np.sum(errs < t) / total for t in x])
            method_basename = method.split('_')[0]
            rfc = 'rfc' in method # or 'Uncal' not in str(manager_class)

            axes_R.plot(x, res,
                        label=manager_class.method_names[method_basename],
                        color=manager_class.method_colors[method_basename],
                        linestyle='solid' if rfc else 'dashed')

        axes_R.axis(xmin=0.0, xmax=180, ymin=0.0, ymax=1.0)
        axes_R.set_ylabel('Portion of samples')
        axes_R.set_xlabel('Rotation error in degrees')
        if title is not None:
            axes_R.set_title(title)
        # axes_R.legend()

    if axes_t is not None:
        x = np.linspace(0, 180, num_bins)
        for method in manager_class.methods:
            df_method = df.loc[df['method_name'] == method]
            errs = df_method['t_err'].to_numpy()
            total = len(errs)
            res = np.array([np.sum(errs < t) / total for t in x])
            method_basename = method.split('_')[0]
            rfc = 'rfc' in method # or 'Uncal' not in str(manager_class)

            axes_t.plot(x, res,
                        label=manager_class.method_names[method_basename],
                        color=manager_class.method_colors[method_basename],
                        linestyle='solid' if rfc else 'dashed')

        axes_t.axis(xmin=0.0, xmax=180, ymin=0.0, ymax=1.0)
        axes_t.set_ylabel('Portion of samples')
        axes_t.set_xlabel('Translation error in degrees')
        if title is not None:
            axes_t.set_title(title)
        # axes_t.legend()


def print_speed_table(df, manager_class):

    for method in manager_class.methods:
        df_method = df.loc[df['method_name'] == method]
        elapsed = 1000 * df_method['f_elapsed'].to_numpy().astype(np.float32)
        iters = df_method['iters'].to_numpy().astype(np.float32)

        iters = iters[iters != np.inf]

        method_basename = method.split('_')[0]
        method_str = f'{method_basename} \t & \t' if not 'rfc' in method else f'{method_basename} \t & \\checkmark  \t'

        # manager_class.print_method(method, f'{100 * t_auc:.2f} \t {100 * R_auc:.2f} \t {100 * p_auc_20:.2f} \t {100 * f_auc_20:.2f}')
        # manager_class.print_method(method_str, f'& \t {100 * p_auc_10:.2f} &\t {100 * p_auc_20:.2f} &\t {100 * f_auc_10:.2f} &\t {100 * f_auc_20:.2f} \\\\')
        # manager_class.print_method(method_str, f' & \t {np.median(elapsed):.2f} & \t {np.mean(elapsed):.2f} & \t {np.median(iters):.2f} & \t {np.nanmean(iters):.2f} \\\\ \\hline')
        manager_class.print_method(method_str, f' & \t {np.mean(elapsed):.2f} & \t {np.nanmean(iters):.2f} \\\\ \\hline')


def print_table(df, manager_class):

    angles = np.arange(1, 21)
    f_points = np.arange(1, 21) / 100

    for method in manager_class.methods:
        df_method = df.loc[df['method_name'] == method]
        t_errs = df_method['t_err'].to_numpy().astype(np.float32)
        R_errs = df_method['R_err'].to_numpy().astype(np.float32)
        p_errs = np.maximum(t_errs, R_errs)
        f_errs = np.array(manager_class.f_errs(df_method), dtype=np.float32)

        p_errs[np.isnan(p_errs)] = 180
        f_errs[np.isnan(f_errs)] = 1.0
        f_errs[f_errs > 1.0] = 1.0

        # t_res = np.array([np.sum(t_errs < t) / len(t_errs) for t in angles])
        # R_res = np.array([np.sum(R_errs < t) / len(R_errs) for t in angles])
        p_res = np.array([np.sum(p_errs < t) / len(p_errs) for t in angles])
        f_res = np.array([np.sum(f_errs < t) / len(f_errs) for t in f_points])



        p_auc_20 = np.mean(p_res)
        f_auc_20 = np.mean(f_res)
        p_auc_10 = np.mean(p_res[:10])
        f_auc_10 = np.mean(f_res[:10])

        method_basename = method.split('_')[0]
        method_str = f'{method_basename} \t & \t' if not 'rfc' in method else f'{method_basename} \t & \\checkmark  \t'

        # manager_class.print_method(method, f'{100 * t_auc:.2f} \t {100 * R_auc:.2f} \t {100 * p_auc_20:.2f} \t {100 * f_auc_20:.2f}')
        # manager_class.print_method(method_str, f'& \t {100 * p_auc_10:.2f} &\t {100 * p_auc_20:.2f} &\t {100 * f_auc_10:.2f} &\t {100 * f_auc_20:.2f} \\\\')
        manager_class.print_method(method_str, f' & \t {np.median(p_errs):.2f}\\degree & \t {100 * p_auc_10:.2f} &\t {100 * p_auc_20:.2f} & \t \t {np.median(f_errs):.3f} &\t {100 * f_auc_10:.2f} &\t {100 * f_auc_20:.2f} \\\\ \\hline')


def save_figs(R_fig, f_fig, t_fig, name, ext='png'):
    if not os.path.exists(f'figs/{name}'):
        os.makedirs(f'figs/{name}')

    f_fig.savefig(f'figs/{name}_f.{ext}')
    R_fig.savefig(f'figs/{name}_R.{ext}')
    t_fig.savefig(f'figs/{name}_t.{ext}')


def get_figs_axes(rows, cols, figsize=(10, 6)):
    figsize = (figsize[0] * cols, figsize[1] * rows)
    f_fig, f_axes = plt.subplots(rows, cols, figsize=figsize)
    R_fig, R_axes = plt.subplots(rows, cols, figsize=figsize)
    t_fig, t_axes = plt.subplots(rows, cols, figsize=figsize)

    if rows == 1 and cols == 1:
        f_axes = [f_axes]
        t_axes = [t_axes]
        R_axes = [R_axes]

    else:
        f_axes = f_axes.flatten()
        t_axes = t_axes.flatten()
        R_axes = R_axes.flatten()

    return R_axes, R_fig, f_axes, f_fig, t_axes, t_fig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--together', action='store_true', default=False)
    parser.add_argument('-nw', '--num_workers', type=int, default=1)
    parser.add_argument('-l', '--load', action='store_true', default=False)
    parser.add_argument('-m', '--matcher', type=str, default='loftr')
    parser.add_argument('dataset_name')

    return parser.parse_args()


def eval_single_manager(manager_class, matcher_string, save_string, subset):
    eng = matlab.engine.start_matlab()
    s = eng.genpath('matlab_utils')
    eng.addpath(s, nargout=0)
    eval_manager = manager_class(subset, eng=eng)
    eval_manager.load_samples(f'saved/{matcher_string}/{save_string.format(subset)}.joblib')
    df = eval_manager.evaluate(exif='exif' in subset)
    if not os.path.exists(f'results/{matcher_string}/{manager_class.manager_str}'):
        os.makedirs(f'results/{matcher_string}/{manager_class.manager_str}')

    eval_manager.save_results(f'results/{matcher_string}/{manager_class.manager_str}/{save_string.format(subset)}.pkl')
    return df


def load_single_manager(manager_class, matcher_string, save_string, subset):
    with open(f'results/{matcher_string}/{manager_class.manager_str}/{save_string.format(subset)}.pkl', 'rb') as f:
        df = pickle.load(f)
    return df


def run_eval(ManagerClass):
    args = parse_args()
    matcher_string = args.matcher
    save_string, subsets, rows, cols = get_subset_string(args.dataset_name)

    if args.load:
        dfs = [load_single_manager(ManagerClass, matcher_string, save_string, subset) for subset in subsets]
    else:
        mgr_classes = [ManagerClass for _ in subsets]
        matcher_strings = [matcher_string for _ in subsets]
        save_strings = [save_string for _ in subsets]
        pool = Pool(args.num_workers)
        dfs = pool.starmap(eval_single_manager, zip(mgr_classes, matcher_strings, save_strings, subsets))



    R_axes, R_fig, f_axes, f_fig, t_axes, t_fig = get_figs_axes(rows, cols)
    for i in range(len(subsets)):
        draw_cumhist(dfs[i], ManagerClass, axes_f=f_axes[i], axes_R=R_axes[i], axes_t=t_axes[i], title=args.dataset_name + ' - ' + subsets[i])

    save_figs(R_fig, f_fig, t_fig, f'{ManagerClass.manager_str}/{matcher_string}/{args.dataset_name}', ext='png')

    all_df = pd.concat(dfs)
    R_axes, R_fig, f_axes, f_fig, t_axes, t_fig = get_figs_axes(1, 1, figsize=(0.8*6, 0.8*4))
    # draw_cumhist(all_df, ManagerClass, axes_f=f_axes[0], axes_R=R_axes[0], axes_t=t_axes[0], title=args.dataset_name + ' - together')
    draw_cumhist(all_df, ManagerClass, axes_f=f_axes[0], axes_R=R_axes[0], axes_t=t_axes[0])
    print_table(all_df, ManagerClass)
    save_figs(R_fig, f_fig, t_fig, f'{ManagerClass.manager_str}/{matcher_string}/together_{args.dataset_name}', ext='pdf')
    print()
    print()
    print_speed_table(all_df, ManagerClass)




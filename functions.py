# includes
from __future__ import absolute_import, division, print_function
from howling_params import compress_files, scores_root_dir, TH_TABLE, HDFS_PARAMS, RND_SEED
from numpy.random import seed
seed(RND_SEED)
import os
import os.path as op
import pandas as pd
import numpy as np
import pickle as pkl
import logging
import datetime
import itertools
import gzip
from tqdm import tqdm
from hashlib import sha1
from parameters import EPSILON
from matplotlib import pyplot as plt
from bisect import bisect
from sklearn.model_selection import KFold
import seaborn as sns


def get_files_meta_with_extensions(input_dirs, ext):
    """
    return a (sorted) pandas DataFrame having all files meta data ( names(idx), paths, extension) contained
    in input_dirs and having extensions ext. PS. names need to be unique.

    Parameters
    ----------
    input_dirs : str, list of strings
    ext : str, list of strings ('.' or '_')
    """

    files_meta = pd.DataFrame(columns=['path', 'extension'])
    # check if only one input directory is fed to the function --> make it a list
    if type(input_dirs) is str:
        input_dirs = [input_dirs]

    if type(ext) is str:
        ext = [ext]

    for fldr in input_dirs:
        for root, dirs, files in os.walk(fldr):
            for f in files:
                if op.splitext(f)[-1] in ext:
                    file_name, file_ext = op.splitext(f)
                    files_meta.loc[file_name] = [root, file_ext]
                elif f.rsplit("_", 1)[-1] in ext:
                    file_name, file_ext = f.rsplit("_", 1)
                    files_meta.loc[file_name] = [root, "_"+file_ext]

    # sort files
    files_meta.sort_index()
    return files_meta


def dump_odf_to_file(object_to_save=None, output='../dir_to_ignore/odfs'):
    object_to_save = object_to_save.squeeze()
    print('saving odf for {} with shape {}.'.format(os.path.split(output)[1], object_to_save.shape))
    pkl.dump(object_to_save, open(output, 'wb'), protocol=2)


def save_hdf(object_to_save=None, output='../dir_to_ignore/obj'):
    # np.savetxt(output, object_to_save)
    dump_obj_to_file(object_to_save, output)


def normalize_to_limits(vec, lower=0, upper=1):
    """
    normalise a vector between lower and upper limits
    :param vec:
    :param lower:
    :param upper:
    :return:
    """
    max_value = np.max(vec)
    min_value = np.min(vec)
    denom = max_value - min_value
    if denom == 0:
        denom = 1
    return lower + (upper - lower) * (vec - min_value) / float(denom)


def normalize_using_params(vec, minimum, maximum, lower=0, upper=1):
    denom = maximum - minimum
    if denom == 0:
        denom = 1
    return lower + (upper - lower) * (vec - minimum) / float(denom)


def instantiate_logger(name='default_logger', path=''):
    """
    creates a logger generating logs with dates at the path "path"
    :param name:
    :param path:
    :return: the logger
    """
    dtstr = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(op.join(path, '{}logs.txt'.format(dtstr)))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    frmtr1 = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    frmtr2 = logging.Formatter('%(levelname)s %(message)s')
    fh.setFormatter(frmtr1)
    ch.setFormatter(frmtr2)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def create_folder_if_not_existing(paths):
    if type(paths) is str:
        paths = [paths]
    folder_exist_flags = {}
    for path in paths:
        # # for printing
        # fldrsplit = os.path.split(path)
        # fldrhash = fldrsplit[-1]
        # if len(fldrhash) > 15:
        #     fldrname = os.path.split(fldrsplit[0])[-1]
        # else:
        #     fldrname = fldrhash
        # createcreateFolderIfNotExisting
        if not os.path.exists(path):
            os.makedirs(path)
            folder_exist_flags[path] = False
        elif len([item for item in os.listdir(path) if not item.startswith('.')]) > 0:    # neglecting os files '.'
            # print('folder ** ' + fldrname + ' ** exists and contains files')
            folder_exist_flags[path] = True
        else:
            # print('folder **' + fldrname + '** exists (empty)')
            folder_exist_flags[path] = False
    return folder_exist_flags


def dump_obj_to_file(object_to_save=None, output='../dir_to_ignore/obj'):
    """
    saves an obj to a file
    :param object_to_save:
    :param output: the path
    :return:
    """
    if compress_files:
        f = gzip.open(output, 'wb')
    else:
        f = open(output, 'wb')
    pkl.dump(object_to_save, f, protocol=2)
    f.close()


def undump_obj_from_file(input_file='../dir_to_ignore/obj'):
    try:
        with gzip.open(input_file, 'rb') as f:
            obj = pkl.load(f)
    except:
        with open(input_file, 'rb') as f:
            obj = pkl.load(f)
    return obj


def fft_stacker(tup):
    if len(tup) > 1:
        results = {'win_len_{}'.format(fft.num_bins*2): fft for fft in tup}
    else:
        results = tup[0]
    return results


def calc_avg_slope_per_bin(frame):
    frames_count = frame.shape[1]
    cand_count = frames_count - 1
    weights = 1.0 / (cand_count - np.arange(cand_count - 1, -1, -1))
    return np.mean(np.expand_dims(weights, axis=1)
                   * (frame[:, 1:].transpose() - frame[:, 0].squeeze()), axis=0).transpose()


def calc_imsd(spec_frame):
    frames_count = spec_frame.shape[1]
    log_spec = 20 * np.log10(spec_frame + EPSILON)
    long_term_slope_avg = calc_avg_slope_per_bin(log_spec)
    short_term_slope_avg = np.asarray(
        [calc_avg_slope_per_bin(log_spec[:, idx:])
         for idx in np.arange(-2, -frames_count, -1)])
    imsd = np.abs(np.mean(long_term_slope_avg - short_term_slope_avg, axis=0))
    return imsd


def spec_anns_stacker(tup):
    results = {'spec': tup[0], 'anns': tup[1]}
    return results


def create_hashes(**params):
    """
    a function to calculate hashes depending on mixtures of parameters
    :param params: list(s), lists of values for each input parameter
    :return: pandas.DataFrame, a table containing hashes with respective params values mixing
    """
    combinations = list(itertools.product(*[params[k] for k in params]))

    index = [sha1(repr(l)).hexdigest() for l in combinations]

    return pd.DataFrame.from_records(combinations, index=index, columns=params.keys())


def ninos_sparsity(spectrogram):
    l2norm = np.power(np.sum(spectrogram ** 2, axis=1), 1.0/2)
    l4norm = np.power(np.sum(spectrogram ** 4, axis=1), 1.0/4)
    l4norm[l4norm == 0] = np.exp(-100)
    ratio = (np.true_divide(l2norm, l4norm)-1)/(np.power(spectrogram.shape[1], 1.0/4)-1)
    return ratio


def ninos_sparsity_energy(spectrogram):
    l2norm = np.power(np.sum(spectrogram ** 2, axis=1), 1.0/2)
    l4norm = np.power(np.sum(spectrogram ** 4, axis=1), 1.0/4)
    l4norm[l4norm == 0] = np.exp(-100)
    ratio = (np.true_divide(l2norm, l4norm)-1)/(np.power(spectrogram.shape[1], 1.0/4)-1) * l2norm
    return ratio


def keep_max_n_elements_per(input_mtx, n, col_row=0):
    """
    a function to keep only the largest n elements per row (or column) and zero the rest
    :param input_mtx:
    :param n:
    :param col_row:
    :return:
    """
    ravel = 'FC'
    output = np.zeros((input_mtx.size,))
    max_indices = np.argsort(input_mtx, axis=col_row)
    if not col_row:
        max_indices = max_indices[-n:, :]
        max_indices_linear = \
            np.ravel(max_indices + np.arange(input_mtx.shape[1 - col_row]) * input_mtx.shape[col_row], ravel[col_row])
    else:
        max_indices = max_indices[:, -n:]
        max_indices_linear = np.ravel(
            max_indices.transpose() + np.arange(input_mtx.shape[1-col_row])*input_mtx.shape[col_row], ravel[col_row])
    output[np.asarray(max_indices_linear)] = np.ravel(input_mtx, ravel[col_row])[np.asarray(max_indices_linear)]
    output = np.reshape(output, input_mtx.shape, ravel[col_row])
    return output


def mtx_to_linear_indexing(indices, shape, col_row=0):
    ravel = 'FC'
    if col_row:
        indices_local = indices.transpose()
    else:
        indices_local = indices
    return np.ravel(indices_local + np.arange(shape[1 - col_row]) * shape[col_row], ravel[col_row])


def keep_max_n_elements_per_zeroing_m_neighbours(input_mtx, n, m, col_row=0):
    ravel = 'FC'
    output = np.zeros((input_mtx.size,))
    max_indices_linear_global = np.array([])
    input_mtx_local = np.array(input_mtx)
    for _ in np.arange(0, n, 1):
        # picking the maximas
        max_indices = np.argmax(input_mtx_local, axis=col_row)
        max_vals = np.max(input_mtx_local, axis=col_row)
        # leaving out maximas = 0 "max_vals > EPSILON"
        max_indices_linear = mtx_to_linear_indexing(max_indices, input_mtx.shape, col_row)
        max_indices_linear_zeroing_g = max_indices_linear[
            np.logical_and(max_indices < (input_mtx.shape[col_row] - m), max_vals > EPSILON)]
        max_indices_linear_zeroing_l = max_indices_linear[
            np.logical_and(max_indices >= m,  max_vals > EPSILON)]
        max_indices_linear = max_indices_linear[max_vals > EPSILON]

        max_indices_linear_global = np.concatenate((max_indices_linear_global, max_indices_linear)).astype(int)
        max_indices_linear_zeroing_g = \
            np.ravel(max_indices_linear_zeroing_g + np.expand_dims(np.arange(0, m+1, 1), axis=1))
        max_indices_linear_zeroing_l = \
            np.ravel(max_indices_linear_zeroing_l + np.expand_dims(np.arange(-m, 1, 1), axis=1))
        input_mtx_local = np.ravel(input_mtx_local, ravel[col_row])
        input_mtx_local[np.concatenate((max_indices_linear_zeroing_g, max_indices_linear_zeroing_l))] = 0
        input_mtx_local = np.reshape(input_mtx_local, input_mtx.shape, ravel[col_row])

    output[np.asarray(max_indices_linear_global)] = \
        np.ravel(input_mtx, ravel[col_row])[np.asarray(max_indices_linear_global)]
    output = np.reshape(output, input_mtx.shape, ravel[col_row])
    return output


def onpick(event, lined, figure):
    vis = False
    # on the pick event, find the orig line corresponding to the
    # legend proxy line, and toggle the visibility
    legline = event.artist
    orig_line = dict()
    for key in lined[legline]:
        orig_line[key] = lined[legline][key]

    for key in orig_line:
        vis = not orig_line[key].get_visible()
        orig_line[key].set_visible(vis)

    # Change the alpha on the line in the legend so we can see what lines
    # have been toggled
    if vis:
        legline.set_alpha(0.7)
    else:
        legline.set_alpha(0.2)
    figure.canvas.draw()

 # # picking on the legend line
# lined = dict()
# num_lines_per_graphs = int(len(lines_roc)/3)     # 3 for full, cand and early
# for l in leg_lines[:num_lines_per_graphs]:
#     lined[l] = dict()
# leg_lines = leg_lines[:num_lines_per_graphs]*3
# counter = 0
# for legline, line_roc, line_pr in zip(leg_lines, lines_roc, lines_pr):
#     legline.set_picker(5)  # 5 pts tolerance
#     lined[legline].update({'roc{}'.format(counter): line_roc, 'pr{}'.format(counter): line_pr})
#     counter +=1
# fig.canvas.mpl_connect('pick_event', lambda event: onpick(event, lined, fig))

def interpolate_pr(tp_count, fp_count, gt_pos_count, interp_points_count):
    tp_flip_axis = np.argmax(tp_count.shape)
    fp_flip_axis = np.argmax(fp_count.shape)
    interp_points_mask = np.zeros((interp_points_count+tp_count.shape[tp_flip_axis],), dtype=bool)
    if gt_pos_count > 0:
        # make tp and fp in ascending order (th in descending)
        tp_count_flipped = np.flip(tp_count, axis=tp_flip_axis)
        fp_count_flipped = np.flip(fp_count, axis=fp_flip_axis)
        # fill in p and r with test points
        r_count_interp = tp_count_flipped / float(gt_pos_count)
        zero_tp_fp = np.logical_and(tp_count_flipped == 0, fp_count_flipped == 0)
        p_count_interp = np.true_divide(tp_count_flipped, fp_count_flipped + tp_count_flipped + zero_tp_fp)
        p_count_interp[zero_tp_fp] = 0
        # calculate tp interpolation points
        interp_points = np.floor(np.arange(0, gt_pos_count, float(gt_pos_count)/interp_points_count)).astype(int)[:interp_points_count]
        # calculate interp points positions
        interp_points_position = np.array([bisect(tp_count_flipped, tp) for tp in interp_points]) \
                                 + np.arange(0, interp_points_count, 1)
        for pt_idx, tp in enumerate(interp_points):
            position_idx = bisect(tp_count_flipped, tp)
            tp_a = tp_count_flipped[position_idx - 1]
            tp_b = tp_count_flipped[position_idx]
            fp_a = fp_count_flipped[position_idx - 1]
            fp_b = fp_count_flipped[position_idx]
            # calculate slope
            local_slope = (fp_b - fp_a) / float(tp_b - tp_a)
            # calculate fp
            fp = fp_a + (tp_b - tp_a) * local_slope
            # calculate recall and precision
            r_count_interp = np.insert(r_count_interp, interp_points_position[pt_idx], tp / float(gt_pos_count))
            if tp_a == 0 and fp_a == 0:
                p_count_interp = np.insert(p_count_interp, interp_points_position[pt_idx], 1.0 / (1 + local_slope))
            else:
                p_count_interp = np.insert(p_count_interp, interp_points_position[pt_idx], tp / (tp + fp))
        interp_points_mask[len(p_count_interp) - 1- interp_points_position] = True
    else:
        p_count_interp = np.nan * np.ones((interp_points_count+len(tp_count),))
        r_count_interp = np.nan * np.ones((interp_points_count+len(tp_count),))
        interp_points_mask[np.arange(0,interp_points_count)] = True

    return np.flip(p_count_interp, axis=tp_flip_axis), np.flip(r_count_interp, axis=tp_flip_axis), interp_points_mask


def area_under_the_curve(x_array, y_array):
    bases = np.abs(x_array[1:] - x_array[:-1])
    heights = (y_array[1:] + y_array[:-1])/2.0
    return np.sum(bases * heights)


def calculate_roc_pr_metrics(tp_count, fp_count, gt_positives_count, gt_negatives_count, interp_points=0):
    points_len = len(tp_count)
    # calculate tpr and fpr considering 4 cases - no po and no neg - no positives - no negatives - normal
    if gt_positives_count == 0 or gt_negatives_count == 0:
        # neglect this evaluation as all gt is either positives or negatives
        tpr = np.nan * np.ones((points_len, ))
        fpr = np.nan * np.ones((points_len, ))
        precision = np.ones((points_len + interp_points, ))
        recall = np.ones((points_len + interp_points, ))
        f1 = np.nan * np.ones(points_len, 1)   # f1-score should be reported for real test not for interpolation
        interp_points_mask = np.ones((points_len + interp_points, ))
    else:
        tpr = tp_count / float(gt_positives_count)
        fpr = fp_count / float(gt_negatives_count)
        # interpolate precision and recall for the PR curves
        precision, recall, interp_points_mask = interpolate_pr(tp_count, fp_count, gt_positives_count, interp_points)
        precision_test = precision[~interp_points_mask]
        f1 = 2 * precision_test * tpr / (precision_test + tpr + EPSILON).astype(float)

    return {'f1': f1,
            'tpr': tpr,
            'fpr': fpr,
            'precision': precision,     # precision from interpolation
            'recall': recall,    # recall from interpolation
            'interp_points_mask': interp_points_mask
            }


def calculate_precision_and_recall(tp_count, fp_count, gt_positives_count):
    r = tp_count / float(gt_positives_count)
    zero_tp_fp = np.logical_and(tp_count == 0, fp_count == 0)
    p = np.true_divide(tp_count, fp_count + tp_count + zero_tp_fp)
    p[zero_tp_fp] = 0
    return p, r


def calculate_score_fixed_metrics(tp_count, fp_count, gt_positives_count, gt_negatives_count):
    points_len = len(tp_count)
    # calculate tpr and fpr considering 4 cases - no po and no neg - no positives - no negatives - normal
    if gt_positives_count == 0 or gt_negatives_count == 0:
        # neglect this evaluation as all gt is either positives or negatives
        tpr = np.nan * np.ones((points_len,))
        fpr = np.nan * np.ones((points_len,))
        precision = np.ones((points_len,))
        recall = np.ones((points_len,))
        f1 = np.nan * np.ones((points_len,))  # f1-score should be reported for real test not for interpolation
    else:
        tpr = tp_count / float(gt_positives_count)
        fpr = fp_count / float(gt_negatives_count)
        # interpolate precision and recall for the PR curves
        precision, recall = calculate_precision_and_recall(tp_count, fp_count, gt_positives_count)
        f1 = 2 * precision * recall / (precision + recall + EPSILON).astype(float)

    return {'f1': f1,
            'tpr': tpr,
            'fpr': fpr,
            'precision': precision,
            'recall': recall,
            }


def get_events_metrics(gt, det_mtx, **params):
    gt_mask = params.get('gt_mask', np.ones_like(gt))
    gt_local = gt * gt_mask
    gt_positives_count = np.sum(np.sum(gt_local, axis=0) > 0)
    gt_negatives_count = np.sum(gt_mask) - np.sum(gt_local)
    # calculate metrics
    tp = det_mtx * gt_local
    tp_count = np.sum(np.squeeze(np.sum(tp, axis=1) > 0), axis=1)
    fp = (gt_local - det_mtx * gt_mask) == -1
    fp_count = np.sum(np.squeeze(np.sum(fp, axis=1)), axis=1)
    fixed_metrics = calculate_score_fixed_metrics(tp_count, fp_count, gt_positives_count, gt_negatives_count)
    roc_auc = area_under_the_curve(fixed_metrics['fpr'], fixed_metrics['tpr'])

    return {'tp_count': tp_count,
            'fp_count': fp_count,
            'gt_positives_count': gt_positives_count,
            'gt_negatives_count': gt_negatives_count,
            'f1': fixed_metrics['f1'],
            'tpr': fixed_metrics['tpr'],
            'fpr': fixed_metrics['fpr'],
            'precision': fixed_metrics['precision'],
            'recall': fixed_metrics['recall'],
            'roc_auc': roc_auc
            }


def normalise_based_on_snippet(spec_feat, snippet):
    spec_feat_no_nans = np.zeros_like(snippet)
    no_nans = np.invert(np.isnan(snippet))
    spec_feat_no_nans[no_nans] = snippet[no_nans]
    feat_max = np.max(spec_feat_no_nans)
    feat_min = np.min(spec_feat_no_nans)
    normalised = normalize_using_params(spec_feat, feat_min, feat_max)
    return normalised


def plot_analysis_curves(hdf, hdfs_hashes_exp, scores_hashes_exp, th_table, interp_points = 10):
    keys = ('full', 'early', 'cand')
    valid_files_count = {key: 0 for key in keys}
    # init figure
    fig, axes = plt.subplots(2, 3)
    fig.suptitle(hdf, fontsize=10)
    lines_roc = []
    lines_pr = []

    hdfs = hdfs_hashes_exp.loc[hdfs_hashes_exp['hdf'] == hdf]
    scores = scores_hashes_exp.loc[scores_hashes_exp['hdf_hash'].isin(hdfs.index)]

    # plot the baseline per curve
    for idx, key in enumerate(keys):
        axes[0, idx].plot((0, 1), (0, 1), 'k--', alpha=0.1)
        axes[1, idx].plot((0, 1), (0.012, 0.012), 'k--', alpha=0.1)
        axes[1, idx].plot((0, 1), (0.0015, 0.0015), 'k--', alpha=0.1)

    scores_len = len(scores)
    cmap = plt.cm.get_cmap('jet', scores_len)
    for scr_idx, score in enumerate(scores.index):
        color = cmap(float(scr_idx)/scores_len)
        # init constants
        th_list = th_table[scores.loc[score, 'th_list']]
        th_count = len(th_list)
        files_count = len(os.listdir(op.join(scores_root_dir, score)))
        interp_plus_orig_count = interp_points + th_count

        # initialise containers
        tp_count = {key: np.zeros((files_count, th_count)) for key in keys}
        fp_count = {key: np.zeros((files_count, th_count)) for key in keys}
        gt_pos = {key: np.zeros((files_count,)) for key in keys}
        tpr = {key: np.zeros((files_count, th_count)) for key in keys}
        fpr = {key: np.zeros((files_count, th_count)) for key in keys}
        f1 = {key: np.zeros((files_count, th_count)) for key in keys}
        precision_test = {key: np.zeros((files_count, th_count)) for key in keys}
        precision = {key: np.zeros((files_count, interp_plus_orig_count)) for key in keys}
        recall = {key: np.zeros((files_count, interp_plus_orig_count)) for key in keys}
        interp_points_mask = {key: np.zeros((files_count, interp_plus_orig_count), dtype=bool) for key in keys}
        valid_files_count = {key: 0 for key in keys}

        # read file and calc metrics per file
        for f_idx, f in enumerate(os.listdir(op.join(scores_root_dir, score))):
            try:
                with gzip.open(op.join(scores_root_dir, score, f), 'rb') as temp_file:
                    score_file = pkl.load(temp_file)
            except:
                with open(op.join(scores_root_dir, score, f), 'rb') as temp_file:
                    score_file = pkl.load(temp_file)

            for key in keys:
                tp_count[key][f_idx, :] = score_file[key]['tp_count']
                fp_count[key][f_idx, :] = score_file[key]['fp_count']
                gt_pos[key][f_idx] = score_file[key]['gt_positives_count']
                tmp_eval_metrics = calculate_roc_pr_metrics(score_file[key]['tp_count'],
                                                            score_file[key]['fp_count'],
                                                            score_file[key]['gt_positives_count'],
                                                            score_file[key]['gt_negatives_count'],
                                                            interp_points=interp_points)
                tpr[key][f_idx, :] = tmp_eval_metrics['tpr']
                fpr[key][f_idx, :] = tmp_eval_metrics['fpr']
                f1[key][f_idx, :] = tmp_eval_metrics['f1']
                # interpolated p and r and interp mask
                precision[key][f_idx, :] = tmp_eval_metrics['precision']
                recall[key][f_idx, :] = tmp_eval_metrics['recall']
                interp_points_mask[key][f_idx, :] = tmp_eval_metrics['interp_points_mask']

        # calc averages
        for key in keys:
            # clean out nans files
            tp_count[key] = np.reshape(tp_count[key][~np.isnan(tpr[key])], (-1, th_count))
            fp_count[key] = np.reshape(fp_count[key][~np.isnan(tpr[key])], (-1, th_count))
            gt_pos[key] = gt_pos[key][~np.isnan(tpr[key][:, 0])]
            tpr[key] = np.reshape(tpr[key][~np.isnan(tpr[key])], (-1, th_count))
            fpr[key] = np.reshape(fpr[key][~np.isnan(fpr[key])], (-1, th_count))
            precision[key] = np.reshape(precision[key][~np.isnan(precision[key])], (-1, interp_plus_orig_count))
            interp_points_mask[key] = np.reshape(interp_points_mask[key][~np.isnan(precision[key])],
                                                 (-1, interp_plus_orig_count))
            recall[key] = np.reshape(recall[key][~np.isnan(recall[key])], (-1, interp_plus_orig_count))
            f1[key] = np.reshape(f1[key][~np.isnan(f1[key])], (-1, th_count))
            # count valid files
            valid_files_count[key] = tpr[key].shape[0]
            # avg metrics
            if valid_files_count[key] > 0:
                tp_count[key] = np.mean(tp_count[key], axis=0)
                fp_count[key] = np.mean(fp_count[key], axis=0)
                gt_pos[key] = np.mean(gt_pos[key])
                tpr[key] = np.mean(tpr[key], axis=0)
                fpr[key] = np.mean(fpr[key], axis=0)
                precision_test[key] = np.mean(
                    np.reshape(precision[key][~interp_points_mask[key]], (-1, th_count), 'C')
                    , axis=0)
                p_temp, r_temp, mask_temp = interpolate_pr(tp_count[key], fp_count[key], gt_pos[key], interp_points)

                precision[key] = p_temp
                recall[key] = r_temp
                precision_test[key] = p_temp[~mask_temp]
                f1[key] = np.mean(f1[key], axis=0)

        # plot the different curves
        for idx, key in enumerate(keys):
            if valid_files_count[key]:
                line_roc, = axes[0, idx].plot(fpr[key], tpr[key], '-', alpha=0.4, c=color,
                                              label='F:{:0.3f}({})-A:{:0.3f}'.format(np.max(f1[key]),
                                                                                valid_files_count[key],
                                                                                area_under_the_curve(fpr[key],
                                                                                                     tpr[key])))
                line_pr, = axes[1, idx].plot(recall[key], precision[key], '-', alpha=0.4, c=color,
                                              label='A:{:0.3f}'.format(area_under_the_curve(recall[key],
                                                                                            precision[key])))

                # plot markers
                axes[0, idx].plot(fpr[key], tpr[key], 'x', alpha=0.5, c=color)
                axes[1, idx].plot(tpr[key], precision_test[key], 'x', alpha=0.5, c=color)

                # mark best scoring threshold
                idx_best = np.argwhere(f1[key] >= (np.max(f1[key])-EPSILON)).flatten()
                if idx_best.size > 1:
                    th_best_range = '{:0.2f}-{:0.2f}'.format(th_list[np.min(idx_best)], th_list[np.max(idx_best)])
                else:
                    th_best_range = '{:0.2f}'.format(th_list[idx_best[0]])
                axes[0, idx].plot(fpr[key][idx_best], tpr[key][idx_best], 'o', alpha=0.35, c=color)
                if tpr[key].size > 0:
                    axes[1, idx].plot(tpr[key][idx_best], precision_test[key][idx_best], 'o', alpha=0.35, c=color)
                    axes[1, idx].annotate(th_best_range,
                                          xy=(tpr[key][idx_best[0]], precision_test[key][idx_best[0]]), xytext=(-20, 20),
                                          textcoords='offset points', ha='right', va='bottom',
                                          bbox=dict(boxstyle='round,pad=0.5', fc=color, alpha=0.2),
                                          arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
                lines_roc = lines_roc + [line_roc]
                lines_pr = lines_pr + [line_pr]

    # arrange axis
    for idx, ax in enumerate(list(axes.transpose().flatten())):
        ax.set_xlim(-0.025, 1.025)
        ax.set_ylim(-0.025, 1.025)
        ax.set_aspect('equal')
        ax.grid()
        ax.legend(loc='best')
        if idx % 2 == 0:
            ax.set_title(keys[int(idx/2)])

    axes[0, 0].set_ylabel('ROC')
    axes[1, 0].set_ylabel('PR')

    plt.tight_layout()

    leg_roc = {}
    leg_lines = []
    for idx, key in enumerate(keys):
        if valid_files_count[key]:
            leg_roc[key] = axes[0, idx].legend(loc='best')
            leg_roc[key].get_frame().set_alpha(0.4)
            leg_lines += leg_roc[key].get_lines()

    return fig, leg_lines, lines_roc, lines_pr


def plot_analysis_per_hdf(hdf, scores, interp_points_count, th_table, scores_root_dir):
    keys = ('full', 'early')
    # init figure
    fig, axes = plt.subplots(2, 2)
    fig.suptitle(hdf, fontsize=10)

    # track different plots
    lines_roc = []
    lines_pr = []

    # plot the baseline per curve
    for idx, key in enumerate(keys):
        axes[idx, 0].plot((0, 1), (0, 1), 'k--', alpha=0.1)
        axes[idx, 1].plot((0, 1), (0.012, 0.012), 'k--', alpha=0.1)  # todo:fix
        axes[idx, 1].plot((0, 1), (0.0015, 0.0015), 'k--', alpha=0.1)

    # prepare coloring
    scores_len = len(scores)
    cmap = plt.cm.get_cmap('jet', scores_len)



    # plotting loop
    for idx, score_hash in enumerate(scores.index):
        scr_idx = scores.loc[score_hash, 'scr_idx']
        color = cmap(float(idx) / scores_len)

        th_list = th_table[scores.loc[score_hash, 'th_list']]
        # read score summary files
        summary = undump_obj_from_file(op.join(scores_root_dir, score_hash, 'summary.sum'))
        summary_interp = undump_obj_from_file(op.join(scores_root_dir, score_hash,
                                                      'interp_{}_summary.sum'.format(interp_points_count)))
        for idx, key in enumerate(keys):
            fpr = summary['comb'][key]['fpr']
            tpr = summary['comb'][key]['tpr']
            f1 = summary['comb'][key]['f1']
            valid_files_count = summary['comb'][key]['valid_files_count']
            roc_auc = summary['comb'][key]['roc_auc']
            recall = summary['comb'][key]['recall']
            precision = summary['comb'][key]['precision']
            r_interp = summary_interp['comb'][key]['r_interp']
            p_interp = summary_interp['comb'][key]['p_interp']
            pr_auc = summary_interp['comb'][key]['pr_auc']

            line_roc, = axes[idx, 0].plot(summary['comb'][key]['fpr'],
                                          summary['comb'][key]['tpr'],
                                          '-', alpha=0.4, c=color,
                                          label='({})F:{:0.3f}({})-A:{:0.3f}'.format(
                                              scr_idx, np.max(f1), valid_files_count, roc_auc)
                                          )
            line_pr, = axes[idx, 1].plot(r_interp, p_interp,
                                         '-', alpha=0.4, c=color,
                                          label='({})A:{:0.3f}'.format(scr_idx, pr_auc))

            # plot markers
            axes[idx, 0].plot(fpr, tpr, 'x', alpha=0.5, c=color)
            axes[idx, 1].plot(recall, precision, 'x', alpha=0.5, c=color)

            # mark best scoring threshold
            idx_best = np.argwhere(f1 >= (np.max(f1)-EPSILON)).flatten()
            if idx_best.size > 1:
                th_best_range = '{:0.2f}-{:0.2f}'.format(th_list[np.min(idx_best)], th_list[np.max(idx_best)])
            else:
                th_best_range = '{:0.2f}'.format(th_list[idx_best[0]])
            axes[idx, 0].plot(fpr[idx_best], tpr[idx_best], 'o', alpha=0.35, c=color)
            axes[idx, 1].plot(recall[idx_best], precision[idx_best], 'o', alpha=0.35, c=color)
            axes[idx, 1].annotate(th_best_range,
                                  xy=(recall[idx_best[0]], precision[idx_best[0]]), xytext=(-20, 20),
                                  textcoords='offset points', ha='right', va='bottom',
                                  bbox=dict(boxstyle='round,pad=0.5', fc=color, alpha=0.2),
                                  arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            lines_roc = lines_roc + [line_roc]
            lines_pr = lines_pr + [line_pr]

    # arrange axis
    for idx, ax in enumerate(list(axes.transpose().flatten())):
        ax.set_xlim(-0.025, 1.025)
        ax.set_ylim(-0.025, 1.025)
        ax.set_aspect('equal')
        ax.grid()
        ax.legend(loc='best')


    axes[0, 0].set_title('ROC')
    axes[0, 1].set_title('PR')
    axes[0, 0].set_ylabel(keys[0])
    axes[1, 0].set_ylabel(keys[1])

    plt.tight_layout()

    # leg_roc = {}
    # leg_lines = []
    # for idx, key in enumerate(keys):
    #     if valid_files_count[key]:
    #         leg_roc[key] = axes[0, idx].legend(loc='best')
    #         leg_roc[key].get_frame().set_alpha(0.4)
    #         leg_lines += leg_roc[key].get_lines()

    return fig
        # leg_lines, lines_roc, lines_pr


def summarise_scores(scores_hashes_exp, keys, interp_points, scores_root_dir):
    for score_hash in tqdm(scores_hashes_exp.index):
        # init constants per folder
        score_fldr = op.join(scores_root_dir, score_hash)
        if not op.exists(op.join(score_fldr, 'interp_{}_summary.sum'.format(interp_points))):
            th_list = TH_TABLE[scores_hashes_exp.loc[score_hash, 'th_list']]
            th_count = len(th_list)
            files_count = len(os.listdir(score_fldr))
            interp_plus_orig_count = interp_points + th_count
            # initialise containers
            tp_count = {key: np.zeros((files_count, th_count)) for key in keys}
            fp_count = {key: np.zeros((files_count, th_count)) for key in keys}
            gt_positives_count = {key: np.zeros((files_count,)) for key in keys}
            gt_negatives_count = {key: np.zeros((files_count,)) for key in keys}
            tpr = {key: np.zeros((files_count, th_count)) for key in keys}
            fpr = {key: np.zeros((files_count, th_count)) for key in keys}
            roc_auc = {key: np.zeros((files_count,)) for key in keys}
            f1 = {key: np.zeros((files_count, th_count)) for key in keys}
            precision = {key: np.zeros((files_count, th_count)) for key in keys}
            recall = {key: np.zeros((files_count, th_count)) for key in keys}
            p_interp = {key: np.zeros((files_count, interp_plus_orig_count)) for key in keys}
            r_interp = {key: np.zeros((files_count, interp_plus_orig_count)) for key in keys}
            pr_auc = {key: np.zeros((files_count,)) for key in keys}
            mask = {key: np.zeros((files_count, interp_plus_orig_count), dtype=bool) for key in keys}

            fixed_score_table = {}
            interp_score_table = {}

            for f in os.listdir(score_fldr):
                f_name = op.splitext(f)[0]
                # read score file
                score_file = undump_obj_from_file(op.join(score_fldr, f))
                # add fixed metrics to fixed table metrics
                fixed_score_table[f_name] = score_file
                # init interp table for current f_name
                interp_score_table[f_name] = {}
                # loop over keys
                for key in keys:
                    # calculate interpolated P_interp, R_interp, PR_AUC
                    p_interp_tmp, r_interp_tmp, interp_mask = interpolate_pr(score_file[key]['tp_count'],
                                                                             score_file[key]['fp_count'],
                                                                             score_file[key]['gt_positives_count'],
                                                                             interp_points)
                    pr_auc_tmp = area_under_the_curve(r_interp_tmp, p_interp_tmp)
                    # add interp metrics to interp table metrics
                    interp_score_table[f_name][key] = {'p_interp': p_interp_tmp,
                                                       'r_interp': r_interp_tmp,
                                                       'mask': interp_mask,
                                                       'pr_auc': pr_auc_tmp}

            # init avg, std and comb fields
            for type in ('avg', 'std', 'comb'):
                fixed_score_table[type] = {}
                interp_score_table[type] = {}
                for key in keys:
                    fixed_score_table[type][key] = {}
                    interp_score_table[type][key] = {}

            # add avg and comb to tables
            for key in keys:
                # loop over files
                for f_idx, f in enumerate(os.listdir(score_fldr)):
                    f_name = op.splitext(f)[0]
                    # loop over metrics
                    for metric in ('tp_count', 'fp_count', 'tpr', 'fpr', 'precision', 'recall', 'f1'):
                        # build metrics tables
                        eval(metric)[key][f_idx, :] = fixed_score_table[f_name][key][metric]
                    for metric in ('gt_positives_count', 'gt_negatives_count', 'roc_auc'):
                        # build metrics tables
                        eval(metric)[key][f_idx] = fixed_score_table[f_name][key][metric]
                    for metric in ('p_interp', 'r_interp', 'mask'):
                        # build metrics tables
                        eval(metric)[key][f_idx, :] = interp_score_table[f_name][key][metric]
                    # build metrics tables
                    pr_auc[key][f_idx] = interp_score_table[f_name][key]['pr_auc']

                # clean invalid files scores before averaging
                valid_files = np.where(~np.isnan(np.sum(tpr[key], axis=1)))[0]
                for metric in ('tp_count', 'fp_count', 'tpr', 'fpr', 'precision', 'recall', 'f1',
                               'p_interp', 'r_interp', 'mask'):
                    eval(metric)[key] = eval(metric)[key][valid_files, :]
                for metric in ('gt_positives_count', 'gt_negatives_count', 'roc_auc', 'pr_auc'):
                    eval(metric)[key] = eval(metric)[key][valid_files]

                # calculate avgs / sums
                if valid_files.size > 0:
                    # avg and std of fixed
                    for metric in ('tp_count', 'fp_count', 'gt_positives_count', 'gt_negatives_count'):
                        fixed_score_table['avg'][key][metric] = np.mean(eval(metric)[key], axis=0)
                        fixed_score_table['std'][key][metric] = np.std(eval(metric)[key], axis=0)

                    # some metrics need weighted avg
                    avg_weights = {}
                    avg_weights['tpr'] = gt_positives_count[key] / float(np.sum(gt_positives_count[key]))
                    avg_weights['fpr'] = gt_negatives_count[key] / float(np.sum(gt_negatives_count[key]))

                    # use full dataset count to average f1, p, roc_auc
                    avg_all = (gt_positives_count[key]+gt_negatives_count[key]) / \
                              float(np.sum(gt_positives_count[key])+np.sum(gt_negatives_count[key]))
                    avg_weights['f1'] = avg_all
                    avg_weights['precision'] = avg_all
                    avg_weights['roc_auc'] = avg_all
                    avg_weights['recall'] = avg_weights['tpr']
                    for metric in ('tpr', 'fpr','precision', 'recall', 'f1', 'roc_auc'):
                        w_mean, w_std = weighted_avg_and_std(eval(metric)[key], avg_weights[metric], axis=0)
                        fixed_score_table['avg'][key][metric] = w_mean
                        fixed_score_table['std'][key][metric] = w_std

                    # avg and std of pr_auc
                    w_mean, w_std = weighted_avg_and_std(pr_auc[key], avg_all, axis=0)
                    interp_score_table['avg'][key]['pr_auc'] = w_mean
                    interp_score_table['std'][key]['pr_auc'] = w_std

                    # comb for fixed
                    for metric in ('tp_count', 'fp_count', 'gt_positives_count', 'gt_negatives_count'):
                        fixed_score_table['comb'][key][metric] = np.sum(eval(metric)[key], axis=0)
                    p_comb, r_comb = calculate_precision_and_recall(fixed_score_table['comb'][key]['tp_count'],
                                                                    fixed_score_table['comb'][key]['fp_count'],
                                                                    fixed_score_table['comb'][key][
                                                                        'gt_positives_count'])
                    fixed_score_table['comb'][key]['precision'] = p_comb
                    fixed_score_table['comb'][key]['recall'] = r_comb
                    fixed_score_table['comb'][key]['f1'] = \
                        2 * p_comb * r_comb / (p_comb + r_comb + EPSILON).astype(float)
                    fixed_score_table['comb'][key]['tpr'] = \
                        fixed_score_table['comb'][key]['tp_count'] \
                        / float(fixed_score_table['comb'][key]['gt_positives_count'])
                    fixed_score_table['comb'][key]['fpr'] = \
                        fixed_score_table['comb'][key]['fp_count'] \
                        / float(fixed_score_table['comb'][key]['gt_negatives_count'])
                    fixed_score_table['comb'][key]['roc_auc'] = area_under_the_curve(
                        fixed_score_table['comb'][key]['fpr'], fixed_score_table['comb'][key]['tpr'])

                    # comb for interp
                    p_comb_interp, r_comb_interp, comb_mask = interpolate_pr(fixed_score_table['comb'][key]['tp_count'],
                                                                             fixed_score_table['comb'][key]['fp_count'],
                                                                             fixed_score_table['comb'][key][
                                                                                 'gt_positives_count'],
                                                                             interp_points)
                    interp_score_table['comb'][key]['p_interp'] = p_comb_interp
                    interp_score_table['comb'][key]['r_interp'] = r_comb_interp
                    interp_score_table['comb'][key]['mask'] = comb_mask
                    interp_score_table['comb'][key]['pr_auc'] = area_under_the_curve(r_comb_interp, p_comb_interp)

                    # save valid files count
                    for type in ('avg', 'std', 'comb'):
                        fixed_score_table[type][key]['valid_files_count'] = valid_files.size
                        interp_score_table[type][key]['valid_files_count'] = valid_files.size
            # save tables
            dump_obj_to_file(fixed_score_table, op.join(score_fldr, 'summary.sum'))
            dump_obj_to_file(interp_score_table, op.join(score_fldr, 'interp_{}_summary.sum'.format(interp_points)))


def bar_plot_compare_AUC(params_hashed_table, scores_root_dir, interp_points_count, eval_methods, boxplot, roc_pr = 'ROC'):
    auc_scores = {}
    col_labels = ["{}-{}".format(hdf, idx) for hdf, idx in zip(
        list(params_hashed_table['hdf']),
        list(params_hashed_table['scr_idx'])
    )]
    for e_method in eval_methods:
        auc_scores[e_method] = pd.DataFrame(columns=col_labels)

    for score_hash in params_hashed_table.index:
        label = "{}-{}".format(params_hashed_table.loc[score_hash, 'hdf'],
                               params_hashed_table.loc[score_hash, 'scr_idx'])
        # read score summary files
        if roc_pr == 'ROC':
            summary = undump_obj_from_file(op.join(scores_root_dir, score_hash, 'summary.sum'))
        elif roc_pr == 'PR':
            summary = undump_obj_from_file(op.join(scores_root_dir, score_hash,
                                                      'interp_{}_summary.sum'.format(interp_points_count)))
        else:
            summary = []
            print('error')

        files = [f for f in list(set(summary.keys()) - {'avg', 'std', 'comb'})]
        for file in files:
            for e_method in eval_methods:
                if roc_pr == 'ROC':
                    auc_scores[e_method].loc[file, label] = summary[file][e_method]['roc_auc']
                elif roc_pr == 'PR':
                    auc_scores[e_method].loc[file, label] = summary[file][e_method]['pr_auc']


    # plot
    fig, axes = plt.subplots(len(eval_methods), 1)
    fig.suptitle('{}-AUC'.format(roc_pr), fontsize=10)

    if boxplot:
        for idx, e_method in enumerate(eval_methods):
            sns.boxplot(ax=axes[idx], notch=True, data=auc_scores[e_method])
            plt.setp(axes[idx].xaxis.get_majorticklabels(), rotation=30)
            axes[idx].set_title(e_method)
            axes[idx].grid()
    else:
        for idx, e_method in enumerate(eval_methods):
            if idx:
                auc_scores[e_method].plot.bar(ax=axes[idx])
            else:
                auc_scores[e_method].plot.bar(ax=axes[idx], legend=False)
            plt.setp(axes[idx].xaxis.get_majorticklabels(), rotation=30)
            axes[idx].set_title(e_method)
    return fig


def weighted_avg_and_std(values, weights, axis):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights, axis=axis)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights, axis=axis)
    return (average, np.sqrt(variance))


def plot_compare_algos(test_name, exp_hashes_table, interp_points_count, eval_methods, th_table=TH_TABLE):
    # init
    scores_summary = \
        pd.DataFrame([],
                     columns=['full_f1', 'full_pr_auc', 'full_roc_auc', 'early_f1', 'early_pr_auc', 'early_roc_auc' ])
    eval_methods_len = len(eval_methods)
    markers = '.vX*P1234567'
    # create #x2 figure #eval_methodsx(ROC, PR)
    fig, axes = plt.subplots(eval_methods_len, 2)
    if len(axes.shape) == 1:
        axes = np.expand_dims(axes, 0)
    fig.suptitle(test_name, fontsize=10)

    # plot the baseline per curve "ROC"
    for idx in np.arange(0, eval_methods_len, 1):
        axes[idx, 0].plot((0, 1), (0, 1), 'k--', alpha=0.1)

    # get hdfs
    hdfs = exp_hashes_table.hdf.unique()
    cmap = plt.cm.get_cmap('jet', len(hdfs))

    # loop over hdfs
    for hdf_idx, hdf in enumerate(hdfs):
        line_idx = -1
        color = cmap(float(hdf_idx) / len(hdfs))
        # get available hdf parameterisation
        params = exp_hashes_table.loc[exp_hashes_table['hdf'] == hdf]
        # loop over different params
        for score_hash in params.index:
            line_idx += 1
            th_list = th_table[exp_hashes_table.loc[score_hash, 'th_list']]
            # read score summary files
            summary = undump_obj_from_file(op.join(scores_root_dir, score_hash, 'summary.sum'))
            summary_interp = undump_obj_from_file(op.join(scores_root_dir, score_hash,
                                                          'interp_{}_summary.sum'.format(interp_points_count)))

            for idx, e_method in enumerate(eval_methods):
                fpr = summary['comb'][e_method]['fpr']
                tpr = summary['comb'][e_method]['tpr']
                f1 = summary['comb'][e_method]['f1']
                valid_files_count = summary['comb'][e_method]['valid_files_count']
                roc_auc = summary['comb'][e_method]['roc_auc']
                recall = summary['comb'][e_method]['recall']
                precision = summary['comb'][e_method]['precision']
                r_interp = summary_interp['comb'][e_method]['r_interp']
                p_interp = summary_interp['comb'][e_method]['p_interp']
                pr_auc = summary_interp['comb'][e_method]['pr_auc']

                line_roc, = axes[idx, 0].plot(summary['comb'][e_method]['fpr'],
                                              summary['comb'][e_method]['tpr'],
                                              '-', alpha=0.4, c=color,
                                              label='({})F:{:0.3f}({})-A:{:0.3f}'.format(
                                                  exp_hashes_table.loc[score_hash, 'scr_idx'],
                                                  np.max(f1), valid_files_count, roc_auc)
                                              )
                line_pr, = axes[idx, 1].plot(r_interp, p_interp,
                                             '-', alpha=0.4, c=color,
                                             label='({})A:{:0.3f}'.format(exp_hashes_table.loc[score_hash, 'scr_idx'],
                                                                          pr_auc))

                # save scores_summary
                scores_summary.loc[hdf, '{}_f1'.format(e_method)] = np.max(f1)
                scores_summary.loc[hdf, '{}_pr_auc'.format(e_method)] = np.max(pr_auc)
                scores_summary.loc[hdf, '{}_roc_auc'.format(e_method)] = np.max(roc_auc)

                # plot markers
                axes[idx, 0].plot(fpr, tpr, markers[line_idx], alpha=0.5, c=color)
                axes[idx, 1].plot(recall, precision, markers[line_idx], alpha=0.5, c=color)

                # mark best scoring threshold
                idx_best = np.argwhere(f1 >= (np.max(f1) - EPSILON)).flatten()
                if idx_best.size > 1:
                    th_best_range = '{:0.2f}-{:0.2f}'.format(th_list[np.min(idx_best)], th_list[np.max(idx_best)])
                else:
                    th_best_range = '{:0.2f}'.format(th_list[idx_best[0]])
                axes[idx, 0].plot(fpr[idx_best], tpr[idx_best], 'o', alpha=0.35, c=color)
                axes[idx, 1].plot(recall[idx_best], precision[idx_best], 'o', alpha=0.35, c=color)
                axes[idx, 1].annotate(th_best_range,
                                      xy=(recall[idx_best[0]], precision[idx_best[0]]), xytext=(-20, 20),
                                      textcoords='offset points', ha='right', va='bottom',
                                      bbox=dict(boxstyle='round,pad=0.5', fc=color, alpha=0.2),
                                      arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    for idx, ax in enumerate(list(axes.transpose().flatten())):
        ax.set_xlim(-0.025, 1.025)
        ax.set_ylim(-0.025, 1.025)
        ax.set_aspect('equal')
        ax.grid()
        ax.legend(loc='best')

    axes[0, 0].set_title('ROC')
    axes[0, 1].set_title('PR')

    for idx in np.arange(0, eval_methods_len, 1):
        axes[idx, 0].set_ylabel(eval_methods[idx])


    plt.tight_layout()
    return scores_summary

def create_folds(file_names, folds_count, rnd_seed=RND_SEED):
    # init
    tr_folds = dict()
    val_folds = dict()
    kf = KFold(folds_count, shuffle=True,random_state=rnd_seed)
    folds = kf.split(file_names)
    for fold_idx in range(0, folds_count, 1):
        train, validation = folds.next()
        tr_folds[fold_idx] = [file_names[file_idx] for file_idx in train]
        val_folds[fold_idx] = [file_names[file_idx] for file_idx in validation]

    return tr_folds, val_folds


def get_scores_per_hdf(hdf, hdf_hashes_table, scores_hashes_table):
    # get available hdf parameterisation
    hdfs = hdf_hashes_table.loc[hdf_hashes_table['hdf'] == hdf]
    scores = scores_hashes_table.loc[scores_hashes_table['hdf_hash'].isin(hdfs.index)]
    return scores


def get_total_pr_auc_for(fldr, files, interp_count, th_count, eval_method='full'):
    tp_count = np.zeros((th_count,))
    fp_count = np.zeros((th_count,))
    gt_positives_count = 0

    summary = undump_obj_from_file(op.join(fldr, 'summary.sum'))

    for f in files:
        tp_count += summary[f][eval_method]['tp_count']
        fp_count += summary[f][eval_method]['fp_count']
        gt_positives_count += summary[f][eval_method]['gt_positives_count']
    p, r, _ = interpolate_pr(tp_count, fp_count, gt_positives_count, interp_count)
    return area_under_the_curve(r, p)



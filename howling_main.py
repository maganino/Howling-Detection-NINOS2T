# includes
from os import path as op
import os
import time
from tqdm import tqdm
import pickle as pkl
import pandas as pd
from madmom.processors import IOProcessor, process_batch
from functions import create_folder_if_not_existing, instantiate_logger, dump_obj_to_file, create_hashes,\
    onpick, plot_analysis_per_hdf, undump_obj_from_file, summarise_scores, \
    bar_plot_compare_AUC, plot_compare_algos, create_folds, get_scores_per_hdf, get_total_pr_auc_for
from classes import Dataset, FFTProcessor, HowlingAnnotationProcessor, ScoreProcessor
# seems not used as used by eval()
from classes import NinosSpaProcessor, PTPRProcessor, PAPRProcessor, PHPRProcessor, PNPRProcessor, IPMPProcessor, \
    IMSDProcessor, Ninos2Processor
from howling_params import *
import matplotlib.pyplot as plt

from numpy.random import seed
from hashlib import sha1
seed(RND_SEED)
import matplotlib2tikz as tikz_save


plt.close('all')


def main():
    #local init
    scores_summary = pd.DataFrame([])
    # create a Logger
    create_folder_if_not_existing([log_path])
    logger = instantiate_logger("main", log_path)
    logger.info("Expriment started!!!")

    # read hashes file
    hashes_table = {}
    for hash_file in hashes_paths.keys():
        hashes_path = hashes_paths[hash_file]
        if op.exists(hashes_path):
            f_handle = open(hashes_path, 'rb')
            hashes_table[hash_file] = pkl.load(f_handle)
            f_handle.close()
        else:
            create_folder_if_not_existing(op.split(hashes_path)[0])
            hashes_table[hash_file] = pd.DataFrame([])

    # Read Dataset + Annotations paths into datasets
    datasets = [Dataset('howd', op.join(ds_parent_fldr[ds], ds), anns_ext=['.csv'], files_names= FILES_NAMES) for ds in DATASET_NAMES]
    logger.info("Datasets metadata loaded")

    # Create hashes and neglect already calculated
    logger.info("Calculating hashes...")
    # calculate fft/gt hashes
    ffts_hashes_exp = create_hashes(dataset=[''.join([ds.name+str(ds.files_count) for ds in datasets])],
                                    frames_per_sec=FRAMES_PER_SEC, win_size=WIN_SIZE)
    ffts_hashes = ffts_hashes_exp[~ffts_hashes_exp.index.isin(hashes_table['fft'].index)]
    logger.info("   FFT & Ground Truth")
    # calculate HDF hashes
    hdfs_hashes_exp = pd.DataFrame([])
    for hdf in HDFS:
        hdfs_hashes_exp = hdfs_hashes_exp.append(
            create_hashes(fft_hash=ffts_hashes_exp.index, hdf=[hdf], **HDFS_PARAMS[hdf]))
    hdfs_hashes = hdfs_hashes_exp[~hdfs_hashes_exp.index.isin(hashes_table['hdf'].index)]
    logger.info("   HDFS")
    # calculate scores hashes
    scores_hashes_exp = create_hashes(hdf_hash=hdfs_hashes_exp.index, **SCORES_PARAMS)
    # move algorithm scoring params to score hashes
    scores_hashes_exp['th_list'] = np.NaN
    for hdf_hash in hdfs_hashes_exp.index:
        corresp_score_idx = scores_hashes_exp[scores_hashes_exp['hdf_hash'] == hdf_hash].index
        scores_hashes_exp.at[corresp_score_idx, 'th_list'] = hdfs_hashes_exp['th_list'][hdf_hash]
    scores_hashes = scores_hashes_exp[~scores_hashes_exp.index.isin(hashes_table['scores'].index)]
    logger.info("   Scores")

    # Pre-process (FFT)
    if len(ffts_hashes):
        tic = time.time()
        # get file list to process
        audio_filepaths = []
        anns_filepaths = []
        for ds in datasets:
            meta = ds.meta
            audio_filepaths += [op.join(meta.loc[k, 'path_audio'], k + meta.loc[k, 'extension_audio'])
                                for k in meta.index]
            anns_filepaths += [op.join(meta.loc[k, 'path_anns'], k + meta.loc[k, 'extension_anns']) for k in meta.index]
        # create fft/gt folders
        create_folder_if_not_existing([op.join(ffts_root_dir, fft_hash) for fft_hash in ffts_hashes.index])
        create_folder_if_not_existing([op.join(gts_root_dir, fft_hash) for fft_hash in ffts_hashes.index])
        # Preprocess
        for fft_hash in tqdm(ffts_hashes.index):
            # Calculate FFT with different windows and overlaps
            pre_process = FFTProcessor(fps=ffts_hashes.loc[fft_hash].frames_per_sec,
                                       window_sizes=ffts_hashes.loc[fft_hash].win_size)
            io_proc = IOProcessor(pre_process, dump_obj_to_file)
            # run the fft extraction
            process_batch(io_proc, audio_filepaths, output_dir=op.join(ffts_root_dir, fft_hash),
                          strip_ext=True, output_suffix='.fft', num_workers=NUM_WORKERS)
            # create GT
            annotator = HowlingAnnotationProcessor(fps=ffts_hashes.loc[fft_hash].frames_per_sec,
                                                   window_size=ffts_hashes.loc[fft_hash].win_size)
            io_proc = IOProcessor(annotator, dump_obj_to_file)
            # run the fft extraction
            process_batch(io_proc, anns_filepaths, output_dir=op.join(gts_root_dir, fft_hash),
                          strip_ext=True, output_suffix='.gt', num_workers=NUM_WORKERS)
            # update and save new fft hashes
            hashes_table['fft'] = hashes_table['fft'].append(ffts_hashes.loc[fft_hash])
            f_handle = open(hashes_paths['fft'], 'wb')
            pkl.dump(hashes_table['fft'], f_handle)
            f_handle.close()
        toc = time.time() - tic
        logger.info('Preprocessing and annotations ended and hashes table updated: %f sec', toc)
    else:
        logger.info('Preprocessing and annotations already done')

    # Calculate howling detection functions (HDF)
    if len(hdfs_hashes):
        tic = time.time()
        # create folders
        create_folder_if_not_existing([op.join(hdfs_root_dir, hdf_hash) for hdf_hash in hdfs_hashes.index])
        # calc hdfs
        for hdf_hash in tqdm(hdfs_hashes.index):
            hdf_params = hdfs_hashes.loc[hdf_hash].to_dict()
            hdf = hdf_params.pop('hdf')
            fft_hash = hdf_params.pop('fft_hash')
            fft_path = op.join(ffts_root_dir, fft_hash)
            filepaths = [op.join(fft_path, f) for f in os.listdir(fft_path)]
            hdf_params['lists_table'] = LISTS_TABLE
            hdf_params['norm_snippet_frames'] = int(np.ceil(NORM_SNIPPET * ffts_hashes_exp.loc[fft_hash, 'frames_per_sec']))
            hdf_processor = eval(hdf + 'Processor')(**hdf_params)
            io_proc = IOProcessor(hdf_processor, dump_obj_to_file)
            # run the hdf extraction
            process_batch(io_proc, filepaths, output_dir=op.join(hdfs_root_dir, hdf_hash), strip_ext=True,
                          output_suffix='.hdf', num_workers=NUM_WORKERS)
            # save new hdf hashes
            hashes_table['hdf'] = hashes_table['hdf'].append(hdfs_hashes.loc[hdf_hash])
            f_handle = open(hashes_paths['hdf'], 'wb')
            pkl.dump(hashes_table['hdf'], f_handle)
            f_handle.close()
        toc = time.time() - tic
        logger.info('HDFs and hashes table updated: %f sec', toc)
    else:
        logger.info('HDFs already done')

    # Scores calculation
    if len(scores_hashes):
        tic = time.time()
        # create folders
        create_folder_if_not_existing([op.join(scores_root_dir, score_hash) for score_hash in scores_hashes.index])

        for score_hash in tqdm(scores_hashes.index):
            score_params = scores_hashes.loc[score_hash].to_dict()
            hdf_hash = score_params.pop('hdf_hash')
            gt_folder = op.join(gts_root_dir,  hdfs_hashes_exp.loc[hdf_hash]['fft_hash'])
            fft_folder = op.join(ffts_root_dir,  hdfs_hashes_exp.loc[hdf_hash]['fft_hash'])
            hdf_folder = op.join(hdfs_root_dir, hdf_hash)
            filepaths = [op.join(hdf_folder, f) for f in os.listdir(hdf_folder)]
            score_params['th_table'] = TH_TABLE
            score_params['norm_snippet'] = NORM_SNIPPET
            score_params['gt_neglect'] = GT_NEGLECT
            scores_processor = ScoreProcessor(gt_folder, fft_folder, **score_params)
            io_proc = IOProcessor(scores_processor, dump_obj_to_file)
            # run the hdf extraction
            process_batch(io_proc, filepaths, output_dir=op.join(scores_root_dir, score_hash), strip_ext=True,
                          output_suffix='.scr', num_workers=NUM_WORKERS)
            # save new scores hashes
            hashes_table['scores'] = hashes_table['scores'].append(scores_hashes.loc[score_hash])
            f_handle = open(hashes_paths['scores'], 'wb')
            pkl.dump(hashes_table['scores'], f_handle)
            f_handle.close()
        toc = time.time() - tic
        logger.info('Scores calculated and hashes table updated: %f sec', toc)
    else:
        logger.info('Scores already done')

    # Summarize Scores
    keys = ('full', 'early', 'cand')
    # init containers
    tic = time.time()
    summarise_scores(scores_hashes_exp, keys, INTERP_POINTS, scores_root_dir)
    toc = time.time() - tic
    logger.info('Scores summarised: %f sec', toc)

    # Summarize hashes
    print('Summarising hashes in one table ...')
    hashes_all = pd.DataFrame(columns=list(ffts_hashes_exp.columns)
                                       + list(hdfs_hashes_exp.columns)
                                       + list(scores_hashes_exp.columns[:-1]) + ['scr_idx'])
    for scr_idx, score_hash in enumerate(scores_hashes_exp.index):
        hdf_hash = scores_hashes_exp.loc[score_hash]['hdf_hash']
        fft_hash = hdfs_hashes_exp.loc[hdf_hash]['fft_hash']
        hashes_all = hashes_all.append(pd.DataFrame(
            {
                'frames_per_sec': [ffts_hashes_exp.loc[fft_hash]['frames_per_sec']],
                'win_size': [ffts_hashes_exp.loc[fft_hash]['win_size']],
                'dataset': [ffts_hashes_exp.loc[fft_hash]['dataset']],
                'coef_sel_percent': [hdfs_hashes_exp.loc[hdf_hash]['coef_sel_percent']],
                'fft_hash': [fft_hash],
                'hdf': [hdfs_hashes_exp.loc[hdf_hash]['hdf']],
                'log': [hdfs_hashes_exp.loc[hdf_hash]['log']],
                'states': [hdfs_hashes_exp.loc[hdf_hash]['states']],
                'th_list': [hdfs_hashes_exp.loc[hdf_hash]['th_list']],
                'hdf_hash': [hdf_hash],
                'scr_candidates': [scores_hashes_exp.loc[score_hash]['scr_candidates']],
                'candidates': [scores_hashes_exp.loc[score_hash]['candidates']],
                'harmonics': [hdfs_hashes_exp.loc[hdf_hash]['harmonics']],
                'neighbours': [hdfs_hashes_exp.loc[hdf_hash]['neighbours']],
                'scr_idx': [scr_idx]
        }, index = [score_hash]
        ))

    # Parameterise
    print('###################### parameterisation ######################')
    # check if already calculated
    analysis_hash = PARAM_ON_EVAL + '_' +sha1(repr(scores_hashes_exp.index)).hexdigest()
    analysis_path = op.join(analysis_root_dir, analysis_hash)
    analysisFlag = create_folder_if_not_existing(analysis_path)
    if not analysisFlag[analysis_path]:
        # create folds
        file_names = []
        for ds in datasets:
            file_names += ds.files_names
        tr_folds, val_folds = create_folds(file_names, FOLDS_COUNT)
        best_params = {}
        # loop on hdfs
        for hdf in HDFS:
            print('for hdf={} ...'.format(hdf))
            # get parameterisation
            scores = get_scores_per_hdf(hdf, hdfs_hashes_exp, scores_hashes_exp)
            # init
            tr_scores = {}
            tr_scores['full'] = pd.DataFrame(index=scores.index, columns=['fold{}'.format(i) for i in range(0, FOLDS_COUNT, 1)])
            tr_scores['early'] = pd.DataFrame(index=scores.index,
                                             columns=['fold{}'.format(i) for i in range(0, FOLDS_COUNT, 1)])
            # loop on parameterisations
            print('     get training score per fold')
            for score_hash in tqdm(scores.index):
                # loop on folds
                for fold in tr_folds:
                    for key in ('full', 'early'):
                        # get PR_AUC per training fold
                        tr_scores[key].loc[score_hash, 'fold{}'.format(fold)] = get_total_pr_auc_for(
                            op.join(scores_root_dir, score_hash),
                            tr_folds[fold],
                            INTERP_POINTS,
                            len(TH_TABLE['th_list1']),
                            eval_method=key)
            # get the parameters maximising the PR_AUC per fold (per key)
            print('     get params maximising training score per fold')
            best_params[hdf] = pd.DataFrame(index=['fold{}'.format(fold) for fold in np.arange(0,FOLDS_COUNT)])
            for key in  ('full', 'early'):
                best_params[hdf] = best_params[hdf].join(pd.DataFrame(tr_scores[key].idxmax(), columns=['params_{}'.format(key)]))
            # loop on folds
            print('     get corresponding testing score')
            for fold in tqdm(val_folds):
                for key in ('full', 'early'):
                    # get the test PR_AUC corresponding to the optimal params
                    best_params[hdf].loc['fold{}'.format(fold), 'val_score_{}'.format(key)] = get_total_pr_auc_for(
                        op.join(scores_root_dir, best_params[hdf].loc['fold{}'.format(fold), 'params_{}'.format(PARAM_ON_EVAL)]),
                        val_folds[fold],
                        INTERP_POINTS,
                        len(TH_TABLE['th_list1']),
                        eval_method=key)
            # calculate avg
            print('     get averages')
            for key in ('full', 'early'):
                val_score_key = 'val_score_{}'.format(key)
                best_params[hdf].loc['avg', val_score_key] = best_params[hdf][val_score_key].mean()
                best_params[hdf].loc['std', val_score_key] = best_params[hdf][val_score_key].std()
                best_params[hdf].loc['avg', 'params_{}'.format(key)] = '---'
                best_params[hdf].loc['std', 'params_{}'.format(key)] = '---'

            # get best param for all and its relative score
            print('     get best params and score for all files together')
            all_files_scoring = pd.DataFrame(index=scores.index, columns=['score_full', 'score_early'])
            for score_hash in tqdm(scores.index):
                summary_all = undump_obj_from_file(op.join(scores_root_dir, score_hash,
                                                           'interp_{}_summary.sum'.format(INTERP_POINTS)))
                all_files_scoring.loc[score_hash, 'score_full'] = summary_all['comb']['full']['pr_auc']
                all_files_scoring.loc[score_hash, 'score_early'] = summary_all['comb']['early']['pr_auc']
            for key in ('full', 'early'):
                best_params[hdf].loc['all_files', 'val_score_{}'.format(key)] = all_files_scoring['score_{}'.format(key)].max()
                best_params[hdf].loc['all_files', 'params_{}'.format(key)] = all_files_scoring['score_{}'.format(key)].idxmax()

        # select params for graph visualisation
        print('Get visualisation parameters per hdf')
        best_hashes = {'all': pd.DataFrame()}
        for idx, hdf in enumerate(HDFS):
            # use only the per fold score/param and not the comb
            best_params_temp = best_params[hdf][~ best_params[hdf].index.isin(['avg','std', 'all_files'])]\
                [['params_{}'.format(PARAM_ON_EVAL), 'val_score_{}'.format(PARAM_ON_EVAL)]]
            best_params_temp = best_params_temp.rename(columns={'params_{}'.format(PARAM_ON_EVAL):'params'})
            # choose parameterisation per hdf
            best_hashes[hdf] = hashes_all.loc[hashes_all.index.isin(best_params_temp['params'])]
            best_hashes['all'] = best_hashes['all'].append(hashes_all.loc[
                                                               hashes_all.index == best_params_temp.groupby(
                                                                   ['params']).mean().idxmax()[0]])

        # save besthashes best params
        dump_obj_to_file(best_params, op.join(analysis_root_dir, analysis_hash, 'best_params.sum'))
        dump_obj_to_file(best_hashes, op.join(analysis_root_dir, analysis_hash, 'best_hashes.sum'))
    else:
        print('Load saved parameterisation')
        best_params = undump_obj_from_file(op.join(analysis_root_dir, analysis_hash, 'best_params.sum'))
        best_hashes = undump_obj_from_file(op.join(analysis_root_dir, analysis_hash, 'best_hashes.sum'))
    # print parameterisation table cols(fold, params(multi), score)
    for hdf in HDFS:
        print('---> {}'.format(hdf))
        print(best_params[hdf])
    ################################# Plot ##########################################
    if local_server == 'local':

        # plot per hdf
        if PLOT_SINGLE_HDF:
            for idx, hdf in enumerate(HDFS):
                fig = plot_analysis_per_hdf(hdf,
                                            best_hashes[hdf],
                                            INTERP_POINTS,
                                            TH_TABLE,
                                            scores_root_dir)
                print("#################### Fig.{}-{} ##########################".format(idx, hdf))
                print(best_hashes[hdf][
                          ['scr_idx', 'hdf', 'frames_per_sec', 'win_size', 'states', 'coef_sel_percent', 'scr_candidates']].to_string(
                    index=False))

        # plot ROC-auc
        fig = bar_plot_compare_AUC(best_hashes['all'], scores_root_dir, INTERP_POINTS, ('full', 'early'), boxplot=PLOT_PR_AUC_PER_HDF_or_FILE, roc_pr='ROC')
        print("#################### Fig. {} #######################".format("ROC/PR - files"))
        print(
            best_hashes['all'][['scr_idx', 'hdf', 'frames_per_sec', 'win_size', 'states', 'coef_sel_percent', 'scr_candidates']].to_string(
                index=False))

        # plot PR-auc
        fig = bar_plot_compare_AUC(best_hashes['all'], scores_root_dir, INTERP_POINTS, ('full', 'early'), boxplot=PLOT_PR_AUC_PER_HDF_or_FILE, roc_pr='PR')

        # plot roc, pr to compare algos
        if compare_all_best:
            best_hashes_for_all = pd.DataFrame()
            for key in best_hashes:
                if key != 'all':
                    best_hashes_for_all = best_hashes_for_all.append(best_hashes[key])
        else:
            best_hashes_for_all = best_hashes['all']

        scores_summary = plot_compare_algos(exp_name, best_hashes_for_all, INTERP_POINTS, ('full', 'early'), th_table=TH_TABLE)

        if SAVEPLOT2TIKZ:
            tikz_save.save('{}_PR_ROC.tex'.format(exp_name),
                           encoding='utf-8',
                           figureheight='\\figureheight',
                           figurewidth='\\figurewidth', externalize_tables=True, override_externals=True,
                           float_format="{:.3f}")
        print("#################### Fig. {} ####".format("Algorithms Comparison Full/early"))
        print(
            best_hashes_for_all[['scr_idx', 'hdf', 'win_size', 'states', 'coef_sel_percent', 'scr_candidates', 'harmonics', 'neighbours']].to_string(
                index=False))

        #plot candidates
        # plot_compare_algos(exp_name, best_hashes_for_all, INTERP_POINTS, ['cand'], th_table=TH_TABLE)
        # print("#################### Fig. {} ####".format("Algorithms Comparison Candidates"))
        # print(
        #     best_hashes_for_all[['scr_idx', 'hdf', 'win_size', 'states', 'coef_sel_percent', 'scr_candidates']].to_string(
        #         index=False))
        plt.show()

    ################################# Print tables for latex  ##########################################
    if PRINTTABLESLATEX:
        print(scores_summary.astype(float).round(2).to_latex())
# main
if __name__ == '__main__':
    main()
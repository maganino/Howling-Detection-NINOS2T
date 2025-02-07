import numpy as np
import os.path as op
import csv
import itertools

# for reproducability
RND_SEED = 42

# experiment name
sm= 'Speech'
exp_name =  'test_{}'.format(sm) #'test_eval' #''
# usable files names
usable_file_names = '{}_ds.csv'.format(sm)
                        # ''

# Parameterisation
FOLDS_COUNT = 5
INTERP_POINTS = 50
# Flags
local_server = 'local'
compress_files = False
# processing
NUM_WORKERS = 2
compare_all_best = False

# Framing
SAMPLE_RATE = 16000
DET_TIME_OFFSET = 5     # seconds
NORM_SNIPPET = 2
FRAMES_PER_SEC = [
    # 21.5,
    50,
    # 100
]
WIN_SIZE = [
    512,
    1024,
    2048,
    4096
]
# logarithmic loudness
EPSILON = np.spacing(1)
CMP_FACTOR = 1


# Root Paths
root_input = {'local': '/Users/mina/Dropbox/Howling_NINOS/Dataset/Processed',
              'server': '/esat/dspdata/mshehata/no_backup/input/Tests/Howling',
              'server_dsp_user': '/esat/dspdata/mshehata/no_backup/input/Tests/Howling'
              }
root_output = {
    'local': '/Users/mina/Documents/Rosso/KU Leuven/zPhD_Code/Code/dir_to_ignore',
    'server': '/users/sista/mshehata/Code Backup/Output/Howling',
    'server_dsp_user': '/esat/dspdata/mshehata/temp/output/Howling'
               }
# Datasets directories
ds_parent_fldr = {}
ds_parent_fldr.update(dict.fromkeys(['Music', 'Speech'], root_input[local_server]))
ds_parent_fldr.update(dict.fromkeys(['MusicTrain', 'SpeechTrain', 'ToonExample', 'ToonExamplePlusSpeech',
                                     'ToonExampleDouble', 'onlySpeech', 'ToonExample-multiple'], root_input[local_server]))



# Paths
hashes_root_path = op.join(root_output[local_server], exp_name, 'hashes')
hashes_paths = {"fft": op.join(hashes_root_path, 'fft.hash'),
                "hdf": op.join(hashes_root_path, 'hdf.hash'),
                "scores": op.join(hashes_root_path, 'score.hash')}
log_path = op.join(root_output[local_server], exp_name)
ffts_root_dir = op.join(root_output[local_server], exp_name, 'ffts')
gts_root_dir = op.join(root_output[local_server], exp_name, 'gts')
hdfs_root_dir = op.join(root_output[local_server], exp_name, 'hdfs')
peaks_root_dir = op.join(root_output[local_server], exp_name, 'peaks')
scores_root_dir = op.join(root_output[local_server], exp_name, 'scores')
analysis_root_dir = op.join(root_output[local_server], exp_name, 'analysis')

# Experiment
DATASET_NAMES = [
    # 'MusicTrain',
    # 'SpeechTrain',
    # 'ToonExample',
    sm
                 ]      # datasets to use

# read usable file names
FILES_NAMES = []
try:
    with open(usable_file_names, 'rb') as f:
        reader = csv.reader(f)
        usable_files_meta = list(reader)
        processing_type = usable_files_meta[0][0]
        if processing_type == 'filter':
            file_type = usable_files_meta[1][0]
            file_nums = usable_files_meta[2]
            air_nums = usable_files_meta[3]
            exclude_nums = usable_files_meta[4]
            FILES_NAMES = [file_type + '_{}{}'.format(f_num.strip(), air_num.strip())
                           for f_num, air_num in itertools.product(file_nums, air_nums)]
            if exclude_nums:
                files_to_exclude = [file_type + '_{}'.format(num.strip()) for num in exclude_nums]
                FILES_NAMES = list(set(FILES_NAMES) - set(files_to_exclude))
        elif processing_type == 'list':
            file_names = usable_files_meta[1]
            FILES_NAMES = [f.strip() for f in file_names]
        FILES_NAMES.sort()
except:
    pass

HDFS = [
    'NinosSpa',
    'Ninos2',
    'PTPR',
    'PAPR',
    'PHPR',
    'PNPR',
    'IPMP',
    'IMSD'
        ]

HDFS_PARAMS = {'NinosSpa': {'states': [4,8,16,32,64,96], 'coef_sel_percent': [50,75,90,95,100], 'log':[ 1], 'th_list': ['th_list1']},
               'Ninos2': {'states': [4,8,16,32,64,96], 'coef_sel_percent': [50,75,90,95,100], 'log':[ 1], 'th_list': ['th_list1']},
               'PTPR': {'th_list': ['th_list1']},
               'PAPR': {'th_list': ['th_list1']},
               'PHPR': {'th_list': ['th_list1'], 'harmonics': ['harm1','harm2','harm3']},
               'PNPR': {'th_list': ['th_list1'], 'neighbours': ['nei1','nei2','nei3']},
               'IPMP': {'th_list': ['th_list1'], 'states': [4,8,16,32,64,96], 'max_candidates': [3]},
               'IMSD': {'th_list': ['th_list1'], 'states': [4,8,16,32,64,96]},
               }

LISTS_TABLE = {'harm1': np.array([2]),
               'harm2': np.array([2, 3]),
               'harm3': np.array([2, 3, 4]),
               'nei1': np.array([2]),
               'nei2': np.array([2, 3]),
               'nei3': np.array([2, 3, 4])
               }


TH_TABLE = {'th_list1': np.array([-np.inf] + list(np.arange(0, 1.05, 0.05)) + [np.inf]),
            'th_list2': np.array([-np.inf] + list(np.arange(28, 65, 6)) + [np.inf]), # not suitable as differently implemented by toon
            'th_list3': np.array([-np.inf] + list(np.arange(32, 55, 2)) + [np.inf]),
            'th_list4': np.array([-np.inf, 9] + list(np.arange(17, 46, 4)) + [np.inf]),
            'th_list5': np.array([-np.inf] + list(np.arange(6, 22, 3)) + [np.inf]),
            'th_list6': np.arange(0, 1.126, 0.125),
            'th_list7': np.array([0, 0.05] + list(np.arange(0.1, 0.6, 0.1)) + [1, 2, np.inf]),
            'th_list8': np.array(list(np.arange(0,0.8,0.01)) + list(np.arange(0.8, 01.01, 0.01))),
            }
SCORES_PARAMS = {'candidates': [3], 'scr_candidates': [1,3,'all']}

GT_NEGLECT = np.max(np.concatenate((np.asarray(HDFS_PARAMS['NinosSpa']['states']),
                                np.asarray(HDFS_PARAMS['Ninos2']['states']),
                                np.asarray(HDFS_PARAMS['IPMP']['states']),
                                np.asarray(HDFS_PARAMS['IMSD']['states'])))) \
             / float(np.min(np.asarray(FRAMES_PER_SEC)))
# GT_NEGLECT = 0
PARAM_ON_EVAL = 'early'
PLOT_PR_AUC_PER_HDF_or_FILE = True
PLOT_SINGLE_HDF = False
SAVEPLOT2TIKZ = False
PRINTTABLESLATEX = False


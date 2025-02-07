# includes
from functions import get_files_meta_with_extensions, normalize_to_limits, \
    ninos_sparsity, fft_stacker, keep_max_n_elements_per, undump_obj_from_file, calc_imsd, \
    normalize_using_params, get_events_metrics, keep_max_n_elements_per_zeroing_m_neighbours, \
    normalise_based_on_snippet, ninos_sparsity_energy
from madmom.processors import SequentialProcessor, ParallelProcessor, Processor
from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.spectrogram import SpectrogramProcessor, LogarithmicSpectrogramProcessor
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from six import string_types
from csv import DictReader
from bisect import bisect
import pandas as pd
# local includes
from parameters import *
from howling_params import DET_TIME_OFFSET, GT_NEGLECT

# Dataset class
class Dataset:
    """
    Class holding meta data about dataset

    Variables
    ---------

    """
    def __init__(self, dataset_type, path, audio_ext=('.wav', '.flac'), anns_ext=('.onsets', 'onsets.txt'),
                 files_names = ()):
        self.dataset_type = dataset_type
        self.path = path
        self.audio_ext = audio_ext
        self.anns_ext = anns_ext
        self.name = op.basename(path)
        self.files_names = files_names
        self.meta = self.load_meta()
        self.files_count = len(self.meta)

    # load files meta data
    def load_meta(self):
        audio_meta = get_files_meta_with_extensions(self.path, self.audio_ext)
        anns_meta = get_files_meta_with_extensions(self.path, self.anns_ext)

        # filter datasets and retain specific files
        if len(self.files_names) > 0:
            audio_meta = audio_meta.loc[audio_meta.index.isin(self.files_names)]
            anns_meta = anns_meta.loc[anns_meta.index.isin(self.files_names)]
        else:
            self.files_names = list(audio_meta.index)

        meta = audio_meta
        # put meta data in one table
        meta = meta.join(anns_meta, how='inner', lsuffix='_audio', rsuffix='_anns')

        # check for missing files
        more_audio = len(audio_meta) - len(meta)
        more_anns = len(anns_meta) - len(meta)
        if more_audio > 0:
            print('{}: Missing annotations (or wrong extensions): {}files'.format(self.path, more_audio))
        if more_anns > 0:
            print('{}: Missing audio: {}files'.format(self.path, more_audio))
        return meta

    # Todo: overload print datasets


# Picklable Objects
class SpectrogramPkl:
    def __init__(self, spec):
        self.spec = np.asarray(spec).transpose()
        self.frequencies = spec.bin_frequencies
        self.num_frames = spec.num_frames
        self.num_bins = spec.num_bins


class HowlingAnnotationPkl:
    def __init__(self, meta, fps, window_size):
        self.howling_freq = float(meta[' MSG frequency'])
        self.howling_start = float(meta[' Start transition time']) + float(meta[' Length transition time']) / 2
        self.sampling_freq = int(meta[' Sampling frequency'])
        # calculate howling onset
        sig_length = float(meta[' Audio length'])
        num_frames = int(np.ceil(sig_length * fps))
        num_bins = int(np.floor(window_size/2.0))
        self.howling_start_index = int(np.floor(self.howling_start * fps))
        # calc howling frequency index
        frequencies = np.fft.fftfreq(num_bins*2, 1. / self.sampling_freq)[:num_bins]
        # calculate annotations
        anns = np.zeros((num_bins, num_frames))
        if self.howling_freq != -1:
            howling_freq_index = bisect(frequencies, self.howling_freq)
            anns[howling_freq_index - 1:howling_freq_index + 1, self.howling_start_index:] = 1
            # # enlarging the GT with the nearest frequency
            # if abs(self.howling_freq - frequencies[howling_freq_index-1]) \
            #         < abs(frequencies[howling_freq_index+1] - self.howling_freq):
            #     nearest_freq_idx = howling_freq_index - 2
            # else:
            #     nearest_freq_idx = howling_freq_index + 2
            # anns[nearest_freq_idx, self.howling_start_index:] = 1
        self.anns = anns
        # early detection time threshold
        self.early_det_frame = max(3, np.ceil((float(meta[' Start transition time'])
                                               + float(meta[' Length transition time'])
                                               + DET_TIME_OFFSET) * fps))
        self.fps = fps


# Annotation classes
class HowlingAnnotationProcessor(Processor):
    def __init__(self, fps, window_size):
        self.fps = fps
        self.window_size = window_size

    def process(self, data, **kwargs):
        meta_file = open(data)
        meta_reader = DictReader(meta_file)
        meta = [m for m in meta_reader][0]
        meta_file.close()
        return HowlingAnnotationPkl(meta, self.fps, self.window_size)


# Frequency Transform classes
class FFTProcessor(SequentialProcessor):
    """
    Framing and FFT
    """
    def __init__(self, fps=FRAMES_PER_SEC, window_sizes=WIN_SIZE, window=None,
                 log_magnitude=False, compression_factor=CMP_FACTOR):
        if ~isinstance(window_sizes, list):
            window_sizes = [window_sizes]
        # read signal
        sig = SignalProcessor(num_channels=1)
        multi = ParallelProcessor([])
        for frame_size in window_sizes:
            frames = FramedSignalProcessor(frame_size=frame_size, fps=fps, end='normal')
            stft = ShortTimeFourierTransformProcessor(window=window)  # hanning window is applied
            if log_magnitude:
                spec = LogarithmicSpectrogramProcessor(log=np.log, add=compression_factor)
            else:
                spec = SpectrogramProcessor()
            multi.append(SequentialProcessor([frames, stft, spec]))

        stack = fft_stacker
        # instantiate a SequentialProcessor
        super(FFTProcessor, self).__init__([sig, multi, stack])


# ODF classes
class NINOS2(SequentialProcessor):
    """
    Reading the signal, pre-processing, calculating the NINOS2 odf and saving them
    """
    def __init__(self, fps=FRAMES_PER_SEC, compression_factor=CMP_FACTOR, window_size=WIN_SIZE,
                 sel_coef_percent=SEL_COEF_PERCENT, norm_before_combine=False, online=True):
        # read signal
        sig = SignalProcessor(num_channels=1)
        multi = ParallelProcessor([])
        for frame_size in window_size:
            frames = FramedSignalProcessor(frame_size=frame_size, fps=fps)
            stft = ShortTimeFourierTransformProcessor()  # hanning window is applied
            spec = LogarithmicSpectrogramProcessor(log=np.log, add=compression_factor)
            ninos2 = Ninos2FuncProcessor(sel_coef_percent, norm_before_combine, online)
            # process each frame size with spec and diff sequentially
            multi.append(SequentialProcessor([frames, stft, spec, ninos2]))
        stack = np.dstack
        # instantiate a SequentialProcessor
        super(NINOS2, self).__init__([sig, multi, stack])


class Ninos2FuncProcessor(Processor):
    """
    a processor function calculating the normalised ninos2 odf (sparsity-energy)

    Parameters
    ----------
    sel_coef_percent: int [0,100]
                      the percentage of bins to use when calculating the ninos2
    online: bool
            to limit data dependency only on past samples/frames
    norm_before_combine: bool, only if online=False
                         to normalise the sparsity and energy measures separately before multiplying them
    """
    def __init__(self, sel_coef_percent=SEL_COEF_PERCENT, norm_before_combine=False, online=True):
        self.sel_coef_percent = sel_coef_percent
        self.online = online
        self.norm_before_combine = norm_before_combine

    def process(self, data, **kwargs):
        sel_coeff = (self.sel_coef_percent*data.shape[1])/100
        spec_sorted = np.sort(data, axis=1)
        spec_sorted_sel = spec_sorted[:, :sel_coeff]
        l4norm = np.power(np.sum(spec_sorted_sel ** 4, axis=1), 1.0/4)
        l4norm[l4norm == 0] = 1
        if self.online:     # normalisation can't be applied as it depends on future samples
            l2norm2 = np.sum(spec_sorted_sel ** 2, axis=1)
            ninos2 = np.true_divide(l2norm2, l4norm)
        elif self.norm_before_combine:
            l2norm = np.power(np.sum(spec_sorted_sel ** 2, axis=1), 1.0/2)
            energy_norm = normalize_to_limits(l2norm)
            spa_norm = normalize_to_limits(np.true_divide(l2norm, l4norm))
            ninos2 = energy_norm * spa_norm
        else:
            l2norm2 = np.sum(spec_sorted_sel ** 2, axis=1)
            ninos2 = normalize_to_limits(np.true_divide(l2norm2, l4norm))
        return ninos2


# HDF classes
class Ninos2Processor(Processor):
    """
    a processor function calculating the ninos_spa hdf (sparsity)

    Parameters
    ----------
    sel_coef_percent: int [0,100]
                      the percentage of bins to use when calculating the ninos2
    status: the memory states used in calculation
    """
    def __init__(self, **params):
        self.sel_coef_percent = int(params.get("coef_sel_percent", 100))
        self.states = int(params.get("states", 0))
        self.log = bool(params.get("log", 0))
        self.norm_snippet = params.get('norm_snippet_frames', 0)

    def process(self, data, **kwargs):
        if isinstance(data, string_types):
            data = undump_obj_from_file(data)
        if self.log:
            data += CMP_FACTOR
            np.log10(data, data)
        sel_coeff = (self.sel_coef_percent*self.states)/100

        spec_framed = [data[i:i + self.states, :]
                       for i in np.arange(0, data.shape[0] - self.states, 1, dtype=int)]
        ninos_odf_mesh_s = np.zeros([data.shape[1], len(spec_framed)])
        for idx, spec_frame in enumerate(spec_framed):
            if self.sel_coef_percent < 100:
                spec_sorted = np.sort(spec_frame, axis=0)
                spec_sorted_sel = spec_sorted[:sel_coeff,:]
            else:
                spec_sorted_sel = spec_frame
            ninos_odf_mesh_s[:, idx] = ninos_sparsity_energy(spec_sorted_sel.transpose())

        # norm using a snippet of the signal
        if self.norm_snippet > 0:
            # neglecting the DC component in the snippet and first (garbage frames)
            ninos_odf_mesh_s = normalise_based_on_snippet(ninos_odf_mesh_s, ninos_odf_mesh_s[1:, int(
                0.02 * self.norm_snippet):self.norm_snippet])[:,
                               self.norm_snippet:]
        return ninos_odf_mesh_s


class NinosSpaProcessor(Processor):
    """
    a processor function calculating the ninos_spa hdf (sparsity)

    Parameters
    ----------
    sel_coef_percent: int [0,100]
                      the percentage of bins to use when calculating the ninos2
    status: the memory states used in calculation
    """
    def __init__(self, **params):
        self.sel_coef_percent = float(params.get("coef_sel_percent", 100))
        self.states = int(params.get("states", 0))
        self.log = bool(params.get("log", 0))

    def process(self, data, **kwargs):
        if isinstance(data, string_types):
            data = undump_obj_from_file(data)
        if self.log:
            data += CMP_FACTOR
            np.log10(data, data)

        spec_framed = [data[i:i + self.states, :]
                       for i in np.arange(0, data.shape[0] - self.states, 1, dtype=int)]
        ninos_odf_mesh_s = np.zeros([data.shape[1], len(spec_framed)])

        sel_coeff = int((self.sel_coef_percent * self.states) / 100.0)

        for idx, spec_frame in enumerate(spec_framed):
            if self.sel_coef_percent < 100:
                spec_sorted = np.sort(spec_frame, axis=0)
                spec_sorted_sel = spec_sorted[:sel_coeff,:]
            else:
                spec_sorted_sel = spec_frame
            ninos_odf_mesh_s[:, idx] = ninos_sparsity(spec_sorted_sel.transpose())
        return ninos_odf_mesh_s


class IPMPProcessor(Processor):
    def __init__(self, **params):
        self.states = int(params.get("states", 0)) + 1      # adding one for the candidate
        self.candidates = int(params.get("max_candidates", 3))
    def process(self, data, **kwargs):
        if isinstance(data, string_types):
            data = undump_obj_from_file(data)
        # mark maxima's (candidates with ones)
        maximas = keep_max_n_elements_per(data, self.candidates, col_row=1) != 0
        maximas_framed = np.asarray([maximas[i:i + self.states, :]
                       for i in np.arange(0, maximas.shape[0] - self.states, 1, dtype=int)])
        ipmp = np.sum(maximas_framed, axis=1)/float(self.states)
        # already normalised measure
        return ipmp.transpose()


class IMSDProcessor(Processor):
    def __init__(self, **params):
        self.states = int(params.get("states", 0)) + 1      # adding one for the candidate
        self.norm = params.get('normalisation', [])
        self.norm_snippet = params.get('norm_snippet_frames', 0)

    def process(self, data, **kwargs):
        if isinstance(data, string_types):
            data = undump_obj_from_file(data)

        spec_framed = np.asarray([data[i:i + self.states, :]
                       for i in np.arange(0, data.shape[0] - self.states, 1, dtype=int)])
        imsd = np.zeros([data.shape[1], len(spec_framed)])
        for idx, spec_frame in enumerate(spec_framed):
            imsd[:, idx] = calc_imsd(spec_frame.transpose())

        # NB. imsd is an inverse measure

        if self.norm_snippet > 0:
            # neglecting the DC component in the snippet and first (garbage frames)
            imsd = normalise_based_on_snippet(imsd, imsd[1:, int(0.02*self.norm_snippet):self.norm_snippet])[:, self.norm_snippet:]

        if self.norm:
            imsd = normalize_using_params(imsd, self.norm[0], self.norm[1])

        # to invert the measure
        imsd = np.max(imsd) - imsd
        return imsd


class PTPRProcessor(Processor):
    def __init__(self, **params):
        self.norm = params.get('normalisation', [])
        self.norm_snippet = params.get('norm_snippet_frames', 0)

    def process(self, data, **kwargs):
        if isinstance(data, string_types):
            data = np.asarray(undump_obj_from_file(data))
        ptpr = 20 * np.log10(data + EPSILON).transpose()
        # norm using a snippet of the signal
        if self.norm_snippet > 0:
            # neglecting the DC component in the snippet and first (garbage frames)
            ptpr = normalise_based_on_snippet(ptpr, ptpr[1:, int(0.02*self.norm_snippet):self.norm_snippet])[:, self.norm_snippet:]
        # norm with values
        if self.norm:
            ptpr = normalize_using_params(ptpr, self.norm[0], self.norm[1])
        return ptpr


class PAPRProcessor(Processor):
    def __init__(self, **params):
        self.norm = params.get('normalisation', [])
        self.norm_snippet = params.get('norm_snippet_frames', 0)

    def process(self, data, **kwargs):
        if isinstance(data, string_types):
            data = np.asarray(undump_obj_from_file(data))
        # Todo: i have a feeling the definition in the paper is wrong -- at least ... works better
        papr = 10 * np.log10(
            (np.power(data, 2)/ np.expand_dims(np.mean(np.power(data, 2), axis=1) + EPSILON, axis=1)) + EPSILON).transpose()

        if self.norm_snippet > 0:
            # neglecting the DC component in the snippet and first (garbage frames)
            papr = normalise_based_on_snippet(papr, papr[1:, int(0.02 * self.norm_snippet):self.norm_snippet])[:,
                   self.norm_snippet:]
        if self.norm:
            papr = normalize_using_params(papr, self.norm[0], self.norm[1])
        return papr


class PHPRProcessor(Processor):
    def __init__(self, **params):
        self.lists_table = params.get('lists_table', {'harm0': [2, 3]})
        self.harmonics = self.lists_table[params.get('harmonics', 'harm0')]
        self.norm = params.get('normalisation', [])
        self.norm_snippet = params.get('norm_snippet_frames', 0)

    def process(self, data, **kwargs):
        # Todo: make PHPR for fractional sub-harmonics
        row_end_list = []
        row = []
        if isinstance(data, string_types):
            data = np.asarray(undump_obj_from_file(data)).transpose()
        div_harm_mtx = np.zeros((len(self.harmonics), data.shape[0], data.shape[1]))
        for idx, harm in enumerate(self.harmonics):
            for row in np.arange(1, data.shape[0]/harm +1):
                div_harm_mtx[idx, row-1, :] = \
                    (data[int(np.floor(row*harm-1)), :] + data[int(np.ceil(row*harm-1)), :])/2.0
            row_end_list += [row]

        phpr = 20 * (np.log10(data + EPSILON) - np.log10(div_harm_mtx + EPSILON))
        for idx, r in enumerate(row_end_list):
            phpr[idx, r:, :] = np.nan      # put out of range harmonics to nan

        if self.norm_snippet > 0:
            # neglecting the DC component in the snippet and first (garbage frames)
            phpr = normalise_based_on_snippet(phpr, phpr[:, 1:, int(0.02 * self.norm_snippet):self.norm_snippet])[:, :, self.norm_snippet:]
        if self.norm:
            phpr = normalize_using_params(phpr, self.norm[0], self.norm[1])
        return phpr


class PNPRProcessor(Processor):
    def __init__(self, **params):
        self.lists_table = params.get('lists_table', {'nei0': [2, 3, 4]})
        self.neighbours = self.lists_table[params.get('neighbours', 'nei0')]
        self.norm = params.get('normalisation', [])
        self.norm_snippet = params.get('norm_snippet_frames', 0)

    def process(self, data, **kwargs):
        row_end_list_after = []
        row_end_list_before = []
        row = []
        if isinstance(data, string_types):
            data = np.asarray(undump_obj_from_file(data)).transpose()
        div_nei_mtx = np.zeros((len(self.neighbours)*2, data.shape[0], data.shape[1]))
        # + neighbours
        for idx, nei in enumerate(self.neighbours):
            for row in np.arange(0, data.shape[0]-nei):
                div_nei_mtx[idx, row, :] = data[row + nei, :]
            row_end_list_after += [row+1]
        # - neighbours
        for idx, nei in enumerate(self.neighbours):
            for row in (data.shape[0] -1 - np.arange(0, data.shape[0] - nei)):
                div_nei_mtx[idx + len(self.neighbours), row, :] = data[row - nei, :]
            row_end_list_before += [row]

        pnpr = 20 * (np.log10(data + EPSILON) - np.log10(div_nei_mtx + EPSILON))
        for idx, r in enumerate(row_end_list_after):
            pnpr[idx, r:, :] = np.nan
        for idx, r in enumerate(row_end_list_before):
            pnpr[idx + len(self.neighbours), :r, :] = np.nan

        if self.norm_snippet > 0:
            # neglecting the DC component in the snippet and first (garbage frames)
            pnpr = normalise_based_on_snippet(pnpr, pnpr[:, 1:, int(0.02 * self.norm_snippet):self.norm_snippet])[:, :,
                   self.norm_snippet:]
        if self.norm:
            pnpr = normalize_using_params(pnpr, self.norm[0], self.norm[1])
        return pnpr


# Scores classes
class ScoreProcessor(Processor):
    def __init__(self, gt_folder, fft_folder, **params):
        self.gt_folder = gt_folder
        self.fft_folder = fft_folder
        self.th_table = params.get('th_table', {'th_list0': [-np.inf, 0, 0.5, 1, np.inf]})
        self.candidates = params.get('candidates', 3)
        self.scr_candidates = params.get('scr_candidates', 'all')
        self.thresholds = self.th_table[params.get('th_list', 'th_list0')]
        self.gt_neglect = params.get('gt_neglect', 0)
        self.norm_snippet = params.get('norm_snippet', 0)
    def process(self, data, **kwargs):
        # manipulate hdf
        hdf = undump_obj_from_file(data)

        if len(hdf.shape) < 3:
            hdf = np.expand_dims(hdf, axis=0)

        # zero hdf DC component
        hdf[:, 0, :] = 0

        # hdf_mtx full
        hdf_mtx_full = np.repeat(np.expand_dims(hdf, 0), self.thresholds.size, axis=0)
        hdf_mtx_full[np.isnan(hdf_mtx_full)] = -np.inf     # fix for th comparison
        hdf_mtx_full_thresholded = (hdf_mtx_full.transpose() >= self.thresholds).transpose()
        hdf_mtx_full_thresholded = np.all(hdf_mtx_full_thresholded, axis=1)


        if self.scr_candidates != 'all':
            hdf_temp = hdf
            hdf_temp[np.isnan(hdf_temp)] = -np.inf
            for hdf_layer in np.arange(0, hdf_temp.shape[0], 1):
                hdf_temp[hdf_layer, :, :] = keep_max_n_elements_per(hdf_temp[hdf_layer, :, :], self.scr_candidates)

            # hdf_mtx with score candidates
            hdf_mtx = np.repeat(np.expand_dims(hdf_temp, 0), self.thresholds.size, axis=0)
            hdf_mtx_thresholded = (hdf_mtx.transpose() >= self.thresholds).transpose()
            hdf_mtx_thresholded = np.all(hdf_mtx_thresholded, axis=1)
        else:
            hdf_mtx_thresholded = hdf_mtx_full_thresholded

        # manipulate gt
        file_name = op.splitext(op.split(data)[-1])[0]
        gt_path = op.join(self.gt_folder, file_name + '.gt')
        gt_file = undump_obj_from_file(gt_path)
        gt = np.asarray(gt_file.anns)
        # fair comparison considering memory + aligning gt and detections
        gt_hdf_len_diff = gt.shape[1] - hdf.shape[2]
        eval_start_frame = int(gt_file.fps * (self.gt_neglect + self.norm_snippet))
        gt = gt[:, max(eval_start_frame, gt_hdf_len_diff):]
        hdf_mtx_thresholded = hdf_mtx_thresholded[:, :, max(0, eval_start_frame-gt_hdf_len_diff):]
        hdf_mtx_full_thresholded = hdf_mtx_full_thresholded[:, :, max(0, eval_start_frame-gt_hdf_len_diff):]

        eval_metrics = dict()
        # gt count on full frame
        eval_metrics['full'] = get_events_metrics(gt, hdf_mtx_thresholded)

        # gt count on candidates (Toon's way)
        fft_path = op.join(self.fft_folder, file_name + '.fft')
        fft_file = undump_obj_from_file(fft_path).transpose()[:, :gt.shape[1]]
        maximas = keep_max_n_elements_per_zeroing_m_neighbours(fft_file, self.candidates, 1, col_row=0) != 0
        # neglecting DC candidates
        maximas[0, :] =0
        eval_metrics['cand'] = get_events_metrics(gt, hdf_mtx_full_thresholded, gt_mask=maximas)

        # gt for early howling detection
        early_mask = np.zeros_like(gt)
        early_mask[:, :max(0,int(gt_file.early_det_frame-eval_start_frame+1))] = 1
        eval_metrics['early'] = get_events_metrics(gt, hdf_mtx_thresholded, gt_mask=early_mask)
        return eval_metrics

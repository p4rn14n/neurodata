import os
from configparser import ConfigParser
import tables
import cv2
import cvui
import numpy as np
import matplotlib
# matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from matplotlib import colors
from matplotlib.dates import DateFormatter
from roipoly import RoiPoly
from scipy import signal
import pandas as pd
from scipy.ndimage import gaussian_filter
#from numba import vectorize
#import numba
#import seaborn as sns
import multiprocessing as mp
from joblib import Parallel, delayed
import re
from numpy import sin, linspace, pi
# from pylab import plot, show, title, xlabel, ylabel, subplot
from scipy.fft import fft
from PIL import Image as PILImage
from videofig import videofig
import time
from enum import Enum
import io
import ffmpeg

aa = np.array([])
threshArray = np.array([])
timeArray = np.array([])
expectedRewardArray = np.array([])
actualRewardArray = np.array([])
threshCrossArray = np.array([])
spontThreshArray = np.array([])
avgROI1Array = []
avgROI2Array = []
avgROI3Array = []
avgROI4Array = []
avgROI5Array = []
avgROI6Array = []
avgROI7Array = []
avgROI8Array = []
avgROI9Array = []
avgROI10Array = []
avgDffROI1 = np.array([])
avgDffROI2 = np.array([])
avgDffROI3 = np.array([])
avgDffROI4 = np.array([])
avgDffROI5 = np.array([])
avgDffROI6 = np.array([])
avgDffROI7 = np.array([])
avgDffROI8 = np.array([])
avgDffROI9 = np.array([])
avgDffROI10 = np.array([])
roi_corr_all = np.array([])
epochSize = []
stackedTrialsNA = []
stackedTrialsAN = []
stackedTrialsNN = []
tiledTrialsNA = []
tiledTrialsAN = []
tiledTrialsNN = []
avgROI1RewardArray = np.array([])
avgROI2RewardArray = np.array([])
avgROI1RewardArrayColor = []
avgROI2RewardArrayColor = []
spontThreshArrayColor = []
avgROI1TrialsNN = []
avgROI2TrialsNN = []
avgROI1TrialsAN = []
avgROI2TrialsAN = []
avgROI1TrialsNA = []
avgROI2TrialsNA = []
fr = []

import math
import numpy as np
import tables

def _iter_time_chunks(node, chunk_size):
    """Yield (start, stop, np.array) over time dimension for a 4D node (T,H,W,C)."""
    T = node.shape[0]
    for start in range(0, T, chunk_size):
        stop = min(start + chunk_size, T)
        # read only this slice into RAM
        yield start, stop, node[start:stop, :, :, :]

def calculate_dff0_to_hdf5(image_path, out_path, chunk_size=512):
    """
    Two-pass, low-RAM dF/F writer:
      - Pass 1: compute per-pixel/channel mean and std over time (streaming)
      - Pass 2: compute dF/F in chunks, mask blue by green std, write (G-B) to out_path as 'dffc_images'
    Memory stays ~O(H*W*C); chunk_size controls RAM used per block.
    """
    with tables.open_file(image_path, mode='r') as h5in:
        # pick the right dataset name
        if hasattr(h5in.root, 'raw_images'):
            node = h5in.root.raw_images
        elif hasattr(h5in.root, 'dffc_images'):
            # allow reusing an existing dff file as input if someone passes it
            node = h5in.root.dffc_images
            if node.ndim == 3:
                raise ValueError("Input appears to be dffc (T,H,W), not raw (T,H,W,C). Provide the raw movie.")
        else:
            raise tables.NoSuchNodeError("Neither 'raw_images' nor 'dffc_images' found in input file")

        if node.ndim != 4:
            raise ValueError(f"Expected 4D dataset (T,H,W,C), got shape {node.shape}")

        T, H, W, C = node.shape
        if C < 2:
            raise ValueError("Expected at least two channels (blue & green).")

        # ---- Pass 1: streaming mean & std (per pixel, per channel)
        sum_ = np.zeros((H, W, C), dtype=np.float64)
        sumsq = np.zeros((H, W, C), dtype=np.float64)

        for _, _, chunk in _iter_time_chunks(node, chunk_size):
            chunk = chunk.astype(np.float32, copy=False)
            sum_   += chunk.sum(axis=0, dtype=np.float64)
            sumsq  += (chunk * chunk).sum(axis=0, dtype=np.float64)

        mean = (sum_ / float(T)).astype(np.float32)
        var  = np.maximum(sumsq / float(T) - mean.astype(np.float64)**2, 0.0)
        std  = np.sqrt(var).astype(np.float32)
        # keep green std for masking blue dF/F
        std_g = std[:, :, 1]  # (H,W)

    # ---- Pass 2: compute dF/F and write (green - blue) to disk in chunks
    filters = tables.Filters(complevel=5, complib='zlib')
    with tables.open_file(image_path, mode='r') as h5in, \
         tables.open_file(out_path, mode='w') as h5out:

        node = h5in.root.raw_images  # we already checked presence above
        T, H, W, C = node.shape

        atom = tables.Float32Atom()
        # chunk along time; H,W full per slice
        chunkshape = (min(chunk_size, T), H, W)
        dffc = h5out.create_carray(
            h5out.root, 'dffc_images', atom, shape=(T, H, W),
            filters=filters, chunkshape=chunkshape
        )

        # guard for division
        mean_nz = np.where(mean != 0, mean, 1.0)

        for start, stop, chunk in _iter_time_chunks(node, chunk_size):
            chunk = chunk.astype(np.float32, copy=False)  # (n,H,W,C)
            # dF/F = (F - mean) / mean
            dff = (chunk - mean) / mean_nz
            # clean nans/infs
            dff[~np.isfinite(dff)] = 0.0

            # mask blue channel where green std==0 (no signal in green)
            # blue is channel 0, green is channel 1 (as in your code)
            dff[..., 0] = np.where(std_g > 0.0, dff[..., 0], 0.0)

            # corrected dF/F = green - blue
            dffc_block = dff[..., 1] - dff[..., 0]   # (n,H,W)
            dffc[start:stop, :, :] = dffc_block.astype(np.float32)

    return out_path  # convenience


import numpy as np
import tables
from scipy.ndimage import gaussian_filter

def filter_and_mask_dffc_hdf5(
    dffc_path,
    mask_array,
    out_dataset='dffc_filt',
    sigma_spatial=3.0,
    chunk_size=512,
    spatial_only=True,
    overwrite=True,
):
    """
    Stream (T,H,W) dffc from HDF5, apply Gaussian filter and brain mask in chunks,
    and write result to a new dataset in the same file.

    - spatial_only=True -> sigma=(0, sigma_spatial, sigma_spatial) so no temporal smoothing
    - mask_array should be (H,W); values >0 inside brain, 0 outside (or NaN if you want NaNs)
    """
    # prepare mask
    mask = mask_array
    if mask.ndim == 3:  # PNGs sometimes come in with 3 channels
        mask = mask[..., 0]
    # prefer binary float mask to avoid NaN explosions unless you explicitly want NaNs
    if np.isnan(mask).any():
        # keep NaNs if user intended NaN-masking
        mask_for_mul = mask
    else:
        mask_for_mul = (mask > 0).astype(np.float32)

    with tables.open_file(dffc_path, mode='a') as h5:
        if not hasattr(h5.root, 'dffc_images'):
            raise tables.NoSuchNodeError("Expected dataset 'dffc_images' in file")

        src = h5.root.dffc_images  # (T,H,W), float32
        T, H, W = src.shape

        # Remove destination if it exists and overwrite requested
        if hasattr(h5.root, out_dataset):
            if overwrite:
                h5.remove_node('/', out_dataset)
            else:
                raise RuntimeError(f"Dataset '{out_dataset}' already exists")

        atom = tables.Float32Atom()
        filters = tables.Filters(complevel=5, complib='zlib')
        dst = h5.create_carray(
            '/', out_dataset, atom, shape=(T, H, W),
            filters=filters, chunkshape=(min(chunk_size, T), H, W)
        )

        # choose sigma
        sigma = (0.0, float(sigma_spatial), float(sigma_spatial)) if spatial_only else float(sigma_spatial)

        for start in range(0, T, chunk_size):
            stop = min(start + chunk_size, T)
            block = src[start:stop, :, :].astype(np.float32, copy=False)  # (n,H,W)

            # filter (no temporal smoothing if sigma[0]==0)
            block = gaussian_filter(block, sigma=sigma)

            # mask (broadcast (H,W) -> (n,H,W))
            block *= mask_for_mul[None, :, :].astype(np.float32, copy=False)

            # ensure finite
            block[~np.isfinite(block)] = 0.0
            dst[start:stop, :, :] = block



def clear_vars():
    global thresh, threshArray, timeArray, expectedRewardArray, actualRewardArray, threshCrossArray, spontThreshArray, \
        avgROI1Array, avgROI2Array, avgROI3Array, avgROI4Array, avgROI5Array, avgROI6Array, avgROI7Array, avgROI8Array, \
        avgROI9Array, avgROI10Array, epochSize, stackedTrialsNA, stackedTrialsAN, stackedTrialsNN, tiledTrialsNA, tiledTrialsAN, \
        tiledTrialsNN, avgROI1RewardArray, avgROI2RewardArray, avgROI1TrialsNN, avgROI2TrialsNN, avgROI1TrialsAN, avgROI2TrialsAN, \
        avgROI1TrialsNA, avgROI2TrialsNA, avgROI1RewardArrayColor, avgROI2RewardArrayColor, \
        spontThreshArrayColor, avgDffROI1, avgDffROI2, avgDffROI3, avgDffROI4, avgDffROI5, avgDffROI6, avgDffROI7, avgDffROI8, avgDffROI9, avgDffROI10
    threshArray = np.array([])
    timeArray = np.array([])
    expectedRewardArray = np.array([])
    actualRewardArray = np.array([])
    threshCrossArray = np.array([])
    spontThreshArray = np.array([])
    epochSize = 5
    stackedTrialsNA = []
    stackedTrialsAN = []
    stackedTrialsNN = []
    tiledTrialsNA = []
    tiledTrialsAN = []
    tiledTrialsNN = []
    avgROI1RewardArray = np.array([])
    avgROI2RewardArray = np.array([])
    avgROI1RewardArrayColor = []
    avgROI2RewardArrayColor = []
    spontThreshArrayColor = []
    avgROI1TrialsNN = []
    avgROI2TrialsNN = []
    avgROI1TrialsAN = []
    avgROI2TrialsAN = []
    avgROI1TrialsNA = []
    avgROI2TrialsNA = []
    avgDffROI1 = np.array([])
    avgDffROI2 = np.array([])
    avgDffROI3 = np.array([])
    avgDffROI4 = np.array([])
    avgDffROI5 = np.array([])
    avgDffROI6 = np.array([])
    avgDffROI7 = np.array([])
    avgDffROI8 = np.array([])
    avgDffROI9 = np.array([])
    avgDffROI10 = np.array([])

def GetConfigValues(data_dir):
    global image_stream_filename, dffHistory, roi_names, roi1, roi2, roi3, roi4, roi5, roi6, roi7, roi8, roi9, roi10, reward_threshold, fr
    config = ConfigParser()
    config.read(data_dir + os.sep + 'config.ini')
    cfg = config.get('configsection', 'config')
    cfgDict = dict(config.items(cfg))
    data_root = config.get(cfg, 'data_root')
    image_stream_filename = config.get(cfg, 'raw_image_file')
    res = list(map(int, config.get(cfg, 'resolution').split(', ')))
    # fr = int(config.get(cfg, 'framerate'))
    # dffHistory = int(config.get(cfg, 'dff_history'))
    anchor = cvui.Point()
    # roi_names = re.split('[-+/%]',cfgDict['roi_operation'])
    # rec = list(map(int, config.get(cfg, roi_names[0]).split(',')))
    # roi1 = cvui.Rect(75, 38, rec[2], rec[3])
    # roi2 = cvui.Rect(75, 149, rec[2], rec[3])
    # roi3 = cvui.Rect(161, 38, rec[2], rec[3])
    # roi4 = cvui.Rect(75, 195, rec[2], rec[3])
    # roi5 = cvui.Rect(75, 125, rec[2], rec[3])
    # roi6 = cvui.Rect(161,125, rec[2], rec[3])
    # roi7 = cvui.Rect(161,149, rec[2], rec[3])
    # roi8 = cvui.Rect(161,195, rec[2], rec[3])
    # roi9 = cvui.Rect(55,195, rec[2], rec[3])
    # roi10 = cvui.Rect(181,195, rec[2], rec[3])
    # n_tones = int(config.get(cfg, 'n_tones'))

    return cfgDict

def get_freqs(n_tones):
    # quarter-octave increment factor
    qo = 2 ** (1 / 4)
    # initial audio frequency
    freqs = [1000]
    freqDict = {}
    
    #import pdb; pdb.set_trace()
    for i in range(1, n_tones):
        binSize = int(100 / n_tones)
        freq = freqs[-1] * qo

        freqDict.update({i: freq for i in range(binSize * (i - 1), 101)})
        freqs.append(freq)

    return freqDict

def get_spont_reward_threshold_dff(df, dff0):
    global thresh, nRewards
    # determine threshold for spontaneous activity
    spontaneous_reward_rate = 2
    spontaneous_thresh = 0
    # threshold the activity above reward point
    # df[df['activity'] < 40] = 0
    #df['time'] = pd.to_datetime(df['time'])
    # sess_dur = (df['time'].iloc[-1] - df['time'].iloc[0]).total_seconds() / 60
    sess_dur = df['time'].iloc[-1] / 60
    dff_steps = np.arange(0,0.2,0.001)
    reward_steps = []
    for thresh in dff_steps:
        nRewards = sum(dff0 >thresh)
        reward_rate = float(nRewards / sess_dur)
        reward_steps.append(reward_rate)
        if reward_rate <= spontaneous_reward_rate and not spontaneous_thresh:
            spontaneous_thresh = thresh
            # break
    # print("[INFO] Spontaneous threshold is: " + str(spontaneous_thresh))
    return spontaneous_thresh, dff_steps, reward_steps

def get_spont_reward_threshold(videoTimestampDF):
    global thresh, nRewards
    # determine threshold for spontaneous activity
    spontaneous_reward_rate = 10
    spontaneous_thresh = 0
    # threshold the activity above reward point
    # df[df['activity'] < 40] = 0
    #df['time'] = pd.to_datetime(df['time'])
    sess_dur = (videoTimestampDF['time'].iloc[-1] - videoTimestampDF['time'].iloc[0]).total_seconds() / 60
    for thresh in range(1, 100):
        nRewards = sum(i > thresh for i in videoTimestampDF['roi_activity'])
        reward_rate = float(nRewards / sess_dur)
        if reward_rate <= spontaneous_reward_rate:
            spontaneous_thresh = thresh
            break
    # print("[INFO] Spontaneous threshold is: " + str(spontaneous_thresh))
    return spontaneous_thresh

def plt_spontaneous_thresholds(mouse_id, data_dir):
    global data_file, df, spontThreshArray,spontThreshArrayColor, threshArray, actualRewardArray
    data_file = 'VideoTimestamp.txt'
    df = pd.read_csv(data_dir + os.sep + data_file, sep='\t', parse_dates=['time'])

    spont_thresh = get_spont_reward_threshold(df)
    spontThreshArray = np.append(spontThreshArray, spont_thresh)
    threshArray = np.append(threshArray, reward_threshold)
    if reward_threshold > 0:
        actualRewardFrames = df.frame[df['roi_activity'] > reward_threshold]
    else:
        actualRewardFrames = []
    actualRewardArray = np.append(actualRewardArray, len(actualRewardFrames))
    if 'sham_roi_activity' in df:
        spontThreshArrayColor = np.append(spontThreshArrayColor, 'r')
    else:
        if 'sham_freq' in df:
            spontThreshArrayColor = np.append(spontThreshArrayColor, 'y')
        else:
            spontThreshArrayColor = np.append(spontThreshArrayColor, 'g')

def plt_threshold_crossings(mouse_id, data_dir):
    global data_file, df, threshCrossArray,avgROIRewardArrayColor
    data_file = 'VideoTimestamp.txt'
    df = pd.read_csv(data_dir + os.sep + data_file, sep='\t', parse_dates=['time'])

    fix_thresh = 25
    actualRewardFrames = df.frame[df['roi_activity'] > fix_thresh]
    threshCrossArray = np.append(threshCrossArray, len(actualRewardFrames))
    if 'sham_roi_activity' in df:
        avgROIRewardArrayColor = np.append(avgROIRewardArrayColor, 'r')
    else:
        if 'sham_freq' in df:
            avgROIRewardArrayColor = np.append(avgROIRewardArrayColor, 'y')
        else:
            avgROIRewardArrayColor = np.append(avgROIRewardArrayColor, 'g')

def plt_roi_activity(mouse_id, data_dir):
    global data_file, df, aa
    aa = pd.DataFrame()
    data_file = 'VideoTimestamp.txt'
    df = pd.read_csv(data_dir + os.sep + data_file, sep='\t', parse_dates=['time'])
    aa = aa.append(df['roi_activity'])
    # a.append(df['roi_activity'].mean())
    ax = plt.axes()
    plt.hlines(y=reward_threshold, xmin=min(df['time']), xmax=max(df['time']), color='k', linestyle='-', linewidth=5.0)
    ax.plot(df['time'], df['roi_activity'])
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
    ax.xaxis.set_major_locator(plt.MaxNLocator(18))
    ax.grid()
    plt.xticks(rotation=30)
    plt.xlabel('Time',fontweight="bold", fontsize=22)
    plt.setp(ax.get_xticklabels(), fontsize=15)
    plt.ylabel('Activation(%)',fontweight="bold", fontsize=22)
    plt.setp(ax.get_yticklabels(), fontsize=15)
    plt.ylim([0, 100])
    plt.title('Mouse ' + str(mouse_id),fontweight="bold", fontsize=22)

# @vectorize(['float32(float32, float32)'], target='cuda')
def moving_average(a, n=5) :
    ret = np.cumsum(a, axis=0, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ret[n -1:] = ret[n - 1:] / n
    return ret

def get_dark_frames(green_frames):
    """for first and last 1000 frames, find index of dark frames by comparing mean to a threshold
    """
    threshold = 4
    green_frames = np.reshape(green_frames, (green_frames.shape[0], green_frames.shape[1]*green_frames.shape[2]))
    temporal_means = np.mean(green_frames, axis=1)
    start_index = 0
    end_index = 0
    for mean, i in zip(temporal_means, range(0, temporal_means.shape[0]-1)):
        if mean < threshold and i < 1000:
            start_index= i
        elif mean < threshold and i >= 1000:
            end_index = i
            break
    return (start_index, end_index)

#@vectorize(['float64(uint8, float64)'], target='cuda')
def sub_mean(images, mn):
    return np.subtract(images, mn)

def div_mean(im,mn):
    return np.divide(im, mn, out=np.zeros_like(mn), where=mn!=0)

def dff_correct(dff):
    return dff[:,:,:,1] - dff[:,:,:,0]

#@vectorize(['float32(float32)'], target='cuda')
def gauss_filter(dff_filt):
    # return gaussian_filter(dff_filt, 1, truncate=8.0)
    return gaussian_filter(dff_filt, 7)

# @vectorize(['float64(float64,bool)'], target='cuda')
def apply_mask(dff_filt, mask):
    dff_filt[:, ~(mask>0)] = 0
    return dff_filt

def plotline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--', label='Slope:'+str(slope))

def get_trials_ix(data_dir):
    data_file = 'VideoTimestamp.txt'
    df = pd.read_csv(data_dir + os.sep + data_file, sep='\t', parse_dates=['time'])
    tr_st = tr_en = 0
    if 'trial' in df.columns:
        tr = df.trial.diff()
        tr_st = tr[tr>0].index
        tr_en = tr[tr<0].index
    else:
        tr_st = [0]
        tr_en = [df.shape[0]-1]
    return tr_st, tr_en

def plt_roi_fluoro(mouse_id, data_dir, processing_dir):
    global thresh, threshArray, timeArray, expectedRewardArray, actualRewardArray, avgROI1Array, avgROI2Array, avgROI3Array, \
        avgROI4Array, avgROI5Array, avgROI6Array, avgROI7Array, avgROI8Array, avgROI9Array, avgROI10Array, roi_corr_all, epochSize, stackedTrialsNA,\
    stackedTrialsAN, stackedTrialsNN, tiledTrialsNA, tiledTrialsAN, tiledTrialsNN, avgROI1RewardArray, avgROI2RewardArray, avgROI1TrialsNN, avgROI2TrialsNN,\
        avgROI1TrialsAN, avgROI2TrialsAN, avgROI1TrialsNA, avgROI2TrialsNA, avgROI1RewardArrayColor, avgROI2RewardArrayColor, \
    avgDffROI1, avgDffROI2, avgDffROI3, avgDffROI4, avgDffROI5, avgDffROI6, avgDffROI7, avgDffROI8, avgDffROI9, avgDffROI10

    # set the number of processes
    n_proc = mp.cpu_count()

    if os.path.exists(processing_dir + os.sep + 'image_stream_dffcorrected_runwin.hdf5'):
        dff_filt = ReadImageStream(processing_dir + os.sep + 'image_stream_dffcorrected_runwin.hdf5')
        mask = get_brain_mask(mouse_id, [])

    else:
        images = ReadImageStream(data_dir + os.sep + image_stream_filename)
        start, stop = get_dark_frames(images[:,:,:,0])
        images = images[start:stop, :,:,:]




        dffCorrected = dff_corrected_moving(images, n_proc)

        # dffG = dff[:,:,:,1]
        # dffG = cheby1_filter_parallel(dffG, 0.1, 6, fr, order=3, rp=1, n_jobs=1)
        # dffB = dff[:,:,:,0]
        # dffB = cheby1_filter_parallel(dffB, 0.1, 6, fr, order=3, rp=1, n_jobs=1)
        # dffCorrected = dffG - dffB
        # dffCorrected = dffG
        mask = get_brain_mask(mouse_id, images[1000,:,:,1])
        dff_filt = dffCorrected
        dff_filt = gauss_filter(dff_filt)

        filters = tables.Filters(complevel=9, complib='zlib')
        dff_hdf5_file = tables.open_file(processing_dir + os.sep + 'image_stream_dffcorrected_runwin.hdf5', mode = "w")
        data_storage = dff_hdf5_file.create_carray(dff_hdf5_file.root, 'raw_images', obj = dff_filt, filters=filters)
        dff_hdf5_file.close()
    dff_filt = apply_mask(dff_filt, mask)
    # Test seed plots############
    image = dff_filt[0,:,:]
    # plt.imshow(images[0,:,:,0])
    # br = plt.ginput()
    # plt.close()
    # br = Position(int(br[0][1]), int(br[0][0]))
    br = Position(118, 128)
    seeds = generate_seeds(br, 25.6, 'u')
    normal_seeds_x_R = []
    normal_seeds_y_R = []
    normal_seeds_x_L = []
    normal_seeds_y_L = []
    image[br.row, br.col] = 255
    for seed in seeds:
        if "-R" in seed.name:
            image[seed.row, seed.col] = 255
            normal_seeds_x_R.append(seed.col)
            normal_seeds_y_R.append(seed.row)
        else:
            image[seed.row, seed.col] = 128
            normal_seeds_x_L.append(seed.col)
            normal_seeds_y_L.append(seed.row)
    seeds_fig = plt.figure(figsize=(8, 8))
    seeds_ax = seeds_fig.add_subplot(111)
    seeds_ax.matshow(image)
    # seeds_ax.plot(br.col, br.row, '.k')
    # seeds_ax.plot(normal_seeds_x_L, normal_seeds_y_L, '.b')
    # seeds_ax.plot(normal_seeds_x_R, normal_seeds_y_R, '.r')
    plt.savefig(processing_dir + os.sep + 'seeds.svg')
    plt.close()
    ########################
    dffROI1 = dff_filt[:, roi1.y: roi1.y + roi1.height, roi1.x: roi1.x + roi1.width]
    dffROI2 = dff_filt[:, roi2.y: roi2.y + roi2.height, roi2.x: roi2.x + roi2.width]
    dffROI3 = dff_filt[:, roi3.y: roi3.y + roi3.height, roi3.x: roi3.x + roi3.width]
    dffROI4 = dff_filt[:, roi4.y: roi4.y + roi4.height, roi4.x: roi4.x + roi4.width]
    dffROI5 = dff_filt[:, roi5.y: roi5.y + roi5.height, roi5.x: roi5.x + roi5.width]
    dffROI6 = dff_filt[:, roi6.y: roi6.y + roi6.height, roi6.x: roi6.x + roi6.width]
    dffROI7 = dff_filt[:, roi7.y: roi7.y + roi7.height, roi7.x: roi7.x + roi7.width]
    dffROI8 = dff_filt[:, roi8.y: roi8.y + roi8.height, roi8.x: roi8.x + roi8.width]
    dffROI9 = dff_filt[:, roi9.y: roi9.y + roi9.height, roi9.x: roi9.x + roi9.width]
    dffROI10 = dff_filt[:, roi10.y: roi10.y + roi10.height, roi10.x: roi10.x + roi10.width]
    avgDffROI1 = dffROI1.mean(axis=(1,2))
    avgDffROI2 = dffROI2.mean(axis=(1,2))
    avgDffROI3 = dffROI3.mean(axis=(1,2))
    avgDffROI4 = dffROI4.mean(axis=(1,2))
    avgDffROI5 = dffROI5.mean(axis=(1,2))
    avgDffROI6 = dffROI6.mean(axis=(1,2))
    avgDffROI7 = dffROI7.mean(axis=(1,2))
    avgDffROI8 = dffROI8.mean(axis=(1,2))
    avgDffROI9 = dffROI9.mean(axis=(1,2))
    avgDffROI10 = dffROI10.mean(axis=(1,2))

    rois_df = pd.DataFrame({'avgDffROI1': avgDffROI1, 'avgDffROI2': avgDffROI2, 'avgDffROI3': avgDffROI3, \
                            'avgDffROI4': avgDffROI4, 'avgDffROI5': avgDffROI5, 'avgDffROI6': avgDffROI6, \
                            'avgDffROI7': avgDffROI7, 'avgDffROI8': avgDffROI8, 'avgDffROI9': avgDffROI9, 'avgDffROI10': avgDffROI10})
    roi_corr = rois_df.corr()
    if len(roi_corr_all) == 0:
        roi_corr_all = roi_corr
    else:
        roi_corr_all = np.dstack([roi_corr_all, roi_corr])


    # avgROI1Array = np.append(avgROI1Array, avgDffROI1)
    # avgROI1Array[-1] = 0.5
    # avgROI2Array = np.append(avgROI2Array, avgDffROI2)
    # avgROI2Array[-1] = 0.5
    # avgROI3Array = np.append(avgROI3Array, avgDffROI3)
    # avgROI3Array[-1] = 0.5
    # avgROI4Array = np.append(avgROI4Array, avgDffROI4)
    # avgROI4Array[-1] = 0.5
    # avgROI5Array = np.append(avgROI5Array, avgDffROI5)
    # avgROI5Array[-1] = 0.5
    # avgROI6Array = np.append(avgROI6Array, avgDffROI6)
    # avgROI6Array[-1] = 0.5
    # avgROI7Array = np.append(avgROI7Array, avgDffROI7)
    # avgROI7Array[-1] = 0.5
    # avgROI8Array = np.append(avgROI8Array, avgDffROI8)
    # avgROI8Array[-1] = 0.5
    # avgROI9Array = np.append(avgROI9Array, avgDffROI9)
    # avgROI9Array[-1] = 0.5
    # avgROI10Array = np.append(avgROI10Array, avgDffROI10)
    # avgROI10Array[-1] = 0.5
    avgROI1Array.append(avgDffROI1)
    avgROI2Array.append(avgDffROI2)
    avgROI3Array.append(avgDffROI3)
    avgROI4Array.append(avgDffROI4)
    avgROI5Array.append(avgDffROI5)
    avgROI6Array.append(avgDffROI6)
    avgROI7Array.append(avgDffROI7)
    avgROI8Array.append(avgDffROI8)
    avgROI9Array.append(avgDffROI9)
    avgROI10Array.append(avgDffROI10)
    data_file = 'VideoTimestamp.txt'
    df = pd.read_csv(data_dir + os.sep + data_file, sep='\t', parse_dates=['time'])

    # a.append(df['roi_activity'].mean())
    # ax = plt.gcf().axes

    # ax[0].plot(avgDffROI1)
    # ax[0].plot(avgDffROI2)
    # ax.plot(avgROIArray)
    # ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
    # ax.xaxis.set_major_locator(plt.MaxNLocator(18))
    # ax.grid()
    # plt.xticks(rotation=30)
    # plt.xlabel('Time',fontweight="bold", fontsize=22)
    # plt.setp(ax.get_xticklabels(), fontsize=15)
    # plt.ylabel('raw fluorescence(%)',fontweight="bold", fontsize=22)
    # plt.setp(ax.get_yticklabels(), fontsize=15)
    # plt.ylim([0, 100])
    # ax[0].set_title('Mouse' + str(mouse_id),fontweight="bold", fontsize=22)

    ### Plot DFF, ROI activity and audio frequency ################################
    # fig = plt.figure(3)
    # nullfmt = plt.NullFormatter()
    # gridspec_layout1= gridspec.GridSpec(2,1)
    # # gridspec_layout1.update(left=0.05, right=0.97, hspace=0.01, wspace=0.018)
    # pyplot_top = fig.add_subplot(gridspec_layout1[0])
    # pyplot_top.set_ylabel('Frequency', fontsize=20)
    # pyplot_top.xaxis.set_major_formatter(nullfmt)
    # if 'sham_freq' in df.columns:
    #     pyplot_top.plot(df.sham_freq, label='Audio freq', color='c')
    # else:
    #     pyplot_top.plot(df.freq, label='Audio freq', color='c')
    # pyplot_mid = fig.add_subplot(gridspec_layout1[1])
    # pyplot_mid.set_ylabel('% ROI activity', fontsize=20)
    # pyplot_mid.set_ylim(0,100)
    # pyplot_mid.plot(df.roi_activity, label='ROI activity', color='g')
    # pyplot_mid.set_xlabel('Frames', fontsize=20)
    #
    # pyplot_mid1 = pyplot_mid.twinx()
    # pyplot_mid1.set_ylabel('DFF', fontsize=20)
    # pyplot_mid1.set_ylim(-0.15, 0.35)
    # pyplot_mid1.plot(avgDffROI, label='DFF', color='b')
    #
    #
    # # pyplot_bottom = fig.add_subplot(gridspec_layout1[2])
    # # pyplot_bottom.set_ylabel('DFF')
    # # pyplot_bottom.set_ylim(-0.15, 0.35)
    # # pyplot_bottom.plot(avgDffROI, label='DFF', color='b')
    # # pyplot_bottom.set_xlabel('Frames')
    #
    # rewardFrames = []
    # if 'sham_roi_activity' in df.columns:
    #     rewardFrames = df.frame[df['sham_roi_activity'] > reward_threshold]
    # else:
    #     rewardFrames = df.frame[df['roi_activity'] > reward_threshold]
    # for xc in rewardFrames:
    #     epoch = np.arange(df.frame[xc]-epochSize,df.frame[xc]+epochSize)
    #     if avgDffROI[xc] > avgDffROI[epoch[0]]:
    #         pyplot_mid1.axvline(x=xc, color='r', linestyle='-', alpha=0.9, ymax=0.40)
    # if not rewardFrames.empty:
    #     pyplot_mid1.axvline(x=xc, color='r', label='Reward', linestyle='-', alpha=0.9, ymax=0.40)
    # fig.legend()
    # pyplot_top.set_title('Mouse' + str(mouse_id),fontweight="bold", fontsize=22)
    # pathParts = [x for x in data_dir.split('/') if x]
    # plt.savefig('/home/pankaj/' + str(mouse_id) + '_' + pathParts[4] +'_'+ pathParts[5] + '.png', format='png')
    # plt.close(fig)
    #####################################################################################################

    actualRewardFrames = []
    rewPt = df.loc[df['reward'] ==1]
    # rewPt= df.loc[-df['roi1dff-roi2dff'] > df['rew_threshold']]
    if 'sham_roi_activity' in df.columns:
        actualRewardFrames = df.frame[df['sham_roi_activity'] > reward_threshold]
        expectedRewardFrames = df.frame[df['roi_activity'] > reward_threshold]
    else:
        actualRewardFrames = expectedRewardFrames = rewPt['frame'].values
    actualRewardArray = np.append(actualRewardArray, len(actualRewardFrames))
    expectedRewardArray = np.append(expectedRewardArray, len(expectedRewardFrames))

    actualRewardIx = np.array(rewPt['frame'].index)
    actualRewardIx = actualRewardIx[1:][np.diff(actualRewardIx) > epochSize]

    # plt.plot(df.loc[rewardIx].time, avgROI[rewardIx], 'r*')
    timeArray = np.append(timeArray, df['time'][int(len(df)/2)])

    for xc in actualRewardIx:
        # ax[0].axvline(x=xc, color='r', linestyle='-')
        epoch = np.arange(df.frame[xc]-epochSize,df.frame[xc]+epochSize)
        if not (np.prod(epoch >= 0) and np.prod(epoch < len(avgDffROI1))):
            continue

        pretrialImg = dff_filt[epoch[0]-1,:,:]
        trial = dff_filt[epoch, :, :] - pretrialImg

        brainEpoch = trial.transpose((1,2, 0)).reshape((256, epochSize*2*256), order='F')
        # brainEpoch = dff[epoch, :, :].transpose((1,2, 0)).reshape((256, epochSize*2*256), order='F')
        # brainEpoch = dff_roi[epoch, :, :].transpose((1,2, 0)).reshape((10, epochSize*2*13), order='F')
        trialAvgDffROI1 = trial[:, roi1.y: roi1.y + roi1.height, roi1.x: roi1.x + roi1.width].mean(axis=(1,2))
        trialAvgDffROI2 = trial[:, roi2.y: roi2.y + roi2.height, roi2.x: roi2.x + roi2.width].mean(axis=(1,2))

        avgROI1RewardArray = np.append(avgROI1RewardArray, avgDffROI1[xc])
        avgROI2RewardArray = np.append(avgROI2RewardArray, avgDffROI2[xc])
        avgROI1RewardArrayColor = np.append(avgROI1RewardArrayColor, 'r')
        avgROI2RewardArrayColor = np.append(avgROI2RewardArrayColor, 'g')

        if len(stackedTrialsNN) == 0:
            stackedTrialsNN = trial[np.newaxis,...]
            tiledTrialsNN = brainEpoch
            avgROI1TrialsNN = trialAvgDffROI1
            avgROI2TrialsNN = trialAvgDffROI2
        else:
            stackedTrialsNN = np.concatenate((stackedTrialsNN, trial[np.newaxis,...]))
            tiledTrialsNN = np.vstack((tiledTrialsNN, brainEpoch))
            avgROI1TrialsNN = np.vstack((avgROI1TrialsNN, trialAvgDffROI1))
            avgROI2TrialsNN = np.vstack((avgROI2TrialsNN, trialAvgDffROI2))

    # plt.tight_layout()


def dff_corrected_moving_multiproc(images, dff_history):
    n_proc = mp.cpu_count()
    # mn = images.mean(axis=0)
    mn = moving_average(images, dff_history)
    # this often can't be devided evenly (handle this in the for-loop below)
    chunksize = images.shape[0] // n_proc
    # devide into chunks
    images_chunks = []
    mn_chunks = []
    for i_proc in range(n_proc):
        chunkstart = i_proc * chunksize
        # make sure to include the division remainder for the last process
        chunkend = (i_proc + 1) * chunksize if i_proc < n_proc - 1 else None

        images_chunks.append(images[slice(chunkstart, chunkend)])
        mn_chunks.append(mn[slice(chunkstart, chunkend)])
    assert sum(map(len, images_chunks)) == len(images)  # make sure all data is in the chunks
    assert sum(map(len, mn_chunks)) == len(mn)  # make sure all data is in the chunks
    # distribute work to the worker processes
    result_chunks = Parallel(n_jobs=n_proc, backend='threading')(
        delayed(sub_mean)(imc, mnc) for (imc, mnc) in zip(images_chunks, mn_chunks))
    dff = np.concatenate(result_chunks, axis=0)
    assert len(dff) == len(images)  # make sure we got a result for each frame
    # dff = sub_mean(images, mn)
    dff_chunks = []
    mn_chunks = []
    for i_proc in range(n_proc):
        chunkstart = i_proc * chunksize
        # make sure to include the division remainder for the last process
        chunkend = (i_proc + 1) * chunksize if i_proc < n_proc - 1 else None

        dff_chunks.append(dff[slice(chunkstart, chunkend)])
        mn_chunks.append(mn[slice(chunkstart, chunkend)])
    assert sum(map(len, dff_chunks)) == len(dff)  # make sure all data is in the chunks
    assert sum(map(len, mn_chunks)) == len(mn)  # make sure all data is in the chunks
    # distribute work to the worker processes
    dff_chunks = Parallel(n_jobs=n_proc, backend='threading')(
        delayed(div_mean)(dffc, mnc) for (dffc, mnc) in zip(dff_chunks, mn_chunks))
    dff = np.concatenate(dff_chunks, axis=0)
    assert len(dff) == len(images)  # make sure we got a result for each frame
    # dff = np.divide(dff, mn, out=np.zeros_like(dff), where=mn!=0)
    dff_chunks = []
    for i_proc in range(n_proc):
        chunkstart = i_proc * chunksize
        # make sure to include the division remainder for the last process
        chunkend = (i_proc + 1) * chunksize if i_proc < n_proc - 1 else None
        dff_chunks.append(dff[slice(chunkstart, chunkend)])
    assert sum(map(len, dff_chunks)) == len(dff)  # make sure all data is in the chunks
    result_chunks = Parallel(n_jobs=n_proc, backend='threading')(delayed(dff_correct)(dffc) for dffc in dff_chunks)
    dffCorrected = np.concatenate(result_chunks, axis=0)
    assert len(dff) == len(images)  # make sure we got a result for each frame
    return dffCorrected

def dff_corrected_multiproc(images):
    n_proc = mp.cpu_count()
    mn = images.mean(axis=0)
    # this often can't be devided evenly (handle this in the for-loop below)
    chunksize = images.shape[0] // n_proc
    # devide into chunks
    images_chunks = []
    mn_chunks = []
    for i_proc in range(n_proc):
        chunkstart = i_proc * chunksize
        # make sure to include the division remainder for the last process
        chunkend = (i_proc + 1) * chunksize if i_proc < n_proc - 1 else None

        images_chunks.append(images[slice(chunkstart, chunkend)])
        mn_chunks.append(mn[slice(chunkstart, chunkend)])
    assert sum(map(len, images_chunks)) == len(images)  # make sure all data is in the chunks
    assert sum(map(len, mn_chunks)) == len(mn)  # make sure all data is in the chunks
    # distribute work to the worker processes
    result_chunks = Parallel(n_jobs=n_proc, backend='threading')(
        delayed(sub_mean)(imc, mnc) for (imc, mnc) in zip(images_chunks, mn_chunks))
    dff = np.concatenate(result_chunks, axis=0)
    assert len(dff) == len(images)  # make sure we got a result for each frame
    # dff = sub_mean(images, mn)
    dff_chunks = []
    mn_chunks = []
    for i_proc in range(n_proc):
        chunkstart = i_proc * chunksize
        # make sure to include the division remainder for the last process
        chunkend = (i_proc + 1) * chunksize if i_proc < n_proc - 1 else None

        dff_chunks.append(dff[slice(chunkstart, chunkend)])
        mn_chunks.append(mn[slice(chunkstart, chunkend)])
    assert sum(map(len, dff_chunks)) == len(dff)  # make sure all data is in the chunks
    assert sum(map(len, mn_chunks)) == len(mn)  # make sure all data is in the chunks
    # distribute work to the worker processes
    dff_chunks = Parallel(n_jobs=n_proc, backend='threading')(
        delayed(div_mean)(dffc, mnc) for (dffc, mnc) in zip(dff_chunks, mn_chunks))
    dff = np.concatenate(dff_chunks, axis=0)
    assert len(dff) == len(images)  # make sure we got a result for each frame
    # dff = np.divide(dff, mn, out=np.zeros_like(dff), where=mn!=0)
    dff_chunks = []
    for i_proc in range(n_proc):
        chunkstart = i_proc * chunksize
        # make sure to include the division remainder for the last process
        chunkend = (i_proc + 1) * chunksize if i_proc < n_proc - 1 else None
        dff_chunks.append(dff[slice(chunkstart, chunkend)])
    assert sum(map(len, dff_chunks)) == len(dff)  # make sure all data is in the chunks
    result_chunks = Parallel(n_jobs=n_proc, backend='threading')(delayed(dff_correct)(dffc) for dffc in dff_chunks)
    dffCorrected = np.concatenate(result_chunks, axis=0)
    assert len(dff) == len(images)  # make sure we got a result for each frame
    return dffCorrected

def dff_corrected(images):
    mn = images.mean(axis=0)

    dff = sub_mean(images, mn)
    assert len(dff) == len(images)  # make sure we got a result for each frame
    dff = np.divide(dff, mn, out=np.zeros_like(dff), where=mn!=0)
    assert len(dff) == len(images)  # make sure we got a result for each frame

    # Make the nans, inf and -inf with zero.
    dff[np.where(np.isinf(dff))] = 0
    dff[np.where(np.isneginf(dff))] = 0
    dff = np.nan_to_num(dff)
    framesStd = np.std(dff, axis=0)  # calculate std of dff0 to mask the blue dff0 with green channel
    dff[:, :, :, 0] = apply_mask(dff[:, :, :, 0], framesStd[:, :, 1])  # mask blue dff0 where green dff0 is zero

    dffCorrected = dff_correct(dff)
    assert len(dffCorrected) == len(images)  # make sure we got a result for each frame

    return dffCorrected

# def calculate_dff0(frames):
#     #frames = frames.astype(np.float32)
#     baseline = np.mean(frames, axis=0)
#     # frames = np.divide(np.subtract(frames, baseline), baseline, out=np.zeros_like(frames), where=baseline!=0)
#     frames = np.divide(np.subtract(frames, baseline), baseline)

#     # Make the nans, inf and -inf with zero.
#     frames[np.where(np.isinf(frames))] = 0
#     frames[np.where(np.isneginf(frames))] = 0
#     frames = np.nan_to_num(frames)
#     framesStd = np.std(frames, axis=0)  # calculate std of dff0 to mask the blue dff0 with green channel
#     frames[:, :, :, 0] = apply_mask(frames[:, :, :, 0], framesStd[:, :, 1])  # mask blue dff0 where green dff0 is zero
#     return frames, framesStd

# def calculate_dff0(frames):
#     import numpy as np

#     # Ensure float32 to avoid upcasting
#     if frames.dtype != np.float32:
#         frames = np.asarray(frames, dtype=np.float32)

#     baseline = frames.mean(axis=0, dtype=np.float32)

#     # In-place subtract and divide to avoid temporaries
#     eps = 1e-6
#     np.subtract(frames, baseline, out=frames, casting='unsafe')
#     np.divide(frames, baseline + eps, out=frames, casting='unsafe')

#     # Replace NaNs/infs in-place
#     np.nan_to_num(frames, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

#     # Compute std in float32 to avoid float64 accumulator
#     framesStd = np.std(frames, axis=0, dtype=np.float32)

#     # Apply mask to blue channel only
#     frames[:, :, :, 0] = apply_mask(frames[:, :, :, 0], framesStd[:, :, 1])

#     return frames, framesStd


# def compute_std_over_time_chunked(frames, chunk_size=500):
#     """
#     Compute std over time axis (axis=0) using Welford's algorithm, in chunks.
#     Returns: std array of shape (H, W, C)
#     """
#     import numpy as np

#     n_frames, H, W, C = frames.shape
#     mean = np.zeros((H, W, C), dtype=np.float32)
#     M2 = np.zeros((H, W, C), dtype=np.float32)
#     count = 0

#     for i in range(0, n_frames, chunk_size):
#         end = min(i + chunk_size, n_frames)
#         chunk = frames[i:end]

#         for frame in chunk:
#             count += 1
#             delta = frame - mean
#             mean += delta / count
#             delta2 = frame - mean
#             M2 += delta * delta2

#     std = np.sqrt(M2 / (count - 1))
#     return std

# def compute_mean_chunked(frames, chunk_size=500):
#     n_frames, H, W, C = frames.shape
#     mean_accumulator = np.zeros((H, W, C), dtype=np.float32)
    
#     for i in range(0, n_frames, chunk_size):
#         end = min(i + chunk_size, n_frames)
#         raw_chunk = frames[i:end]
#         chunk = np.asarray(raw_chunk, dtype=np.float32)

#         mean_accumulator += chunk.sum(axis=0)
    
#     mean_result = mean_accumulator / n_frames
#     return mean_result


# ### WORKED BUT RAM USAGE HIGH
# def calculate_dff0(frames, chunk_size=500):
#     import numpy as np

#     n_frames, H, W, C = frames.shape
#     eps = 1e-6
#     # Ensure float32 to avoid upcasting

#     frames = np.asarray(frames, dtype=np.float32)
#     # for i in range(0, n_frames, chunk_size):
#     #     # raw_chunk = frames[i:i+chunk_size]
        
#     #     end = min(i + chunk_size, n_frames)
#     #     raw_chunk = frames[i:end]
#     #     # chunk = raw_chunk.astype(np.float32)
#     #     chunk = np.asarray(raw_chunk, dtype=np.float32)
#     #     frames[i:end] = chunk


#     # 1. Compute global mean baseline (across time)
#     baseline = np.mean(frames, axis=0)
#     # baseline = compute_mean_chunked(frames)
#     # 2. Apply dF/F in chunks
#     for i in range(0, n_frames, chunk_size):
#         # raw_chunk = frames[i:i+chunk_size]
        
#         end = min(i + chunk_size, n_frames)
#         raw_chunk = frames[i:end]
#         # chunk = raw_chunk.astype(np.float32)
#         chunk = np.asarray(raw_chunk, dtype=np.float32)
#         # Subtract and divide baseline in-place
#         np.subtract(chunk, baseline, out=chunk, casting='unsafe')
#         np.divide(chunk, baseline + eps, out=chunk, casting='unsafe')
        
#         # chunk = np.divide(np.subtract(chunk, baseline), baseline)
#         # Replace non-finite values in-place
#         chunk[~np.isfinite(chunk)] = 0
#         np.nan_to_num(chunk, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

#         # # Make the nans, inf and -inf with zero.
#         # chunk[np.where(np.isinf(chunk))] = 0
#         # chunk[np.where(np.isneginf(chunk))] = 0
#         # chunk = np.nan_to_num(chunk)


#         # Assign the processed chunk back
#         frames[i:end] = chunk
        
#     # 3. Compute STD over time (after dF/F)
#     # framesStd = np.std(frames, axis=0)
#     framesStd = compute_std_over_time_chunked(frames)
#     # 4. Apply mask to blue channel using green STD
#     frames[:, :, :, 0] = apply_mask(frames[:, :, :, 0], framesStd[:, :, 1])

#     return frames, framesStd



import numpy as np
from scipy.ndimage import uniform_filter1d

def compute_mean(frames, axis=0, window=None):
    """
    Baseline over 'axis'.
    - If window is None: global mean (cheap and exact).
    - If window is int: centered moving average along time (axis=0) with edge 'nearest'.
      NOTE: when used per-chunk (no halo), edges of each chunk will be biased.
    """
    frames = np.asarray(frames, dtype=np.float32, order='C')

    if window is None or window <= 1:
        return frames.mean(axis=axis, dtype=np.float32)

    if axis != 0:
        raise ValueError("Moving window mean only implemented for axis=0 (time).")

    # Moving mean along time; returns same shape as frames
    # (centered window, edges handled by 'nearest')
    baseline = uniform_filter1d(frames, size=window, axis=0, origin=1, mode='nearest').astype(np.float32, copy=False)
    return baseline


def compute_std_over_time_chunked(frames):
    """
    Std over time for a *chunk* that already holds dF/F values.
    Simple and fast since chunk_size is small (e.g. 500-1500 frames).
    """
    frames = np.asarray(frames, dtype=np.float32, order='C')
    return frames.std(axis=0, dtype=np.float32)


def calculate_dff0(frames, baseline, window=None, eps=1e-6):
    """
    Computes dF/F for a *chunk* (t, H, W, C) in float32 and masks blue using
    green-channel temporal std computed *within the chunk*.

    window:
      - None or <=1  -> global mean baseline over the chunk
      - int (e.g. 450) -> moving mean baseline along time within the chunk (no halo)
    """
    import numpy as np

    frames = np.asarray(frames, dtype=np.float32, order='C')   # work in float32, contiguous
    t, H, W, C = frames.shape

    # 1) Baseline
    # baseline = compute_mean(frames, axis=0, window=window)     # shape: (H,W,C) if global OR (t,H,W,C) if moving

    # 2) dF/F in-place (broadcast works against (H,W,C) or (t,H,W,C))
    # np.subtract(frames, baseline, out=frames, casting='unsafe')
    # np.divide(frames, (baseline + eps), out=frames, casting='unsafe')
    np.subtract(frames, baseline, out=frames, dtype=np.float32)
    np.divide(frames, (baseline + eps), out=frames, dtype=np.float32)

    # 3) Clean non-finite
    frames[~np.isfinite(frames)] = 0.0

    # 4) Std over time for this chunk (used to mask blue with green variability)
    framesStd = compute_std_over_time_chunked(frames)          # (H,W,C), float32

    # 5) Apply mask to blue using green STD (your function)
    #    (assumes: apply_mask(blue_chunk, green_std) -> masked_blue_chunk)
    frames[..., 0] = apply_mask(frames[..., 0], framesStd[..., 1])

    return frames











# import numpy as np
# from scipy.ndimage import percentile_filter
# from scipy.ndimage import uniform_filter1d  # for moving average

# def iter_core_slices(T, chunk_size, half_window):
#     # yields: (write_start, write_end, read_start, read_end, core_slice)
#     for start in range(0, T, chunk_size):
#         read_s = max(0, start - half_window)
#         read_e = min(T, start + chunk_size + half_window)
#         # which part of the padded array corresponds to the "core" we keep?
#         left  = start - read_s
#         right = read_e - (start + chunk_size)
#         if start == 0:
#             core = slice(0, min(chunk_size, T))
#         elif read_e == T:
#             core = slice(left, read_e - (start))
#         else:
#             core = slice(left, -right)
#         write_s = start
#         write_e = min(start + chunk_size, T)
#         yield write_s, write_e, read_s, read_e, core

# def moving_baseline_chunked(frames, window, method="percentile", p=10,
#                             chunk_size=1000, eps=1e-6, ch=None):
#     """
#     frames: array-like with slicing [t0:t1] -> (t, H, W, C) or (t, H, W)
#     window: int (#frames, e.g. 450)
#     method: "percentile" or "mean"
#     ch:     None if frames already single-channel; or channel index (e.g., 0=blue,1=green)
#     returns a generator of (write_slice, baseline_core) for each chunk core.
#     """
#     T = frames.shape[0]
#     half = window // 2

#     for w_s, w_e, r_s, r_e, core in iter_core_slices(T, chunk_size, half):
#         # read padded chunk
#         chunk = frames[r_s:r_e]                      # (pad_t, H, W, C?) or (pad_t,H,W)
#         if ch is not None:
#             chunk = chunk[..., ch]
#         chunk = np.asarray(chunk, dtype=np.float32)  # safe dtype

#         if method == "percentile":
#             # along time only
#             base = percentile_filter(chunk, percentile=p, size=(window, 1, 1), mode='nearest')
#         elif method == "mean":
#             # fast moving average along time only
#             # uniform_filter1d is equivalent to a sliding mean
#             base = uniform_filter1d(chunk, size=window, axis=0, mode='nearest')
#         else:
#             raise ValueError("method must be 'percentile' or 'mean'")

#         yield slice(w_s, w_e), base[core]            # baseline just for the core frames




# def compute_mean_chunked(frames, chunk_size=1000):
#     """Mean over time (axis=0), streaming."""
#     T, H, W, C = frames.shape
#     acc = np.zeros((H, W, C), dtype=np.float64)
#     n = 0
#     for i in range(0, T, chunk_size):
#         j = min(i + chunk_size, T)
#         # cast only the chunk
#         chunk = frames[i:j].astype(np.float32, copy=False)
#         acc += chunk.sum(axis=0, dtype=np.float64)
#         n += (j - i)
#     mean = (acc / n).astype(np.float32)
#     return mean

# def compute_std_over_time_chunked(array_like, chunk_size=1000):
#     """Std over time of an already-normalized array (axis=0), streaming Welford."""
#     T, H, W, C = array_like.shape
#     mean = np.zeros((H, W, C), dtype=np.float32)
#     M2   = np.zeros((H, W, C), dtype=np.float32)
#     count = 0
#     for i in range(0, T, chunk_size):
#         j = min(i + chunk_size, T)
#         chunk = array_like[i:j].astype(np.float32, copy=False)
#         for k in range(chunk.shape[0]):
#             x = chunk[k]
#             count += 1
#             delta  = x - mean
#             mean  += delta / count
#             delta2 = x - mean
#             M2    += (delta * delta2)
#     std = np.sqrt(M2 / max(count - 1, 1))
#     return std

# def calculate_dff0_chunked(frames, chunk_size=1000, memmap_path=None, eps=1e-6):
#     """
#     Chunked ΔF/F:
#       1) baseline = mean(frames, axis=0) computed streaming
#       2) write ΔF/F to a float32 memmap (no 24 GB in RAM)
#       3) framesStd = std over time (streaming) of ΔF/F
#       4) mask blue with green-std
#     Returns (dff_memmap, framesStd).  dff_memmap behaves like a numpy array but lives on disk.
#     """
#     T, H, W, C = frames.shape

#     # 0) prepare memmap output (float32) so we never hold the whole result in RAM
#     if memmap_path is None:
#         memmap_path = os.path.abspath("dff_tmp.float32.memmap")
#     dff = np.memmap(memmap_path, dtype=np.float32, mode='w+', shape=(T, H, W, C))

#     # 1) global mean baseline in chunks (no big cast)
#     baseline = compute_mean_chunked(frames, chunk_size=chunk_size)  # float32 (H, W, C)

#     # 2) ΔF/F per chunk → write to memmap
#     for i in range(0, T, chunk_size):
#         j = min(i + chunk_size, T)
#         raw = frames[i:j]                          # likely uint8 from HDF5
#         chunk = raw.astype(np.float32, copy=False) # cast only the chunk

#         # (chunk - baseline) / (baseline + eps), all in float32
#         np.subtract(chunk, baseline, out=chunk, casting='unsafe')
#         np.divide(chunk, baseline + eps, out=chunk, casting='unsafe')

#         # clean non-finite in-place (no giant masks kept)
#         chunk[~np.isfinite(chunk)] = 0.0

#         # write to memmap
#         dff[i:j] = chunk

#     # 3) std over time from the dff memmap (streaming)
#     framesStd = compute_std_over_time_chunked(dff, chunk_size=chunk_size)  # (H, W, C) float32

#     # 4) apply your mask to blue using green std, streaming so we don't read all at once
#     #    (assumes apply_mask_fn(img_blue, green_std) -> masked_blue)
#     for i in range(0, T, chunk_size):
#         j = min(i + chunk_size, T)
#         blue = dff[i:j, :, :, 0]
#         # apply mask function; should operate elementwise/broadcast with framesStd[...,1]
#         dff[i:j, :, :, 0] = apply_mask(blue, framesStd[:, :, 1])

#     # Ensure writes are flushed; keep the memmap open for caller to use
#     dff.flush()
#     return dff, framesStd

# def calculate_dff0(frames):
#     import numpy as np

#     # 1) Keep everything in float32 (one copy if needed)
#     if frames.dtype != np.float32:
#         frames = np.asarray(frames, dtype=np.float32)  # ensures writable, contiguous float32

#     # 2) Baseline as float32 (np.mean defaults to float64 otherwise)
#     baseline = frames.mean(axis=0, dtype=np.float32)

#     # 3) In-place dF/F to avoid large temporaries
#     eps = 1e-6
#     np.subtract(frames, baseline, out=frames, casting='unsafe')           # frames = frames - baseline
#     np.divide(frames, baseline + eps, out=frames, casting='unsafe')       # frames /= (baseline + eps)

#     # 4) Clean bad values in-place
#     np.nan_to_num(frames, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

#     # 5) Std in float32 (again, avoid float64)
#     framesStd = frames.std(axis=0, dtype=np.float32)

#     # 6) Mask blue using green std (unchanged logic)
#     frames[:, :, :, 0] = apply_mask(frames[:, :, :, 0], framesStd[:, :, 1])

#     # (Optional) Help GC free baseline ASAP
#     del baseline

#     return frames, framesStd


def ReadImageStream(imagePath, ix=None):
    # open the hdf5 file
    # imagePath is path of HDF5 file
    # ix is array of indexes (images) to be read

    h5file = tables.open_file(imagePath, mode='r')
    # h5file.root.raw_images[10][10:200,10:200,0]
    # images = h5file.root.raw_images[:, :, :, 1]
    if ix is None:
        images = h5file.root.raw_images[:, :, :]
    else:
        images = h5file.root.raw_images[ix, :, :]
    h5file.close()
    return images

def get_brain_mask(file_mask, image=None, seeds=None):
    if seeds is None:
        seeds = []
    if image is None:
        image = []
    # mask = np.load('brain/masks/' + str(mouse_id) + 'Mask.npy')
    if os.path.isfile(file_mask):
        mask = plt.imread(file_mask)
    else:
        if len(image):
            print('mask file not found')
            for seedname in seeds:
                image[seeds[seedname]['AP'], seeds[seedname]['ML']] = 255
            # Show the image
            fig = plt.figure(figsize=(10, 10))
            plt.imshow(image, cmap='gray', interpolation='nearest', vmin=0, vmax=80)
            plt.grid(True)
            plt.xticks([])
            plt.yticks([])
            plt.colorbar()
            plt.title("left click: line segment         right click or double click: close region")
            plt.show(block=False)

            # Let user draw first ROI
            left_hem = RoiPoly(color='b', fig=fig)

            # Show the image with the first ROI
            fig = plt.figure(figsize=(10, 10))
            plt.imshow(image, cmap='gray', interpolation='nearest', vmin=0, vmax=80)
            plt.grid(True)
            plt.xticks([])
            plt.yticks([])
            plt.colorbar()
            left_hem.display_roi()
            plt.title('draw second ROI')
            plt.show(block=False)

            # Let user draw second ROI
            right_hem = RoiPoly(color='r', fig=fig)

            # Show the image with both ROIs and their mean values
            plt.imshow(image, cmap='gray', interpolation='nearest', vmin=0, vmax=150)
            plt.colorbar()
            for roi in [left_hem, right_hem]:
                roi.display_roi()
                roi.display_mean(image)
            plt.title('The two ROIs')
            plt.show()

            # Show ROI masks
            mask = np.logical_or(left_hem.get_mask(image), right_hem.get_mask(image))
            plt.title("Mouse Mask")
            plt.imshow(mask)
            plt.xticks([])
            plt.yticks([])
            plt.show(block=True)
            im = PILImage.fromarray(mask)
            im.save(file_mask, "PNG")
        else:
            print('get_brain_mask: Image is empty')
    
    return mask

def cheby1_bandpass(low_limit, high_limit, frame_rate, order=4, rp=0.1):
    nyq = frame_rate*0.5
    low_limit = low_limit/nyq
    high_limit = high_limit/nyq
    Wn=[low_limit, high_limit]

    b, a = signal.cheby1(order, rp, Wn, btype='bandpass', analog=False)
    return b, a

def cheby1_bandpass_filter(data, b, a):
    # y = signal.lfilter(b, a, data, axis=0)
    y = signal.filtfilt(b, a, data, axis=0)
    return y

def cheby1_filter_parallel(FRAMES, low_limit, high_limit, frame_rate, order=5, rp=0.1, n_jobs=-1):
    b, a = cheby1_bandpass(low_limit, high_limit, frame_rate, order, rp)

    # if n_jobs < 0:
    #     n_jobs = max(cpu_count() + 1 + n_jobs, 1)

    # fd = delayed(cheby1_bandpass_filter)
    # ret = Parallel(n_jobs=n_jobs, verbose=0)(
    #     fd(FRAMES[:, s], b, a)
    #     for s in gen_even_slices(FRAMES.shape[1], n_jobs))
    #
    # return np.hstack(ret).reshape(n_frames, height, width)
    return cheby1_bandpass_filter(FRAMES, b, a)

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band', analog=False)
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    # y = signal.lfilter(b, a, data)
    y = signal.filtfilt(b, a, data)
    return y

def animate(iframe, ax, allframes):
    """
    Animation function. Takes the current frame number (to select the potion of
    data to plot) and a line object to update.
    """

    # Not strictly neccessary, just so we know we are stealing these from
    # the global scope
    # global all_data, image

    # We want up-to and _including_ the frame'th element
    # ax.set_array(allframes[iframe])
    ax.imshow(allframes[iframe],cmap=cm.jet, vmin=-0.15, vmax=0.15)
    rect1 = patches.Rectangle((eval(roi_names[0]+'.x'),eval(roi_names[0]+'.y')),eval(roi_names[0]+'.width'),eval(roi_names[0]+'.height'),
                              linewidth=0.5,edgecolor='r',facecolor='none')
    rect2 = patches.Rectangle((eval(roi_names[1]+'.x'),eval(roi_names[1]+'.y')),eval(roi_names[1]+'.width'),eval(roi_names[1]+'.height'),
                              linewidth=0.5,edgecolor='r',facecolor='none')
    ax.add_patch(rect1)
    ax.add_patch(rect2)

    # Create a circle patch
    circ = patches.Circle((10, 10), 10, alpha=0.8, fc=None, visible=False)
    # Add the patch to the Axes
    ax.add_patch(circ)
    if iframe == epochSize+1:
        circ.set_visible(True)
        circ.set_facecolor('green')
    else:
        circ.set_visible(False)
        circ.set_facecolor(None)

    return ax

#@title define a function which returns an image as numpy array from figure
def get_img_from_fig(fig, dpi=180):
    """
        function which returns an image as numpy array from figure
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

# def stack_plot_brain_frames_rois(FRxHxW_stack, rois, fps, vmin, vmax, title='', dorsal_map=None):
#     """
#         Takes stack, plots it in figure, draws ROIs, optional overlay brain map and
#         obtains stack of RGB images from figure
#     """
#     frames = []
#     plt.ion()
#     fig = plt.figure()
#     ax1 = fig.add_subplot(1, 1, 1)
#     divnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
#     im = plt.imshow(FRxHxW_stack[0, :, :], cmap='jet', norm=divnorm, origin='upper')
#     if dorsal_map is not None:
#         [plt.plot(r[0][:, 0]-1, r[0][:, 1]-1, 'w') for r in dorsal_map['edgeOutlineRegionsLR']] # -1 because this is exported from matlab where indexing starts at 1
#     for roi in rois:
#         rect1 = patches.Rectangle((int(roi.x), int(roi.y)), 10, 10, linewidth=1, edgecolor='k', facecolor='none')
#         ax1.add_patch(rect1)
#         ax1.text(int(roi.x) - 10, int(roi.y)-10, roi.name, fontsize=20)
#     plt.xlim([0, FRxHxW_stack.shape[1]]); plt.ylim([FRxHxW_stack.shape[2], 0])
#     ax1.set_axis_off()
#     cbar = plt.colorbar()
#     cbar.ax.tick_params(labelsize=15)
#     for i in np.arange(FRxHxW_stack.shape[0]):
#         # updating the value of x and y
#         im.set_data(FRxHxW_stack[i, :, :])
#         ax1.set_title(
#             f"{title} {(i - FRxHxW_stack.shape[0] // 2)/fps:.2f} sec.",
#             fontsize=20
#             )
#         # re-drawing the figure
#         fig.canvas.draw()
#         # to flush the GUI events
#         fig.canvas.flush_events()
#         # plt.title("{rule} {time:.2f} sec.".format(rule=title, time=(i - FRxHxW_stack.shape[0] // 2)/fps), fontsize=20)

#         # plt.subplots_adjust(0, 0, 1, 1, 0, 0);
#         # plt.show(block=False)
#         frames.append(get_img_from_fig(fig, dpi=66.67))  # dpi 66.67 to match behavior frame along Y
#     plt.close()
#     frames = np.stack(frames)
#     return frames

def stack_plot_brain_frames_rois(FRxHxW_stack, rois, fps, vmin, vmax, title='', dorsal_map=None):
    """
        Takes stack, plots it in figure, draws ROIs, optional overlay brain map and
        obtains stack of RGB images from figure
    """
    frames = []
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    divnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    im = plt.imshow(FRxHxW_stack[0, :, :], cmap='jet', norm=divnorm, origin='upper')
    if dorsal_map is not None:
        [plt.plot(r[0][:, 0]-1, r[0][:, 1]-1, 'w') for r in dorsal_map['edgeOutlineRegionsLR']] # -1 because this is exported from matlab where indexing starts at 1
    for roi in rois:
        rect1 = patches.Rectangle((int(roi.x), int(roi.y)), 10, 10, linewidth=1, edgecolor='k', facecolor='none')
        ax1.add_patch(rect1)
        ax1.text(int(roi.x) - 10, int(roi.y)-10, roi.name, fontsize=20)
    plt.xlim([0, FRxHxW_stack.shape[1]]); plt.ylim([FRxHxW_stack.shape[2], 0])
    ax1.set_axis_off()
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=15)
    for i in np.arange(FRxHxW_stack.shape[0]):
        # updating the value of x and y
        im.set_data(FRxHxW_stack[i, :, :])
        # re-drawing the figure
        fig.canvas.draw()
        # to flush the GUI events
        fig.canvas.flush_events()
        plt.title("{rule} {time:.2f} sec.".format(rule=title, time=(i - FRxHxW_stack.shape[0] // 2)/fps), fontsize=20)
        # plt.subplots_adjust(0, 0, 1, 1, 0, 0);
        # plt.show(block=False)
        frames.append(get_img_from_fig(fig, dpi=66.67))  # dpi 66.67 to match behavior frame along Y
    plt.close()
    frames = np.stack(frames)
    return frames



def ffmpeg_write_video(rgb_stack, video_path, fps=30):
    """
        Takes RGB stack and writes video file at the path specified
    """
    process = (
        ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', framerate=fps, s='{}x{}'.format(rgb_stack.shape[2], rgb_stack.shape[1]))
            .output(video_path, pix_fmt='yuv420p', vf='pad=ceil(iw/2)*2:ceil(ih/2)*2') #vf filte rto fix the issue with height and width being non divisible by 2
            .overwrite_output()
            .run_async(pipe_stdin=True)
    )
    process.stdin.write(
        rgb_stack
            .astype(np.uint8)
            .tobytes()
    )
    process.stdin.close()
    process.wait()

def plotSpectrum(avgDffROI,Fs):
    """
    Plots a Single-Sided Amplitude Spectrum of y(t)
    """
    n = len(avgDffROI) # length of the signal
    k = np.arange(n)
    T = n/fr
    frq = k/T # two sides frequency range
    frq = frq[range(int(n/2))] # one side frequency range

    Y = fft(avgDffROI)/n # fft computing and normalization
    Y = Y[range(int(n/2))]

    plot(frq,abs(Y),'r') # plotting the spectrum
    xlabel('Freq (Hz)')
    ylabel('|Y(freq)|')

def get_dark_frames(green_frames):
    start_index = 0
    end_index = len(green_frames)
    for i in np.arange(0,100):
        if green_frames[i, 100,100] > 10:
            start_index = i
            break
    for i in np.arange(len(green_frames)-100,len(green_frames)):
        if green_frames[i, 100,100] < 10:
            end_index = i-1
            break

    return (start_index, end_index)

def get_dark_frames_gradient_method(behaviour_frames, sigma=15, show_plot=False, spacetime=False):
    if spacetime:
        means = np.mean(behaviour_frames, axis=1)
    else:
        means = np.mean(np.mean(behaviour_frames, axis=1), axis=1)
    grad = np.gradient(means)
    mean = np.mean(grad)
    std = np.std(grad)

    if show_plot:
        plt.figure()
        plt.plot(np.gradient(means))
        plt.show()

    indeces = grad[np.where(np.abs(grad)>mean+std*sigma)]
    start = np.where(grad==indeces[0])
    stop = np.where(grad==indeces[-1])
    assert (len(start)==1)
    assert (len(stop)==1)
    return (start[0][0], stop[0][0])

def stack_movie(dff_stack, play_fps=15,cspace=cm.coolwarm, vmin=-0.1, vmax=0.1):

    def redraw_fn(f, axes):
        if not redraw_fn.initialized:
            redraw_fn.im = axes.imshow(dff_stack[f, :, :], animated=True,cmap=cspace, vmin=vmin, vmax=vmax)
            redraw_fn.initialized = True
        else:
            redraw_fn.im.set_array(dff_stack[f, :, :])


    redraw_fn.initialized = False

    videofig(dff_stack.shape[0], redraw_fn, play_fps=30)


def gsr_mask(frames, mask):
    mask = mask == 1
    mask = np.reshape(mask, mask.shape[0] * mask.shape[1])
    indices = np.where((mask == True))
    indices = np.asarray(indices)
    width = frames.shape[1]
    height = frames.shape[2]
    brain_frames = np.zeros((frames.shape[0], indices.shape[1]))

    frames = np.reshape(frames, (frames.shape[0], frames.shape[1] * frames.shape[2]))
    # brain_frames[:, :] = np.squeeze(frames[:, indices])

    mean_g = np.mean(np.squeeze(frames[:, indices]), axis=1)
    g_plus = np.squeeze(np.linalg.pinv([mean_g]))

    beta_g = np.dot(g_plus, frames)
    global_signal = np.dot(np.asarray([mean_g]).T, [beta_g])
    del mean_g
    frames -= global_signal
    del global_signal
    frames = np.reshape(frames, (frames.shape[0], width, height))
    return frames

def gsr(frames):
    frames[np.isnan(frames)] = 0
    width = frames.shape[1]
    height = frames.shape[2]
    # Reshape into time and space
    frames = np.reshape(frames, (frames.shape[0], width*height))
    mean_g = np.mean(frames, axis=1, dtype=np.float32)
    g_plus = np.squeeze(np.linalg.pinv([mean_g]))
    beta_g = np.dot(g_plus, frames)
    # print('mean_g = '+str(np.shape(mean_g)))
    # print('beta_g = '+str(np.shape(beta_g)))
    global_signal = np.dot(np.asarray([mean_g]).T, [beta_g])
    frames = frames - global_signal
    frames = np.reshape(frames, (frames.shape[0], width, height))
    return frames

# An array/list where bregma[0] is y and bregma[1] is x
# Gives you back a list of seeds, as per matthieu vanni matlab program.
#ppmm is pixels per mm. 256/10mm or so for hyperscanner :/
# Direction is one of 'u', 'd', 'l', 'r' Indicates the direction mouse is facing.
#Seed("V1", -3.2678, 2.5168)
def generate_seeds(bregma, seedMMDict, ppmm, direction=None):
    # All the seeds
    #Seed("OB", 4.6, .86)
    # seeds = [
    #         #Feds seed pixel locations
    #         # Seed("M1", ML=1.03+1, AP=1.365),
    #         # Seed("FL", ML=.16, AP=2.47),
    #         # Seed("HL", ML=-.7, AP=1.85),
    #         # Seed("aBC", ML=-1.36+.575, AP=3.35),
    #         # Seed("pBC", ML=-1.9, AP=3.35),
    #         # Seed("AC", ML=0+1, AP=0.6),
    #         # Seed("RS", ML=-2.8858+1, AP=0.62043),
    #         # Seed("V1", ML=-4.2678+.8, AP=2.5168),
    #         # Seed("mPTA", ML=-2.4962, AP=2.2932),
    #         # Seed("lPTA", ML=-2.4962-0.3, AP=3.35-0.2),
    #         # Seed("Un", ML=-1.5, AP=2.6),
    #
    #         # Allen institute seed pixel locations
    #         Seed("A", ML=2.2932, AP=-2.4962),
    #         # Seed("AC", ML=0.097951, AP=1.8536),
    #         # Seed("AL", ML=3.8271, AP=-3.3393),
    #         Seed("AM", ML=1.6479, AP=-2.696),
    #         # Seed("AU", ML=4.5304, AP=-2.901),
    #         Seed("BC", ML=3.4569, AP=-1.727),
    #         Seed("FL", ML=2.4526, AP=-0.5668),
    #         Seed("HL", ML=1.6942, AP=-1.1457),
    #         # Seed("L", ML=3.7126, AP=-4.2615),
    #         # Seed("LI", ML=4.0586, AP=-4.2293),
    #         Seed("M1", ML=1.8603, AP=0.64181),
    #         Seed("M2", ML=0.87002, AP=1.4205),
    #         Seed('ALM', ML=1.5, AP=2.5),
    #         # Seed("MO", ML=3.4917, AP=0.58712),
    #         # Seed("NO", ML=3.8001, AP=-0.47733),
    #         # Seed("PL", ML=3.5161, AP=-5.2146),
    #         Seed("PM", ML=1.6217, AP=-3.6247),
    #         # Seed("POR", ML=4.2231, AP=-4.755),
    #         Seed("RL", ML=3.1712, AP=-2.849),
    #         Seed("RS", ML=0.62043, AP=-2.8858),
    #         # Seed("S2", ML=4.3977, AP=-1.2027),
    #         # Seed("TEA", ML=4.5657, AP=-4.1622),
    #         Seed("TR", ML=1.8644, AP=-2.0204),
    #         Seed("UN", ML=2.7979, AP=-0.97112),
    #         Seed("V1", ML=2.5168, AP=-4.2678),
    # ]

    seedPxDict = {}
    if direction == 'u' or direction == None:
        for loc in seedMMDict:
            seedPxDict[loc+"_L"] = {"ML": int(bregma.col-ppmm*seedMMDict[loc]["ML"]),
                       "AP": int(bregma.row-ppmm*seedMMDict[loc]["AP"])}
        for loc in seedMMDict:
            seedPxDict[loc+"_R"] = {"ML": int(bregma.col+ppmm*seedMMDict[loc]["ML"]),
                       "AP": int(bregma.row-ppmm*seedMMDict[loc]["AP"])}

    
    elif direction == 'd':
        for loc in seedMMDict:
            seedPxDict[loc+"_L"] = {"ML": int(bregma.col-ppmm*seedMMDict[loc]["ML"]),
                       "AP": int(bregma.row-ppmm*seedMMDict[loc]["AP"])}
        for loc in seedMMDict:
            seedPxDict[loc+"_R"] = {"ML": int(bregma.col+ppmm*seedMMDict[loc]["ML"]),
                       "AP": int(bregma.row-ppmm*seedMMDict[loc]["AP"])}
    elif direction == 'r':
        for loc in seedMMDict:
            seedPxDict[loc+"_L"] = {"ML": int(bregma.col+ppmm*seedMMDict[loc]["ML"]),
                       "AP": int(bregma.row+ppmm*seedMMDict[loc]["AP"])}
        for loc in seedMMDict:
            seedPxDict[loc+"_R"] = {"ML": int(bregma.col-ppmm*seedMMDict[loc]["ML"]),
                       "AP": int(bregma.row+ppmm*seedMMDict[loc]["AP"])}
    else:
        for loc in seedMMDict:
            seedPxDict[loc+"_L"] = {"ML": int(bregma.col-ppmm*seedMMDict[loc]["ML"]),
                       "AP": int(bregma.row-ppmm*seedMMDict[loc]["AP"])}
        for loc in seedMMDict:
            seedPxDict[loc+"_R"] = {"ML": int(bregma.col+ppmm*seedMMDict[loc]["ML"]),
                       "AP": int(bregma.row-ppmm*seedMMDict[loc]["AP"])}
    return seedPxDict

class Position:
    def __init__(self, row, col):
        self.row = row
        self.col = col

class Seed:
    # in mm
    def __init__(self, name, AP=0, ML=0):
        self.name = name
        self.y = AP
        self.x = ML

class ScaledSeed(Seed):
    def __init__(self, name, row, col, bregma):
        self.name = name
        self.row = row
        self.col = col
        self.signal = None
        self.bregma = bregma
        self.corr_map = None


class SessionType(Enum):
    normal_audio_normal_reward = 1
    normal_audio_no_reward = 2
    normal_audio_random_reward = 3
    single_audio_normal_reward = 4
    no_audio_normal_reward = 5
    no_audio_random_reward = 6
    no_audio_no_reward = 7
    continuous_rewards = 8

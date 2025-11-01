from ClosedLoopHelper2ROI import ReadImageStream, get_brain_mask, Position, generate_seeds, GetConfigValues
import ClosedLoopHelper2ROI as clh
import roi_manager
import re
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import imageio
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import ImageGrid
import tables
import seaborn as sns
from sklearn import metrics
import matplotlib
from get_list_data import dic
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# dic = {
#     '505970m3': {'target': (-0.02, 0.08),
#                  'seeds': (-0.04, 0.12),
#                  'maps': (-0.04, 0.04)},
#     '505970m5': {'target': (-0.02, 0.12),
#                  'seeds': (-0.04, 0.14),
#                  'maps': (-0.1, 0.1)},
#     '506554m1': {'target': (-0.08, 0.08),
#                  'seeds': (-0.05, 0.17),
#                  'maps': (-0.1, 0.1)},
#     '506554m3': {'target' : (-0.04, 0.12),
#                  'seeds': (-0.05, 0.26),
#                  'maps': (-0.12, 0.12)}
# }

def make_sess_df(plotting_root):
    sess_df = pd.DataFrame(columns=['mouse_id', 'cal_day', 'day', 'sex', 'roi_type', 'roi_rule', 'trials', 'rewards', 'auc', 'target_var'])
    sess_df = pd.DataFrame({'mouse_id': pd.Series(dtype='str'),
                                'cal_day': pd.Series(dtype='str'),
                                'day': pd.Series(dtype='int'),
                                'sex': pd.Series(dtype='str'),
                                'roi_type': pd.Series(dtype='str'),
                                'roi_rule': pd.Series(dtype='str'),
                                'trials': pd.Series(dtype='int'),
                                'rewards': pd.Series(dtype='int'),
                                'auc': pd.Series(dtype='float'),
                                'target_var': float})

    trials_df = pd.DataFrame(columns=['mouse_id', 'cal_day', 'day', 'sex', 'roi_type', 'roi_rule', 'trial', 'start_ix',
                                    'end_ix', 'tr_duration', 'reward'])

    sess_df_csv = plotting_root + os.sep + 'clnf_sessions_df' + '.csv'
    trials_df_csv = plotting_root + os.sep + 'clnf_trials_df' + '.csv'

    dff_res_all_pkl = plotting_root + os.sep + 'clnf_avg_dff_response_all' + '.pkl'
    df_perf_all_pkl = plotting_root + os.sep + 'clnf_df_perf_all' + '.pkl'

    if os.path.isfile(sess_df_csv):
        sess_df = pd.read_csv(sess_df_csv, dtype = {'mouse_id':str, 'cal_day':str, 'day':int, 'sex':str, 'roi_type':str,
                                                                'roi_rule':str, 'trials':int, 'rewards':int, 'auc':float,
                                                                'target_var':float, 'performance':float})

    if os.path.isfile(trials_df_csv):
        trials_df = pd.read_csv(trials_df_csv, dtype = {'mouse_id':str, 'cal_day':str, 'day':int, 'sex':str,
                                                            'roi_type':str, 'roi_rule':str, 'trial':int, 'start_ix':int,
                                                            'end_ix':int, 'tr_duration':float, 'reward': int})
        
    return sess_df, trials_df, sess_df_csv, trials_df_csv, dff_res_all_pkl, df_perf_all_pkl


def edit_sess_df(sess_df, trials_df, mouse_id, dffCol, df_perf, rec_dir, starts_ix, ends_ix, brain_fps, day, sex, roi_type, roi_name, total_trials, n_rewards, dff_reward_auc):
    if ((sess_df['mouse_id'] == mouse_id) &
        (sess_df['cal_day'] == rec_dir)).any():

        # row already exists, update all values
        sess_df.loc[(sess_df['mouse_id'] == mouse_id) & (sess_df['cal_day'] == rec_dir),
        ['day', 'sex', 'roi_type', 'roi_rule', 'trials', 'rewards', 'auc', 'target_var']] = \
            [day, sex, roi_type, roi_name, total_trials, n_rewards, dff_reward_auc, np.var(df_perf[dffCol].values)]
    else:
        # row does not exist, append it to the dataframe
        sess_df.loc[len(sess_df)] = {
            'mouse_id': mouse_id,
            'cal_day': rec_dir,
            'day': day,
            'sex': sex,
            'roi_type': roi_type,
            'roi_rule': roi_name,
            'trials': total_trials,
            'rewards': n_rewards,
            'auc': dff_reward_auc,
            'target_var': np.var(df_perf[dffCol].values)
        }

    for s, e in zip(starts_ix, ends_ix):
        # check if row exists in dataframe
        if ((trials_df['mouse_id'] == mouse_id) &
            (trials_df['cal_day'] == rec_dir) &
            (trials_df['trial'] == df_perf.iloc[e].trial)).any():

            # row already exists, update all values except the manual_label
            trials_df.loc[(trials_df['mouse_id'] == mouse_id) &
                            (trials_df['cal_day'] == rec_dir) &
                            (trials_df['trial'] == df_perf.iloc[e].trial),
            ['day', 'sex', 'roi_type', 'roi_rule', 'trial', 'start_ix', 'end_ix', 'tr_duration', 'reward']] = \
                [day, sex, roi_type, roi_name, df_perf.iloc[e].trial, s, e, (e - s) / brain_fps, df_perf.iloc[e].reward]
        else:
            # row does not exist, append it to the dataframe
            trials_df.loc[len(trials_df)] = {
                'mouse_id': mouse_id,
                'cal_day': rec_dir,
                'day': day,
                'sex': sex,
                'roi_type': roi_type,
                'roi_rule': roi_name,
                'trial': df_perf.iloc[e].trial,
                'start_ix': s,
                'end_ix': e,
                'tr_duration': (e - s) / brain_fps,
                'reward': df_perf.iloc[e].reward
            }

    return sess_df, trials_df


def readconfigts(sess_dir, rec_dir, roi_type, df, mouse_id, sex):
    cfgDict = clh.GetConfigValues(sess_dir)
    # tr_st, tr_en = clh.get_trials_ix(sess_dir)
    ppmm = float(cfgDict['ppmm'])
    # roi_size in config file is in mm. We convert it to pixel coordinates and round off.
    roi_size = [int(round(x)) for x in np.array([float(i) for i in cfgDict['roi_size'].split(',')]) * ppmm]
    bregma = list(map(int, cfgDict['bregma'].split(', ')))
    cfgDict['seeds_mm'] = json.loads(cfgDict['seeds_mm'])
    # seeds_mm = cfgDict['seeds_mm']
    # total_trials = int(cfgDict['total_trials'])

    br = clh.Position(bregma[0], bregma[1])
    seeds = clh.generate_seeds(br, cfgDict['seeds_mm'] , ppmm, 'u')
        
    # # one green frame for drawing
    # frames = ReadImageStream(os.path.join(sess_dir, 'image_stream.hdf5'), ix=[100])
    # image  = frames[0, :, :, 1]  # green channel

    # make seed markers (optional but useful)
    # br = Position(bregma[0], bregma[1])
    # seeds = generate_seeds(br, seeds_mm, ppmm, 'u')
    rois = []
    if roi_type == '2ROI':
        roi_names = re.split('[-+/%]',cfgDict['roi_operation'])
        roi_op = re.split('([-+/%])',cfgDict['roi_operation'])
        roi_op_str = roi_op[0] + 'dff' + roi_op[1] + roi_op[2] + 'dff'
        roi_name = cfgDict['roi_operation']
        dffCol = roi_op[0] + 'dff' + roi_op[1] + roi_op[2] + 'dff'
        roi1 = roi_names[0]
        roi2 = roi_names[1]
        if 'roi_names' in cfgDict: #exp4 2ROI
            # dffCol      = 'roi1dff-roi2dff'
            for name in roi_names:
                rec = list(map(int, cfgDict[name].split(',')))
                rois.append(roi_manager.Rect(name,
                                            x=rec[0], y= rec[1],w= rec[2],h=rec[3],
                                            color=list(map(int, cfgDict[name+'color'].split(',')))))
        else: #other 2ROI exp
            for name in roi_names:
                seed = seeds[name]
                rois.append(roi_manager.Rect(name,
                                            x= int(seed['ML']-roi_size[0]/2),
                                            y= int(seed['AP']-roi_size[1]/2),
                                            w= roi_size[0],h=roi_size[1],
                                            color=[0,0,255]))
    # target_rule_list.append(roi_name)
    # rois_list.append(rois)
    # brain_fps = int(cfgDict['framerate'])

    # data_file = 'VideoTimestamp.txt'
    # # df = plt_roi_activity(mouse_id, data_dir)
    # # df = pd.read_csv(data_dir + os.sep + data_file, sep='\t', parse_dates=['time'])
    # df = pd.read_csv(sess_dir + os.sep + data_file, sep='\t', parse_dates=['time'])

    
    df['time'] = [a.timestamp() - df.time[0].timestamp() for a in df.time]
    df['mouse_id'] = mouse_id
    df['sex'] = sex
    df['roi_type'] = roi_type
    # df['day'] = day;
    df['rec_dir'] = rec_dir
    df['rule'] = roi_name
    if roi_type == '1ROI':
        df['roi1dff'] = df[roi_name + 'dff']
        df['roi2dff'] = np.nan
    elif roi_type == '2ROI':
        df['roi1dff'] = df[roi_names[0] + 'dff']
        df['roi2dff'] = df[roi_names[1] + 'dff']
        
    # df.rename(columns={dffCol: 'target_dff'}, inplace=True)
    if 'trial' not in df: df['trial'] = 0
    if 'lick' not in df: df['lick'] = 0

    return cfgDict, roi_size, seeds, dffCol, rois, df



def definevars (cfgDict, df):
    roi_name = cfgDict['roi_operation']
    # target_rule_list.append(roi_name)
    # rois_list.append(rois)
    brain_fps = int(cfgDict['framerate'])
    total_trials = int(cfgDict['total_trials'])

    epochSize = int(3 * brain_fps)
    trials_ix = df[df.trial > 0].index
    rests_ix = df[df.trial == 0].index
    starts_ix = np.where(np.diff(df.trial.values) > 0)[0]
    ends_ix = np.where(np.diff(df.trial.values) < 0)[0]
    rew_df = df[df.reward == 1]

    reward_ix = df[df.reward == 1].index
    reward_ix = reward_ix[1:][np.diff(reward_ix) > 2 * epochSize] # make sure indices are far enough from each other
    for ix in reward_ix: # make sure, each epoch is indexing within the range
        epoch = np.arange(ix - epochSize, ix + epochSize)
        if not (np.prod(epoch >= 0) and np.prod(epoch < len(df))):
            reward_ix = np.delete(reward_ix, np.where(reward_ix == ix))
            continue

    fail_ix = df[df.reward == -1].index
    n_rewards = len(df[df['reward'] ==1])
    return roi_name, brain_fps, total_trials, epochSize, trials_ix, rests_ix, starts_ix, ends_ix, rew_df, reward_ix, fail_ix, n_rewards


def createdir (plotting_mouse_dir):
    plt_dir1 = plotting_mouse_dir + os.sep + "average_dff_around_reward" + os.sep
    plt_dir2 = plotting_mouse_dir + os.sep + "tiled_brain_maps" + os.sep
    plt_dir3 = plotting_mouse_dir + os.sep + "tiled_brain_map_all" + os.sep
    plt_dir4 = plotting_mouse_dir + os.sep + "success_rate" + os.sep
    plt_dir5 = plotting_mouse_dir + os.sep + "latency" + os.sep
    plt_dir6 = plotting_mouse_dir + os.sep + "average_rest" + os.sep
    plt_dir7 = plotting_mouse_dir + os.sep + "average_seeds_around_reward" + os.sep
    plt_dir8 = plotting_mouse_dir + os.sep + "example_roi1_vs_roi2" + os.sep
    plt_dir9 = plotting_mouse_dir + os.sep + "example_roi1_minus_roi2" + os.sep
    plt_dir10 = plotting_mouse_dir + os.sep + "gif" + os.sep
    plt_dir11 = plotting_mouse_dir + os.sep + "dff_licks" + os.sep
    plt_dir12 = plotting_mouse_dir + os.sep + "kde_distribution" + os.sep
    plt_dir13 = plotting_mouse_dir + os.sep + "scree_plot" + os.sep
    plt_dir14 = plotting_mouse_dir + os.sep + "seed_corr_map" + os.sep
    plt_dir15 = plotting_mouse_dir + os.sep + "auc_scree" + os.sep
    plt_dir16 = plotting_mouse_dir + os.sep + "seed_auc_results" + os.sep
    plt_dir17 = plotting_mouse_dir + os.sep + "auc_2roi" + os.sep
    os.makedirs(plt_dir1, exist_ok=True)
    os.makedirs(plt_dir2, exist_ok=True)
    os.makedirs(plt_dir3, exist_ok=True)
    os.makedirs(plt_dir4, exist_ok=True)
    os.makedirs(plt_dir5, exist_ok=True)
    os.makedirs(plt_dir6, exist_ok=True)
    os.makedirs(plt_dir7, exist_ok=True)
    os.makedirs(plt_dir8, exist_ok=True)
    os.makedirs(plt_dir9, exist_ok=True)
    os.makedirs(plt_dir10, exist_ok=True)
    os.makedirs(plt_dir11, exist_ok=True)
    os.makedirs(plt_dir12, exist_ok=True)
    os.makedirs(plt_dir13, exist_ok=True)
    os.makedirs(plt_dir14, exist_ok=True)
    os.makedirs(plt_dir15, exist_ok=True)
    os.makedirs(plt_dir16, exist_ok=True)
    os.makedirs(plt_dir17, exist_ok=True)


def plt_dff_licks (df_perf, plotting_mouse_dir, mouse_id, day, rec_dir, dffCol, tr_st, tr_en, rew_df):
    fig = plt.figure(figsize=(12, 6))
    # set_trace()
    avgDff = df_perf[dffCol]
    ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=1, colspan=1)
    ax1.plot(df_perf['time'], df_perf[dffCol], 'tab:green', label='target DFF')
    # ax1.plot(avgDff, 'r', label='offline')
    ax1.plot(df_perf['time'], df_perf['rew_threshold'], c='grey', label='threshold')
    """ SET YLIM """
    # ax1.set_ylim([-0.2, 0.2])
    ax1.set_ylim(dic[mouse_id]['target'])

    plt.title(f'{mouse_id} | {day} - {rec_dir} | avg target activity {dffCol}')
    plt.legend()
    for s, e in zip(df_perf['time'][tr_st], df_perf['time'][tr_en]):
        plt.axvspan(s, e, facecolor='grey', alpha=0.2)
    plt.ylabel(dffCol, fontsize=14)
    plt.yticks(fontsize=14)
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.grid(False)
    ax2 = plt.subplot2grid((4, 1), (1, 0), rowspan=1, colspan=1, sharex=ax1)
    # ax2.plot(df_perf['reward'])
    ax2.scatter(rew_df['time'], rew_df['reward'], s=100, marker='*', color='gold', edgecolor='black', linewidth=0.5)
    # for s, e in zip(df_perf['time'][tr_st], df_perf['time'][tr_en]): plt.axvspan(s, e, facecolor='grey', alpha=0.2)
    plt.ylabel('Reward', fontsize=14)
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax2.grid(False)
    if 'lick' in df_perf:
        ax3 = plt.subplot2grid((4, 1), (2, 0), rowspan=1, colspan=1, sharex=ax1)
        # ax3.plot(df_perf['lick'])
        plt.vlines(x=df_perf[df_perf['lick'] == 1]['time'], color='black', ymin=0, ymax=1, linewidth=0.5)
        # for s, e in zip(df_perf['time'][tr_st], df_perf['time'][tr_en]): plt.axvspan(s, e, facecolor='grey', alpha=0.2)
        plt.ylabel('licks')
        ax3.grid(False)
        plt.setp(ax3.get_xticklabels(), visible=False)
    ax4 = plt.subplot2grid((4, 1), (3, 0), rowspan=1, colspan=1, sharex=ax1)
    # ax2.plot(df_perf['reward'])
    ax4.plot(df_perf['time'], df_perf['freq'], 'tab:orange', label='target DFF')
    for s, e in zip(df_perf['time'][tr_st], df_perf['time'][tr_en]):
        plt.axvspan(s, e, facecolor='grey', alpha=0.2)
    plt.ylabel('Freq', fontsize=14)
    ax4.grid(False)
    plt.xlabel('Time (sec.)', fontsize=14)
    # plt.xlim([20, 250]); #change this to desired range
    # plt.xticks(np.arange(30,250, 40), fontsize=14)
    fig.savefig(plotting_mouse_dir + os.sep + "dff_licks" + os.sep + f'mouse_{mouse_id}_licks_dff.png', dpi=300, bbox_inches='tight')
    plt.close()


def plt_success (df_perf_mouse, mouse_id, plotting_mouse_dir):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Subset data for this mouse
    # mouse_data = df_perf_mouse.sort_values('day')

    if mouse_id == "505970m3":
        color = 'red'
        group = 'Q175'
    else:
        color = 'green'
        group = 'WT'
    # Plot raw rewards (no smoothing)
    ax.plot(df_perf_mouse['day'], df_perf_mouse['success_rate'], marker='o', color=color)
    ax.set_ylim(0,1)
    ax.set_title(f'mouse {mouse_id} ({group})', fontsize=18)
    ax.set_xlabel('Days', fontweight='bold', fontsize=16)
    ax.set_ylabel('Success Rate', fontweight='bold', fontsize=16)
    ax.set_xticks(df_perf_mouse['day'])
    ax.tick_params(axis='both', which='major', labelsize=14)
    sns.despine(offset=10, trim=True)
    fig.savefig(plotting_mouse_dir + os.sep + "success_rate" + os.sep + f'mouse_{mouse_id}_success_rate.png', dpi=300, bbox_inches='tight')
    plt.close()



def plt_avg_dff_reward (df, reward_ix, epochSize, dffCol, brain_fps, rec_dir, mouse_id, day, plotting_mouse_dir):
    roi1Trial = []
    
    for xc in reward_ix:
        # ax[0].axvline(x=xc, color='r', linestyle='-')
        epoch = np.arange(df.frame[xc] - epochSize, df.frame[xc] + epochSize)
        if not (np.prod(epoch >= 0) and np.prod(epoch < len(df))):
            continue
        trial = df.iloc[epoch]
        if len(roi1Trial) == 0:
            roi1Trial = trial[dffCol].values
        else:
            roi1Trial = np.vstack((roi1Trial, trial[dffCol].values))
    t = np.arange(-epochSize, epochSize) / brain_fps
    if len(reward_ix) > 3:
        fig = plt.figure()
        roi1Trial = np.atleast_2d(roi1Trial)

        # plt.plot(t, np.mean(roi1Trial, axis=0))

        mean = np.mean(roi1Trial, axis=0)
        sd  = np.std(roi1Trial, axis=0, ddof=1)
        plt.plot(t, mean, label = dffCol.replace("dff",""))
        plt.fill_between(t, mean - sd, mean + sd, alpha=0.20, linewidth=0)

        plt.ylim(dic[mouse_id]['target'])
        plt.axvline(x=0, color='red', label='reward', linewidth=0.5, alpha=0.4)
        plt.axvline(x=-1, color='grey', label='threshold cross', linewidth=0.5, alpha=0.4)
        # plt.title(mouse_id + ' dir ' + rec_dir + ' day ' + str(day) + ' avg target activity ' + dffCol)
        plt.title(f'{mouse_id} | {day} - {rec_dir} | avg target activity {dffCol}')
        # plt.show()
        fig.savefig(plotting_mouse_dir + os.sep + "average_dff_around_reward" + os.sep + f'{day}_{mouse_id}_{rec_dir}_avg_dff_around_reward.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print("No well-separated rewards in this session; skipping reward-centered outputs.")



def analyzebrain (image_path, processing_session_dir):
    chunk_size = 1000
    image_hdf5_file = tables.open_file(image_path, mode='r')
    images = image_hdf5_file.root.raw_images             # shape (T,H,W,3), dtype uint8
    images_b = image_hdf5_file.root.raw_images[:,:,:,:]
    T, H, W, C = images.shape
    # --- output (final corrected+filtered+masked) ---
    filters = tables.Filters(complevel=5, complib='zlib', shuffle=True)  # keep zlib for portability
    out_path = processing_session_dir + os.sep + 'dff_preproc.hdf5'
    h5out = tables.open_file(out_path, mode='w')
    OUT = h5out.create_carray('/', 'dff_preproc', tables.Float32Atom(),
                            shape=(T, H, W),
                            filters=filters,
                            chunkshape=(min(chunk_size, T), H, W))

    # baseline = np.mean(images_b, axis=0, dtype=np.dtype('float32'))
    accum = np.zeros((H, W, C), dtype=np.float32)
    for i in range(0, T, chunk_size):
        end = min(i + chunk_size, T)
        accum += np.asarray(images_b[i:end], dtype=np.float32).sum(axis=0)
    baseline = accum / T

    del images_b

    for i in range(0, T, chunk_size):

        end = min(i + chunk_size, T)

        # 1) load a chunk and cast to float32
        chunk = np.asarray(images[i:end], dtype=np.float32)        # (t,H,W,3)

        # 2) dF/F on this chunk (choose baseline type: None=global per-chunk; int=windowed per-chunk)
        dff_chunk = clh.calculate_dff0(chunk, baseline)  # or window=450 for moving mean per chunk

        # 3) corrected (G - B)
        corrected = dff_chunk[..., 1] - dff_chunk[..., 0]          # (t,H,W)

        # 4) filter (spatial-only to avoid chunk seams in time)
        # corrected = gaussian_filter(corrected, sigma=(0, 3, 3))    # no temporal blur

        # 5) apply mask (expects mask shape (H,W), values 0/1; avoid NaNs)
        # corrected *= mask

        # 6) write final
        OUT[i:end] = corrected
    del chunk
    del dff_chunk
    dff_filt = OUT.read()
    del OUT
    # framesStd = clh.compute_std_over_time_chunked(OUT)
    # OUT[..., 0] = clh.apply_mask(OUT[..., 0], framesStd[..., 1])
    # OUT = OUT[..., 1] - OUT[..., 0]
    h5out.close()
    image_hdf5_file.close()

    return dff_filt, baseline

def analyzebrain_no_correction (image_path, processing_session_dir):
    chunk_size = 1000
    image_hdf5_file = tables.open_file(image_path, mode='r')
    images = image_hdf5_file.root.raw_images             # shape (T,H,W,3), dtype uint8
    images_b = image_hdf5_file.root.raw_images[:,:,:]
    T, H, W = images.shape
    # --- output (final corrected+filtered+masked) ---
    filters = tables.Filters(complevel=5, complib='zlib', shuffle=True)  # keep zlib for portability
    out_path = processing_session_dir + os.sep + 'dff_preproc_no_correction.hdf5'
    h5out = tables.open_file(out_path, mode='w')
    OUT = h5out.create_carray('/', 'dff_preproc_no_correction', tables.Float32Atom(),
                            shape=(T, H, W),
                            filters=filters,
                            chunkshape=(min(chunk_size, T), H, W))

    # baseline = np.mean(images_b, axis=0, dtype=np.dtype('float32'))
    # del images_b
    accum = np.zeros((H, W), dtype=np.float32)
    for i in range(0, T, chunk_size):
        end = min(i + chunk_size, T)
        accum += np.asarray(images_b[i:end], dtype=np.float32).sum(axis=0)
    baseline = accum / T

    del images_b
    eps = 1e-6
    for i in range(0, T, chunk_size):

        end = min(i + chunk_size, T)

        # 1) load a chunk and cast to float32
        chunk = np.asarray(images[i:end, :, :], dtype=np.float32, order='C')        # (t,H,W,3)

        # 2) dF/F on this chunk (choose baseline type: None=global per-chunk; int=windowed per-chunk)
        #dff_chunk = clh.calculate_dff0(chunk, baseline)  # or window=450 for moving mean per chunk

    #frames = np.asarray(frames, dtype=np.float32, order='C')   # work in float32, contiguous
        # t, H, W = chunk.shape

        np.subtract(chunk, baseline, out=chunk, dtype=np.float32)
        np.divide(chunk, (baseline + eps), out=chunk, dtype=np.float32)

        # 3) Clean non-finite
        chunk[~np.isfinite(chunk)] = 0.0
        # 3) corrected (G - B)
        # corrected = dff_chunk[..., 1] - dff_chunk[..., 0]          # (t,H,W)

        # 4) filter (spatial-only to avoid chunk seams in time)
        # corrected = gaussian_filter(corrected, sigma=(0, 3, 3))    # no temporal blur

        # 5) apply mask (expects mask shape (H,W), values 0/1; avoid NaNs)
        # corrected *= mask

        OUT[i:end] = chunk
    
    del chunk
    dff_filt = OUT.read()
    del OUT
    h5out.close()
    image_hdf5_file.close()

    return dff_filt, baseline



def plt_gif_map (OUT, reward_ix, epochSize, plotting_mouse_dir, mouse_id, rec_dir, day, brain_fps, rois, roi_name):
    if len(reward_ix) > 3:
        reward_stack = [OUT[ix - epochSize:ix + epochSize, :, :] for ix in reward_ix]
        reward_stack_avg = np.mean(np.stack(reward_stack, axis=0), axis=0)
        # ... (vmin/vmax, frames, gif, tiled maps, etc.)
                    # reward_stack_avg = np.stack([dff_filt[ix - epochSize:ix + epochSize, :, :] for ix in reward_ix]).mean(axis=0)
        # vmin = np.min([np.quantile(np.nan_to_num(reward_stack_avg), 0.2), -0.009])
        # vmax = np.quantile(np.nan_to_num(reward_stack_avg), 0.95)
        vmin = dic[mouse_id]['maps'][0]
        vmax = dic[mouse_id]['maps'][1]
        # frame_duration = (1/brain_fps)*2000/1000
        frames = clh.stack_plot_brain_frames_rois(reward_stack_avg, rois, brain_fps, vmin, vmax, title=roi_name)
        # clh.ffmpeg_write_video(frames, outdir + mouse_id + '_' + list_rec_dir[0] + '_rew_avg.mp4')
        imageio.mimwrite(plotting_mouse_dir + os.sep + "gif" + os.sep + f'{day}_{mouse_id}_{rec_dir}_rew_avg.gif', frames, format='GIF', duration=(1/brain_fps)*1000, loop=0)
        # imageio.mimwrite(processing_session_dir + '_rew_avg.gif', frames, format='GIF', duration=1/brain_fps, loop=0)
    else:
        print("No well-separated rewards in this session; skipping reward-centered outputs.")


def plt_avg_rew_map (OUT, reward_ix, epochSize, plotting_mouse_dir, mouse_id, rec_dir, day, brain_fps, rois, avg_reward_dff_timebinned_daily):
    if len(reward_ix) > 3:                
        binlength = 5
        reward_stack = [OUT[ix - epochSize:ix + epochSize, :, :] for ix in reward_ix]
        reward_stack_avg = np.mean(np.stack(reward_stack, axis=0), axis=0)
        end_ignore = reward_stack_avg.shape[0] % binlength
        if end_ignore:
            reward_stack_avg = reward_stack_avg[:-end_ignore]
        avg_dff_timebinned = reward_stack_avg.reshape(reward_stack_avg.shape[0] // binlength, binlength,
                                                        reward_stack_avg.shape[1], reward_stack_avg.shape[2])
        avg_dff_timebinned = avg_dff_timebinned.mean(axis=1)
        avg_reward_dff_timebinned_daily.append(avg_dff_timebinned)
        t = np.round(np.arange(-epochSize, epochSize, binlength) / brain_fps, 1)
        # avg_dff_tiled = avg_dff_timebinned.transpose((1,2,0)).reshape(avg_dff_timebinned.shape[1],avg_dff_timebinned.shape[0]*avg_dff_timebinned.shape[2], order='F')

        # vmin = np.min([np.quantile(np.nan_to_num(avg_dff_timebinned), 0.2), -0.001])
        # vmax = np.quantile(np.nan_to_num(avg_dff_timebinned), 0.95)
        vmin = dic[mouse_id]['maps'][0]
        vmax = dic[mouse_id]['maps'][1]
        # divnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0., vmax=vmax)
        fig = plt.figure(figsize=(16., 4.))
        plt.title(f'{mouse_id} | day {day} - {rec_dir}')
        plt.axis('off')
        grid = ImageGrid(fig, 111, nrows_ncols=(1, avg_dff_timebinned.shape[0]), axes_pad=0.0,
                            cbar_mode='single',
                            cbar_size="5%", cbar_pad="10%", )
        for ix, ax in enumerate(grid):
            ret = ax.imshow(avg_dff_timebinned[ix, :, :], vmin=vmin, vmax=vmax, cmap='jet',
                            rasterized=True)
            # [ax.plot(r[0][:, 0] - 1, r[0][:, 1] - 1, 'w', linewidth=0.5) for r in dorsal_map['edgeOutlineRegionsLR']]  # -1 because this is exported from matlab where indexing starts at 1
            for roi in rois:
                rect1 = patches.Rectangle((int(roi.x), int(roi.y)), roi.w, roi.h, linewidth=0.2, edgecolor='k', facecolor='none')
                ax.add_patch(rect1)
            ax.set_title(t[ix], fontsize=8)
            ax.axis('off')
        grid.cbar_axes[0].colorbar(ret)
        # plt.show(block=False)
        fig.savefig(plotting_mouse_dir + os.sep + "tiled_brain_maps" + os.sep + f'{day}_{mouse_id}_{rec_dir}_tiledbrainmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print("No well-separated rewards in this session; skipping reward-centered outputs.")
    
    return avg_reward_dff_timebinned_daily



def plt_avg_seed_rew (OUT, reward_ix, epochSize, plotting_mouse_dir, mouse_id, rec_dir, day, brain_fps, cfgDict, seeds, dff_res_all, roi_name):
    t = np.arange(-epochSize, epochSize) / brain_fps

    for ixx, se in enumerate(cfgDict['seeds_mm']):
        plt_seeds = plotting_mouse_dir + os.sep + "average_seeds_around_reward" + os.sep + f'{se}' + os.sep
        os.makedirs(plt_seeds, exist_ok=True)
        fig = plt.figure()
        reward_avg_R = []
        reward_avg_L = []
        if len(reward_ix) > 3:
            reward_stack_L = np.stack([OUT[ix - epochSize:ix + epochSize,
                                        int(seeds[se + '_L']['AP']), int(seeds[se + '_L']['ML'])] for ix in
                                        reward_ix])
            reward_stack_R = np.stack([OUT[ix - epochSize:ix + epochSize,
                                        int(seeds[se + '_R']['AP']), int(seeds[se + '_R']['ML'])] for ix in
                                        reward_ix])
            reward_avg_L = reward_stack_L.mean(axis=0)
            reward_avg_R = reward_stack_R.mean(axis=0)

            reward_sd_L  = reward_stack_L.std(axis=0, ddof=1)
            reward_sd_R  = reward_stack_R.std(axis=0, ddof=1)

            plt.fill_between(t, reward_avg_L - reward_sd_L, reward_avg_L + reward_sd_L, alpha=0.20, linewidth=0)
            plt.fill_between(t, reward_avg_R - reward_sd_R, reward_avg_R + reward_sd_R, alpha=0.20, linewidth=0)



            dff_res_all.append({
                'mouse_id': mouse_id,
                'rec_dir': rec_dir,
                'day': day,
                'roi_rule': roi_name,
                'seedname': se,
                'reward_stack_l': reward_stack_L,
                'reward_stack_r': reward_stack_R})
            # -----------------------
            # plt.plot(reward_stack.T, color='g', linestyle='dotted', linewidth=0.9, alpha=0.4)
            plt.plot(t, reward_avg_L, color='g', linestyle='dashed', label=se + '_L reward')
            plt.plot(t, reward_avg_R, color='g', label=se + '_R reward')
            plt.ylim(dic[mouse_id]['seeds'])
            plt.axvline(x=0, color='red', label='reward', linewidth=0.5, alpha=0.4)
            plt.axvline(x=-1, color='grey', label='threshold cross', linewidth=0.5, alpha=0.4)
            plt.title(f'DFF responses: {mouse_id} | {se} | {day} - {rec_dir}')
            plt.legend()
            # plt.show(block=False)
            fig.savefig(plt_seeds + f'{day}_{mouse_id}_{rec_dir}_{se}_average_dff_reward.png', dpi=300, bbox_inches='tight')
            plt.close()

    return dff_res_all

def plt_avg_rew_map_all (epochSize, plotting_mouse_dir, mouse_id, brain_fps, avg_reward_dff_timebinned_daily):
    vmin = dic[mouse_id]['maps'][0]
    vmax = dic[mouse_id]['maps'][1]
    binlength = 5
    t = np.round(np.arange(-epochSize, epochSize, binlength) / brain_fps, 1)
    # divnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0., vmax=vmax)
    fig = plt.figure(figsize=(16., 18.))
    plt.title(mouse_id + ' daily reward avg.')
    plt.axis('off')
    grid = ImageGrid(fig, 111, nrows_ncols=(
    avg_reward_dff_timebinned_daily.shape[0], avg_reward_dff_timebinned_daily.shape[1]),
                    axes_pad=0.0, cbar_mode='single',
                    cbar_size="5%", cbar_pad="10%", )
    for ix, ax in enumerate(grid):
        ret = ax.imshow(np.vstack(avg_reward_dff_timebinned_daily)[ix, :, :], vmin=vmin, vmax=vmax, cmap='jet',
                        rasterized=True, origin='upper')
        # [ax.plot(r[0][:, 0] - 1, r[0][:, 1] - 1, 'w', linewidth=0.5) for r in dorsal_map['edgeOutlineRegionsLR']]  # -1 because this is exported from matlab where indexing starts at 1
        if ix < len(t):
            ax.set_title(t[ix], fontsize=9)
        ax.axis('off')
        grid.cbar_axes[0].colorbar(ret)
        # plt.show()
    fig.savefig(plotting_mouse_dir + os.sep + "tiled_brain_map_all" + os.sep + f'{mouse_id}_brain_map_rew_centered_alldays.png', dpi=300, bbox_inches='tight')
    plt.close()
    np.save(plotting_mouse_dir + os.sep + "tiled_brain_map_all" + os.sep + f'{mouse_id}_avg_reward_dff_timebinned_daily.npy', avg_reward_dff_timebinned_daily)
    return avg_reward_dff_timebinned_daily



def corr2_coeff(A, B):
    """
    Row-wise Pearson correlation between rows of A and rows of B.
    A: (n_vars_A, T), B: (n_vars_B, T)
    returns: (n_vars_A, n_vars_B)
    """
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    ssA = (A_mA ** 2).sum(1)
    ssB = (B_mB ** 2).sum(1)

    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None], ssB[None]))


def seedauc(se, seeds, dff_filt, roi_size, df):
    # average seed ROI from raw dff_filt (no GSR)
    ap0 = int(seeds[se]['AP'] - roi_size[0] / 2)
    ap1 = int(seeds[se]['AP'] + roi_size[0] / 2)
    ml0 = int(seeds[se]['ML'] - roi_size[1] / 2)  # <- width on ML
    ml1 = int(seeds[se]['ML'] + roi_size[1] / 2)

    avgDffSeed = dff_filt[:, ap0:ap1, ml0:ml1].mean(axis=(1, 2))

    dff_spont_thr, dff_steps, dff_reward_steps = clh.get_spont_reward_threshold_dff(df, avgDffSeed)
    auc = metrics.auc(dff_steps, dff_reward_steps)
    return dff_steps, dff_reward_steps, auc


def seedcorr(se, seeds, dff_filt, dff_filt1, roi_size):
    # seed–pixel correlation map from raw dff_filt (no GSR)
    ap0 = int(seeds[se]['AP'] - roi_size[0] / 2)
    ap1 = int(seeds[se]['AP'] + roi_size[0] / 2)
    ml0 = int(seeds[se]['ML'] - roi_size[1] / 2)  # <- width on ML
    ml1 = int(seeds[se]['ML'] + roi_size[1] / 2)

    avgDffSeed = dff_filt[:, ap0:ap1, ml0:ml1].mean(axis=(1, 2))  # (T,)
    # flatten movie to (pixels, T)
    # dff_filt1 = dff_filt.reshape(dff_filt.shape[0], dff_filt.shape[1] * dff_filt.shape[2])
    seedCorr = corr2_coeff(dff_filt1.T, avgDffSeed[np.newaxis, :]).ravel()
    return seedCorr


def plt_seedcorrmap(seedcorrmap_stack, seedcorrmap_max, target_rule_list, rois_list, roi_size, plotting_mouse_dir, mouse_id, rec_dir, day):
    images = []
    for i, (rule, rois) in enumerate(zip(target_rule_list, rois_list)):
        fig = plt.figure()
        fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 5]})
        axs[0].imshow(seedcorrmap_max[i], cmap='jet', vmin=0, vmax=1);
        axs[0].axis('off');
        for roi in rois:
            rect1 = patches.Rectangle((int(roi.x), int(roi.y)), roi_size[0], roi_size[1], linewidth=0.2, edgecolor='k', facecolor='none')
            axs[0].add_patch(rect1)
            axs[0].text(int(roi.x) - 150, int(roi.y), roi.name, fontsize=8)
            # axs[0].annotate(roi.name, xy=(int(roi.x), int(roi.y)), xytext=(int(roi.x)-10, int(roi.y)), \
            #     arrowprops=dict(facecolor='black', shrink=0.05), color='green', fontsize=8)

        im = axs[1].imshow(seedcorrmap_stack[:, :, i], cmap='jet', vmin=0, vmax=1);
        plt.axis('off');
        fig.colorbar(im, ax=axs[1], shrink=0.4);
        plt.title('Day: ' + str(i + 1) + '\nTarget rule: ' + rule)
        # plt.subplots_adjust(0,0,1,1,0,0);

        images.append(clh.get_img_from_fig(fig))
        plt.close()
    imageio.mimwrite(plotting_mouse_dir + os.sep + "seed_corr_map" + os.sep + f'{day}_{mouse_id}_{rec_dir}_seedcorrmap_montage.gif', images, format='GIF', duration=1, loop=0)
    np.save(plotting_mouse_dir + os.sep + "seed_corr_map" + os.sep + f'{mouse_id}_seedcorrmap_stack.npy', seedcorrmap_stack)
    np.save(plotting_mouse_dir + os.sep + "seed_corr_map" + os.sep + f'{mouse_id}_seedcorrmap_max.npy', seedcorrmap_max)
    print('seedcorrmap_stack.gif')

    return seedcorrmap_stack, seedcorrmap_max


def plt_kde_distribution (df_perf_mouse, plotting_mouse_dir, mouse_id, rec_dir, dffCol):
    # plot KDE distribution of DFF all days (.loc[(df_perf_mouse['mouse_id'] == mouse_id) & (df_perf_mouse['exp_label'] == exp_label)]))
    fig = plt.figure(figsize=(6, 3))
    sns.kdeplot(data=df_perf_mouse.loc[(df_perf_mouse['mouse_id'] == mouse_id)], x='target_dff',
                hue='day',
                log_scale=(False, False), shade=False)
    sns.despine(offset=10, trim=True)
    plt.title(rec_dir + ' DFF distribution whole session' + ' ' + dffCol)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('DFF', fontweight='bold', fontsize=14)
    plt.ylabel('density', fontweight='bold', fontsize=14)
    plt.tight_layout(pad=2)
    fig.savefig(plotting_mouse_dir + os.sep + "kde_distribution" + os.sep + f'{mouse_id}_kde.png', dpi=300, bbox_inches='tight')
    plt.close()


def plt_scree(dff_steps, plotting_mouse_dir, mouse_id, dff_reward_steps_roi, auc_arr):
    fig_auc, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    for drsr in dff_reward_steps_roi:
        ax1.plot(dff_steps, drsr)
    ax1.set_title('ROI DFFo scree-plot')
    ax1.set_xlabel('DFF threshold')
    ax1.set_ylabel('Calculated rewards')
    ax2.plot(auc_arr)
    ax2.set_xlabel('Days')
    ax2.set_ylabel('AUC')
    fig_auc.savefig(plotting_mouse_dir + os.sep + "scree_plot" + os.sep + f'{mouse_id}_scree.png', dpi=300, bbox_inches='tight')
    plt.close()




def plt_seed_auc_scree (plotting_mouse_dir, mouse_id, seeds, seedauc_results_arr):
    for ix, se in enumerate(seeds):
        fig_auc, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        for d in np.arange(seedauc_results_arr.shape[2]):
            ax1.plot(seedauc_results_arr[ix,0,d], seedauc_results_arr[ix,1,d])
        ax1.set_title(se + ' DFFo scree-plot')
        ax1.set_xlabel('DFF threshold')
        ax1.set_ylabel('Calculated rewards')
        ax2.plot(seedauc_results_arr[ix,2,:])
        ax2.set_xlabel('Days')
        ax2.set_ylabel('AUC')
        ax2.set_title(se)
        fig_auc.savefig(plotting_mouse_dir + os.sep + "auc_scree" + os.sep + f'{mouse_id}_{se}_scree.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    np.save(plotting_mouse_dir + os.sep + 'seed_auc_results' + os.sep + f'{mouse_id}_{se}_seedauc_results.npy', seedauc_results_arr)
    return seedauc_results_arr



def plt_auc_2roi(auc_2roi_dict, plotting_mouse_dir, mouse_id, rec_dir, day, dffCol):
    auc_2roi_pivot = auc_2roi_dict.pivot(index='r1', columns='r2', values='auc_2roi')
    min = np.min([np.quantile(auc_2roi_pivot, 0.20), -0.001])
    max = np.quantile(auc_2roi_pivot, 0.80)
    divnorm = matplotlib.colors.TwoSlopeNorm(vmin=min, vcenter=0., vmax=max)
    fig = plt.figure()
    sns.heatmap(auc_2roi_pivot, cmap="Greys", norm=divnorm, xticklabels=True, yticklabels=True, rasterized=True)
    plt.title('AUC 2ROI' + ' ' + dffCol)
    plt.tight_layout()
    fig.savefig(plotting_mouse_dir + os.sep + "auc_2roi" + os.sep + f'{day}_{mouse_id}_{rec_dir}_auc_2roi.png', dpi=300, bbox_inches='tight')
    plt.close()
    np.save(plotting_mouse_dir + os.sep + "auc_2roi" + os.sep + f'{day}_{mouse_id}_{rec_dir}_auc_2roi.npy', auc_2roi_pivot)
    return auc_2roi_pivot








def fit_pca_image(X):
    if X.ndim == 3:
        T, H, W = X.shape
        X = X.reshape(T, H * W)
    else:
        raise ValueError("X must be (T, H, W)")

    Xz = X - X.mean(0, keepdims=True)

    # Fit PCA using fast randomized solver by default
    pca = PCA(svd_solver="randomized")
    scores = pca.fit_transform(Xz)  # (T, n_components=F)
    return pca, Xz, scores

def plot_elbow(pca):
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    ks = np.arange(1, len(evr) + 1)

    # Scree
    plt.figure()
    plt.plot(ks, evr, marker="o")
    plt.xlabel("Component")
    plt.ylabel("Explained variance ratio")
    plt.title("Scree (elbow) plot")
    plt.grid(True)
    plt.show()

    # Cumulative
    plt.figure()
    plt.plot(ks, cum, marker="o")
    plt.axhline(0.90, linestyle="--")
    plt.axhline(0.95, linestyle="--")
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title("Cumulative explained variance")
    plt.grid(True)
    plt.show()

    k90 = int(np.argmax(cum >= 0.90) + 1)
    k95 = int(np.argmax(cum >= 0.95) + 1)
    print(f"Components for ≥90% variance: {k90} | for ≥95%: {k95}")


    # if not shape_info["is_image"]:
    #     raise ValueError("plot_component_map is only for (T, H, W) data.")
    # H, W = shape_info["H"], shape_info["W"]
def spatial_map(pca, comp_idx, H, W):
    comp = pca.components_[comp_idx].reshape(H, W)

    plt.figure()
    plt.imshow(comp, vmin=np.percentile(comp, 1), vmax=np.percentile(comp, 99), cmap='coolwarm')
    plt.colorbar()
    plt.title(f"PCA component #{comp_idx} (spatial map)")
    plt.axis("off")
    plt.show()


def plot_timecourse(pca, comp_idx, scores):
    plt.figure()
    plt.plot(scores[:, comp_idx])
    plt.xlabel("Time (samples)")
    plt.ylabel("Score")
    plt.title(f"PCA component #{comp_idx} time course")
    plt.grid(True)
    plt.show()

def reconstruct_image(pca, Xz, drop_indices):
    scores = pca.transform(Xz)                 # (T, n_comp)
    if drop_indices:
        scores[:, drop_indices] = 0.0          # zero bad components
    Xz_clean = pca.inverse_transform(scores)   # back to standardized/centered space
    # If we only mean-centered, add the original mean back:
    X_clean = Xz_clean
    return X_clean

def to_original_shape(X2d, shape_info):
    """
    Inverse of to_2d for reconstruction.
    """
    if not shape_info["is_image"]:
        return X2d
    T, H, W = shape_info["T"], shape_info["H"], shape_info["W"]
    return X2d.reshape(T, H, W)






# ----------------------------
# PCA core
# ----------------------------
# ----------------------------
# Helpers: shape handling
# ----------------------------
def to_2d(X):
    """
    Accepts either (T, F) or (T, H, W). Returns (T, F) and a shape helper.
    """
    if X.ndim == 2:
        T, F = X.shape
        return X, {"is_image": False, "T": T, "H": None, "W": None, "F": F}
    elif X.ndim == 3:
        T, H, W = X.shape
        return X.reshape(T, H * W), {"is_image": True, "T": T, "H": H, "W": W, "F": H * W}
    else:
        raise ValueError("X must be (T, F) or (T, H, W)")




def fit_pca(X, svd_solver="randomized"):
    """
    Fit PCA on X (T, F). Returns (pca, scaler, Xz, scores).
    """
    scaler = None
    # if standardize:
    #     scaler = StandardScaler(with_mean=True, with_std=True)
    #     Xz = scaler.fit_transform(X)
    # else:
    Xz = X - X.mean(0, keepdims=True)

    # Fit PCA using fast randomized solver by default
    pca = PCA(svd_solver=svd_solver)
    scores = pca.fit_transform(Xz)  # (T, n_components=F)
    return pca, scaler, Xz, scores


# ----------------------------
# Plots
# ----------------------------
def plot_elbow(pca):
    """
    Scree plot (per-component explained variance ratio) + cumulative.
    Use the 'knee' visually or a threshold like 90-95% cumulative.
    """
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    ks = np.arange(1, len(evr) + 1)

    # Scree
    plt.figure()
    plt.plot(ks, evr, marker="o")
    plt.xlabel("Component")
    plt.ylabel("Explained variance ratio")
    plt.title("Scree (elbow) plot")
    plt.grid(True)
    plt.show()

    # Cumulative
    plt.figure()
    plt.plot(ks, cum, marker="o")
    plt.axhline(0.90, linestyle="--")
    plt.axhline(0.95, linestyle="--")
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title("Cumulative explained variance")
    plt.grid(True)
    plt.show()

    k90 = int(np.argmax(cum >= 0.90) + 1)
    k95 = int(np.argmax(cum >= 0.95) + 1)
    print(f"Components for ≥90% variance: {k90} | for ≥95%: {k95}")


def plot_component_map(pca, shape_info, comp_idx):
    """
    For imaging data: plot spatial map of a principal component.
    comp_idx is zero-based.
    """
    if not shape_info["is_image"]:
        raise ValueError("plot_component_map is only for (T, H, W) data.")
    H, W = shape_info["H"], shape_info["W"]
    comp = pca.components_[comp_idx].reshape(H, W)

    plt.figure()
    plt.imshow(comp)
    plt.colorbar()
    plt.title(f"PCA component #{comp_idx} (spatial map)")
    plt.axis("off")
    plt.show()




def plot_component_timecourse(scores, comp_idx):
    """
    Plot the time course (score) of a given component across T samples.
    """
    plt.figure()
    plt.plot(scores[:, comp_idx])
    plt.xlabel("Time (samples)")
    plt.ylabel("Score")
    plt.title(f"PCA component #{comp_idx} time course")
    plt.grid(True)
    plt.show()


# ----------------------------
# Remove bad components & reconstruct
# ----------------------------
def reconstruct_without_components(pca, scaler, Xz, drop_indices):
    """
    Zero out selected components and inverse-transform back to original space.
    - Xz: the standardized (or mean-centered) data used to fit PCA
    - drop_indices: list of component indices (0-based) to remove
    Returns X_clean in the original feature space.
    """
    scores = pca.transform(Xz)                 # (T, n_comp)
    if drop_indices:
        scores[:, drop_indices] = 0.0          # zero bad components
    Xz_clean = pca.inverse_transform(scores)   # back to standardized/centered space

    if scaler is not None:
        X_clean = scaler.inverse_transform(Xz_clean)
    else:
        # If we only mean-centered, add the original mean back:
        X_clean = Xz_clean
    return X_clean


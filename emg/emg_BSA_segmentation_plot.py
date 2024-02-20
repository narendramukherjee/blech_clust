# Import stuff!
import numpy as np
import tables
import easygui
import sys
import os
import matplotlib.pyplot as plt
import glob
import json
import pandas as pd
import seaborn as sns
from tqdm import tqdm, trange
# Necessary blech_clust modules
sys.path.append('..')
from utils.blech_utils import imp_metadata

# def create_grid_plots(array, array_name, plot_type = 'line'):
# 
#     inds = list(np.ndindex(array.shape[:3]))
# 
#     # Create plots
#     fig_list = [
#             plt.subplots(
#                 len(unique_lasers), 
#                 len(tastes),
#                 sharex=True, sharey=True,
#                 figsize = (len(tastes)*4, 4*len(unique_lasers))) \
#             for i in range(len(channel_names))] 
# 
#     # Make sure axes are 2D
#     fin_fig_list = []
#     for fig,ax in fig_list:
#         if len(unique_lasers) == 1:
#             ax = ax[np.newaxis,:]
#         elif len(tastes) == 1:
#             ax = ax[:,np.newaxis]
#         fin_fig_list.append((fig,ax))
# 
#     sub_inds = sorted(list(set([x[1:] for x in inds])))
#     for chan_num, (fig,ax) in enumerate(fin_fig_list):
#         fig.suptitle(channel_names[chan_num] + ' : ' + array_name)
#         for row,col in sub_inds:
#             this_ax = ax[row,col]
#             this_dat = array[chan_num, row, col]
#             if col == 0:
#                 this_ax.set_ylabel(f"Laser: {unique_lasers[row]}")
#             if row == ax.shape[0]-1:
#                 this_ax.set_xlabel('Time post-stim (ms)')
#             this_ax.set_title(tastes[col])
#             if plot_type == 'line':
#                 this_ax.plot(x[plot_indices], this_dat[plot_indices])
#                 this_ax.set_ylim([0,1])
#             else:
#                 this_ax.pcolormesh(
#                         x[plot_indices],
#                         np.arange(this_dat.shape[0]),
#                         this_dat[:,plot_indices],
#                         shading = 'nearest'
#                         )
#             this_ax.axvline(0, 
#                     color = 'red', linestyle = '--', 
#                     linewidth = 2, alpha = 0.7)
# 
#     for this_name, this_fig in zip(channel_names, fin_fig_list):
#         this_fig[0].savefig(
#                 os.path.join(
#                     plot_dir, 
#                     f'{this_name}_{array_name}.png'))
#         plt.close(this_fig[0])
#     #plt.show()
# 
# def create_overlay(array, array_name):
# 
#     inds = list(np.ndindex((array.shape[:2])))
# 
#     # Create plots
#     fig,ax = plt.subplots(
#                 len(channel_names),
#                 len(unique_lasers), 
#                 sharex=True, sharey=True,
#                 figsize = (4*len(unique_lasers), 4*len(channel_names))) 
# 
#     # Make sure axes are 2D
#     if len(unique_lasers)*len(channel_names) == 1:
#         ax = np.expand_dims(ax,axis=(0,1)) 
#     elif len(unique_lasers) == 1:
#         ax = np.array([ax]).T
#     elif len(channel_names) == 1:
#         ax = np.array([ax])
# 
#     for chan_num,laser_num in inds:
#         ax_title = channel_names[chan_num] + ' : ' + array_name
#         this_ax = ax[chan_num,laser_num]
#         this_dat = array[chan_num, laser_num] 
#         if laser_num == 0:
#             this_ax.set_ylabel(f"Channel: {channel_names[chan_num]}")
#         if chan_num == 0:
#             this_ax.set_title(f"Laser: {unique_lasers[laser_num]}")
#         if chan_num == ax.shape[0]-1:
#             this_ax.set_xlabel('Time post-stim (ms)')
# 
#         this_ax.plot(x[plot_indices], this_dat[:,plot_indices].T)
#         this_ax.legend(tastes)
#         this_ax.set_ylim([0,1])
#         this_ax.axvline(0, 
#                 color = 'red', linestyle = '--', 
#                 linewidth = 2, alpha = 0.7)
#     fig.savefig(
#             os.path.join(
#                 plot_dir, 
#                 f'{array_name}_overlay.png'))
#     plt.close(fig)


############################################################

# Ask for the directory where the hdf5 file sits, and change to that directory
# Get name of directory with the data files
metadata_handler = imp_metadata(sys.argv)
# dir_name = '/media/bigdata/NM43_2500ms_160515_104159_copy' 
# metadata_handler = imp_metadata([[],dir_name])
dir_name = metadata_handler.dir_name
info_dict = metadata_handler.info_dict
params_dict = metadata_handler.params_dict
os.chdir(dir_name)

emg_merge_df = pd.read_csv('emg_output/emg_env_merge_df.csv', index_col = 0)
emg_merge_df['laser'].fillna(False, inplace = True)

gapes = np.load('emg_output/gape_array.npy')
ltps = np.load('emg_output/ltp_array.npy')

# # Open the hdf5 file
# hf5 = tables.open_file(metadata_handler.hdf5_name, 'r+')
# 
# all_nodes = list(hf5.get_node('/emg_BSA_results')._f_iter_nodes())
# channel_names = [x._v_name for x in all_nodes \
#         if 'group' in str(x.__class__)]
# 
# # Pull the data from the /ancillary_analysis node
# unique_lasers = hf5.root.ancillary_analysis.laser_combination_d_l[:]
# gapes=hf5.root.emg_BSA_results.gapes[:]
# ltps=hf5.root.emg_BSA_results.ltps[:]
# sig_trials=hf5.root.emg_BSA_results.sig_trials[:]
# emg_BSA_results=hf5.root.emg_BSA_results.emg_BSA_results_final[:]

# Reading single values from the hdf5 file seems hard, 
# needs the read() method to be called
pre_stim = params_dict["spike_array_durations"][0]
time_limits = [int(x) for x in params_dict['psth_params']['durations']]
x = np.arange(gapes.shape[-1]) - pre_stim
plot_indices = np.where((x >= -time_limits[0])*(x <= time_limits[1]))[0]

plot_gapes = gapes[:,plot_indices]
plot_ltps = ltps[:,plot_indices]
plot_x = [x[plot_indices]]*len(gapes)

# Multi-column explode is not available in current pandas
# Add time, gapes, and ltps to the dataframe manually
fin_frame_list = []
for i in trange(len(emg_merge_df)):
    this_frame = emg_merge_df.iloc[i]
    this_dict = this_frame.to_dict()
    this_dict['time'] = plot_x[i]
    this_dict['gapes'] = plot_gapes[i]
    this_dict['ltps'] = plot_ltps[i]
    fin_frame_list.append(pd.DataFrame(this_dict))

emg_merge_df_long = pd.concat(fin_frame_list, axis = 0) 

# Melt gapes and ltps into long format
emg_merge_df_long = pd.melt(
        emg_merge_df_long, 
        id_vars = [x for x in emg_merge_df_long.columns if x not in ['gapes', 'ltps']], 
        value_vars = ['gapes', 'ltps'],
        var_name = 'emg_type',
        value_name = 'emg_value')

mean_emg_long = emg_merge_df_long.groupby(
        ['car','taste','laser','emg_type','time']).mean().reset_index()

    # this_frame['time'] = plot_x[i]
    # this_frame['gapes'] = plot_gapes[i]
    # this_frame['ltps'] = plot_ltps[i]

# emg_merge_df['time'] = [x[plot_indices]]*len(emg_merge_df)
# emg_merge_df['gapes'] = list(plot_gapes*1)
# emg_merge_df['ltps'] = list(plot_ltps*1)
# 
# emg_merge_df_long = emg_merge_df.explode(
#         ['gapes', 'ltps', 'time']).reset_index(drop = True)

# For each [CAR, Taste, Laser Condition],
# Plot both Gapes and LTPS
# Single trials and Averages

############################################################
# Plotting
############################################################

plot_dir = os.path.join(
        dir_name,
        'emg_output',
        'plots')
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# plot_data = gapes.copy()
# Plot Grid
for plot_name, plot_data in zip(['gapes', 'ltps'], [gapes, ltps]):
    car_list = [x[1] for x in list(emg_merge_df.groupby('car'))]
    for this_car in car_list:
        taste_laser_groups = list(this_car.groupby(['taste', 'laser']))
        taste_laser_inds = [x[0] for x in taste_laser_groups]
        taste_laser_data = [x[1] for x in taste_laser_groups]
        n_tastes = this_car['taste'].nunique()
        n_lasers = this_car['laser'].nunique()
        fig, ax = plt.subplots(
                n_lasers, n_tastes,
                sharex = True, sharey = True,
                figsize = (4*n_tastes, 4*n_lasers))
        if n_tastes == 1:
            ax = np.expand_dims(ax, axis = 0)
        if n_lasers == 1:
            ax = np.expand_dims(ax, axis = 1)
        for this_ind, this_dat, this_ax in zip(taste_laser_inds, taste_laser_data, ax.flatten()):
            this_data_inds = this_dat.index.values
            this_plot_data = plot_data[this_data_inds]
            this_ax.set_title(f"Taste: {this_ind[0]}, Laser: {this_ind[1]}")
            this_ax.pcolormesh(
                    x[plot_indices], 
                    np.arange(this_plot_data.shape[0]), 
                    this_plot_data[:,plot_indices])
            this_ax.axvline(0, 
                    color = 'red', linestyle = '--', 
                    linewidth = 2, alpha = 0.7)
            # this_ax.set_yticks(np.arange(this_plot_data.shape[0])+0.5)
            # this_ax.set_yticklabels(this_data_inds)
        fig.savefig(
                os.path.join(
                    plot_dir,
                    f'{this_car.car.unique()[0]}_{plot_name}.png'),
                bbox_inches = 'tight')
        plt.close(fig)

# Plot taste overlay per laser condition and CAR
g = sns.relplot(
        data = mean_emg_long,
        x = 'time',
        y = 'emg_value',
        hue = 'taste',
        row = 'laser',
        col = 'car',
        style = 'emg_type',
        kind = 'line',
        linewidth = 5,
        alpha = 0.7,
        )
g.fig.suptitle('Taste Overlay')
# Plot dashed line as x=0
for ax in g.axes.flatten():
    ax.axvline(0, color = 'red', linestyle = '--', linewidth = 2, alpha = 0.7)
g.savefig(
        os.path.join(
            plot_dir,
            'taste_overlay.png'),
        bbox_inches = 'tight')
plt.close(g.fig)


# Plot laser overlay per taste and CAR
g = sns.relplot(
        data = mean_emg_long,
        x = 'time',
        y = 'emg_value',
        hue = 'laser',
        row = 'car',
        col = 'taste',
        style = 'emg_type',
        kind = 'line',
        linewidth = 5,
        alpha = 0.7,
        )
g.fig.suptitle('Laser Overlay')
# Plot dashed line as x=0
for ax in g.axes.flatten():
    ax.axvline(0, color = 'red', linestyle = '--', linewidth = 2, alpha = 0.7)
g.savefig(
        os.path.join(
            plot_dir,
            'laser_overlay.png'),
        bbox_inches = 'tight')
plt.close(g.fig)


# Also maintain a counter of the number of trials in the analysis

# Close the hdf5 file
# hf5.close()



# Get an array of x values to plot the average probability of 
# gaping or licking across time

# Get the indices of x that need to be plotted based on the chosen time limits

# tastes = info_dict['taste_params']['tastes']
# 
# mean_gapes = np.nanmean(gapes, axis=-2)
# mean_ltps = np.nanmean(ltps, axis=-2)
# 
# # Generate grid plots
# create_grid_plots(mean_gapes, 'mean_gapes')
# create_grid_plots(mean_ltps, 'mean_ltps')
# 
# create_grid_plots(gapes, 'gapes', plot_type = 'im')
# create_grid_plots(ltps, 'ltps', plot_type = 'im')
# 
# # todo: Generate overlay plots
# create_overlay(mean_gapes, 'mean_gapes')
# create_overlay(mean_ltps, 'mean_ltps')
# 

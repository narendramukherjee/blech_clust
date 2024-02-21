# Create plots for filtered emg and respective envelope

# Import stuff
import numpy as np
from scipy.signal import butter, filtfilt, periodogram
import easygui
import os
import pylab as plt
import sys
import glob
import json
import tables
import pandas as pd

# Ask for the directory where the data (emg_data.npy) sits
# Get name of directory with the data files

if len(sys.argv) > 1:
    dir_name = os.path.abspath(sys.argv[1])
    if dir_name[-1] != '/':
        dir_name += '/'
else:
    dir_name = easygui.diropenbox(msg = 'Please select data directory')
os.chdir(dir_name)

h5_path = glob.glob(os.path.join(dir_name, '*.h5'))[0]
############################################################
## Load params
############################################################
# Extract info experimental info file
dir_basename = os.path.basename(dir_name[:-1])
json_path = glob.glob(os.path.join(dir_name, dir_basename + '.info'))[0]
with open(json_path, 'r') as params_file:
    info_dict = json.load(params_file)
tastes = info_dict['taste_params']['tastes']
print(f'Tastes : {tastes}'+'\n')

emg_output_dir = os.path.join(dir_name, 'emg_output')
plot_dir = os.path.join(emg_output_dir, 'plots')

# Pull pre_stim duration from params file
params_file_name = glob.glob('./**.params')[0]
with open(params_file_name,'r') as params_file_connect:
    params_dict = json.load(params_file_connect)
durations = params_dict['spike_array_durations']
pre_stim = int(durations[0])
plot_params = params_dict['psth_params']['durations']

fin_inds = [pre_stim - plot_params[0], pre_stim + plot_params[1]]
time_vec = np.arange(-plot_params[0], plot_params[1])
print(f'Plotting from {-plot_params[0]}ms pre_stim to {plot_params[1]}ms post_stim\n')

############################################################
## Load data and generate plots 
############################################################
hf5 = tables.open_file(h5_path, 'r')
# Get all emg_env data
# Structure /emg_data/dig_in_<xx>/processed_emg/<xx>_emg_env
dig_in_nodes = hf5.list_nodes('/emg_data')
dig_in_nodes = [x for x in dig_in_nodes if 'dig_in' in x._v_name]
emg_env_node_names = []
emg_env_data = []
emg_filt_data = []
for node in dig_in_nodes:
    node_list = hf5.list_nodes(
            os.path.join(node._v_pathname, 'processed_emg'),
                               classname='Array')
    node_list = [x for x in node_list if 'emg_env' in x._v_name]
    dat_list = [x.read() for x in node_list]
    emg_env_data.extend(dat_list)
    emg_env_node_names.extend([x._v_pathname for x in node_list])

    node_list = hf5.list_nodes(
            os.path.join(node._v_pathname, 'processed_emg'),
                                classname='Array')
    node_list = [x for x in node_list if 'emg_filt' in x._v_name]
    dat_list = [x.read() for x in node_list]
    emg_filt_data.extend(dat_list)
hf5.close()
dig_in_list = [x.split('/')[2] for x in emg_env_node_names]
car_list = [x.split('_')[-3].split('/')[1] for x in emg_env_node_names]

cut_emg_env = [x[...,fin_inds[0]:fin_inds[1]] for x in emg_env_data]
cut_emg_filt = [x[...,fin_inds[0]:fin_inds[1]] for x in emg_filt_data]


df = pd.DataFrame(
        {
            'dig_in': dig_in_list, 
            'car': car_list, 
            'emg_env': cut_emg_env, 
            'emg_filt': cut_emg_filt
            }
        )

car_group = list(df.groupby('car'))

max_trials = max([x.shape[0] for x in cut_emg_env])
for car_name, car_data in car_group:
    n_digs = car_data.dig_in.nunique()
    fig, ax = plt.subplots(max_trials, n_digs, 
                           sharex = True, sharey = True,
                           figsize = (n_digs*4, max_trials)
                           )
    for i, (dig_name, dig_data) in enumerate(car_data.groupby('dig_in')):
        ax[0, i].set_title(dig_name)
        dig_filt = dig_data.emg_filt.values[0]
        for j, trial in enumerate(dig_filt):
            ax[j, i].plot(time_vec, trial)
            ax[j, i].axvline(0, color = 'r', linestyle = '--')
    fig.suptitle(f'{car_name} EMG Filt')
    # fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    # plt.show()
    plt.savefig(os.path.join(plot_dir, f'{car_name}_emg_filt.png'))
    plt.close()

    fig, ax = plt.subplots(max_trials, n_digs, 
                           sharex = True, sharey = True,
                           figsize = (n_digs*4, max_trials)
                           )
    for i, (dig_name, dig_data) in enumerate(car_data.groupby('dig_in')):
        ax[0, i].set_title(dig_name)
        dig_filt = dig_data.emg_env.values[0]
        for j, trial in enumerate(dig_filt):
            ax[j, i].plot(time_vec, trial)
            ax[j, i].axvline(0, color = 'r', linestyle = '--')
    fig.suptitle(f'{car_name} EMG Env')
    # fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    # plt.show()
    plt.savefig(os.path.join(plot_dir, f'{car_name}_emg_env.png'))
    plt.close()

############################################################
# Plot env using flat_emg_env and emg_env_merge_df

# emg_env_merge_df = pd.read_csv(
#         os.path.join(emg_output_dir, 'emg_env_merge_df.csv'),
#         index_col = 0)
# flat_emg_env = np.load(os.path.join(emg_output_dir, 'flat_emg_env_data.npy'))
# 
# car_group = list(emg_env_merge_df.groupby('car'))
# 
# max_trials = emg_env_merge_df.taste_rel_trial_num.max() + 1 
# 
# for car_name, car_data in car_group:
#     n_digs = car_data.dig_in_num_taste.nunique()
#     fig, ax = plt.subplots(max_trials, n_digs, 
#                            sharex = True, sharey = True,
#                            figsize = (n_digs*4, max_trials)
#                            )
#     for i, (dig_name, dig_data) in enumerate(car_data.groupby('dig_in_name_taste')):
#         ax[0, i].set_title(dig_name)
#         dat_inds = dig_data.index.values
#         dig_filt = flat_emg_env[dat_inds][:, fin_inds[0]:fin_inds[1]] 
#         for j, trial in enumerate(dig_filt):
#             ax[j, i].plot(time_vec, trial)
#             ax[j, i].axvline(0, color = 'r', linestyle = '--')
#     fig.suptitle(f'{car_name} EMG Filt')
#     # fig.tight_layout()
#     fig.subplots_adjust(top=0.9)
#     # plt.show()
#     plt.savefig(os.path.join(plot_dir, f'{car_name}_emg_env_df.png'))
#     plt.close()

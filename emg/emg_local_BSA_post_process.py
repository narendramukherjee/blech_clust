"""
Post processing cleanup of the mess of files created by emg_local_BSA_execute.py. 
All the output files will be saved to p (named by tastes) and omega 
in the hdf5 file under the node emg_BSA_results
"""

# Import stuff
import numpy as np
import easygui
import os
import tables
import glob
import json
import sys
import pandas as pd

sys.path.append('..')
from utils.blech_utils import imp_metadata

# Get name of directory with the data files
# metadata_handler = imp_metadata(sys.argv)
dir_name = '/media/fastdata/KM45/KM45_5tastes_210620_113227_new/'
# dir_name = '/media/bigdata/NM43_2500ms_160515_104159_copy' 
metadata_handler = imp_metadata([[],dir_name])
dir_name = metadata_handler.dir_name
os.chdir(dir_name)
print(f'Processing : {dir_name}')

# Open the hdf5 file
hf5 = tables.open_file(metadata_handler.hdf5_name, 'r+')

# Delete the raw_emg node, if it exists in the hdf5 file, 
# to cut down on file size
try:
    hf5.remove_node('/raw_emg', recursive = 1)
except:
    print("Raw EMG recordings have already been removed, so moving on ..")

# Extract info experimental info file
info_dict = metadata_handler.info_dict
taste_names = info_dict['taste_params']['tastes']

# Use trial count from emg_data to account for chopping down of trials
# emg_data = np.load(os.path.join('emg_output','emg_data.npy'))
# emg_data = np.load(os.path.join('emg_output','flat_emg_env_data.npy'))
# trials = [emg_data.shape[2]]*emg_data.shape[1]
emg_trials_frame = pd.read_csv('emg_output/emg_env_df.csv', index_col = 0)

# Also get trial_info_frame
trial_info_frame = pd.read_csv(os.path.join(dir_name,'trial_info_frame.csv'))

# Load frequency analysis output
results_path = os.path.join(dir_name, 'emg_output', 'emg_BSA_results')
p_files = glob.glob(os.path.join(results_path, '*_p.npy'))
omega_files = glob.glob(os.path.join(results_path, '*_omega.npy'))

p_data = [np.load(x) for x in p_files]
p_data = np.stack(p_data, axis = 0)
omega_data = [np.load(x) for x in omega_files]
# Get first non nan omega
omega = [x for x in omega_data if not any(np.isnan(x))][0]

# Convert p_data to 1d
p_inds = np.where(p_data)
p_flat = np.zeros(p_data.shape[:2])
for this_iter in zip(*p_inds):
    p_flat[this_iter[:2]] = omega[this_iter[-1]]

# Write out p_flat and omega to disk
np.save(os.path.join('emg_output', 'p_flat.npy'), p_flat)
np.save(os.path.join('emg_output', 'omega.npy'), omega)

merge_frame = pd.merge(emg_trials_frame, trial_info_frame, 
                       left_on = ['dig_in', 'trial_inds'], 
                       right_on = ['dig_in_name_taste', 'taste_rel_trial_num'], 
                       how = 'left')
merge_frame.drop(
        columns = ['dig_in','trial_inds', 'start_taste','end_taste','start_laser','end_laser',
                   'laser_duration','laser_lag','start_laser_ms','end_laser_ms',
                   'start_taste_ms','end_taste_ms',
                   ]
        , inplace = True)

# Write out merge frame
merge_frame.to_csv('emg_output/emg_env_merge_df.csv')

############################################################
# ## Create gape and ltp arrays
############################################################
gape_array = np.logical_and(
        p_flat >= 3,
        p_flat <= 5
        )
ltp_flat = p_flat >= 5.5

# Write out
np.save('emg_output/gape_array.npy', gape_array)
np.save('emg_output/ltp_flat.npy', ltp_flat)


comp_dir = '/media/fastdata/KM45/KM45_5tastes_210620_113227_old/emg_output/emg/emg_BSA_results/'
comp_name = 'taste00_trial00'
comp_p_name = comp_name + '_p.npy'
comp_p_path = os.path.join(comp_dir, comp_p_name)
comp_p = np.load(comp_p_path)

# ind = 0
# fig, ax = plt.subplots(2,1, sharex = True)
# ax[0].pcolormesh(np.arange(p_data.shape[1]), omega, p_data[ind].T)
# ax[0].plot(p_flat[ind], color = 'r', linewidth = 2, linestyle = '--')
# ax[0].axhline(3, color = 'yellow', linestyle = '--')
# ax[0].axhline(5, color = 'yellow', linestyle = '--')
# ax[1].plot(gape_array[ind], color = 'b', label = 'Gape')
# ax[1].plot(ltp_flat[ind], color = 'r', label = 'LTP')
# ax[1].legend()
# 
# # plt.show()
# plt.imshow(comp_p.T, aspect = 'auto')
# plt.show()


# ############################################################
# ## Following will be looped over emg channels
# # In case there is more than one pair/location or differencing did not happen
# ############################################################
# output_list = glob.glob(os.path.join(dir_name,'emg_output/*'))
# output_list = [x for x in output_list if 'emg' in os.path.basename(x)]
# channel_dirs = sorted([x for x in output_list if os.path.isdir(x)])
# channels_discovered = [os.path.basename(x) for x in channel_dirs]
# print(f'Creating plots for : {channels_discovered}\n')
# 
# # Add group to hdf5 file for emg BSA results
# if '/emg_BSA_results' in hf5:
#     hf5.remove_node('/','emg_BSA_results', recursive = True)
# hf5.create_group('/', 'emg_BSA_results')
# 
# for num, this_dir in enumerate(channel_dirs):
#     os.chdir(this_dir)
#     this_basename = channels_discovered[num]
#     print(f'Processing data for : {this_basename}')
# 
#     # Load sig_trials.npy to get number of tastes
#     sig_trials = np.load('sig_trials.npy')
#     tastes = sig_trials.shape[0]
# 
#     print(f'Trials taken from emg_data.npy ::: {dict(zip(taste_names, trials))}')
# 
#     # Change to emg_BSA_results
#     os.chdir('emg_BSA_results')
# 
#     # Omega doesn't vary by trial, 
#     # so just pick it up from the 1st taste and trial, 
#     first_omega = 'taste00_trial00_omega.npy'
#     if os.path.exists(first_omega):
#         omega = np.load(first_omega)
# 
#         # Add omega to the hdf5 file
#         if '/emg_BSA_results/omega' not in hf5:
#             atom = tables.Atom.from_dtype(omega.dtype)
#             om = hf5.create_carray('/emg_BSA_results', 'omega', atom, omega.shape)
#             om[:] = omega 
#             hf5.flush()
# 
#         base_dir = '/emg_BSA_results'
#         if os.path.join(base_dir, this_basename) in hf5:
#             hf5.remove_node(base_dir, this_basename, recursive = True)
#         hf5.create_group(base_dir, this_basename)
# 
# 
#         # Load one of the p arrays to find out the time length of the emg data
#         p = np.load('taste00_trial00_p.npy')
#         time_length = p.shape[0]
# 
#         # Go through the tastes and trials
#         # todo: Output to HDF5 needs to be named by channel
#         for i in range(tastes):
#             # Make an array for posterior probabilities for each taste
#             #p = np.zeros((trials[i], time_length, 20))
#             # Make array with highest numbers of trials, so uneven trial numbers
#             # can be accomadated
#             p = np.zeros((np.max(trials), time_length, 20))
#             for j in range(trials[i]):
#                 p[j, :, :] = np.load(f'taste{i:02}_trial{j:02}_p.npy')
#             # Save p to hdf5 file
#             atom = tables.Atom.from_dtype(p.dtype)
#             prob = hf5.create_carray(
#                     os.path.join(base_dir, this_basename), 
#                     'taste%i_p' % i, 
#                     atom, 
#                     p.shape)
#             prob[:, :, :] = p
#         hf5.flush()
# 
#         # TODO: Since BSA returns most dominant frequency, BSA output is 
#         #       HIGHLY compressible. Change to utilizing timeseries rather than
#         #       time-frequency representation
# 
#         # Since BSA is an expensive process, don't delete anything
#         # In case things need to be reanalyzed
# 
#     else:
#         print(f'No data found for channel {this_basename}')
#         print('Computer will self-destruct in T minus 10 seconds')
#     print('\n')
#     print('================================')
# 
# # Close the hdf5 file
# hf5.close()

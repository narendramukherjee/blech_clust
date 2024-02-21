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
metadata_handler = imp_metadata(sys.argv)
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

hf5.close()

# Extract info experimental info file
info_dict = metadata_handler.info_dict
taste_names = info_dict['taste_params']['tastes']

# Use trial count from emg_data to account for chopping down of trials
emg_trials_frame = pd.read_csv('emg_output/emg_env_df.csv', index_col = 0)

# Also get trial_info_frame
trial_info_frame = pd.read_csv(os.path.join(dir_name,'trial_info_frame.csv'))

# Load frequency analysis output
results_path = os.path.join(dir_name, 'emg_output', 'emg_BSA_results')
#p_files = sorted(glob.glob(os.path.join(results_path, '*_p.npy')))
#omega_files = sorted(glob.glob(os.path.join(results_path, '*_omega.npy')))
trial_inds = emg_trials_frame.index.values
p_files = [os.path.join(results_path, f'trial{x:03}_p.npy') for x in trial_inds]
omega_files = [os.path.join(results_path, f'trial{x:03}_omega.npy') for x in trial_inds]

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
# Segment by frequencies

gape_array = np.logical_and(
        p_flat >= 3,
        p_flat <= 5
        )
ltp_flat = p_flat >= 5.5

# Write out
np.save('emg_output/gape_array.npy', gape_array)
np.save('emg_output/ltp_array.npy', ltp_flat)


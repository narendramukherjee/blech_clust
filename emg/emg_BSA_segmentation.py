import tables
import numpy as np
import easygui
import os
import matplotlib.pyplot as plt
import glob
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

# Grab the nodes for the available tastes
# Take trial counts from emg arrays
# TODO: This needs to be done for all emg "channels"
emg_data = np.load('emg_output/emg_data.npy')
num_trials = emg_data.shape[2]
num_tastes = emg_data.shape[1]

# Load the unique laser duration/lag combos and the trials 
# that correspond to them from the ancillary analysis node
trials = hf5.root.ancillary_analysis.trials[:] # laser conditions x trials
unique_lasers = hf5.root.ancillary_analysis.laser_combination_d_l[:] # laser conditions x details

# Iterate over channels
output_list = glob.glob(os.path.join(dir_name,'emg_output/*'))
channel_dirs = sorted([x for x in output_list if os.path.isdir(x)])
channels_discovered = [os.path.basename(x) for x in channel_dirs]
channels_discovered = [x for x in channels_discovered if 'emg' in x]
print(f'Creating plots for : {channels_discovered}\n')

# Save order of EMG channels to emg_data_readme.txt
path_to_info = os.path.join(dir_name,'emg_output','emg_data_readme.txt')
channels_discovered_dict = dict(zip(range(len(channels_discovered)), channels_discovered))
out_str = '\n\n' + f'Output channel order : {channels_discovered_dict}'
with open(path_to_info,'a') as outfile:
    outfile.write(out_str)

final_gapes_list = []
final_ltps_list = []
final_sig_trials_list = []
final_emg_BSA_list = []

for num, this_dir in enumerate(channels_discovered):
    print(f'Processing {this_dir}')
    this_basename = channels_discovered[num]
    emg_BSA_results = [x[:] for x in \
            hf5.get_node('/emg_BSA_results',this_basename)._f_iter_nodes()\
            if 'taste' in x.name]
    emg_BSA_results = np.vstack(emg_BSA_results)

    ## Find the frequency with the maximum EMG power at each time point on each trial
    ## Gapes are anything upto 4.6 Hz
    ## LTPs are from 5.95 Hz to 8.65 Hz
    #Alternatively, gapes from 3.65-5.95 Hz (6-11). LTPs from 5.95 to 8.65 Hz (11-17) 
    gapes = np.sum(emg_BSA_results[:, :, 6:11], axis = 2)/\
            np.sum(emg_BSA_results, axis = 2)
    ltps = np.sum(emg_BSA_results[:, :, 11:], axis = 2)/\
            np.sum(emg_BSA_results, axis = 2)

    # Also load up the array of significant trials 
    # (trials where the post-stimulus response is at least 
    # 4 stdev above the pre-stimulus response)
    # TODO: Needs to refer to sig_trials within a channel
    sig_trials = np.load(f'emg_output/{this_basename}/sig_trials.npy').flatten()

    # Now arrange these arrays by 
    # SHAPE : laser condition X taste X trial X time x freq
    final_emg_BSA_results = np.zeros((  len(trials), 
                                        num_tastes, 
                                        int(num_trials/len(trials)),  
                                        emg_BSA_results.shape[1], 
                                        emg_BSA_results.shape[2]), 
                                    dtype = float) 
    final_gapes = np.zeros((len(trials), 
                            num_tastes, 
                            int(num_trials/len(trials)),  
                            gapes.shape[1]), 
                        dtype = float)
    final_ltps = np.zeros((len(trials), 
                            num_tastes, 
                            int(num_trials/len(trials)), 
                            ltps.shape[1]), 
                        dtype = float)
    final_sig_trials = np.zeros((len(trials), 
                                num_tastes, 
                                int(num_trials/len(trials))), 
                            dtype = float)

    # Instead of messing around with weird indices, use dataframe to 
    # keep track of tastes and laser conditions
    trials_frame = pd.DataFrame(
            data = dict(laser_cond = np.zeros(emg_BSA_results.shape[0]).astype('int'))
            ) 
    for cond_num, inds in enumerate(trials):
        trials_frame.loc[inds, 'laser_cond'] = cond_num
    trials_frame['taste'] = None
    for taste_ind in range(num_tastes):
        this_taste_trials = np.arange(num_trials*taste_ind, num_trials*(taste_ind+1))
        trials_frame.loc[this_taste_trials, 'taste'] = taste_ind

    unique_groups = trials_frame.groupby(['laser_cond','taste']).apply(lambda x : list(np.unique(x)))
    unique_groups = unique_groups.reset_index()[['laser_cond','taste']]

    # For each group, pull out trial inds
    for _ , this_row in unique_groups.iterrows():
        this_laser = this_row['laser_cond']
        this_taste = this_row['taste']
        query_out = trials_frame.query(f" laser_cond == {this_laser} and taste == {this_taste} ")
        wanted_trial_inds = query_out.index.values

        final_emg_BSA_results[this_laser, this_taste] = emg_BSA_results[wanted_trial_inds] 
        final_gapes[this_laser, this_taste] = gapes[wanted_trial_inds] 
        final_ltps[this_laser, this_taste] = ltps[wanted_trial_inds] 
        final_sig_trials[this_laser, this_taste] = sig_trials[wanted_trial_inds] 

    ## Fill up these arrays
    #for cond_num, this_trial_vec in enumerate(trials):
    #    for trial_ind, trial_num in enumerate(this_trial_vec): 
    #        taste_ind = trial_num // num_trials
    #        mod_trial_ind = trial_num % num_trials
    #        final_emg_BSA_results[cond_num, taste_ind, mod_trial_ind] = emg_BSA_results[trial_num] 
    #        final_gapes[cond_num, taste_ind, mod_trial_ind] = gapes[trial_num] 
    #        final_ltps[cond_num, taste_ind, mod_trial_ind] = ltps[trial_num] 
    #        final_sig_trials[cond_num, taste_ind, mod_trial_ind] = sig_trials[trial_num] 

    final_gapes_list.append(final_gapes)
    final_ltps_list.append(final_ltps)
    final_sig_trials_list.append(final_sig_trials)
    final_emg_BSA_list.append(final_emg_BSA_results)

# SHAPE : channel x laser_cond x taste x trial x time
final_gapes_array = np.stack(final_gapes_list)
# SHAPE : channel x laser_cond x taste x trial x time
final_ltps_array = np.stack(final_ltps_list)
# SHAPE : channel x laser_cond x taste x trial 
final_sig_trials_array = np.stack(final_sig_trials_list)
# SHAPE : channel x laser_cond x taste x trial x time x freq
final_emg_BSA_array = np.stack(final_emg_BSA_list)

# Save under emg_BSA_results to segregate output better 
try:
    hf5.remove_node('/emg_BSA_results/gapes')
    hf5.remove_node('/emg_BSA_results/ltps')
    hf5.remove_node('/emg_BSA_results/sig_trials')
    hf5.remove_node('/emg_BSA_results/emg_BSA_results_final')
except:
    pass
hf5.create_array('/emg_BSA_results', 'gapes', final_gapes_array)
hf5.create_array('/emg_BSA_results', 'ltps', final_ltps_array)
hf5.create_array('/emg_BSA_results', 'sig_trials', final_sig_trials_array)
hf5.create_array('/emg_BSA_results', 'emg_BSA_results_final', final_emg_BSA_array)

hf5.flush()

hf5.close()

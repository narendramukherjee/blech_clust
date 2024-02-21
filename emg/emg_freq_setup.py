# Sets up emg data for running the envelope of emg recordings (env.npy) through 
# a local Bayesian Spectrum Analysis (BSA). 
# Needs an installation of R (installing Rstudio on Ubuntu is enough) - 
# in addition, the R library BaSAR needs to be installed from the CRAN 
# archives (https://cran.r-project.org/src/contrib/Archive/BaSAR/)
# This is the starting step for emg_local_BSA_execute.py

# Import stuff
import numpy as np
import os
import multiprocessing
import sys
import shutil
from glob import glob
import json
import tables
import pandas as pd

sys.path.append('..')
from utils.blech_utils import imp_metadata
from utils.blech_process_utils import path_handler

# Get name of directory with the data files
metadata_handler = imp_metadata(sys.argv)
data_dir = metadata_handler.dir_name
os.chdir(data_dir)
print(f'Processing : {data_dir}')

##############################
# Setup params
##############################
# Get paths
this_path_handler = path_handler()
blech_clust_dir = this_path_handler.blech_clust_dir
blech_emg_dir = os.path.join(blech_clust_dir ,'emg')
print(f'blech_emg_dir: {blech_emg_dir}')

print(f'blech_clust_dir: {blech_clust_dir}')
print()
emg_params_path = os.path.join(blech_clust_dir, 'params', 'emg_params.json')

if not os.path.exists(emg_params_path):
    print('=== EMG params file not found. ===')
    print('==> Please copy [[ blech_clust/params/_templates/emg_params.json ]] to [[ blech_clust/params/emg_params.json ]] and update as needed.')
    exit()

emg_params_dict = json.load(open(emg_params_path, 'r'))
use_BSA_bool = emg_params_dict['use_BSA']

hf5 = tables.open_file(metadata_handler.hdf5_name, 'r')

# Get all emg_env data
# Structure /emg_data/dig_in_<xx>/processed_emg/<xx>_emg_env
dig_in_nodes = hf5.list_nodes('/emg_data')
dig_in_nodes = [x for x in dig_in_nodes if 'dig_in' in x._v_name]
emg_env_node_names = []
emg_env_data = []
for node in dig_in_nodes:
    node_list = hf5.list_nodes(
            os.path.join(node._v_pathname, 'processed_emg'),
                               classname='Array')
    node_list = [x for x in node_list if 'emg_env' in x._v_name]
    dat_list = [x.read() for x in node_list]
    emg_env_data.extend(dat_list)
    emg_env_node_names.extend([x._v_pathname for x in node_list])

hf5.close()

# Save everything as pandas dataframe
dig_in_list = [x.split('/')[2] for x in emg_env_node_names]
car_list = [x.split('_')[-3].split('/')[1] for x in emg_env_node_names]
trial_lens = [x.shape[0] for x in emg_env_data]

fin_dig_list = [[x]*y for x,y in zip(dig_in_list, trial_lens)]
fin_car_list = [[x]*y for x,y in zip(car_list, trial_lens)]
fin_dig_list = [item for sublist in fin_dig_list for item in sublist]
fin_car_list = [item for sublist in fin_car_list for item in sublist]
trial_inds = [list(range(x)) for x in trial_lens]
trial_inds = [item for sublist in trial_inds for item in sublist]
flat_emg_env_data = np.stack(
        [item for sublist in emg_env_data for item in sublist]
         )

emg_env_df = pd.DataFrame(
        dict(
            dig_in = fin_dig_list,
            car = fin_car_list,
            trial_inds = trial_inds,
            )
        )


emg_output_dir = os.path.join(data_dir, 'emg_output')
print(f'emg_output_dir: {emg_output_dir}')
os.chdir(emg_output_dir)  

# Write the emg_env data to a file
emg_env_df.to_csv('emg_env_df.csv')
np.save('flat_emg_env_data.npy', flat_emg_env_data)

print('Deleting emg_BSA_results')
if os.path.exists('emg_BSA_results'):
    shutil.rmtree('emg_BSA_results')
os.makedirs('emg_BSA_results')

# Also delete log
print('Deleting results.log')
if os.path.exists('results.log'):
    os.remove('results.log')

# Dump shell file(s) for running GNU parallel job on the 
# user's blech_clust folder on the desktop
# First get number of CPUs - parallel be asked to run num_cpu-1 
# threads in parallel
num_cpu = multiprocessing.cpu_count()
# Then produce the file generating the parallel command
f = open(os.path.join(blech_emg_dir,'blech_emg_jetstream_parallel.sh'), 'w')
format_args = (
        int(num_cpu)-1, 
        data_dir, 
        len(emg_env_df)-1)
print(
        "parallel -k -j {:d} --noswap --load 100% --progress --ungroup --joblog {:s}/results.log bash blech_emg_jetstream_parallel1.sh ::: {{0..{:d}}}".format(*format_args), 
        file = f)
f.close()

# Then produce the file that runs blech_process.py
if use_BSA_bool:
    file_name = 'emg_local_BSA_execute.py'
    print(' === Using BSA for frequency estimation ===')
else:
    file_name = 'emg_local_STFT_execute.py'
    print(' === Using STFT for frequency estimation ===')
f = open(os.path.join(blech_emg_dir,'blech_emg_jetstream_parallel1.sh'), 'w')
print("export OMP_NUM_THREADS=1", file = f)
print(f"python {file_name} $1", file = f)
f.close()

# Finally dump a file with the data directory's location (blech.dir)
# If there is more than one emg group, this will iterate over them
f = open(os.path.join(blech_emg_dir,'BSA_run.dir'), 'w')
print(data_dir, file = f)
f.close()

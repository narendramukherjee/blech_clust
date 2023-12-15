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

sys.path.append('..')
from utils.blech_utils import imp_metadata
from utils.blech_process_utils import path_handler

# Get name of directory with the data files
metadata_handler = imp_metadata(sys.argv)
data_dir = metadata_handler.dir_name
os.chdir(data_dir)
print(f'Processing : {data_dir}')

emg_output_dir = os.path.join(data_dir, 'emg_output')
# Get dirs for each emg CAR
dir_list = glob(os.path.join(emg_output_dir,'emg*'))
dir_list = [x for x in dir_list if os.path.isdir(x)]

for num, dir_name in enumerate(dir_list): 
    #if 'emg_channel' not in os.path.basename(dir_name[:-1]):
    if 'emg_env.npy' not in os.listdir(dir_name):
        raise Exception(f'emg_env.py not found for {dir_name}')
        exit()

    os.chdir(dir_name)

    print('Deleting emg_BSA_results')
    if os.path.exists('emg_BSA_results'):
        shutil.rmtree('emg_BSA_results')
    os.makedirs('emg_BSA_results')

    # Also delete log
    print('Deleting results.log')
    if os.path.exists('results.log'):
        os.remove('results.log')

    # Load the data files
    #env = np.load('./emg_0/env.npy')
    #sig_trials = np.load('./emg_0/sig_trials.npy')
    env = np.load('./emg_env.npy')
    sig_trials = np.load('./sig_trials.npy')

    # Grab Brandeis unet username
    this_path_handler = path_handler()
    blech_clust_dir = this_path_handler.blech_clust_dir
    blech_emg_dir = os.path.join(blech_clust_dir ,'emg')

    # Dump shell file(s) for running GNU parallel job on the 
    # user's blech_clust folder on the desktop
    # First get number of CPUs - parallel be asked to run num_cpu-1 
    # threads in parallel
    num_cpu = multiprocessing.cpu_count()
    # Then produce the file generating the parallel command
    f = open(os.path.join(blech_emg_dir,'blech_emg_jetstream_parallel.sh'), 'w')
    format_args = (
            int(num_cpu)-1, 
            dir_name, 
            sig_trials.shape[0]*sig_trials.shape[1])
    print(
            "parallel -k -j {:d} --noswap --load 100% --progress --ungroup --joblog {:s}/results.log bash blech_emg_jetstream_parallel1.sh ::: {{1..{:d}}}".format(*format_args), 
            file = f)
    f.close()

    # Then produce the file that runs blech_process.py
    f = open(os.path.join(blech_emg_dir,'blech_emg_jetstream_parallel1.sh'), 'w')
    print("export OMP_NUM_THREADS=1", file = f)
    print("python emg_local_BSA_execute.py $1", file = f)
    f.close()

    # Finally dump a file with the data directory's location (blech.dir)
    # If there is more than one emg group, this will iterate over them
    if num == 0:
        f = open(os.path.join(blech_emg_dir,'BSA_run.dir'), 'w')
    else:
        f = open(os.path.join(blech_emg_dir,'BSA_run.dir'), 'a')
    print(dir_name, file = f)
    f.close()

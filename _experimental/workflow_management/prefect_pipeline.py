"""
Creating a Prefect pipeline for running tests
Run python scripts using subprocess as prefect tasks
"""
import os
from subprocess import PIPE, Popen
from prefect import flow, task
from glob import glob

# Define paths
home_dir = os.path.expanduser("~")
desktop_dir = os.path.join(home_dir, "Desktop")
blech_clust_dir = os.path.join(desktop_dir, "blech_clust") 

data_subdir = '_experimental/workflow_management/test_data_handling/test_data/KM45_5tastes_210620_113227_new'
data_dir = os.path.join(blech_clust_dir, data_subdir)


#def raise_error_if_error(process,stderr,stdout):
#    if stderr:
#        decode_err = stderr.decode('utf-8')
#        #if 'Error' in decode_err or 'Traceback' in decode_err:
#        if 'Traceback' in decode_err:
#            raise Exception(decode_err)
def raise_error_if_error(process, stderr, stdout):
    print(stdout.decode('utf-8'))
    if process.returncode:
        decode_err = stderr.decode('utf-8')
        raise Exception(decode_err)

# Define tasks

############################################################
## Common Scripts
############################################################
@task
def run_clean_slate(data_dir):
    script_name = 'blech_clean_slate.py'
    process = Popen(["python", script_name, data_dir],
                               stdout = PIPE, stderr = PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process,stderr,stdout)

@task
def run_blech_clust(data_dir):
    script_name = 'blech_clust.py'
    process = Popen(["python", script_name, data_dir],
                               stdout = PIPE, stderr = PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process,stderr,stdout)

@task
def make_arrays(data_dir):
    script_name = 'blech_make_arrays.py'
    process = Popen(["python", script_name, data_dir],
                               stdout = PIPE, stderr = PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process,stderr,stdout)

############################################################
## Spike Only 
############################################################

@task
def run_CAR(data_dir):
    script_name = 'blech_common_avg_reference.py'
    process = Popen(["python", script_name, data_dir],
                               stdout = PIPE, stderr = PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process,stderr,stdout)

@task
def run_jetstream_bash(data_dir):
    script_name = 'blech_clust_jetstream_parallel.sh'
    process = Popen(["bash", script_name],
                               stdout = PIPE, stderr = PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process,stderr,stdout)

@task
def select_clusters(data_dir):
    script_name = '_experimental/workflow_management/select_some_waveforms.py'
    process = Popen(["python", script_name, data_dir],
                               stdout = PIPE, stderr = PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process,stderr,stdout)

@task
def post_process(data_dir):
    script_name = 'blech_post_process.py'
    plot_flag = '-p ' + 'False'
    dir_flag = '-d' + data_dir
    sorted_units_path = glob(os.path.join(data_dir, '*sorted_units.csv'))[0]
    file_flag = '-f' + sorted_units_path
    process = Popen(["python", script_name, plot_flag, dir_flag, file_flag],
                               stdout = PIPE, stderr = PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process,stderr,stdout)

@task
def units_similarity(data_dir):
    script_name = 'blech_units_similarity.py'
    process = Popen(["python", script_name, data_dir],
                               stdout = PIPE, stderr = PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process,stderr,stdout)

@task
def units_plot(data_dir):
    script_name = 'blech_units_plot.py'
    process = Popen(["python", script_name, data_dir],
                               stdout = PIPE, stderr = PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process,stderr,stdout)


@task
def make_psth(data_dir):
    script_name = 'blech_make_psth.py'
    process = Popen(["python", script_name, data_dir],
                               stdout = PIPE, stderr = PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process,stderr,stdout)

@task
def pal_iden_setup(data_dir):
    script_name = 'blech_palatability_identity_setup.py'
    process = Popen(["python", script_name, data_dir],
                               stdout = PIPE, stderr = PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process,stderr,stdout)


@task
def overlay_psth(data_dir):
    script_name = 'blech_overlay_psth.py'
    process = Popen(["python", script_name, data_dir],
                               stdout = PIPE, stderr = PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process,stderr,stdout)

############################################################
## EMG Only
############################################################
@task
def cut_emg_trials(data_dir):
    script_name = '_experimental/workflow_management/cut_emg_trials.py'
    process = Popen(["python", script_name, data_dir],
                               stdout = PIPE, stderr = PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process,stderr,stdout)

@task
def emg_filter(data_dir):
    script_name = 'emg_filter.py'
    process = Popen(["python", script_name, data_dir],
                               stdout = PIPE, stderr = PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process,stderr,stdout)

@task
def emg_local_BSA(data_dir):
    script_name = 'emg_local_BSA.py'
    process = Popen(["python", script_name, data_dir],
                               stdout = PIPE, stderr = PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process,stderr,stdout)

@task
def emg_jetstream_parallel(data_dir):
    emg_env_path = '/home/abuzarmahmood/anaconda3/envs/emg_env'
    conda_init = 'conda run -p ' + emg_env_path
    script_name = 'bash blech_emg_jetstream_parallel.sh'
    full_str = ' '.join([conda_init, script_name])
    process = Popen(full_str, shell = True, stdout = PIPE, stderr = PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process,stderr,stdout)

@task
def get_laser_info(data_dir):
    script_name = 'emg_get_laser_info.py'
    process = Popen(["python", script_name, data_dir],
                               stdout = PIPE, stderr = PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process,stderr,stdout)

@task
def local_BSA_post(data_dir):
    script_name = 'emg_local_BSA_post_process.py'
    process = Popen(["python", script_name, data_dir],
                               stdout = PIPE, stderr = PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process,stderr,stdout)

@task
def BSA_segmentation(data_dir):
    script_name = 'emg_BSA_segmentation.py'
    process = Popen(["python", script_name, data_dir],
                               stdout = PIPE, stderr = PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process,stderr,stdout)

@task
def BSA_segmentation_plot(data_dir):
    script_name = 'emg_BSA_segmentation_plot.py'
    process = Popen(["python", script_name, data_dir],
                               stdout = PIPE, stderr = PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process,stderr,stdout)

@task
def run_gapes_Li(data_dir):
    script_name = 'get_gapes_Li.py'
    process = Popen(["python", script_name, data_dir],
                               stdout = PIPE, stderr = PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process,stderr,stdout)

@task
def run_QDA_gapes_plot(data_dir):
    script_name = 'gape_classifier_plots.py'
    process = Popen(["python", script_name, data_dir],
                               stdout = PIPE, stderr = PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process,stderr,stdout)

############################################################
## Define Flows
############################################################

@flow
def run_spike_test():
    os.chdir(blech_clust_dir)
    run_clean_slate(data_dir)
    run_blech_clust(data_dir)
    run_CAR(data_dir)
    run_jetstream_bash(data_dir)
    select_clusters(data_dir)
    post_process(data_dir)
    units_similarity(data_dir)
    units_plot(data_dir)
    make_arrays(data_dir)
    make_psth(data_dir)
    pal_iden_setup(data_dir)
    overlay_psth(data_dir)

@flow
def run_emg_main_test():
    os.chdir(blech_clust_dir)
    run_clean_slate(data_dir)
    run_blech_clust(data_dir)
    make_arrays(data_dir)
    cut_emg_trials(data_dir)
    # TODO: Chop number of trials down to preserve time
    os.chdir(os.path.join(blech_clust_dir, 'emg'))
    emg_filter(data_dir)
    get_laser_info(data_dir)

@flow
def run_emg_BSA_test():
    run_emg_main_test()
    emg_local_BSA(data_dir)
    emg_jetstream_parallel(data_dir)
    local_BSA_post(data_dir)
    BSA_segmentation(data_dir)
    BSA_segmentation_plot(data_dir)

@flow
def run_EMG_QDA_test():
    run_emg_main_test()
    os.chdir(os.path.join(blech_clust_dir, 'emg', 'gape_QDA_classifier'))
    run_gapes_Li(data_dir)
    run_QDA_gapes_plot(data_dir)

############################################################
## Run Flows
############################################################
run_spike_test()
run_emg_BSA_test()
run_EMG_QDA_test()

import os
import tables
import numpy as np
import easygui
import ast
import re
import pylab as plt
import matplotlib.image as mpimg
from sklearn.mixture import GaussianMixture
import argparse
import pandas as pd

# Import 3rd party code
from utils import blech_waveforms_datashader
from utils.blech_utils import entry_checker, imp_metadata
import utils.blech_post_process_utils as post_utils

# Set seed to allow inter-run reliability
# Also allows reusing the same sorting sheets across runs
np.random.seed(0)

def cluster_check(x):
    clusters = re.findall('[0-9]+',x)
    return sum([i.isdigit() for i in clusters]) == len(clusters)

# Get directory where the hdf5 file sits, and change to that directory
# Get name of directory with the data files
# Create argument parser
parser = argparse.ArgumentParser(
        description = 'Spike extraction and sorting script')
parser.add_argument('--dir-name',  '-d', help = 'Directory containing data files')
parser.add_argument('--show-plot', '-p', 
        help = 'Show waveforms while iterating (True/False)', default = 'True')
parser.add_argument('--sort-file', '-f', help = 'CSV with sorted units')
args = parser.parse_args()

if args.sort_file is not None:
    if not (args.sort_file[-3:] == 'csv'):
        raise Exception("Please provide CSV file")
    sort_table = pd.read_csv(args.sort_file)
    sort_table.fillna('',inplace=True)
    # Check when more than one cluster is specified
    sort_table['len_cluster'] = \
            [len(re.findall('[0-9]+',str(x))) for x in sort_table.Cluster]
    
    ## Get splits and merges out of the way first
    #sort_table.sort_values(['len_cluster','Split'],ascending=False, inplace=True)
    #true_index = sort_table.index
    #sort_table.reset_index(inplace=True)

if args.dir_name is not None: 
    metadata_handler = imp_metadata([[],args.dir_name])
else:
    metadata_handler = imp_metadata([])
dir_name = metadata_handler.dir_name
os.chdir(dir_name)
file_list = metadata_handler.file_list
hdf5_name = metadata_handler.hdf5_name
# Open the hdf5 file
hf5 = tables.open_file(hdf5_name, 'r+')

# Clean up the memory monitor files, pass if clean up has been done already
post_utils.clean_memory_monitor_data()  

# Delete the raw node, if it exists in the hdf5 file, to cut down on file size
post_utils.delete_raw_recordings(hf5)


# Make the sorted_units group in the hdf5 file if it doesn't already exist
try:
    hf5.create_group('/', 'sorted_units')
except:
    pass

# Run an infinite loop as long as the user wants to 
# pick clusters from the electrodes   
while True:

    unit_details_bool = 0
    # If sort_file given, iterate through that, otherwise ask user
    electrode_num, num_clusters, clusters = \
            post_utils.get_electrode_details(args, sort_table, counter)

    # Print out selections
    print('||| Electrode {}, Solution {}, Cluster {} |||'.\
            format(electrode_num, num_clusters, clusters))


    # Load data from the chosen electrode and solution
    (
        spike_waveforms,
        spike_times,
        pca_slices,
        energy,
        amplitudes,
        predictions,
    ) = post_utils.load_data_from_disk(electrode_num, num_clusters)

    # Re-show images of neurons so dumb people like Abu can make sure they
    # picked the right ones
    #if ast.literal_eval(args.show_plot):
    if args.show_plot == 'False':
        post_utils.gen_select_cluster_plot(electrode_num, num_clusters, clusters)

    # Check if the user wants to merge clusters if more than 1 cluster was chosen. 
    # Else ask if the user wants to split/re-cluster the chosen cluster
    merge = False
    re_cluster = False
    merge_msg = 'a'
    re_cluster_msg = 'a'
    if len(clusters) > 1:
        # Providing more than one cluster will AUTOMATICALLY merge
        merge = True

    else:
        # if sort_file present use that
        if args.sort_file is not None:
            split_element = sort_table.Split[counter]
            if not (split_element.strip() == ''):
                    re_cluster = True
            else:
                    re_cluster = False
        # Otherwise ask user
        else:
            split_msg, continue_bool = entry_checker(\
                    msg = 'SPLIT this cluster? (y/n)',
                    check_func = lambda x: x in ['y','n'],
                    fail_response = 'Please enter (y/n)')
            if continue_bool:
                if split_msg == 'y': 
                    re_cluster = True
                elif split_msg == 'n': 
                    re_cluster = False
            else:
                continue


    # If the user asked to split/re-cluster, 
    # ask them for the clustering parameters and perform clustering
    split_predictions = []
    chosen_split = 0
    if re_cluster: 
        # Get clustering parameters from user
        continue_bool, n_clusters, thresh, n_iter, n_restarts = \
                post_utils.get_clustering_params()
        if not continue_bool: continue

        # Make data array to be put through the GMM - 5 components: 
        # 3 PCs, scaled energy, amplitude
        this_cluster = np.where(predictions == int(clusters[0]))[0]

        data = post_utils.prepare_data(
                            this_cluster,
                            pca_slices,
                            energy,
                            amplitudes,
                            )

        # Cluster the data
        g = GaussianMixture(
                n_components = n_clusters, 
                covariance_type = 'full', 
                tol = thresh, 
                max_iter = n_iter, 
                n_init = n_restarts)
        g.fit(data)
    
        # Show the cluster plots if the solution converged
        if g.converged_:
            ... = post_utils.get_clustering_details(
                            g, 
                            data, 
                            spike_waveforms, 
                            spike_times, 
                            n_clusters, 
                            this_cluster)
        else:
            print("Solution did not converge "\
                    "- try again with higher number of iterations "\
                    "or lower convergence criterion")
            continue


        # Ask the user for the split clusters they want to choose
        continue_bool, chosen_split = \
                post_utils.get_split_cluster_choice(n_clusters)
        if not continue_bool: continue

    ##################################################

    # If the user re-clustered/split clusters, 
    # add the chosen clusters in split_clusters
    if re_cluster:
            hf5.create_group('/sorted_units', unit_name)
            # Waveforms of originally chosen cluster
            cluster_inds = np.where(predictions == int(clusters[0]))[0] 
            fin_inds = np.concatenate(\
                    [np.where(split_predictions == this_split)[0] \
                                for this_split in chosen_split])


            ############################################################ 
            unit_waveforms = spike_waveforms[cluster_inds, :]    
            # Subsetting this set of waveforms to include only the chosen split
            unit_waveforms = unit_waveforms[fin_inds]

            # Do the same thing for the spike times
            unit_times = spike_times[cluster_inds]
            unit_times = unit_times[fin_inds] 
            ############################################################ 

            post_utils.save_unit(
                    unit_waveforms,
                    unit_times,
                    unit_name,
                    electrode_num,
                    unit_description,
                    hf5)


            # To consolidate asking for unit details (single unit vs multi,
            # regular vs fast), set bool and ask for details at the end
            unit_details_bool = 1
            unit_details_file_bool = 0
            
    ##################################################

    # If only 1 cluster was chosen (and it wasn't split), 
    # add that as a new unit in /sorted_units. 
    # Ask if the isolated unit is an almost-SURE single unit
    elif len(clusters) == 1:
            hf5.create_group('/sorted_units', unit_name)
            fin_inds = np.where(predictions == int(clusters[0]))[0]

            post_utils.save_unit(
                    spike_waveforms[fin_inds, :],
                    spike_times[fin_inds],
                    fin_inds, 
                    unit_name,
                    electrode_num,
                    unit_description,
                    hf5)

            # To consolidate asking for unit details (single unit vs multi,
            # regular vs fast), set bool and ask for details at the end
            unit_details_bool = 1
            # If unit was not manipulated (merge/split), read unit details
            # from file if provided
            unit_details_file_bool = 1

    else:
        # If the chosen units are going to be merged, merge them
        if merge:
            fin_inds = np.concatenate(\
                    [np.where(predictions == int(cluster))[0] \
                    for cluster in clusters])

            unit_waveforms = spike_waveforms[fin_inds, :]
            unit_times = spike_times[fin_inds]

            # Generate plot for merged unit
            violations1, violations2 = post_utils.generate_datashader_plot(
                    unit_waveforms, 
                    title = 'Merged Unit',
                    )

            # Warn the user about the frequency of ISI violations 
            # in the merged unit
            print_str = (f':: Merged cluster \n'
                f':: {violations1:.1f} % (<2ms)\n'
                f':: {violations2:.1f} % (<1ms)\n'
                f':: {len(unit_times)} Total Waveforms \n' 
                ':: I want to still merge these clusters into one unit (y/n) :: ')
            proceed_msg, continue_bool = entry_checker(\
                    msg = print_str, 
                    check_func = lambda x: x in ['y','n'],
                    fail_response = 'Please enter (y/n)')
            if continue_bool:
                if proceed_msg == 'y': 
                    proceed = True
                elif proceed_msg == 'n': 
                    proceed = False
            else:
                continue

            # Create unit if the user agrees to proceed, 
            # else abort and go back to start of the loop 
            if proceed:     
                post_utils.save_unit(
                        unit_waveforms,
                        unit_times,
                        unit_name,
                        electrode_num,
                        unit_description,
                        hf5)


                # To consolidate asking for unit details (single unit vs multi,
                # regular vs fast), set bool and ask for details at the end
                unit_details_bool = 1
                unit_details_file_bool = 0

            else:
                continue

    # Ask user for unit details, and ask for HDF5 file 
    if unit_details_bool:
        continue_bool, unit_description = \
                post_utils.get_unit_properties(unit_description, counter)
        if not continue_bool:
            continue

        unit_description.append()
        table.flush()
        hf5.flush()


    print('==== {} Complete ===\n'.format(unit_name))
    print('==== Iteration Ended ===\n')

# Sort unit_descriptor by unit_number
# This will be needed if sort_table is used, as using sort_table
# will add merge/split marked units first
unit_descriptor_handler.sort_table_and_saved_units()

print('== Post-processing exiting ==')
# Close the hdf5 file
hf5.close()

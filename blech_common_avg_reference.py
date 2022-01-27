# Run through all the raw electrode data, and subtract a common average reference from every electrode's recording
# The user specifies the electrodes to be used as a common average group 

# Import stuff!
import tables
import numpy as np
import os
import easygui
from tqdm import tqdm

# Ask for the directory where the hdf5 file sits, and change to that directory
dir_name = easygui.diropenbox()
os.chdir(dir_name)

# Look for the hdf5 file in the directory
file_list = os.listdir('./')
hdf5_name = ''
for files in file_list:
	if files[-2:] == 'h5':
		hdf5_name = files

# Open the hdf5 file
hf5 = tables.open_file(hdf5_name, 'r+')

# Get the names of all files in this directory
file_list = os.listdir('./')

# Get the Intan amplifier ports used in the recordings
ports = list(set(f[4] for f in file_list if f[:3] == 'amp'))
# Sort the ports in alphabetical order
ports.sort()

# Count the number of electrodes on one of the ports (assume all ports have equal number of electrodes)
num_electrodes = [int(f[-7:-4]) for f in file_list if f[:3] == 'amp']
num_electrodes = np.max(num_electrodes) + 1

# Ask the user how many common average groups there are in the data. Group = set of electrodes put in together in the same place in the brain (that work).
num_groups = easygui.multenterbox(msg = "How many common average groups do you have in the dataset?", fields = ["Number of CAR groups"])
num_groups = int(num_groups[0])

# Ask the user to choose the port number and electrodes for each of the groups
group_ports = []
average_electrodes = []
for i in range(num_groups):
	group_ports.append(easygui.multchoicebox(msg = 'Choose the port for common average reference group {:d}'.format(i+1), choices = tuple(ports)))
	average_electrodes.append(easygui.multchoicebox(msg = 'Choose the ELECTRODES TO AVERAGE ACROSS in the common average reference group {:d}. Remember to DESELECT the EMG electrodes'.format(i+1), choices = ([el for el in range(num_electrodes)])))

# Get the emg electrode ports and channel numbers from the user
# If only one amplifier port was used in the experiment, that's the emg_port. Else ask the user to specify
emg_port = ''
if len(ports) == 1:
	emg_port = list(ports[0])
else:
	emg_port = easygui.multchoicebox(msg = 'Which amplifier port were the EMG electrodes hooked up to? Just choose any amplifier port if you did not hook up an EMG at all.', choices = tuple(ports))
# Now get the emg channel numbers, and convert them to integers
emg_channels = easygui.multchoicebox(msg = 'Choose the CHANNEL NUMBERS FOR THE EMG ELECTRODES. Click clear all and ok if you did not use an EMG electrode', choices = tuple([i for i in range(32)]))
if emg_channels:
	for i in range(len(emg_channels)):
		emg_channels[i] = int(emg_channels[i])
# set emg_channels to an empty list if no channels were chosen
if emg_channels is None:
	emg_channels = []
emg_channels.sort()

# Now convert the electrode numbers to be averaged across to the absolute scale (0-63 if there are 2 ports with 32 recordings each with no EMG)
CAR_electrodes = []
# Run through the common average groups
raw_electrodes = hf5.list_nodes('/raw')
all_electrodes = [x for y in average_electrodes for x in y]
all_electrodes_padded = [f'{int(x):02}' for x in all_electrodes]
all_electrodes_inds = np.argsort(all_electrodes_padded)
all_electrodes = [all_electrodes[i] for i in all_electrodes_inds]
avg_electrodes_mapping = dict(zip(all_electrodes, range(len(raw_electrodes))))

for group in range(num_groups):
    # Now run through the electrodes and port chosen for that group, 
    # and convert to the absolute scale
    # To allow for more varied emg layouts, instead of comparing with emg_channels[0]
    # (which assumes emg channels will be together), check if the electrode number
    # is higher than ANY emg_channels, and subtract however many channels it has passed
    this_group_electrodes = []
    for electrode in average_electrodes[group]:
        this_group_electrodes.append(avg_electrodes_mapping[electrode])
        #if len(emg_channels) == 0:
        #	this_group_electrodes.append(int(electrode) + \
                #            num_electrodes*ports.index(group_ports[group][0]))
        #else:
        #	if group_ports[group] == emg_port and int(electrode) < emg_channels[0]:
        #		this_group_electrodes.append(int(electrode) + \
                #                num_electrodes*ports.index(group_ports[group][0]))
        #	else:
        #		this_group_electrodes.append(int(electrode) + \
                #                num_electrodes*ports.index(group_ports[group][0]) - len(emg_channels))
    CAR_electrodes.append(this_group_electrodes)

# Pull out the raw electrode nodes of the HDF5 file

# First get the common average references by averaging across the electrodes picked for each group
print("Calculating common average reference for {:d} groups".format(num_groups))
common_average_reference = np.zeros((num_groups, hf5.root.raw.electrode0[:].shape[0]))
for group in range(num_groups):
	# Stack up the voltage data from all the electrodes that need to be 
    # averaged across in this CAR group	
	# In hindsight, don't stack up all the data, it is a huge memory waste. 
    # Instead first add up the voltage values from each electrode to the same array, 
    # and divide by number of electrodes to get the average	
	for electrode in tqdm(CAR_electrodes[group]):
		exec("common_average_reference[group, :] += hf5.root.raw.electrode{:d}[:]".format(electrode))

	# Average the voltage data across electrodes by dividing by the number of electrodes in this group
	common_average_reference[group, :] /= float(len(CAR_electrodes[group]))

print("Common average reference for {:d} groups calculated".format(num_groups))

# Now run through the raw electrode data and subtract the common average reference from each of them
for electrode in tqdm(raw_electrodes):
	electrode_num = int(str.split(electrode._v_pathname, 'electrode')[-1])
	# Get the common average group number that this electrode belongs to
	# We assume that each electrode belongs to only 1 common average reference group - IMPORTANT!
	group = int([i for i in range(num_groups) if electrode_num in CAR_electrodes[i]][0])

	# Subtract the common average reference for that group from the voltage data of the electrode
	referenced_data = electrode[:] - common_average_reference[group]

	# First remove the node with this electrode's data
	hf5.remove_node("/raw/electrode{:d}".format(electrode_num))

	# Now make a new array replacing the node removed above with the referenced data
	hf5.create_array("/raw", "electrode{:d}".format(electrode_num), referenced_data)
	hf5.flush()

	del referenced_data

hf5.close()
print("Modified electrode arrays written to HDF5 file after subtracting the common average reference")

# Compress the file to clean up all the deleting and creating of arrays
print("Compressing the modified HDF5 file to save up on space")
# Use ptrepack to save a clean and fresh copy of the hdf5 file as tmp.hf5
os.system("ptrepack --chunkshape=auto --propindexes --complevel=9 --complib=blosc " + hdf5_name + " " +  "tmp.h5")

# Delete the old hdf5 file
os.system("rm " + hdf5_name)

# And rename the new file with the same old name
os.system("mv tmp.h5 " + hdf5_name)

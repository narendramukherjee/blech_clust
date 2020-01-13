# Organizes the list of units into a coherent sequence starting at 000. In case you delete units after looking at rasters or overlap between units, this code will reorganize the remaining units

# Import stuff
import os
import tables
import numpy as np
import easygui

# Get directory where the hdf5 file sits, and change to that directory
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

# Get list of units under the sorted_units group
units = hf5.list_nodes('/sorted_units')
# Get the unit numbers and sort them in descending order
unit_nums = -np.sort(-np.array([int(str(unit).split('/')[-1][4:7]) for unit in units]))

# Now run through the units in descending order - the number of units is taken from the unit_descriptor tables, assuming that the user deleted units without deleting
# the corresponding entry in the table
num_units_in_table = len(hf5.root.unit_descriptor) 
for unit in reversed(range(num_units_in_table)):
	# Check if this unit exists in the sorted unit list - don't do anything if it exists
	if unit in unit_nums:
		continue
	else:
		# If the unit does not exist in the sorted unit list
		# First delete the corresponding row from the unit_descriptor table
		hf5.root.unit_descriptor.remove_row(unit)
		hf5.flush()
		# Check if this is the last unit (if you have 30 neurons, is this unit029?) - don't rename units if that's the case
		if unit == num_units_in_table - 1:
			continue
		# Otherwise rename all the previous units
		else:
			# Read all the current unit numbers
			current_units = np.array([int(str(current_unit).split('/')[-1][4:7]) for current_unit in hf5.list_nodes('/sorted_units')])
			# Run a loop from the missing unit number + 1 to the maximum current unit
			for i in range(unit + 1, np.max(current_units) + 1, 1):
				# Rename each of these units to the next lower number
				hf5.rename_node('/sorted_units/unit{:03d}'.format(i), 'unit{:03d}'.format(i - 1))
				hf5.flush()
hf5.close()		
print("Units organized")

# Compress the file
print("File being compressed")
# Use ptrepack to save a clean and fresh copy of the hdf5 file as tmp.hf5
os.system("ptrepack --chunkshape=auto --propindexes --complevel=9 --complib=blosc " + hdf5_name + " " +  "tmp.h5")

# Delete the old hdf5 file
os.remove(hdf5_name)

# And rename the new file with the same old name
os.rename("tmp.h5", hdf5_name)





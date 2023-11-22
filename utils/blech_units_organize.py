"""
 Organizes the list of units into a coherent sequence starting at 000. 
 In case you delete units after looking at rasters or overlap between units, 
 this code will reorganize the remaining units
"""

# Import stuff
import os
import tables
import numpy as np
import easygui
import argparse

# Get directory where the hdf5 file sits, and change to that directory
# Get name of directory with the data files
# Create argument parser

metadata_handler = imp_metadata(sys.argv)
dir_name = metadata_handler.dir_name
os.chdir(dir_name)

# Open the hdf5 file
hf5 = tables.open_file(metadata_handler.hdf5_name, 'r+')

this_descriptor_handler = post_utils.unit_descriptor_handler(hf5, dir_name)
match_bool, match_frame = \
	this_descriptor_handler.check_table_matches_saved_units()

print("Checking if unit_descriptor match saved units")
print(match_frame)
if not match_bool:
    this_descriptor_handler.resort_units()
    print("Units resorted")
else:
    print("Units already sorted")

hf5.close()		

# Compress the file
print("File being compressed")
# Use ptrepack to save a clean and fresh copy of the hdf5 file as tmp.hf5
os.system(
	"ptrepack --chunkshape=auto --propindexes --complevel=9 ' +\
	'--complib=blosc " +\
        hdf5_name + " " +  "tmp.h5"
	)

# Delete the old hdf5 file
os.remove(hdf5_name)

# And rename the new file with the same old name
os.rename("tmp.h5", hdf5_name)

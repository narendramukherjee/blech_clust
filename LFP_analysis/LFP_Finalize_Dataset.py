import easygui
import glob
import tables
from itertools import product
import pandas as pd

# =============================================================================
# #Channel Check Processing
# =============================================================================

dir_name = easygui.diropenbox(msg = 'Select directory with HDF5 file')
hdf5_name = glob.glob(dir_name + '/*.h5')[0]

#Open the hdf5 file
hf5 = tables.open_file(hdf5_name, 'r+')
parsed_lfp_addr = '/Parsed_LFP'
taste_num = len([x for x in hf5.list_nodes(parsed_lfp_addr) \
        if 'dig_in' in str(x)])
channel_num = hf5.list_nodes(parsed_lfp_addr)[0].shape[0]

#Ask user to check LFP traces to ensure channels are not shorted/bad in order to remove said channel from further processing
channel_check =  list(map(int,easygui.multchoicebox(
        msg = 'Choose the channel numbers that you '\
                'want to REMOVE from further analyses. '
                'Click clear all and ok if all channels are good', 
                choices = tuple([i for i in range(channel_num)]))))

taste_check = list(map(int, easygui.multchoicebox(
        msg = 'Chose the taste numbers that you want to '\
                'REMOVE from further analyses. Click clear all '\
                'and ok if all channels are good',
                choices = tuple([i for i in range(taste_num)]))))

flag_frame = pd.DataFrame(list(product(range(taste_num),range(channel_num))),
                columns = ['Dig_In','Channel'])
flag_frame = flag_frame.query('Dig_In in @taste_check or Channel in @channel_check')
flag_frame.to_hdf(hdf5_name, parsed_lfp_addr + '/flagged_channels')

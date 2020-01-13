# Subtracts the two emg signals and filters and saves the results.

# Import stuff
import numpy as np
from scipy.signal import butter, filtfilt, periodogram
import easygui
import os

# Ask for the directory where the data (emg_data.npy) sits
dir_name = easygui.diropenbox()
os.chdir(dir_name)

# Load the data
emg_data = np.load('emg_data.npy')

# Ask the user for stimulus delivery time in each trial, and convert to an integer
pre_stim = easygui.multenterbox(msg = 'Enter the pre-stimulus time included in each trial', fields = ['Pre-stimulus time (ms)']) 
pre_stim = int(pre_stim[0])

# Get coefficients for Butterworth filters
m, n = butter(2, 2.0*300.0/1000.0, 'highpass')
c, d = butter(2, 2.0*15.0/1000.0, 'lowpass')

# check how many EMG channels used in this experiment
check = easygui.ynbox(msg = 'Did you have more than one EMG channel?', title = 'Check YES if you did')

# Bandpass filter the emg signals, and store them in a numpy array. Low pass filter the bandpassed signals, and store them in another array
emg_filt = np.zeros(emg_data.shape[1:])
env = np.zeros(emg_data.shape[1:])
for i in range(emg_data.shape[1]):
	for j in range(emg_data.shape[2]):
		if check:
			emg_filt[i, j, :] = filtfilt(m, n, emg_data[0, i, j, :] - emg_data[1, i, j, :])
		else:
			emg_filt[i, j, :] = filtfilt(m, n, emg_data[0, i, j, :])
		env[i, j, :] = filtfilt(c, d, np.abs(emg_filt[i, j, :]))    
			
# Get mean and std of baseline emg activity, and use it to select trials that have significant post stimulus activity
sig_trials = np.zeros((emg_data.shape[1], emg_data.shape[2]))
m = np.mean(np.abs(emg_filt[:, :, :pre_stim]))
s = np.std(np.abs(emg_filt[:, :, :pre_stim]))
for i in range(emg_data.shape[1]):
	for j in range(emg_data.shape[2]):
		if np.mean(np.abs(emg_filt[i, j, pre_stim:])) > m and np.max(np.abs(emg_filt[i, j, pre_stim:])) > m + 4.0*s:
			sig_trials[i, j] = 1	
	
# Save the highpass filtered signal, the envelope and the indicator of significant trials as a np array
np.save('emg_filt.npy', emg_filt)
np.save('env.npy', env)
np.save('sig_trials.npy', sig_trials)



	
					





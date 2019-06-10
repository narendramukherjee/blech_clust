import os
import tables
import numpy as np
import easygui
import ast
import pylab as plt
from sklearn.mixture import GaussianMixture
import blech_waveforms_datashader

# Get directory where the hdf5 file sits, and change to that directory
dir_name = easygui.diropenbox()
os.chdir(dir_name)

# Clean up the memory monitor files, pass if clean up has been done already
if not os.path.exists('./memory_monitor_clustering/memory_usage.txt'):
	file_list = os.listdir('./memory_monitor_clustering')
	f = open('./memory_monitor_clustering/memory_usage.txt', 'w')
	for files in file_list:
		try:
			mem_usage = np.loadtxt('./memory_monitor_clustering/' + files)
			print('electrode'+files[:-4], '\t', str(mem_usage)+'MB', file=f)
			os.system('rm ' + './memory_monitor_clustering/' + files)
		except:
			pass	
	f.close()

# Look for the hdf5 file in the directory
file_list = os.listdir('./')
hdf5_name = ''
for files in file_list:
	if files[-2:] == 'h5':
		hdf5_name = files

# Open the hdf5 file
hf5 = tables.open_file(hdf5_name, 'r+')

# Delete the raw node, if it exists in the hdf5 file, to cut down on file size
try:
	hf5.remove_node('/raw', recursive = 1)
	# And if successful, close the currently open hdf5 file and ptrepack the file
	hf5.close()
	print("Raw recordings removed")
	os.system("ptrepack --chunkshape=auto --propindexes --complevel=9 --complib=blosc " + hdf5_name + " " + hdf5_name[:-3] + "_repacked.h5")
	# Delete the old (raw and big) hdf5 file
	os.system("rm " + hdf5_name)
	# And open the new, repacked file
	hf5 = tables.open_file(hdf5_name[:-3] + "_repacked.h5", 'r+')
	print("File repacked")
except:
	print("Raw recordings have already been removed, so moving on ..")

# Make the sorted_units group in the hdf5 file if it doesn't already exist
try:
	hf5.create_group('/', 'sorted_units')
except:
	pass

# Define a unit_descriptor class to be used to add things (anything!) about the sorted units to a pytables table
class unit_descriptor(tables.IsDescription):
	electrode_number = tables.Int32Col()
	single_unit = tables.Int32Col()
	regular_spiking = tables.Int32Col()
	fast_spiking = tables.Int32Col()

# Make a table under /sorted_units describing the sorted units. If unit_descriptor already exists, just open it up in the variable table
try:
	table = hf5.create_table('/', 'unit_descriptor', description = unit_descriptor)
except:
	table = hf5.root.unit_descriptor

# Run an infinite loop as long as the user wants to pick clusters from the electrodes	
while True:
	# Get electrode number from user
	electrode_num = easygui.multenterbox(msg = 'Which electrode do you want to choose? Hit cancel to exit', fields = ['Electrode #'])
	# Break if wrong input/cancel command was given
	try:
		electrode_num = int(electrode_num[0])
	except:
		break
	
	# Get the number of clusters in the chosen solution
	num_clusters = easygui.multenterbox(msg = 'Which solution do you want to choose for electrode %i?' % electrode_num, fields = ['Number of clusters in the solution'])
	num_clusters = int(num_clusters[0])

	# Load data from the chosen electrode and solution
	spike_waveforms = np.load('./spike_waveforms/electrode%i/spike_waveforms.npy' % electrode_num)
	spike_times = np.load('./spike_times/electrode%i/spike_times.npy' % electrode_num)
	pca_slices = np.load('./spike_waveforms/electrode%i/pca_waveforms.npy' % electrode_num)
	energy = np.load('./spike_waveforms/electrode%i/energy.npy' % electrode_num)
	amplitudes = np.load('./spike_waveforms/electrode%i/spike_amplitudes.npy' % electrode_num)
	predictions = np.load('./clustering_results/electrode%i/clusters%i/predictions.npy' % (electrode_num, num_clusters))

	# Get cluster choices from the chosen solution
	clusters = easygui.multchoicebox(msg = 'Which clusters do you want to choose?', choices = tuple([str(i) for i in range(int(np.max(predictions) + 1))]))
	
	# Check if the user wants to merge clusters if more than 1 cluster was chosen. Else ask if the user wants to split/re-cluster the chosen cluster
	merge = False
	re_cluster = False
	if len(clusters) > 1:
		merge = easygui.multchoicebox(msg = 'I want to merge these clusters into one unit (True = Yes, False = No)', choices = ('True', 'False'))
		merge = ast.literal_eval(merge[0])
	else:
		re_cluster = easygui.multchoicebox(msg = 'I want to split this cluster (True = Yes, False = No)', choices = ('True', 'False'))
		re_cluster = ast.literal_eval(re_cluster[0])

	# If the user asked to split/re-cluster, ask them for the clustering parameters and perform clustering
	split_predictions = []
	chosen_split = 0
	if re_cluster: #I want to split this cluster
		# Get clustering parameters from user
		clustering_params = easygui.multenterbox(msg = 'Fill in the parameters for re-clustering (using a GMM)', 
                                           fields = ['Number of clusters', 'Maximum number of iterations (1000 is more than enough)', 
                                                     'Convergence criterion (usually 0.0001)', 'Number of random restarts for GMM (10 is more than enough)'])
		n_clusters = int(clustering_params[0])
		n_iter = int(clustering_params[1])
		thresh = float(clustering_params[2])
		n_restarts = int(clustering_params[3]) 

		# Make data array to be put through the GMM - 5 components: 3 PCs, scaled energy, amplitude
		this_cluster = np.where(predictions == int(clusters[0]))[0]
		n_pc = 3
		data = np.zeros((len(this_cluster), n_pc + 2))	
		data[:,2:] = pca_slices[this_cluster,:n_pc]
		data[:,0] = energy[this_cluster]/np.max(energy[this_cluster])
		data[:,1] = np.abs(amplitudes[this_cluster])/np.max(np.abs(amplitudes[this_cluster]))

		# Cluster the data
		g = GaussianMixture(n_components = n_clusters, covariance_type = 'full', tol = thresh, max_iter = n_iter, n_init = n_restarts)
		g.fit(data)
	
		# Show the cluster plots if the solution converged
		if g.converged_:
			split_predictions = g.predict(data)
			x = np.arange(len(spike_waveforms[0])/10) + 1
			for cluster in range(n_clusters):
				split_points = np.where(split_predictions == cluster)[0]				
				# plt.figure(cluster)
				slices_dejittered = spike_waveforms[this_cluster, :]		# Waveforms and times from the chosen cluster
				times_dejittered = spike_times[this_cluster]
				times_dejittered = times_dejittered[split_points]		# Waveforms and times from the chosen split of the chosen cluster
				ISIs = np.ediff1d(np.sort(times_dejittered))/30.0
				violations1 = 100.0*float(np.sum(ISIs < 1.0)/split_points.shape[0])
				violations2 = 100.0*float(np.sum(ISIs < 2.0)/split_points.shape[0])
				fig, ax = blech_waveforms_datashader.waveforms_datashader(slices_dejittered[split_points, :], x)
				# plt.plot(x-15, slices_dejittered[split_points, :].T, linewidth = 0.01, color = 'red')
				ax.set_xlabel('Sample (30 samples per ms)')
				ax.set_ylabel('Voltage (microvolts)')
				ax.set_title("Split Cluster{:d}, 2ms ISI violations={:.1f} percent".format(cluster, violations2) + "\n" + "1ms ISI violations={:.1f}%, Number of waveforms={:d}".format(violations1, split_points.shape[0]))
		else:
			print("Solution did not converge - try again with higher number of iterations or lower convergence criterion")
			continue

		plt.show()
		# Ask the user for the split clusters they want to choose
		chosen_split = easygui.multchoicebox(msg = 'Which split cluster do you want to choose? Hit cancel to exit', choices = tuple([str(i) for i in range(n_clusters)]))
		try:
			chosen_split = [int(chosen_split[i]) for i in range(len(chosen_split))]
			split_merge = False
#			if len(chosen_split) > 1:
#				split_merge = easygui.multchoicebox(msg = 'I want to merge these splited-clusters into one unit (True = Yes, False = No)', choices = ('True', 'False'))
#				split_merge = ast.literal_eval(split_merge[0])
		except:
			continue


	# Get list of existing nodes/groups under /sorted_units
	node_list = hf5.list_nodes('/sorted_units')

	# If node_list is empty, start naming units from 000
	unit_name = ''
	max_unit = 0
	if node_list == []:		
		unit_name = 'unit%03d' % 0
	# Else name the new unit by incrementing the last unit by 1 
	else:
		unit_numbers = []
		for node in node_list:
			unit_numbers.append(node._v_pathname.split('/')[-1][-3:])
			unit_numbers[-1] = int(unit_numbers[-1])
		unit_numbers = np.array(unit_numbers)
		max_unit = np.max(unit_numbers)
		unit_name = 'unit%03d' % int(max_unit + 1)

	# Get a new unit_descriptor table row for this new unit
	unit_description = table.row	

	# If the user re-clustered/split clusters, add the chosen clusters in split_clusters
	if re_cluster:
#		split_merge = False
		if len(chosen_split) == 1:
			hf5.create_group('/sorted_units', unit_name)
			unit_waveforms = spike_waveforms[np.where(predictions == int(clusters[0]))[0], :]	# Waveforms of originally chosen cluster
			unit_waveforms = unit_waveforms[np.where(split_predictions == chosen_split[0])[0], :]	# Subsetting this set of waveforms to include only the chosen split
			unit_times = spike_times[np.where(predictions == int(clusters[0]))[0]]			# Do the same thing for the spike times
			unit_times = unit_times[np.where(split_predictions == chosen_split[0])[0]]
			waveforms = hf5.create_array('/sorted_units/%s' % unit_name, 'waveforms', unit_waveforms)
			times = hf5.create_array('/sorted_units/%s' % unit_name, 'times', unit_times)
			unit_description['electrode_number'] = electrode_num
			single_unit = easygui.multchoicebox(msg = 'I am almost-SURE that this is a beautiful single unit (True = Yes, False = No)', choices = ('True', 'False'))
			unit_description['single_unit'] = int(ast.literal_eval(single_unit[0]))
			# If the user says that this is a single unit, ask them whether its regular or fast spiking
			unit_description['regular_spiking'] = 0
			unit_description['fast_spiking'] = 0
			if int(ast.literal_eval(single_unit[0])):
				unit_type = easygui.multchoicebox(msg = 'What type of unit is this (Regular spiking = Pyramidal cells, Fast spiking = PV+ interneurons)?', choices = ('regular_spiking', 'fast_spiking'))
				unit_description[unit_type[0]] = 1		
			unit_description.append()
			table.flush()
			hf5.flush()

		elif len(chosen_split) > 1:
			split_merge = easygui.multchoicebox(msg = 'I want to merge these splited-clusters into one unit (True = Yes, False = No)', choices = ('True', 'False'))
			split_merge = ast.literal_eval(split_merge[0])
            
			if split_merge: #I want to merge these splited-clusters
				unit_waveforms = []
				unit_times = []
				cluster_unit_waveforms = spike_waveforms[np.where(predictions == int(clusters[0]))[0], :]
				cluster_unit_times = spike_times[np.where(predictions == int(clusters[0]))[0]]
				for split_cluster in chosen_split:
					if unit_waveforms == []:
						unit_waveforms = cluster_unit_waveforms[np.where(split_predictions == split_cluster)[0], :]
						unit_times = cluster_unit_times[np.where(split_predictions == split_cluster)[0]]
					else:
						unit_waveforms = np.concatenate((unit_waveforms, cluster_unit_waveforms[np.where(split_predictions == split_cluster)[0], :]))
						unit_times = np.concatenate((unit_times, cluster_unit_times[np.where(split_predictions == split_cluster)[0]]))

			# Show the merged cluster to the user, and ask if they still want to merge
				x = np.arange(len(unit_waveforms[0])/10) + 1
				fig, ax = blech_waveforms_datashader.waveforms_datashader(unit_waveforms, x)
				# plt.plot(x - 15, unit_waveforms[:, ::10].T, linewidth = 0.01, color = 'red')
				ax.set_xlabel('Sample (30 samples per ms)')
				ax.set_ylabel('Voltage (microvolts)')
				ax.set_title('Merged cluster, No. of waveforms={:d}'.format(unit_waveforms.shape[0]))
				plt.show()
 
			# Warn the user about the frequency of ISI violations in the merged unit
				ISIs = np.ediff1d(np.sort(unit_times))/30.0
				violations1 = 100.0*float(np.sum(ISIs < 1.0)/len(unit_times))
				violations2 = 100.0*float(np.sum(ISIs < 2.0)/len(unit_times))
				proceed = easygui.multchoicebox(msg = 'My merged cluster has %.1f percent (<2ms) and %.1f percent (<1ms) ISI violations out of %i total waveforms. I want to still merge these clusters into one unit (True = Yes, False = No)' % (violations2, violations1, len(unit_times)), choices = ('True', 'False'))
				proceed = ast.literal_eval(proceed[0])

				# Create unit if the user agrees to proceed, else include each split_cluster as a separate unit 
				if proceed:	
					hf5.create_group('/sorted_units', unit_name)
					waveforms = hf5.create_array('/sorted_units/%s' % unit_name, 'waveforms', unit_waveforms)
					times = hf5.create_array('/sorted_units/%s' % unit_name, 'times', unit_times)
					unit_description['electrode_number'] = electrode_num
					single_unit = easygui.multchoicebox(msg = 'I am almost-SURE that this is a beautiful single unit (True = Yes, False = No)', choices = ('True', 'False'))
					unit_description['single_unit'] = int(ast.literal_eval(single_unit[0]))
					# If the user says that this is a single unit, ask them whether its regular or fast spiking
					unit_description['regular_spiking'] = 0
					unit_description['fast_spiking'] = 0
					if int(ast.literal_eval(single_unit[0])):
						unit_type = easygui.multchoicebox(msg = 'What type of unit is this (Regular spiking = Pyramidal cells, Fast spiking = PV+ interneurons)?', choices = ('regular_spiking', 'fast_spiking'))
						unit_description[unit_type[0]] = 1
					unit_description.append()
					table.flush()
					hf5.flush()
############
			else: # if not merge, then include each split_cluster as a separate unit
				for split_cluster in chosen_split:
					hf5.create_group('/sorted_units', unit_name)
					unit_waveforms = spike_waveforms[np.where(predictions == int(clusters[0]))[0], :]	# Waveforms of originally chosen cluster
					unit_waveforms = unit_waveforms[np.where(split_predictions == chosen_split)[0], :]	# Subsetting this set of waveforms to include only the chosen split
					unit_times = spike_times[np.where(predictions == int(clusters[0]))[0]]			# Do the same thing for the spike times
					unit_times = unit_times[np.where(split_predictions == chosen_split)[0]]
					waveforms = hf5.create_array('/sorted_units/%s' % unit_name, 'waveforms', unit_waveforms)
					times = hf5.create_array('/sorted_units/%s' % unit_name, 'times', unit_times)
					unit_description['electrode_number'] = electrode_num
					single_unit = easygui.multchoicebox(msg = 'I am almost-SURE that electrode: %i cluster: %i split: %i is a beautiful single unit (True = Yes, False = No)' % (electrode_num, int(cluster), int(split_cluster)), choices = ('True', 'False'))
					unit_description['single_unit'] = int(ast.literal_eval(single_unit[0]))
					# If the user says that this is a single unit, ask them whether its regular or fast spiking
					unit_description['regular_spiking'] = 0
					unit_description['fast_spiking'] = 0
					if int(ast.literal_eval(single_unit[0])):
						unit_type = easygui.multchoicebox(msg = 'What type of unit is this (Regular spiking = Pyramidal cells, Fast spiking = PV+ interneurons)?', choices = ('regular_spiking', 'fast_spiking'))
						unit_description[unit_type[0]] = 1
					unit_description.append()
					table.flush()
					hf5.flush()				

					# Finally increment max_unit and create a new unit name
					max_unit += 1
					unit_name = 'unit%03d' % int(max_unit + 1)

					# Get a new unit_descriptor table row for this new unit
					unit_description = table.row
					#continue
###########                
		else:
			continue
# =============================================================================
# 
# 		else:#if not split_merge:
# 			hf5.create_group('/sorted_units', unit_name)
# 			unit_waveforms = spike_waveforms[np.where(predictions == int(clusters[0]))[0], :]	# Waveforms of originally chosen cluster
# 			unit_waveforms = unit_waveforms[np.where(split_predictions == chosen_split[0])[0], :]	# Subsetting this set of waveforms to include only the chosen split
# 			unit_times = spike_times[np.where(predictions == int(clusters[0]))[0]]			# Do the same thing for the spike times
# 			unit_times = unit_times[np.where(split_predictions == chosen_split[0])[0]]
# 			waveforms = hf5.create_array('/sorted_units/%s' % unit_name, 'waveforms', unit_waveforms)
# 			times = hf5.create_array('/sorted_units/%s' % unit_name, 'times', unit_times)
# 			unit_description['electrode_number'] = electrode_num
# 			single_unit = easygui.multchoicebox(msg = 'I am almost-SURE that this is a beautiful single unit (True = Yes, False = No)', choices = ('True', 'False'))
# 			unit_description['single_unit'] = int(ast.literal_eval(single_unit[0]))
# 			# If the user says that this is a single unit, ask them whether its regular or fast spiking
# 			unit_description['regular_spiking'] = 0
# 			unit_description['fast_spiking'] = 0
# 			if int(ast.literal_eval(single_unit[0])):
# 				unit_type = easygui.multchoicebox(msg = 'What type of unit is this (Regular spiking = Pyramidal cells, Fast spiking = PV+ interneurons)?', choices = ('regular_spiking', 'fast_spiking'))
# 				unit_description[unit_type[0]] = 1		
# 			unit_description.append()
# 			table.flush()
# 			hf5.flush()
# 
# =============================================================================
		

	# If only 1 cluster was chosen (and it wasn't split), add that as a new unit in /sorted_units. Ask if the isolated unit is an almost-SURE single unit
	elif len(clusters) == 1:
		hf5.create_group('/sorted_units', unit_name)
		unit_waveforms = spike_waveforms[np.where(predictions == int(clusters[0]))[0], :]
		unit_times = spike_times[np.where(predictions == int(clusters[0]))[0]]
		waveforms = hf5.create_array('/sorted_units/%s' % unit_name, 'waveforms', unit_waveforms)
		times = hf5.create_array('/sorted_units/%s' % unit_name, 'times', unit_times)
		unit_description['electrode_number'] = electrode_num
		single_unit = easygui.multchoicebox(msg = 'I am almost-SURE that this is a beautiful single unit (True = Yes, False = No)', choices = ('True', 'False'))
		unit_description['single_unit'] = int(ast.literal_eval(single_unit[0]))
		# If the user says that this is a single unit, ask them whether its regular or fast spiking
		unit_description['regular_spiking'] = 0
		unit_description['fast_spiking'] = 0
		if int(ast.literal_eval(single_unit[0])):
			unit_type = easygui.multchoicebox(msg = 'What type of unit is this (Regular spiking = Pyramidal cells, Fast spiking = PV+ interneurons)?', choices = ('regular_spiking', 'fast_spiking'))
			unit_description[unit_type[0]] = 1
		unit_description.append()
		table.flush()
		hf5.flush()


	else:
		# If the chosen units are going to be merged, merge them
		if merge:
			unit_waveforms = []
			unit_times = []
			for cluster in clusters:
				if unit_waveforms == []:
					unit_waveforms = spike_waveforms[np.where(predictions == int(cluster))[0], :]			
					unit_times = spike_times[np.where(predictions == int(cluster))[0]]
				else:
					unit_waveforms = np.concatenate((unit_waveforms, spike_waveforms[np.where(predictions == int(cluster))[0], :]))
					unit_times = np.concatenate((unit_times, spike_times[np.where(predictions == int(cluster))[0]]))

			# Show the merged cluster to the user, and ask if they still want to merge
			x = np.arange(len(unit_waveforms[0])/10) + 1
			fig, ax = blech_waveforms_datashader.waveforms_datashader(unit_waveforms, x)
			# plt.plot(x - 15, unit_waveforms[:, ::10].T, linewidth = 0.01, color = 'red')
			ax.set_xlabel('Sample (30 samples per ms)')
			ax.set_ylabel('Voltage (microvolts)')
			ax.set_title('Merged cluster, No. of waveforms={:d}'.format(unit_waveforms.shape[0]))
			plt.show()
 
			# Warn the user about the frequency of ISI violations in the merged unit
			ISIs = np.ediff1d(np.sort(unit_times))/30.0
			violations1 = 100.0*float(np.sum(ISIs < 1.0)/len(unit_times))
			violations2 = 100.0*float(np.sum(ISIs < 2.0)/len(unit_times))
			proceed = easygui.multchoicebox(msg = 'My merged cluster has %.1f percent (<2ms) and %.1f percent (<1ms) ISI violations out of %i total waveforms. I want to still merge these clusters into one unit (True = Yes, False = No)' % (violations2, violations1, len(unit_times)), choices = ('True', 'False'))
			proceed = ast.literal_eval(proceed[0])

			# Create unit if the user agrees to proceed, else abort and go back to start of the loop 
			if proceed:	
				hf5.create_group('/sorted_units', unit_name)
				waveforms = hf5.create_array('/sorted_units/%s' % unit_name, 'waveforms', unit_waveforms)
				times = hf5.create_array('/sorted_units/%s' % unit_name, 'times', unit_times)
				unit_description['electrode_number'] = electrode_num
				single_unit = easygui.multchoicebox(msg = 'I am almost-SURE that this is a beautiful single unit (True = Yes, False = No)', choices = ('True', 'False'))
				unit_description['single_unit'] = int(ast.literal_eval(single_unit[0]))
				# If the user says that this is a single unit, ask them whether its regular or fast spiking
				unit_description['regular_spiking'] = 0
				unit_description['fast_spiking'] = 0
				if int(ast.literal_eval(single_unit[0])):
					unit_type = easygui.multchoicebox(msg = 'What type of unit is this (Regular spiking = Pyramidal cells, Fast spiking = PV+ interneurons)?', choices = ('regular_spiking', 'fast_spiking'))
					unit_description[unit_type[0]] = 1
				unit_description.append()
				table.flush()
				hf5.flush()
			else:
				continue

		# Otherwise include each cluster as a separate unit
		else:
			for cluster in clusters:
				hf5.create_group('/sorted_units', unit_name)
				unit_waveforms = spike_waveforms[np.where(predictions == int(cluster))[0], :]
				unit_times = spike_times[np.where(predictions == int(cluster))[0]]
				waveforms = hf5.create_array('/sorted_units/%s' % unit_name, 'waveforms', unit_waveforms)
				times = hf5.create_array('/sorted_units/%s' % unit_name, 'times', unit_times)
				unit_description['electrode_number'] = electrode_num
				single_unit = easygui.multchoicebox(msg = 'I am almost-SURE that electrode: %i cluster: %i is a beautiful single unit (True = Yes, False = No)' % (electrode_num, int(cluster)), choices = ('True', 'False'))
				unit_description['single_unit'] = int(ast.literal_eval(single_unit[0]))
				# If the user says that this is a single unit, ask them whether its regular or fast spiking
				unit_description['regular_spiking'] = 0
				unit_description['fast_spiking'] = 0
				if int(ast.literal_eval(single_unit[0])):
					unit_type = easygui.multchoicebox(msg = 'What type of unit is this (Regular spiking = Pyramidal cells, Fast spiking = PV+ interneurons)?', choices = ('regular_spiking', 'fast_spiking'))
					unit_description[unit_type[0]] = 1
				unit_description.append()
				table.flush()
				hf5.flush()				

				# Finally increment max_unit and create a new unit name
				max_unit += 1
				unit_name = 'unit%03d' % int(max_unit + 1)

				# Get a new unit_descriptor table row for this new unit
				unit_description = table.row

# Close the hdf5 file
hf5.close()
	 



	





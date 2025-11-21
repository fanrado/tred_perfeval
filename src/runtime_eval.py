import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

def get_runtime_perbatch(data: dict):
	"""
	Organize runtime evaluation data from a dictionary.

	Args:
		data (dict): Dictionary containing runtime evaluation data.
	Returns: lists of organized data.
		event_ids, N_segments, N_qblock, runtimes
	"""
	organized_data = {}
	batch_size = data['batch_size']
	nbchunk = data['nbchunk']
	nbchunk_conv = data['nbchunk_conv']
	del data['batch_size']
	del data['nbchunk']
	del data['nbchunk_conv']

	event_ids 				= []
	N_segments 				= []
	N_qblocks 				= []
	runtimes_perbatch 		= []
	## Process the TPC data
	for key, value in data.items():
		## TPC level
		for key1, value1 in value.items():
			if isinstance(value1, dict):
				## Batch level
				for key2, value2 in value1.items():
					if not isinstance(value2, dict):
						continue
					if value2['event_id'] != 1003:
						continue
					event_ids.append(value2['event_id'])
					N_segments.append(value2['N_segments'])
					N_qblocks.append(value2['N_qblock'])
					runtimes_perbatch.append(value2['runtime_sec']) ## runtime_sec replaces peak_memory_MB
	organized_data['event_ids'] 			= event_ids
	organized_data['N_segments'] 			= N_segments
	organized_data['N_qblocks'] 			= N_qblocks
	organized_data['runtimes_perbatch'] 	= runtimes_perbatch
	organized_data['batch_size'] 			= batch_size
	organized_data['nbchunk'] 			= nbchunk
	organized_data['nbchunk_conv'] 		= nbchunk_conv
	return organized_data

def overlay_plots(*args, title, xlabel, ylabel,output_file=''):
	plt.figure(figsize=(10, 8))
	hep.style.use("CMS") 
	filled_markers = ['.', '<', '+', '*', '^', '<', '8', 's', 'p', 'h', 'H', 'D', 'd', 'P', 'X']
	for i,(x_data, y_data, label, color) in enumerate(args):
		plt.scatter(x_data, y_data,label=label, color=color, alpha=0.7, marker=filled_markers[i], s=50)
		# coeffs = np.polyfit(x_data, y_data, 1)
		# poly_func = np.poly1d(coeffs)
		# x_fit = np.linspace(min(x_data), max(x_data), 100)
		# y_fit = poly_func(x_fit)
		# plt.plot(x_fit, y_fit, color=color, linestyle='--', alpha=0.7)
	plt.xlabel(xlabel, fontsize=20)
	plt.ylabel(ylabel, fontsize=20)
	plt.title(title, fontsize=20)
	plt.xticks(fontsize=20)
	plt.yticks(fontsize=20)
	plt.tight_layout()
	plt.legend(loc='upper left', fontsize=15)
	plt.grid(True)
	plt.savefig(output_file)
	plt.close()

def load_json(input_file=''):
	with open(input_file, 'r') as f:
		data = json.load(f)
	return data

def benchmark_runtime(input_path: str):
	"""
		Benchmark runtime evaluation from the input path.
	"""
	list_files 				= [f for f in os.listdir(input_path) if (f.endswith('.json')) and ('runtime_vs_Nsegments' not in f) and ('majorOps' not in f)]
	list_tuples 			= []
	colors 					= ['maroon', 'green', 'black', 'purple', 'red', 'blue', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
	for i, f in enumerate(list_files):
		file_path = '/'.join([input_path, f])
		f_split = f.split('.')[0].split('_')[3:]
		data = load_json(file_path)
		organized_data 	 	= get_runtime_perbatch(data=data)
		# label 				= f'Batch size: {organized_data["batch_size"]}, \nnbchunk: {organized_data["nbchunk"]}, nbchunk_conv: {organized_data["nbchunk_conv"]}'
		label 				= f'nbchunk: {organized_data["nbchunk"]}, nbchunk_conv: {organized_data["nbchunk_conv"]}'
		# tuple_data 			= (organized_data['N_segments'], organized_data['runtimes_perbatch'], label, colors[i])
		x_data = organized_data['N_segments']
		y_data = organized_data['runtimes_perbatch']
		sort_indices = np.argsort(x_data)
		x_sorted = np.array(x_data)[sort_indices]
		y_sorted = np.array(y_data)[sort_indices]
		tuple_data = (x_sorted, y_sorted, label, colors[i])
		list_tuples.append(tuple_data)
	## Save data in a json file
	data_dict = {}
	for x_data, y_data, label, color in list_tuples:
		data_dict[label] = {
			'N_segments' : x_data.tolist(),
			'runtimes_perbatch' : y_data.tolist()
		}
	output_json_file = '/'.join([input_path, 'runtime_vs_Nsegments.json'])
	with open(output_json_file, 'w') as f:
		json.dump(data_dict, f)
	output_file = '/'.join([input_path, 'runtime_vs_Nsegments.png'])
	overlay_plots(*list_tuples, title='Runtime vs N_segments', xlabel='N segments', ylabel='Runtime (sec)', output_file=output_file)

def get_runtime_majorOperations(data: dict):
	"""
		Get the runtime of major operations from the data dictionary. The output of this function will be used to draw the pie chart of the runtime.
		For now, let's assume the format of the runtime dict is similar to peak memory dict.
		The variables of interests are those corresponding to each_operation
			-- recombination
			-- drift
			-- chunking_conv:
			    ** ichunk_0:
				  	** rasterize
					** chunksum_qblock
					** conv
						** total runtime
						** details:
						    ** ichunk_conv0:
								** conv
								** chunksum_i
							...
					** sumcurrent
				...
			-- sum_current
			-- chunksum_readout
			-- concat_readout (concatenate waveforms)
			-- formingBlock_readout_current
	"""
	organized_data = {}
	batch_size = data['batch_size']
	nbchunk = data['nbchunk']
	nbchunk_conv = data['nbchunk_conv']
	del data['batch_size']
	del data['nbchunk']
	del data['nbchunk_conv']
	recombination 		= []
	drift 				= []
	chunking_conv 		= {
		'rasterize' 		: [],
		'chunksum_qblock' 	: [],
		'conv_conv'			: [],
		'conv_chunksum_i'	: [],
		'sumcurrent'		: []
	}
	sum_current 		= []
	chunksum_readout 	= []
	concat_readout 		= []
	formingBlock_readout_current = []
	## Process the TPC data
	for keytpc, tpcdata in data.items():
		# TPC level
		for keybatch, batch_data in tpcdata.items():
			## Batch level
			# print(keybatch)
			if not isinstance(batch_data, dict):
				continue
			for key, value in batch_data.items():
				# replace _MB with _sec
				each_operation_runtime = value['each_operation_sec'] ## replace each_operation_MB with each_operation_sec once the data is ready
				recombination.append(each_operation_runtime['recomb_sec'])
				drift.append(each_operation_runtime['drifter_sec'])
				sum_current.append(each_operation_runtime['sum_current_sec'])
				chunksum_readout.append(each_operation_runtime['chunksum_readout_sec'])
				concat_readout.append(each_operation_runtime['concat_readout_sec']) # concatenating waveforms
				formingBlock_readout_current.append(each_operation_runtime['formingBlock_readout_current_sec'])
				chunking_conv_data = each_operation_runtime['chunking_conv']
				for ichunk, chunkdata in chunking_conv_data.items():
					chunking_conv['rasterize'].append(chunkdata['raster_sec'])
					chunking_conv['chunksum_qblock'].append(chunkdata['chunksum_qblock_sec'])
					chunking_conv['sumcurrent'].append(chunkdata['sumcurrent_sec'])
					conv_data = chunkdata['conv_sec']
					for keyconv, valconv in conv_data.items():
						if not isinstance(valconv, dict):
							continue
						## Convolution level
						for key1, value1 in valconv.items():
							chunking_conv['conv_conv'].append(value1['conv_time'])
							chunking_conv['conv_chunksum_i'].append(value1['chunksum_i_time'])
	organized_data['batch_size'] 					= batch_size
	organized_data['nbchunk'] 						= nbchunk
	organized_data['nbchunk_conv'] 					= nbchunk_conv
	#
	organized_data['recombination'] 				= recombination
	organized_data['drift'] 						= drift
	organized_data['chunking_conv'] 				= chunking_conv
	## Recall. This is needed especially for the function runtimeshare_majorOp below.
	# chunking_conv 		= {
	# 	'rasterize' 		: [],
	# 	'chunksum_qblock' 	: [],
	# 	'conv_conv'			: [],
	# 	'conv_chunksum_i'	: [],
	# 	'sumcurrent'		: []
	# }
	organized_data['sum_current'] 					= sum_current
	organized_data['chunksum_readout'] 				= chunksum_readout
	organized_data['concat_readout'] 				= concat_readout
	organized_data['formingBlock_readout_current'] 	= formingBlock_readout_current

	# ## Calculate major operations runtime
	# Electronic_readout = np.sum(organized_data['chunksum_readout']) + np.sum(organized_data['concat_readout']) + np.sum(organized_data['formingBlock_readout_current'])
	# Induced_current_calculation = np.sum( organized_data['chunking_conv']['conv_chunksum_i'] ) + np.sum( organized_data['chunking_conv']['conv_conv'] ) + np.sum( organized_data['chunking_conv']['sumcurrent'] ) + np.sum( organized_data['sum_current'] )
	# Rasterization_of_ionization_charges = np.sum( organized_data['chunking_conv']['chunksum_qblock'] ) + np.sum( organized_data['chunking_conv']['rasterize'] )
	# Recombination_attenuation_and_drift = np.sum( organized_data['recombination'] ) + np.sum( organized_data['drift'] )

	# major_operations_runtime = {
	# 	'Electronic_readout' : Electronic_readout,
	# 	'Induced_current_calculation' : Induced_current_calculation,
	# 	'Rasterization_of_ionization_charges' : Rasterization_of_ionization_charges,
	# 	'Recombination_attenuation_and_drift' : Recombination_attenuation_and_drift
	# }
	## Calculate major operations runtime
	Electronic_readout = np.sum(organized_data['chunksum_readout']) + np.sum(organized_data['concat_readout']) + np.sum(organized_data['formingBlock_readout_current'])
	# Induced_current_calculation = np.sum( organized_data['chunking_conv']['conv_chunksum_i'] ) + np.sum( organized_data['chunking_conv']['conv_conv'] ) + np.sum( organized_data['chunking_conv']['sumcurrent'] ) + np.sum( organized_data['sum_current'] )
	Convolution = np.sum(  organized_data['chunking_conv']['conv_conv'] )
	Chunksum_current = np.sum( organized_data['chunking_conv']['conv_chunksum_i'] ) + np.sum( organized_data['chunking_conv']['sumcurrent'] ) + np.sum( organized_data['sum_current'] )
	# Rasterization_of_ionization_charges = np.sum( organized_data['chunking_conv']['chunksum_qblock'] ) + np.sum( organized_data['chunking_conv']['rasterize'] )
	Chunksum_qblock = np.sum( organized_data['chunking_conv']['chunksum_qblock'] )
	Rasterization = np.sum( organized_data['chunking_conv']['rasterize'] )
	Recombination_attenuation_and_drift = np.sum( organized_data['recombination'] ) + np.sum( organized_data['drift'] )

	major_operations_runtime = {
		'Electronic_readout' : Electronic_readout,
		'Convolution' : Convolution,
		'Chuksum and current accumulation' : Chunksum_current,
		# 'Induced_current_calculation' : Induced_current_calculation,
		# 'Rasterization_of_ionization_charges' : Rasterization_of_ionization_charges,
		'Chunksum on charge blocks' : Chunksum_qblock,
		'Rasterization' : Rasterization,
		'Recombination_attenuation_and_drift' : Recombination_attenuation_and_drift
	}
	return major_operations_runtime

def runtimeshare_majorOp(organized_data: dict, output_file=''):
	"""
		This function generate a piechart of the runtime share of major operations.
		Args:
			organized_data (dict): output of get_runtime_majorOperations function.
			output_file (str): output file name for the pie chart.
		Major operations:
			-- Electronic readout: prepare wf i/io (not here because we don't touch anything related to output wf here)
			                      + chunksum readout
								  + concat readout
								  + formingBlock readout current
			-- Induced current calculation: conv_chunksum_i + conv_conv + sumcurrent + sum_current (current accumulations)
			-- Rasterization of ionization charges: chunksum_qblock + rasterize
			-- Recombination, attenuation, and drift: recombination + drift
	"""
	# Electronic_readout = np.sum(organized_data['chunksum_readout']) + np.sum(organized_data['concat_readout']) + np.sum(organized_data['formingBlock_readout_current'])
	# Induced_current_calculation = np.sum( organized_data['chunking_conv']['conv_chunksum_i'] ) + np.sum( organized_data['chunking_conv']['conv_conv'] ) + np.sum( organized_data['chunking_conv']['sumcurrent'] ) + np.sum( organized_data['sum_current'] )
	# Rasterization_of_ionization_charges = np.sum( organized_data['chunking_conv']['chunksum_qblock'] ) + np.sum( organized_data['chunking_conv']['rasterize'] )
	# Recombination_attenuation_and_drift = np.sum( organized_data['recombination'] ) + np.sum( organized_data['drift'] )
	Electronic_readout = organized_data['Electronic_readout']
	Induced_current_calculation = organized_data['Induced_current_calculation']
	Rasterization_of_ionization_charges = organized_data['Rasterization_of_ionization_charges']
	Recombination_attenuation_and_drift = organized_data['Recombination_attenuation_and_drift']

	labels = ['Electronic readout', 'Induced current calculation', 'Rasterization of ionization charges', 'Recombination, attenuation, and drift']
	sizes = [Electronic_readout, Induced_current_calculation, Rasterization_of_ionization_charges, Recombination_attenuation_and_drift]
	colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
	explode = (0.05, 0.05, 0.05, 0.05)  # explode all slices slightly
	plt.figure(figsize=(8, 8))
	wedges, texts, autotexts = plt.pie(sizes, explode=explode, labels=None, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=140, pctdistance=1.1)
	plt.title('Runtime Distribution of Major Operations', fontsize=16)
	plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
	plt.legend(wedges, labels, title="Operations", loc="upper right", bbox_to_anchor=(1.0, 1.0))
	plt.tight_layout()
	plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
	plt.savefig(output_file)
	plt.close()

def runtimeshare_majorOp_details(organized_data: dict, output_file=''):
	"""
		This function generate a piechart of the runtime share of major operations.
		Args:
			organized_data (dict): output of get_runtime_majorOperations function.
			output_file (str): output file name for the pie chart.
		Major operations:
			-- Electronic readout: prepare wf i/io (not here because we don't touch anything related to output wf here)
			                      + chunksum readout
								  + concat readout
								  + formingBlock readout current
			-- Induced current calculation: conv_chunksum_i + conv_conv + sumcurrent + sum_current (current accumulations)
			-- Rasterization of ionization charges: chunksum_qblock + rasterize
			-- Recombination, attenuation, and drift: recombination + drift
	"""
	# Electronic_readout = np.sum(organized_data['chunksum_readout']) + np.sum(organized_data['concat_readout']) + np.sum(organized_data['formingBlock_readout_current'])
	# Induced_current_calculation = np.sum( organized_data['chunking_conv']['conv_chunksum_i'] ) + np.sum( organized_data['chunking_conv']['conv_conv'] ) + np.sum( organized_data['chunking_conv']['sumcurrent'] ) + np.sum( organized_data['sum_current'] )
	# Rasterization_of_ionization_charges = np.sum( organized_data['chunking_conv']['chunksum_qblock'] ) + np.sum( organized_data['chunking_conv']['rasterize'] )
	# Recombination_attenuation_and_drift = np.sum( organized_data['recombination'] ) + np.sum( organized_data['drift'] )
	Electronic_readout = organized_data['Electronic_readout']
	# Induced_current_calculation = organized_data['Induced_current_calculation']
	Convolution = organized_data['Convolution']
	Chunksum_current = organized_data['Chuksum and current accumulation']
	# Rasterization_of_ionization_charges = organized_data['Rasterization_of_ionization_charges']
	Chunksum_qblock = organized_data['Chunksum on charge blocks']
	Rasterization = organized_data['Rasterization']
	Recombination_attenuation_and_drift = organized_data['Recombination_attenuation_and_drift']

	labels = ['Electronic readout', 'Convolution', 'Chunksum current and accumulation','Chunksum of charge blocks', 'Rasterization', 'Recombination, attenuation, and drift']
	sizes = [Electronic_readout, Convolution, Chunksum_current, Chunksum_qblock, Rasterization, Recombination_attenuation_and_drift]
	colors = ['#ff9999','#66b3ff','#99ff99',"#e7c907", "#036515", "#8B005A"]
	explode = (0.05, 0.05, 0.05, 0.05, 0.05, 0.05)  # explode all slices slightly
	plt.figure(figsize=(12,10))
	plt.subplots_adjust(right=0.7)
	wedges, texts, autotexts = plt.pie(sizes, explode=explode, labels=None, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=140, pctdistance=1.1)
	plt.title('Runtime Distribution of Major Operations', fontsize=20)
	plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
	plt.legend(wedges, labels, title="Operations", loc="upper right", bbox_to_anchor=(0.6, 0.5), fontsize=18)
	plt.tight_layout()
	plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
	plt.savefig(output_file)
	plt.close()
if __name__=='__main__':
	pass

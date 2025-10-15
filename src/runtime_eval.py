import os, sys
import numpy as np
import matplotlib.pyplot as plt

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

def overlay_plots(data_tuple, title, xlabel, ylabel,output_file=''):
	plt.figure(figsize=(10, 6))
	for (x_data, y_data, label, color) in data_tuple:
		plt.scatter(x_data, y_data, label=label, color=color, alpha=0.7)
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.legend()
	plt.grid(True)
	plt.savefig(output_file)
	plt.close()

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
				each_operation_runtime = value['each_operation_MB'] ## replace each_operation_MB with each_operation_sec once the data is ready
				recombination.append(each_operation_runtime['recomb_MB'])
				drift.append(each_operation_runtime['drifter_MB'])
				sum_current.append(each_operation_runtime['sum_current_MB'])
				chunksum_readout.append(each_operation_runtime['chunksum_readout_MB'])
				concat_readout.append(each_operation_runtime['concat_readout_MB'])
				formingBlock_readout_current.append(each_operation_runtime['formingBlock_readout_current_MB'])
				chunking_conv_data = each_operation_runtime['chunking_conv']
				for ichunk, chunkdata in chunking_conv_data.items():
					chunking_conv['rasterize'].append(chunkdata['raster_MB'])
					chunking_conv['chunksum_qblock'].append(chunkdata['chunksum_qblock_MB'])
					chunking_conv['sumcurrent'].append(chunkdata['sumcurrent_MB'])
					conv_data = chunkdata['conv_MB']
					for keyconv, valconv in conv_data.items():
						if not isinstance(valconv, dict):
							continue
						## Convolution level
						for key1, value1 in valconv.items():
							chunking_conv['conv_conv'].append(value1['conv_MB'])
							chunking_conv['conv_chunksum_i'].append(value1['chunksum_i_MB'])
	organized_data['batch_size'] 					= batch_size
	organized_data['nbchunk'] 						= nbchunk
	organized_data['nbchunk_conv'] 					= nbchunk_conv
	organized_data['recombination'] 				= recombination
	organized_data['drift'] 						= drift
	organized_data['chunking_conv'] 				= chunking_conv
	organized_data['sum_current'] 					= sum_current
	organized_data['chunksum_readout'] 				= chunksum_readout
	organized_data['concat_readout'] 				= concat_readout
	organized_data['formingBlock_readout_current'] 	= formingBlock_readout_current
	return organized_data

if __name__=='__main__':
	pass

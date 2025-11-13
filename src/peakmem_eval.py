import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

def organize_peakmem_data(data: dict):
	"""
	Organize peak memory data from a dictionary.

	Args:
		data (dict): Dictionary containing peak memory data.
	Returns: lists of organized data.
		event_ids, N_segments, N_qblock, peak_memory
	"""
	organized_data 			= {}
	batch_size 				= data['batch_size']
	nbchunk 				= data['nbchunk']
	nbchunk_conv = data['nbchunk_conv']
	params_for_dynamic_chunking_batching = {
		'mem_limit_MB': None,
		'shape_limit': None,
		'xyz_limit': None
	}
	del data['batch_size']
	del data['nbchunk']
	del data['nbchunk_conv']

	event_ids 				= []
	N_segments 				= []
	N_qblocks 				= []
	peak_memory_perbatch 	= []
	## Process the TPC data
	for key, value in data.items():
		## TPC level
		for key1, value1 in value.items():
			if isinstance(value1, dict):
				## Batch level
				# print(value1['batch_label2'].keys())
				# sys.exit()
				for key2, value2 in value1.items():
					if not isinstance(value2, dict):
						continue
					# print(value2.keys())
					for key in ['mem_limit_MB', 'shape_limit', 'xyz_limit']:
						if params_for_dynamic_chunking_batching[key] is None:
							try:
								params_for_dynamic_chunking_batching[key] = value2[key]
							except:
								# print(f'key {key} not found in the data {value}')
								pass
					event_ids.append(value2['event_id'])
					N_segments.append(value2['N_segments'])
					N_qblocks.append(value2['N_qblock'])
					peak_memory_perbatch.append(value2['peak_memory_MB'])
	organized_data['event_ids'] 				= event_ids
	organized_data['N_segments'] 				= N_segments
	organized_data['N_qblocks'] 				= N_qblocks
	for key, value in params_for_dynamic_chunking_batching.items():
		organized_data[key] = value
	organized_data['peak_memory_perbatch'] 		= peak_memory_perbatch
	organized_data['batch_size'] 				= batch_size
	organized_data['nbchunk'] 					= nbchunk
	organized_data['nbchunk_conv'] 				= nbchunk_conv
	return organized_data

def plot_peakmem_vs_nsegs(n_segments, peak_memory, title):
	"""
	Plot peak memory vs number of segments For one batch scheme.

	Args:
		n_segments (list): List of number of segments.
		peak_memory (list): List of peak memory usage.
		title (str): Title for the plot.
	"""
	plt.figure(figsize=(10,6))
	hep.style.use("CMS") 
	plt.scatter(n_segments, peak_memory, color='blue', alpha=0.7)
	plt.xlabel('Number of Segments')
	plt.ylabel('Peak Memory Usage (MB)')
	plt.title(title)
	plt.grid(True)
	plt.savefig('peak_memory_vs_n_segments.png')
	plt.close()

def overlay_plots(*args, title='', xlabel='', ylabel='', output_file='overlay_plot.png', nodynChunkBatch=True):
	"""
	Overlay multiple plots on the same figure. One needs to provide in the title whether the dynamic chunking and batching are used.
	If a comparison between with/without dynamic batching or chunking is made, it should be mentioned in the label.

	Args:
		*args: Tuples of (x_data, y_data, label, color) for each plot.
		title (str): Title for the plot.
		xlabel (str): Label for the x-axis.
		ylabel (str): Label for the y-axis.
		output_file (str): Filename to save the plot.
	"""
	plt.figure(figsize=(12,12))
	hep.style.use("CMS") 
	i = 1
	filled_markers = ['.', 'o', 'v', '*', '^', '<', '8', 's', 'p', 'h', 'H', 'D', 'd', 'P', 'X']
	j = 0
	for x_data, y_data, label, color in args:
		if nodynChunkBatch:
			plt.scatter(x_data, y_data, label=label, color=color, alpha=0.4, s=100, marker=filled_markers[i])
		else:
			plt.scatter(x_data, y_data, label=label, color=color, alpha=0.4, s=100, marker=filled_markers[j])
		i += 2
		j+=1
	plt.xlabel(xlabel, fontsize=20)
	plt.ylabel(ylabel, fontsize=20)
	plt.title(title, fontsize=20)
	plt.xticks(fontsize=20)
	plt.yticks(fontsize=20)
	plt.tight_layout()
	plt.legend(loc='upper right', fontsize=15)
	plt.grid(True)
	plt.savefig(output_file)
	plt.close()

def load_json(file_path: str):
	"""
	Load JSON data from a file.

	Args:
		file_path (str): Path to the JSON file.
	Returns:
		dict: Loaded JSON data.
	"""
	with open(file_path, 'r') as f:
		data = json.load(f)
	return data

def benchmark_peak_memory_nodynamic_chunking_and_batching(input_path: str):
	"""
	Benchmark peak memory usage without dynamic chunking and batching.
	"""
	list_of_files = [f for f in os.listdir(input_path) if f.endswith('.json')]
	list_of_tuple_data = []
	colors = ['red', 'green', 'black', 'purple', 'yellow', 'maroon']
	title = None
	for i,f in enumerate(list_of_files):
		if f=='peak_memory_usage_batch8192_nbchunk50_nbchunkconv10.json':
			continue
		file_path = '/'.join([input_path, f])
		f_split = f.split('.')[0].split('_')[3:]
		if title is None:
			title = f'Batch size: {f_split[0].replace("batch","")}'
		label = f'nbchunk: {f_split[1].replace("nbchunk","")}, nbchunk_conv: {f_split[2].replace("nbchunkconv","")}'
		data = load_json(file_path)
		organized_data = organize_peakmem_data(data=data)
		tuple_data = ()
		# if len(f) > len(colors):
		# 	tuple_data = (organized_data['N_segments'], organized_data['peak_memory_perbatch'], label, np.random.rand(3,))
		# else:
		tuple_data = (organized_data['N_segments'], organized_data['peak_memory_perbatch'], label, colors[i])
		list_of_tuple_data.append(tuple_data)
	## Save the data in a json
	data_dict = {}
	for x_data, y_data, label, color in list_of_tuple_data:
		data_dict[label] = {
			'N_segments': x_data,
			'peak_memory_perbatch': y_data
		}
	output_file_json = '/'.join([input_path, 'peakmem_vs_Nsegments_nodynamic_chunking_batching.json'])
	with open(output_file_json, 'w') as f:
		json.dump(data_dict, f, indent=4)

	overlay_plots(*list_of_tuple_data, title=f'Peak memory vs N_segments, {title}', xlabel='N segments', ylabel='Peak Memory (MB)', output_file='/'.join([input_path, 'peakmem_vs_Nsegments_nodynamic_chunking_batching.png']), nodynChunkBatch=True)

def benchmark_peak_memory_dynamic_chunking_and_batching(input_path: str):
	"""
	  Benchmark peak memory usage with dynamic chunking and batching enabled.
	"""
	list_of_files 			= [f for f in os.listdir(input_path) if f.endswith('.json')]
	list_of_tuple_data 		= []
	colors 					= ['red', 'green', 'blue', 'purple', 'maroon', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
	# if len(list_of_files) > len(colors):
	# 	colors = []
	# 	for i in range(len(list_of_files)):
	# 		colors.append(np.random.rand(3,))
	for i, f in enumerate(list_of_files):
		# if 'xyzLimit10_' in f:
		# 	continue
		print(f'Processing file: {f}')
		file_path = '/'.join([input_path, f])
		f_split = f.split('.')[0].split('_')[3:]
		data = load_json(file_path)
		organized_data 	 	= organize_peakmem_data(data=data)
		# label 				= f'mem_limit : {organized_data['mem_limit_MB']/1024} GB, \nshape_limit : {np.format_float_scientific(organized_data['shape_limit'], precision=4)}, xyz_limit : {np.format_float_scientific(organized_data['xyz_limit'], precision=4)}'
		label 				= f'xyz_limit : {organized_data['xyz_limit']}'
		# if organized_data['mem_limit_MB'] < 10*1024:
		# 	continue
		
		tuple_data 			= (organized_data['N_segments'], organized_data['peak_memory_perbatch'], label, colors[i])
		list_of_tuple_data.append(tuple_data)
	output_file = '/'.join([input_path, 'peakmem_vs_Nsegments_dynamic_chunking_batching.png'])
	# overlay_plots(*list_of_tuple_data, title='Peak memory vs N_segments (with dynamic chunking and batching)', xlabel='N segments', ylabel='Peak Memory (MB)', output_file=output_file)
	overlay_plots(*list_of_tuple_data, title='Peak memory vs N_segments', xlabel='N segments', ylabel='Peak Memory (MB)', output_file=output_file, nodynChunkBatch=False)

if __name__=='__main__':
	pass

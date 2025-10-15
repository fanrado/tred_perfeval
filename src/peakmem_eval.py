import os, sys
import numpy as np
import matplotlib.pyplot as plt

def organize_peakmem_data(data: dict):
	"""
	Organize peak memory data from a dictionary.

	Args:
		data (dict): Dictionary containing peak memory data.
	Returns: lists of organized data.
		event_ids, N_segments, N_qblock, peak_memory
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
					event_ids.append(value2['event_id'])
					N_segments.append(value2['N_segments'])
					N_qblocks.append(value2['N_qblock'])
					peak_memory_perbatch.append(value2['peak_memory_MB'])
	organized_data['event_ids'] 				= event_ids
	organized_data['N_segments'] 				= N_segments
	organized_data['N_qblocks'] 				= N_qblocks
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
	plt.scatter(n_segments, peak_memory, color='blue', alpha=0.7)
	plt.xlabel('Number of Segments')
	plt.ylabel('Peak Memory Usage (MB)')
	plt.title(title)
	plt.grid(True)
	plt.savefig('peak_memory_vs_n_segments.png')
	plt.close()

def overlay_plots(*args, title='', xlabel='', ylabel='', output_file='overlay_plot.png'):
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
	plt.figure(figsize=(10,6))
	for x_data, y_data, label, color in args:
		plt.scatter(x_data, y_data, label=label, color=color)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.legend()
	plt.grid(True)
	plt.savefig(output_file)
	plt.close()

if __name__=='__main__':
	pass

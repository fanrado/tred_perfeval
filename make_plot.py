import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

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

## Memory consumption
def overlay_plots_mem(*args, title='', xlabel='', ylabel='', output_file='overlay_plot.png', nodynChunkBatch=True):
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

def benchmark_peak_memory_nodynamic_chunking_and_batching(path_to_file: str=''):
	data = load_json(path_to_file) ## load peakmem_vs_Nsegments_nodynamic_chunking_batching.json
	colors = ['red', 'green', 'black', 'purple', 'blue', 'maroon']
	list_of_tuple_data = []
	for i, key in enumerate(data.keys()):
		x_data = np.array(data[key]['N_segments'])
		y_data = np.array(data[key]['peak_memory_perbatch'])
		label = key
		color = colors[i]
		list_of_tuple_data.append((x_data, y_data, label, color))
	output_file = path_to_file.replace('.json', '.png')
	overlay_plots_mem(*list_of_tuple_data, title='Peak Memory vs N_segments', xlabel='N segments', ylabel='Peak Memory (MB)', output_file=output_file, nodynChunkBatch=True)

## Runtime evaluation
def overlay_plots_runtime(*args, title, xlabel, ylabel,output_file=''):
	plt.figure(figsize=(10, 8))
	hep.style.use("CMS") 
	filled_markers = ['.', '<', '+', '*', '^', '<', '8', 's', 'p', 'h', 'H', 'D', 'd', 'P', 'X']
	for i,(x_data, y_data, label, color) in enumerate(args):
		plt.scatter(x_data, y_data,label=label, color=color, alpha=0.7, marker=filled_markers[i], s=50)
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

def benchmark_runtime(path_to_file: str=''):
	data = load_json(path_to_file) ## load runtime_vs_Nsegments.json
	list_of_tuple_data = []
	colors 					= ['maroon', 'green', 'black', 'purple', 'red', 'blue', 'orange']
	for i, key in enumerate(data.keys()):
		x_data = np.array(data[key]['N_segments'])
		y_data = np.array(data[key]['runtimes_perbatch'])
		label = key
		color = colors[i]
		list_of_tuple_data.append((x_data, y_data, label, color))
	output_file = path_to_file.replace('.json', '.png')
	overlay_plots_runtime(*list_of_tuple_data, title='Runtime vs N_segments', xlabel='N segments', ylabel='Runtime (sec)', output_file=output_file)

def runtimeshare_majorOp(path_to_file: str=''):
	data = load_json(path_to_file) ## load runtime_majorOps_batchsize8192_NBCHUNK100_NBCHUNKCONV50.json
	Electronic_readout = data['Electronic_readout']
	Induced_current_calculation = data['Induced_current_calculation']
	Rasterization_of_ionization_charges = data['Rasterization_of_ionization_charges']
	Recombination_attenuation_and_drift = data['Recombination_attenuation_and_drift']

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
	plt.savefig(path_to_file.replace('.json', '.png'))
	plt.close()

## Effective charge evaluation
def overlay_hists_deltaQ(*deltaQ_list, title, xlabel, ylabel, output_file=''):
	"""
		Overlay the distributions of the deltaQ = data - ref.
		Args:
		    *deltaQ_list: a list of tuples, each tuple contains (deltaQ_array, label, color)
			title (str): Title of the plot.
			xlabel (str): Label for the x-axis.
			ylabel (str): Label for the y-axis.
			output_file (str): Filename to save the plot.
	"""
	plt.figure(figsize=(10, 6))
	for (deltaQ, label, color, linestyle) in deltaQ_list:
		## concatenate the accumulated charge from all TPCs
		all_tpcs_deltaQ = np.array([], dtype=np.float32)
		for itpc in deltaQ.keys():
			all_tpcs_deltaQ = np.concatenate((all_tpcs_deltaQ, deltaQ[itpc]), axis=0)

		if linestyle is None:
			linestyle='-'
		plt.hist(all_tpcs_deltaQ, bins=100, histtype='step', color=color, label=label, linewidth=1.5, linestyle=linestyle)
		# ax[1].hist(all_tpcs_deltaQ, bins=100, histtype='step', color=color, label=label)
		# plt.hist(deltaQ, bins=100, range=(-1000, 1000), histtype='step', color=color, label=label)
	plt.xlabel(xlabel, fontsize=18)
	plt.ylabel(ylabel, fontsize=18)
	plt.yscale('log')
	# plt.xlim([-2, 5])
	# plt.xlim([-0.15, 0])
	plt.xticks(fontsize=18)
	plt.yticks(fontsize=18)
	plt.title(title, fontsize=18)
	plt.grid(True)
	plt.legend(fontsize=14)
	plt.tight_layout()
	plt.savefig(output_file)
	plt.close()

def effq_evaluation(path_to_file: str=''):
	data = load_json(path_to_file) ## load effq_accuracy_data.json
	colors = ['maroon', 'green', 'red', 'purple', 'black', 'blue', 'orange']
	list_of_deltaQ_tuple = []
	list_of_dQ_over_Q_tuple = []
	for i, key in enumerate(data.keys()):
		deltaQ_data = data[key]['deltaQ']
		dQ_over_Q = data[key]['dQ_over_Q']
		label = key
		color = colors[i]
		list_of_deltaQ_tuple.append((deltaQ_data, label, color, None))
		list_of_dQ_over_Q_tuple.append((dQ_over_Q, label, color, None))
	output_file_deltaQ = path_to_file.replace('.json', '_deltaQ.png')
	overlay_hists_deltaQ(*list_of_deltaQ_tuple, title='Delta Q Distribution', xlabel='DeltaQ (ke-)', ylabel='Counts', output_file=output_file_deltaQ)
	output_file_dQ_over_Q = path_to_file.replace('.json', '_dQ_over_Q.png')
	overlay_hists_deltaQ(*list_of_dQ_over_Q_tuple, title='dQ/Q Distribution', xlabel='dQ/Q', ylabel='Counts', output_file=output_file_dQ_over_Q)

if __name__ == "__main__":
	## Peak memory evaluation without dynamic chunking and batching
	input_path_peakmem = 'data4plots/peakmem_vs_Nsegments_nodynamic_chunking_batching.json'  ## path to peakmem_vs_Nsegments_nodynamic_chunking_batching.json
	benchmark_peak_memory_nodynamic_chunking_and_batching(input_path_peakmem)

	## Runtime evaluation
	input_path_runtime = 'data4plots/runtime_vs_Nsegments.json'  ## path to runtime_vs_Nsegments.json
	# benchmark_runtime(input_path_runtime)

	input_path_runtime = 'data4plots/runtime_majorOps_batchsize8192_NBCHUNK100_NBCHUNKCONV50.json'  ## path to runtime_majorOps_batchsize8192_NBCHUNK100_NBCHUNKCONV50.json
	runtimeshare_majorOp(input_path_runtime)

	## Effective charge evaluation
	path_to_file = 'data4plots/effq_accuracy_data_effq_out_nt_1.json'
	effq_evaluation(path_to_file)
	
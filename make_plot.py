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
	filled_markers = ['.', 'p', 'v', '*', '^', '<', '8', 's', 'p', 'h', 'H', 'D', 'd', 'P', 'X']
	j = 0
	for x_data, y_data, label, color in args:
		if nodynChunkBatch:
			plt.scatter(x_data, y_data, label=label, color=color, alpha=0.4, s=100, marker=filled_markers[j])
		else:
			plt.scatter(x_data, y_data, label=label, color=color, alpha=0.4, s=100, marker=filled_markers[j])
		i += 2
		j+=1
	plt.xlabel(xlabel, fontsize=24)
	plt.ylabel(ylabel, fontsize=24)
	plt.title(title, fontsize=24)
	plt.ylim([-500, 20000])
	plt.xticks(fontsize=24)
	plt.yticks(fontsize=24)
	
	plt.legend(loc='best', fontsize=24)
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(output_file, dpi=300, bbox_inches='tight')
	plt.close()

def benchmark_peak_memory_nodynamic_chunking_and_batching(path_to_file: str=''):
	map_key = {
		'nbchunk': 'nbchunk',
		'nbchunk_conv': 'nbchunk_conv',
	}
	data = load_json(path_to_file) ## load peakmem_vs_Nsegments_nodynamic_chunking_batching.json
	colors = ['red', 'green', 'black', 'blue', 'maroon', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
	list_of_tuple_data = []
	j = 0
	for i, key in enumerate(data.keys()):
		x_data = np.array(data[key]['N_segments'])
		y_data = np.array(data[key]['peak_memory_perbatch'])
		tmp_label = key.split(',')
		label = f'{map_key['nbchunk']} : {tmp_label[0].split(':')[1]}, {map_key['nbchunk_conv']} : {tmp_label[1].split(':')[1]}'
		print(label,'\t', f'{map_key['nbchunk']} : 100, {map_key['nbchunk_conv']} : 10')
		# if ('300' in tmp_label[0]) or (label==f'{map_key['nbchunk']} :  100, {map_key['nbchunk_conv']} :  10'):
		# 	continue
		if (label==f'{map_key['nbchunk']} :  10, {map_key['nbchunk_conv']} :  10') or (label==f'{map_key['nbchunk']} :  300, {map_key['nbchunk_conv']} :  10') or (label==f'{map_key['nbchunk']} :  50, {map_key['nbchunk_conv']} :  50'):
			continue
		color = colors[j]
		list_of_tuple_data.append((x_data, y_data, label, color))
		j += 1
	output_file = path_to_file.replace('.json', '.png')
	overlay_plots_mem(*list_of_tuple_data, title='Peak Memory as a function of N_segments', xlabel='N segments', ylabel='Peak Memory (MB)', output_file=output_file, nodynChunkBatch=True)
	return list_of_tuple_data

## Runtime evaluation
def overlay_plots_runtime(*args, title, xlabel, ylabel,output_file=''):
	plt.figure(figsize=(12, 9))
	hep.style.use("CMS") 
	filled_markers = ['.', 'p', 'v', '*', '^', '<', '8', 's', 'p', 'h', 'H', 'D', 'd', 'P', 'X']
	for i,(x_data, y_data, label, color) in enumerate(args):
		plt.scatter(x_data, y_data,label=label, color=color, alpha=0.5, marker=filled_markers[i], s=100)
	plt.xlabel(xlabel, fontsize=24)
	plt.ylabel(ylabel, fontsize=24)
	plt.title(title, fontsize=24)
	# plt.ylim([-1, 8])
	plt.xticks(fontsize=24)
	plt.yticks(fontsize=24)
	plt.tight_layout()
	plt.legend(loc='upper left', fontsize=24)
	plt.grid(True)
	plt.savefig(output_file, dpi=300)
	plt.close()

def benchmark_runtime(path_to_file: str=''):
	map_key = {
		'nbchunk': 'nbchunk',
		'nbchunk_conv': 'nbchunk_conv',
	}
	data = load_json(path_to_file) ## load runtime_vs_Nsegments.json
	list_of_tuple_data = []
	colors 					= ['red', 'green', 'black', 'blue', 'maroon', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
	j = 0
	for i, key in enumerate(data.keys()):
		x_data = np.array(data[key]['N_segments'])
		y_data = np.array(data[key]['runtimes_perbatch'])
		tmp_label = key.split(',')
		label = f'{map_key['nbchunk']} : {tmp_label[0].split(':')[1]}, {map_key['nbchunk_conv']} : {tmp_label[1].split(':')[1]}'
		if (label==f'{map_key['nbchunk']} :  10, {map_key['nbchunk_conv']} :  10') or (label==f'{map_key['nbchunk']} :  300, {map_key['nbchunk_conv']} :  10') or (label==f'{map_key['nbchunk']} :  50, {map_key['nbchunk_conv']} :  50') or (label==f'{map_key['nbchunk']} :  300, {map_key['nbchunk_conv']} :  100') or (label==f'{map_key['nbchunk']} :  300, {map_key['nbchunk_conv']} :  150'):
			continue
		color = colors[j]
		list_of_tuple_data.append((x_data, y_data, label, color))
		j += 1
	output_file = path_to_file.replace('.json', '.png')
	overlay_plots_runtime(*list_of_tuple_data, title='Runtime vs N_segments', xlabel='N segments', ylabel='Runtime (sec)', output_file=output_file)
	return list_of_tuple_data

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
	plt.figure(figsize=(12,10))
	plt.subplots_adjust(right=0.7)
	# hep.style.use("CMS")
	wedges, texts, autotexts = plt.pie(sizes, explode=explode, labels=None, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=140, pctdistance=1.1)
	for autotext in autotexts:
		autotext.set_fontsize(18)
	plt.title('Runtime Distribution of Major Operations', fontsize=20)
	plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
	plt.legend(wedges, labels, title="Operations", loc="upper right", bbox_to_anchor=(1.1, 0.5), fontsize=18)
	plt.tight_layout()
	plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
	plt.savefig(path_to_file.replace('.json', '.png'), bbox_inches='tight')
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
	hep.style.use("CMS")
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
	plt.xlabel(xlabel, fontsize=24)
	plt.ylabel(ylabel, fontsize=24)
	plt.yscale('log')
	# plt.xlim([-0.15, 0])
	# plt.xlim([-2,5])
	plt.xticks(fontsize=24)
	plt.yticks(fontsize=24)
	plt.title(title, fontsize=24)
	plt.grid(True)
	plt.legend(fontsize=24)
	plt.tight_layout()
	plt.savefig(output_file)
	plt.close()

def effq_evaluation(path_to_file: str=''):
	data = load_json(path_to_file) ## load effq_accuracy_data.json
	colors = ['maroon', 'green', 'red', 'purple', 'black', 'blue', 'orange']
	linestyles = ['solid', 'dotted', 'dashed', 'dashdot', 'loosely dashed']
	list_of_deltaQ_tuple = []
	list_of_dQ_over_Q_tuple = []
	for i, key in enumerate(data.keys()):
		deltaQ_data = data[key]['deltaQ']
		dQ_over_Q = data[key]['dQ_over_Q']
		label = key
		color = colors[i]
		linestyle = linestyles[i]
		list_of_deltaQ_tuple.append((deltaQ_data, label, color, linestyle))
		list_of_dQ_over_Q_tuple.append((dQ_over_Q, label, color, linestyle))
	output_file_deltaQ = path_to_file.replace('.json', '_deltaQ.png')
	overlay_hists_deltaQ(*list_of_deltaQ_tuple, title='Delta Q Distribution', xlabel='DeltaQ (ke-)', ylabel='Counts', output_file=output_file_deltaQ)
	output_file_dQ_over_Q = path_to_file.replace('.json', '_dQ_over_Q.png')
	overlay_hists_deltaQ(*list_of_dQ_over_Q_tuple, title='dQ/Q Distribution', xlabel='dQ/Q', ylabel='Counts', output_file=output_file_dQ_over_Q)

def runtime_chunksum(path_to_file: str=''):
	xlabel = 'Number of bins in chunksum qblock'
	ylabel = 'Runtime (sec)'
	title = 'Chunksum qblock Runtime vs Number of bins'
	def plot_chunksum_runtime_vs_Nbins(Nbins: list, t_mean: list, t_std: list, output_path: str, xlabel: str='Number of bins', ylabel: str='Runtime (sec)', title: str='Chunksum Runtime vs Number of bins'):
		plt.figure(figsize=(10, 8))
		hep.style.use("CMS") 
		# plt.errorbar(Nbins, t_mean, yerr=t_std, fmt='o--', ecolor='r', capsize=5, label=r'Mean $\pm stdev$')
		plt.errorbar(Nbins, t_mean, yerr=None, fmt='o--', ecolor='r', capsize=5, label=r'Average of runtime per batch')
		# plt.plot(Nbins_sorted, t_mean_sorted, '*--', label=r'Mean $\pm stdev$')
		plt.xlabel(xlabel, fontsize=24)
		plt.ylabel(ylabel, fontsize=24)
		plt.title(title, fontsize=24)
		plt.xticks(fontsize=24)
		plt.yticks(fontsize=24)
		plt.legend(fontsize=24)
		plt.grid(True)
		plt.tight_layout()
		plt.savefig(f'{output_path}')
		plt.close()
	data = load_json(path_to_file) ## load chunksum_runtime_vs_Nbins.json
	Nbins = data['Nbins']
	t_mean = data['t_mean']
	t_std = data['t_std']
	output_path = path_to_file.replace('.json', '.png')
	plot_chunksum_runtime_vs_Nbins(Nbins, t_mean, t_std, output_path=output_path, xlabel=xlabel, ylabel=ylabel, title=title)

def plot(peak_mem_tuple, runtime_tuple):
	map_key_color = {'nbchunk :  100, nbchunk_conv :  10': 'red',
					'nbchunk :  100, nbchunk_conv :  50': 'green',
					'nbchunk :  100, nbchunk_conv :  100' : 'black',
					'nbchunk :  300, nbchunk_conv :  50' : 'blue'
					}
	map_key_marker = {'nbchunk :  100, nbchunk_conv :  10': '.',
					'nbchunk :  100, nbchunk_conv :  50': 'p',
					'nbchunk :  100, nbchunk_conv :  100' : '<',
					'nbchunk :  300, nbchunk_conv :  50' : '*'
					}
	hep.style.use("CMS")
	filled_markers = ['.', 'p', 'v', '*', '^', '<', '8', 's', 'p', 'h', 'H', 'D', 'd', 'P', 'X']
	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True, gridspec_kw={'hspace': 0.05}, constrained_layout=True)
	# Top plot
	for i, (x_data, y_data, label, color) in enumerate(peak_mem_tuple):
		color = map_key_color[label]
		marker = map_key_marker[label]
		ax1.scatter(x_data, y_data, label=label, color=color, alpha=0.5, s=100, marker=marker)
	# ax1.set_xlabel('N segments', fontsize=24)
	ax1.set_ylabel('Peak memory consumption (MB)', fontsize=24)
	ax1.set_ylim([0, 25000])
	ax1.legend(loc='best')
	ax1.grid(True, alpha=0.3)

	# Bottom plot
	for i, (x_data, y_data, label, color) in enumerate(runtime_tuple):
		color = map_key_color[label]
		marker = map_key_marker[label]
		ax2.scatter(x_data, y_data, label=label, color=color, alpha=0.5, s=100, marker=marker)
	# ax2.set_xlabel('N segments', fontsize=24)
	ax2.set_xlabel('Track segments', fontsize=24)
	ax2.set_ylabel('Runtime per batch(sec)', fontsize=24)
	# ax2.set_ylim([0, 25])
	ax2.legend(loc='best')
	ax2.grid(True, alpha=0.3)

	# plt.tight_layout()
	plt.savefig('/home/rrazakami/work/ND-LAr/starting_over/plots4paper/runtime_memoryConsumption.png', dpi=300)

if __name__ == "__main__":
	## Peak memory evaluation without dynamic chunking and batching
	input_path_peakmem = '/home/rrazakami/work/ND-LAr/starting_over/plots4paper/gpu_maxmemory_consumption_event1003/peakmem_vs_Nsegments_nodynamic_chunking_batching.json'  ## path to peakmem_vs_Nsegments_nodynamic_chunking_batching.json
	peak_mem_tuple = benchmark_peak_memory_nodynamic_chunking_and_batching(input_path_peakmem)

	## Runtime evaluation
	input_path_runtime = '/home/rrazakami/work/ND-LAr/starting_over/plots4paper/runtime_event1003/runtime_vs_Nsegments.json'  ## path to runtime_vs_Nsegments.json
	runtime_tuple = benchmark_runtime(input_path_runtime)

	# ## Overlay peak memory and runtime plots with shared x-axis
	plot(peak_mem_tuple, runtime_tuple)

	# input_path_runtime = 'data4plots/convo_8x8x2560/runtime_majorOps_batchsize8192_NBCHUNK100_NBCHUNKCONV50.json'  ## path to runtime_majorOps_batchsize8192_NBCHUNK100_NBCHUNKCONV50.json
	# runtimeshare_majorOp(input_path_runtime)

	# ## Effective charge evaluation
	# path_to_file = 'data4plots/effq_accuracy_data_effq_out_nt_10.json'
	# effq_evaluation(path_to_file)

	## Chunksum runtime evaluation
	# # path_to_file = 'data4plots/chunksum_i_runtime_vs_Nbins.json'
	# path_to_file = 'tests/tmp/chunksum_qblock_runtime_vs_Nbins.json'
	# runtime_chunksum(path_to_file)
	
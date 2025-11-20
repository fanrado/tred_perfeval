import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

def load_json(fname):
	with open(fname, 'r') as f:
		return json.load(f)

def get_chunksum_runtime(data_json: dict, output_path: str='../tests'):
	organized_data = {}
	batch_size = data_json['batch_size']
	nbchunk = data_json['nbchunk']
	nbchunk_conv = data_json['nbchunk_conv']
	chunksum_qblock_shape = data_json['chunksum_qblock_shape']
	chunksum_i_shape = data_json['chunksum_i_shape']
	chunksum_readout_shape = data_json['chunksum_readout_shape']
	del data_json['chunksum_readout_shape']
	del data_json['chunksum_i_shape']
	del data_json['chunksum_qblock_shape']
	del data_json['batch_size']
	del data_json['nbchunk']
	del data_json['nbchunk_conv']

	event_ids 				= []
	N_segments 				= []
	N_qblocks 				= []
	runtimes_perbatch 		= []
	#
	runtime_chunksum_readout_sec = []
	runtime_chunksum_qblock_sec  = []
	runtime_chunksum_i_sec       = []
	runtime_sum_current = []
	qblock_size = []

	## Process the TPC data_json
	for key, value in data_json.items():
		# print(f'key : {key}')
		## TPC level
		# runtimes_perbatch.append(value['runtime_per_batch']) ## runtime_sec replaces peak_memory_MB
		for key1, value1 in value.items():
			if not isinstance(value1, dict):
				continue
			# print(value.keys())
			# sys.exit()
			## Batch level
			for key2, value2 in value1.items():
				if not isinstance(value2, dict):
					continue
				event_ids.append(value2['event_id'])
				N_segments.append(value2['N_segments'])
				N_qblocks.append(value2['N_qblock'])
				runtimes_perbatch.append(value2['runtime_sec']) ## runtime_sec replaces peak_memory_MB
				#
				runtime_chunksum_readout_sec.append(value2['each_operation_sec']['chunksum_readout_sec'])
				t_chunksum_qblock_perbatch = 0.0
				t_chunksum_i_perbatch = 0.0
				t_sumcurrent_perbatch = 0.0 ## summing per batch
				# print('--')
				for ibatch, batch_data in value2['each_operation_sec'].items():
					if not isinstance(batch_data, dict):
						continue
					## Under chunking conv
					tmp_chunksum_qblock_sec = 0.0
					tmp_sumcurrent_sec = 0.0
					for ichunk, chunk_data in batch_data.items():
						# print(ichunk)
						# print(ichunk, chunk_data['chunksum_qblock_sec'])
						tmp_chunksum_qblock_sec += chunk_data['chunksum_qblock_sec']
						tmp_sumcurrent_sec += chunk_data['sumcurrent_sec']
						ichunk_chunksum_i_sec = 0.0
						try:
							qblock_size.append((chunk_data['qblock_shape_x'], chunk_data['qblock_shape_y'], chunk_data['qblock_shape_z']))
						except:
							pass
						continue
						for keyfinal, val_final in chunk_data['conv_sec']['details'].items():
							# print(f'---- {keyfinal} : {val_final}')
							ichunk_chunksum_i_sec += val_final['chunksum_i_time']
						t_chunksum_i_perbatch += ichunk_chunksum_i_sec
					t_chunksum_qblock_perbatch += tmp_chunksum_qblock_sec
					t_sumcurrent_perbatch += tmp_sumcurrent_sec ## summing per batch

				runtime_chunksum_qblock_sec.append(t_chunksum_qblock_perbatch)
				runtime_chunksum_i_sec.append(t_chunksum_i_perbatch)
				runtime_sum_current.append(t_sumcurrent_perbatch)
	qblock_size_array = np.array(qblock_size, dtype=[('x', 'i4'), ('y', 'i4'), ('z', 'i4')])

	## plot distributions of the runtimes
	output_dist = f'{output_path}/dist_runtime_chunksum/{chunksum_readout_shape}_{chunksum_qblock_shape}_{chunksum_i_shape}_dist.png'
	plot_dist = False
	if plot_dist:
		plt.figure(figsize=(15, 5))
		plt.subplot(1, 3, 1)
		plt.hist(runtime_chunksum_readout_sec, bins=30, color='blue', alpha=0.7)
		plt.xlabel('Runtime Chunksum Readout (sec)')
		plt.ylabel('#')
		plt.title('Distribution of Chunksum Readout Runtime')
		#
		plt.subplot(1, 3, 2)
		plt.hist(runtime_chunksum_qblock_sec, bins=30, color='green', alpha=0.7)
		plt.xlabel('Runtime Chunksum Qblock (sec)')
		plt.ylabel('#')
		plt.title('Distribution of Chunksum Qblock Runtime')
		#
		plt.subplot(1, 3, 3)
		plt.hist(runtime_chunksum_i_sec, bins=30, color='red', alpha=0.7)
		plt.xlabel('Runtime Chunksum current (sec)')
		plt.ylabel('#')
		plt.title('Distribution of Chunksum current Runtime')
		#
		plt.tight_layout()
		plt.savefig(output_dist)
		plt.close()
	Runtime_summary = {
		'Nbin_chunksum_readout': int(chunksum_readout_shape.split('x')[-1]),
		'Nbin_chunksum_qblock': int(chunksum_qblock_shape.split('x')[-1]),
		'Nbin_chunksum_i': int(chunksum_i_shape.split('x')[-1]),
		'runtime_chunksum_readout_sec_mean': np.mean(runtime_chunksum_readout_sec),
		'runtime_chunksum_readout_sec_std' : np.std(runtime_chunksum_readout_sec),
		'runtime_chunksum_qblock_sec_mean' : np.mean(runtime_chunksum_qblock_sec),
		'runtime_chunksum_qblock_sec_std'  : np.std(runtime_chunksum_qblock_sec),
		'runtime_chunksum_i_sec_mean'      : np.mean(runtime_chunksum_i_sec),
		'runtime_chunksum_i_sec_std'       : np.std(runtime_chunksum_i_sec),
		'runtime_perbatch_sec_mean'      : np.mean(runtimes_perbatch),
		'runtime_sumcurrent_sec_mean'      : np.mean(runtime_sum_current),
		'qblock_size': qblock_size_array
	}
	return Runtime_summary

def plot_chunksum_runtime_vs_Nbins(Nbins: list, t_mean: list, t_std: list, output_path: str, xlabel: str='Number of bins', ylabel: str='Runtime (sec)', title: str='Chunksum Runtime vs Number of bins', filename: str='chunksum_runtime_vs_Nbins.png'):
	plt.figure(figsize=(10, 8))
	hep.style.use("CMS") 
	# plt.errorbar(Nbins, t_mean, yerr=t_std, fmt='o--', ecolor='r', capsize=5, label=r'Mean $\pm stdev$')
	plt.errorbar(Nbins, t_mean, yerr=None, fmt='o--', ecolor='r', capsize=5, label=r'Average of runtime per batch')
	# plt.plot(Nbins_sorted, t_mean_sorted, '*--', label=r'Mean $\pm stdev$')
	plt.xlabel(xlabel, fontsize=20)
	plt.ylabel(ylabel, fontsize=20)
	# plt.xlim([100, 500])
	plt.title(title, fontsize=20)
	plt.xticks(fontsize=15)
	plt.yticks(fontsize=15)
	plt.legend(fontsize=15)
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(f'{output_path}/{filename}')
	plt.close()

def get_runtime_chunksum_qblock(path_to_file: str, output_path: str):
	list_json = [f for f in os.listdir(path_to_file) if f.endswith('.json')]
	Nbins 	= []
	t_mean 	= []
	t_std 	= []
	runtime_per_batch = []
	qblock_size = np.array([], dtype=[('x', 'i4'), ('y', 'i4'), ('z', 'i4')])
	for f in list_json:
		runtime = get_chunksum_runtime(data_json=load_json(os.path.join(path_to_file, f)), output_path=output_path)
		if runtime['Nbin_chunksum_qblock'] not in Nbins:
			if len(Nbins)==0:
				qblock_size = np.concatenate((qblock_size, runtime['qblock_size']))
			Nbins.append(runtime['Nbin_chunksum_qblock'])
			t_mean.append(runtime['runtime_chunksum_qblock_sec_mean'])
			t_std.append(runtime['runtime_chunksum_qblock_sec_std'])
			runtime_per_batch.append(runtime['runtime_perbatch_sec_mean'])
			
	Nbins_dict = {index: Nbin for index, Nbin in enumerate(Nbins)}
	sorted_indices = sorted(Nbins_dict, key=Nbins_dict.get)
	Nbins_sorted = [Nbins_dict[i] for i in sorted_indices]
	t_mean_sorted = [t_mean[i] for i in sorted_indices]
	t_std_sorted = [t_std[i] for i in sorted_indices]

	# dump data to a json file
	output_data = {
		'Nbins': Nbins_sorted,
		't_mean': t_mean_sorted,
		't_std': t_std_sorted
	}
	with open(f'{output_path}/chunksum_qblock_runtime_vs_Nbins.json', 'w') as f:
		json.dump(output_data, f)
	##
	try:
		## plot distribution of the qblock sizes
		plt.figure(figsize=(12, 8))
		hep.style.use("CMS")
		mean_x = np.round(np.mean(qblock_size['x']), 1)
		mean_y = np.round(np.mean(qblock_size['y']),1)
		mean_z = np.round(np.mean(qblock_size['z']),1)
		max_x = np.max(qblock_size['x'])
		max_y = np.max(qblock_size['y'])
		max_z = np.max(qblock_size['z'])

		plt.hist(qblock_size['x'], bins=np.arange(np.min(qblock_size['x']), np.max(qblock_size['x']), 1), alpha=0.7, histtype='step', color='blue')
		plt.hist(qblock_size['y'], bins=np.arange(np.min(qblock_size['y']), np.max(qblock_size['y']), 1), alpha=0.7, histtype='step', color='green')
		plt.hist(qblock_size['z'], bins=np.arange(np.min(qblock_size['z']), np.max(qblock_size['z']), 1), alpha=0.7, histtype='step', color='red')
		plt.xlabel('Qblock size', fontsize=20)
		plt.ylabel('#', fontsize=20)
		plt.title('Distribution of Qblock sizes', fontsize=20)
		plt.xticks(fontsize=15)
		plt.yticks(fontsize=15)
		plt.legend([f'x size, mean = {mean_x}, max = {max_x}', f'y size, mean = {mean_y}, max = {max_y}', f'z size, mean = {mean_z}, max = {max_z}'], fontsize=15)
		plt.grid(True)
		plt.tight_layout()
		plt.savefig(f'{output_path}/distribution_qblock_sizes.png')
		plt.close()
	except:
		pass
	# sys.exit()
	# plot
	plot_chunksum_runtime_vs_Nbins(Nbins=Nbins_sorted, t_mean=t_mean_sorted, t_std=t_std_sorted, output_path=output_path, xlabel='Number of bins in chunksum_qblock', ylabel='Runtime (sec)', title='Chunksum_qblock Runtime vs Number of bins', filename='chunksum_qblock_runtime_vs_Nbins.png')

def get_runtime_chunksum_i(path_to_file: str, output_path: str):
	list_json = [f for f in os.listdir(path_to_file) if f.endswith('.json')]
	Nbins 	= []
	t_mean 	= []
	t_std 	= []
	runtime_per_batch = []
	runtime_sum_current = []
	for f in list_json:
		runtime = get_chunksum_runtime(data_json=load_json(os.path.join(path_to_file, f)), output_path=output_path)
		# print(runtime)
		if runtime['Nbin_chunksum_i'] not in Nbins:
			if runtime['Nbin_chunksum_i'] < 32:
				continue
			# print(runtime['Nbin_chunksum_i'], runtime['runtime_chunksum_i_sec_mean'])
			Nbins.append(runtime['Nbin_chunksum_i'])
			t_mean.append(runtime['runtime_chunksum_i_sec_mean'])
			t_std.append(runtime['runtime_chunksum_i_sec_std'])
			runtime_per_batch.append(runtime['runtime_perbatch_sec_mean'])
			runtime_sum_current.append(runtime['runtime_sumcurrent_sec_mean'])
	plt.figure(figsize=(10, 8))
	# plt.scatter(Nbins, runtime_per_batch)
	plt.scatter(Nbins, runtime_sum_current)
	plt.savefig('tests/runtime_sumcurrent_per_batch_vs_Nbins.png')
	plt.close()
	sys.exit()

	Nbins_dict = {index: Nbin for index, Nbin in enumerate(Nbins)}
	sorted_indices = sorted(Nbins_dict, key=Nbins_dict.get)
	Nbins_sorted = [Nbins_dict[i] for i in sorted_indices]
	t_mean_sorted = [t_mean[i] for i in sorted_indices]
	t_std_sorted = [t_std[i] for i in sorted_indices]

	# dump data in a json file
	data_dict = {
		'Nbins': Nbins_sorted,
		't_mean': t_mean_sorted,
		't_std': t_std_sorted
	}
	with open(f'{output_path}/chunksum_i_runtime_vs_Nbins.json', 'w') as f:
		json.dump(data_dict, f)
	#
	plot_chunksum_runtime_vs_Nbins(Nbins=Nbins_sorted, t_mean=t_mean_sorted, t_std=t_std_sorted, output_path=output_path, xlabel='Number of bins in chunksum_current', ylabel='Runtime (sec)', title='Chunksum current Runtime vs Number of bins', filename='chunksum_current_runtime_vs_Nbins.png')

def get_runtime_chunksum_readout(path_to_file: str, output_path: str):
	list_json = [f for f in os.listdir(path_to_file) if f.endswith('.json')]
	Nbins 	= []
	t_mean 	= []
	t_std 	= []
	for f in list_json:
		runtime = get_chunksum_runtime(data_json=load_json(os.path.join(path_to_file, f)), output_path=output_path)
		if runtime['Nbin_chunksum_readout'] not in Nbins:
			print(runtime['Nbin_chunksum_readout'], runtime['runtime_chunksum_readout_sec_mean'])
			Nbins.append(runtime['Nbin_chunksum_readout'])
			t_mean.append(runtime['runtime_chunksum_readout_sec_mean'])
			t_std.append(runtime['runtime_chunksum_readout_sec_std'])
	Nbins_dict = {index: Nbin for index, Nbin in enumerate(Nbins)}
	sorted_indices = sorted(Nbins_dict, key=Nbins_dict.get)
	Nbins_sorted = [Nbins_dict[i] for i in sorted_indices]
	t_mean_sorted = [t_mean[i] for i in sorted_indices]
	t_std_sorted = [t_std[i] for i in sorted_indices]
	# dump data in a json file
	data_dict = {
		'Nbins': Nbins_sorted,
		't_mean': t_mean_sorted,
		't_std': t_std_sorted
	}
	with open(f'{output_path}/chunksum_readout_runtime_vs_Nbins.json', 'w') as f:
		json.dump(data_dict, f)
	# plot
	plot_chunksum_runtime_vs_Nbins(Nbins=Nbins_sorted, t_mean=t_mean_sorted, t_std=t_std_sorted, output_path=output_path, xlabel='Number of bins in chunksum_readout', ylabel='Runtime (sec)', title='Chunksum_readout Runtime vs Number of bins', filename='chunksum_readout_runtime_vs_Nbins.png')


if __name__=='__main__':
	path_to_file = '/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/CHUNKSUM_EVAL/combined'
	# list_json = [f for f in os.listdir(path_to_file) if f.endswith('.json')]
	# ##
	# ## chunksum current
	# Nbins 	= []
	# t_mean 	= []
	# t_std 	= []
	# for f in list_json:
	# 	runtime = get_chunksum_runtime(data_json=load_json(os.path.join(path_to_file, f)))
	# 	if runtime['Nbin_chunksum_qblock'] not in Nbins:
	# 		Nbins.append(runtime['Nbin_chunksum_qblock'])
	# 		t_mean.append(runtime['runtime_chunksum_qblock_sec_mean'])
	# 		t_std.append(runtime['runtime_chunksum_qblock_sec_std'])
	# Nbins_dict = {index: Nbin for index, Nbin in enumerate(Nbins)}
	# sorted_indices = sorted(Nbins_dict, key=Nbins_dict.get)
	# Nbins_sorted = [Nbins_dict[i] for i in sorted_indices]
	# t_mean_sorted = [t_mean[i] for i in sorted_indices]
	# t_std_sorted = [t_std[i] for i in sorted_indices]
	# plt.figure(figsize=(10, 8))
	# hep.style.use("CMS") 
	# plt.errorbar(Nbins_sorted, t_mean_sorted, yerr=t_std_sorted, fmt='o--', ecolor='r', capsize=5, label=r'Mean $\pm stdev$')
	# plt.xlabel('Number of bins in chunksum_qblock', fontsize=20)
	# plt.ylabel('Runtime (sec)', fontsize=20)
	# plt.title('Chunksum_qblock Runtime vs Number of bins', fontsize=20)
	# plt.xticks(fontsize=15)
	# plt.yticks(fontsize=15)
	# plt.legend(fontsize=15)
	# plt.grid(True)
	# plt.tight_layout()
	# plt.savefig('../tests/chunksum_qblock_runtime_vs_Nbins.png')
	# plt.close()
	# get_runtime_chunksum_qblock(path_to_file=path_to_file, output_path='../tests/')
	get_runtime_chunksum_i(path_to_file=path_to_file, output_path='../tests/')
	# get_runtime_chunksum_readout(path_to_file=path_to_file, output_path='../tests/')
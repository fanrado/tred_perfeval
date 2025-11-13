import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

def load_json(fname):
	with open(fname, 'r') as f:
		return json.load(f)
# def get_chunksum_runtime(data_json: dict):
# 	"""
# 	Extract the runtime information from the chunksum evaluation JSON data_json.
# 	Args:
# 		data_json (dict): The JSON data containing chunksum evaluation results.
# 	Returns:
# 		dict: A dictionary with chunk sizes as keys and their corresponding runtimes.
# 	"""
# 	chunksum_readout_shape = ''
# 	chunksum_i_shape = ''
# 	chunksum_qblock_shape = ''
# 	chunksum_shape = {
# 		'chunksum_readout_shape': '',
# 		'chunksum_i_shape': '',
# 		'chunksum_qblock_shape': ''
# 	}
# 	for itpc, tpc_data in data_json.items():
# 		print(itpc)
# 		if not isinstance(tpc_data, dict):
# 			print(f'Not a dict : {itpc}')
# 			if 'chunksum' in itpc:
# 				chunksum_shape[itpc] = tpc_data
# 			continue
# 		for key, val in tpc_data.items():
# 			print(f'key : {key}')
# 			continue
# 			if not isinstance(val, dict):
# 				continue
# 			for key1, val1 in val.items():
# 				print(f'key1 : {key1}')
# 				continue
# 				if not isinstance(val1, dict):
# 					continue
# 				for key2, val2 in val1.items():
# 					## Batch level
# 					# print(f'key2 : {key2}')
# 					if not isinstance(val2, dict):
# 						continue
# 					for key3, val3 in val2.items():
# 						print(f'key3 : {key3}')

def get_chunksum_runtime(data_json: dict):
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
	## Process the TPC data_json
	for key, value in data_json.items():
		# print(f'key : {key}')
		## TPC level
		for key1, value1 in value.items():
			if not isinstance(value1, dict):
				continue

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
				# print('--')
				for ibatch, batch_data in value2['each_operation_sec'].items():
					if not isinstance(batch_data, dict):
						continue
					## Under chunking conv
					tmp_chunksum_qblock_sec = 0.0
					for ichunk, chunk_data in batch_data.items():
						# print(ichunk)
						# print(ichunk, chunk_data['chunksum_qblock_sec'])
						tmp_chunksum_qblock_sec += chunk_data['chunksum_qblock_sec']
						ichunk_chunksum_i_sec = 0.0

						for keyfinal, val_final in chunk_data['conv_sec']['details'].items():
							# print(f'---- {keyfinal} : {val_final}')
							ichunk_chunksum_i_sec += val_final['chunksum_i_time']
						t_chunksum_i_perbatch += ichunk_chunksum_i_sec
					t_chunksum_qblock_perbatch += tmp_chunksum_qblock_sec
				runtime_chunksum_qblock_sec.append(t_chunksum_qblock_perbatch)
				runtime_chunksum_i_sec.append(t_chunksum_i_perbatch)
	Runtime_summary = {
		'Nbin_chunksum_readout': int(chunksum_readout_shape.split('x')[-1]),
		'Nbin_chunksum_qblock': int(chunksum_qblock_shape.split('x')[-1]),
		'Nbin_chunksum_i': int(chunksum_i_shape.split('x')[-1]),
		'runtime_chunksum_readout_sec_mean': np.mean(runtime_chunksum_readout_sec),
		'runtime_chunksum_readout_sec_std' : np.std(runtime_chunksum_readout_sec),
		'runtime_chunksum_qblock_sec_mean' : np.mean(runtime_chunksum_qblock_sec),
		'runtime_chunksum_qblock_sec_std'  : np.std(runtime_chunksum_qblock_sec),
		'runtime_chunksum_i_sec_mean'      : np.mean(runtime_chunksum_i_sec),
		'runtime_chunksum_i_sec_std'       : np.std(runtime_chunksum_i_sec)
	}
	return Runtime_summary

def get_runtime_chunksum_qblock(path_to_file: str, output_path: str):
	list_json = [f for f in os.listdir(path_to_file) if f.endswith('.json')]
	Nbins 	= []
	t_mean 	= []
	t_std 	= []
	for f in list_json:
		runtime = get_chunksum_runtime(data_json=load_json(os.path.join(path_to_file, f)))
		if runtime['Nbin_chunksum_qblock'] not in Nbins:
			Nbins.append(runtime['Nbin_chunksum_qblock'])
			t_mean.append(runtime['runtime_chunksum_qblock_sec_mean'])
			t_std.append(runtime['runtime_chunksum_qblock_sec_std'])
	Nbins_dict = {index: Nbin for index, Nbin in enumerate(Nbins)}
	sorted_indices = sorted(Nbins_dict, key=Nbins_dict.get)
	Nbins_sorted = [Nbins_dict[i] for i in sorted_indices]
	t_mean_sorted = [t_mean[i] for i in sorted_indices]
	t_std_sorted = [t_std[i] for i in sorted_indices]
	plt.figure(figsize=(10, 8))
	hep.style.use("CMS") 
	plt.errorbar(Nbins_sorted, t_mean_sorted, yerr=t_std_sorted, fmt='o', ecolor='r', capsize=5, label=r'Mean $\pm stdev$')
	plt.xlabel('Number of bins in chunksum_qblock', fontsize=20)
	plt.ylabel('Runtime (sec)', fontsize=20)
	plt.title('Chunksum_qblock Runtime vs Number of bins', fontsize=20)
	plt.xticks(fontsize=15)
	plt.yticks(fontsize=15)
	plt.legend(fontsize=15)
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(f'{output_path}/chunksum_qblock_runtime_vs_Nbins.png')
	plt.close()

def get_runtime_chunksum_i(path_to_file: str, output_path: str):
	list_json = [f for f in os.listdir(path_to_file) if f.endswith('.json')]
	Nbins 	= []
	t_mean 	= []
	t_std 	= []
	for f in list_json:
		runtime = get_chunksum_runtime(data_json=load_json(os.path.join(path_to_file, f)))
		if runtime['Nbin_chunksum_i'] not in Nbins:
			print(runtime['Nbin_chunksum_i'], runtime['runtime_chunksum_i_sec_mean'])
			Nbins.append(runtime['Nbin_chunksum_i'])
			t_mean.append(runtime['runtime_chunksum_i_sec_mean'])
			t_std.append(runtime['runtime_chunksum_i_sec_std'])
	Nbins_dict = {index: Nbin for index, Nbin in enumerate(Nbins)}
	sorted_indices = sorted(Nbins_dict, key=Nbins_dict.get)
	Nbins_sorted = [Nbins_dict[i] for i in sorted_indices]
	t_mean_sorted = [t_mean[i] for i in sorted_indices]
	t_std_sorted = [t_std[i] for i in sorted_indices]
	plt.figure(figsize=(10, 8))
	hep.style.use("CMS") 
	plt.errorbar(Nbins_sorted, t_mean_sorted, yerr=t_std_sorted, fmt='o', ecolor='r', capsize=5, label=r'Mean $\pm stdev$')
	plt.xlabel('Number of bins in chunksum_current', fontsize=20)
	plt.ylabel('Runtime (sec)', fontsize=20)
	plt.title('Chunksum_current Runtime vs Number of bins', fontsize=20)
	plt.xticks(fontsize=15)
	plt.yticks(fontsize=15)
	plt.legend(fontsize=15)
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(f'{output_path}/chunksum_current_runtime_vs_Nbins.png')
	plt.close()

def get_runtime_chunksum_readout(path_to_file: str, output_path: str):
	list_json = [f for f in os.listdir(path_to_file) if f.endswith('.json')]
	Nbins 	= []
	t_mean 	= []
	t_std 	= []
	for f in list_json:
		runtime = get_chunksum_runtime(data_json=load_json(os.path.join(path_to_file, f)))
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
	plt.figure(figsize=(10, 8))
	hep.style.use("CMS") 
	plt.errorbar(Nbins_sorted, t_mean_sorted, yerr=t_std_sorted, fmt='o', ecolor='r', capsize=5, label=r'Mean $\pm stdev$')
	plt.xlabel('Number of bins in chunksum_readout', fontsize=20)
	plt.ylabel('Runtime (sec)', fontsize=20)
	plt.title('Chunksum_readout Runtime vs Number of bins', fontsize=20)
	plt.xticks(fontsize=15)
	plt.yticks(fontsize=15)
	plt.legend(fontsize=15)
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(f'{output_path}/chunksum_readout_runtime_vs_Nbins.png')
	plt.close()

if __name__=='__main__':
	path_to_file = '/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/CHUNKSUM_EVAL'
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
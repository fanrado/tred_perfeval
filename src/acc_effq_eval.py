import os, sys, json
import numpy as np
import numpy.lib.recfunctions as rfn
import matplotlib.pyplot as plt

def load_npz(input_file=''):
	data = np.load(input_file)
	hits_keys = [k for k in data.keys() if ('hits' in k) and ('location' not in k)]
	hits = np.array([], dtype=np.float32)
	# print(hits_keys)
	for k in hits_keys:
		hits = rfn.append_fields(hits, k, data[k][:, -1], usemask=False)
	return hits

def get_deltaQ(npz_data, npz_ref):
	"""
		Calculate the deltaQ between two datasets stored in npz format.
		The charges are from the hits information.
	"""
	data 		= load_npz(input_file=npz_data)	
	ref_data 	= load_npz(input_file=npz_ref)
	common_fields = set(data.dtype.names) & set(ref_data.dtype.names)
	# deltaQ = data[common_fields] - ref_data[common_fields]
	# return deltaQ
	# deltaQ = {}
	deltaQ = np.array([], dtype=np.float32)
	for field in common_fields:
		N = len(data[field])
		if N > len(ref_data[field]):
			N = len(ref_data[field])
		# print('data[field][:N] = ', data[field][:N])
		# sys.exit()
		# try:
		deltaQ = np.concatenate((deltaQ, data[field][:N] - ref_data[field][:N]))
		# except:
		# 	print(field)
		# 	print(data[field].shape, ref_data[field].shape)
	return deltaQ

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
	for (deltaQ, label, color) in deltaQ_list:
		plt.hist(deltaQ, bins=100, histtype='step', color=color, label=label)
		# plt.hist(deltaQ, bins=100, range=(-1000, 1000), histtype='step', color=color, label=label)
	plt.xlabel(xlabel, fontsize=18)
	plt.ylabel(ylabel, fontsize=18)
	plt.yscale('log')
	plt.xticks(fontsize=18)
	plt.yticks(fontsize=18)
	plt.title(title, fontsize=18)
	plt.tight_layout()
	plt.legend(fontsize=18)
	plt.grid(True)
	plt.savefig(output_file)
	plt.close()

if __name__=='__main__':
	# path_to_ref = '/'.join(['/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/ACC_EFFQ/node_2x2x2', '10x10_partitions.npz'])
	# path_to_data = '/'.join(['/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/ACC_EFFQ', '10x10_partitions_afterconversion.npz'])
	# # path_to_6x6 = '/'.join(['/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/ACC_EFFQ', '6x6_partitions.npz'])
	# # deltaQ = get_deltaQ(npz_data=path_to_data, npz_ref=path_to_ref)
	# deltaQ = get_deltaQ(npz_data=path_to_data, npz_ref=path_to_ref)
	# # deltaQ_6x6 = get_deltaQ(npz_data=path_to_6x6, npz_ref=path_to_ref)
	# # print(deltaQ)
	# # k = load_npz(input_file=path_to_data)
	# plt.figure(figsize=(10,6))
	# plt.hist(deltaQ, bins=100, range=(-1000, 1000), histtype='step', color='blue', label='deltaQ (10x10_changeinconfig - 10x10_changeinpy)')
	# plt.xlabel('Delta Q [ke-]')
	# plt.ylabel('Counts')
	# plt.yscale('log')
	# plt.title('Delta Q Distribution between 10x10_after and 10x10_before')
	# plt.legend()
	# plt.grid()
	# plt.savefig('deltaQ_10x10_vs_10x10.png')
	# plt.close()
	root_path = '/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/ACC_EFFQ/benchmark_plots'
	path_to_ref = '/'.join([root_path, '10x10_partitions_2x2x2.npz'])
	path_to_10x10_4x4x2 = '/'.join([root_path, '10x10_partitions_4x4x2.npz'])
	path_to_8x8_2x2x2 = '/'.join([root_path, '8x8_partitions_2x2x2.npz'])
	path_to_6x6_2x2x2 = '/'.join([root_path, '6x6_partitions_2x2x2.npz'])

	deltaQ_10x10_4x4x2 = get_deltaQ(npz_data=path_to_10x10_4x4x2, npz_ref=path_to_ref)
	deltaQ_8x8_2x2x2 = get_deltaQ(npz_data=path_to_8x8_2x2x2, npz_ref=path_to_ref)
	plt.figure(figsize=(10,6))
	plt.hist(deltaQ_10x10_4x4x2, bins=100, range=(-1000, 1000), histtype='step', color='blue', label='deltaQ 810x10 4x4x2')
	plt.hist(deltaQ_8x8_2x2x2, bins=100, range=(-1000, 1000), histtype='step', color='red', label='deltaQ 8x8 2x2x2')
	plt.xlabel('Delta Q [ke-]')
	plt.ylabel('Counts')
	plt.yscale('log')
	plt.title('Delta Q Distribution between 10x10_after and 10x10_before')
	plt.legend()
	plt.grid()
	plt.savefig('test.png')
	plt.close()
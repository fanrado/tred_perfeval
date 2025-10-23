import os, sys, json
import numpy as np
import numpy.lib.recfunctions as rfn
import matplotlib.pyplot as plt
import h5py

def load_npz(input_file=''):
	"""
		Accumulated charge per TPC per event.
		Args:
			input_file (str): Path to the npz file containing the current data.
		Output structure: 
			Dictionary of 70 TPCs, each TPC contains a structured array with fields:
				-- [('pixels_locations', '<i4', (3,)), ('accumulated_charge', '<f4')]
				-- Accumulated charge per pixel : shape (1340,)
				-- Pixel locations : shape (1340, 3)
	"""
	data = np.load(input_file)
	current_keys = [k for k in data.keys() if ('current' in k) and ('location' not in k)]
	current_location_keys = [k for k in data.keys() if ('current' in k) and ('location' in k)]

	tpcs_keys = {f'tpc{i}': [] for i in range(70)}
	for i in range(70):
		kk = [k for k in current_keys if f'_tpc{i}_' in k]
		tpcs_keys[f'tpc{i}'] = kk
	tpcs_locations_keys = {f'tpc{i}': [] for i in range(70)}
	for i in range(70):
		kk = [k for k in current_location_keys if f'_tpc{i}_' in k]
		tpcs_locations_keys[f'tpc{i}'] = kk
	
	all_tpc_data = {}
	for itpc in range(70):
		# if itpc > 3:
		# 	continue
		tpc_key = f'tpc{itpc}'
		tmp_tpc_data = {
			'pixels_locations': np.array([], dtype=np.int32),
			'accumulated_charge': np.array([], dtype=np.float32)
		}
		for i, key in enumerate(tpcs_keys[tpc_key]):
			acc_charge = np.sum(data[key], axis=-1).reshape(-1) # sum over the last axis to get the accumulated charge per pixel
			if i==0:
				tmp_tpc_data['accumulated_charge'] = acc_charge
				tmp_tpc_data['pixels_locations'] = data[f'{key}_location']
			else:
				tmp_tpc_data['pixels_locations'] = np.concatenate((tmp_tpc_data['pixels_locations'], data[f'{key}_location']), axis=0)
				tmp_tpc_data['accumulated_charge'] = np.concatenate((tmp_tpc_data['accumulated_charge'], acc_charge), axis=0)
		# Create a numpy structured array with fields 'pixels_locations' and 'accumulated_charge'
		tpc_data = np.zeros(len(tmp_tpc_data['pixels_locations']), dtype=[('pixels_locations', np.int32, (3,)), ('accumulated_charge', np.float32)])
		## This try-except is to handle empty TPC data
		try:
			tpc_data['pixels_locations'] = tmp_tpc_data['pixels_locations']
			tpc_data['accumulated_charge'] = tmp_tpc_data['accumulated_charge']
		except:
			print('Empty TPC data for ', tpc_key)
			# continue

		all_tpc_data[tpc_key] = tpc_data
	return all_tpc_data

def get_all_pixels_locations(data_tpc, ref_data_tpc):
	"""
		Get all unique pixel locations from both data and reference data for a given TPC.
	"""
	combined = np.vstack((data_tpc['pixels_locations'], ref_data_tpc['pixels_locations']))
	pix_locs = np.unique(combined, axis=0)
	
	return pix_locs


def get_deltaQ_fromNPZ(npz_data, npz_ref, saveHDF5=False, output_hdf5=''):
	data = load_npz(input_file=npz_data)
	ref_data = load_npz(input_file=npz_ref)

	deltaQ_allTPCs = {f'tpc{i}': None for i in range(70)}  # Only first 3 TPCs for now
	dQ_over_Q_allTPCs = {f'tpc{i}': None for i in range(70)}  # Only first 3 TPCs for now
	for itpc in deltaQ_allTPCs.keys():
		print('Processing ', itpc)
		data_tpc = data[itpc]
		ref_data_tpc = ref_data[itpc]
		pix_locs_array = get_all_pixels_locations(data_tpc, ref_data_tpc)
		deltaQ = np.array([], dtype=np.float32)
		dQ_over_Q = np.array([], dtype=np.float32)
		for pix_loc in pix_locs_array:
			locs_in_data = np.where(np.all(data_tpc['pixels_locations']==pix_loc, axis=1))[0]
			locs_in_ref = np.where(np.all(ref_data_tpc['pixels_locations']==pix_loc, axis=1))[0]
			if len(locs_in_data)==0:
				tmp_charge = np.array([(pix_loc, 0.0)], dtype=data_tpc.dtype)
				data_tpc = rfn.stack_arrays((data_tpc, tmp_charge), usemask=False)
			if len(locs_in_ref)==0:
				tmp_charge = np.array([(pix_loc, 0.0)], dtype=ref_data_tpc.dtype)
				ref_data_tpc = rfn.stack_arrays((ref_data_tpc, tmp_charge), usemask=False)
		
			locs_in_data = np.where(np.all(data_tpc['pixels_locations']==pix_loc, axis=1))[0]
			locs_in_ref = np.where(np.all(ref_data_tpc['pixels_locations']==pix_loc, axis=1))[0]
			charge_in_ref = np.sum(ref_data_tpc['accumulated_charge'][locs_in_ref])
			charge_in_data = np.sum(data_tpc['accumulated_charge'][locs_in_data])
			deltaQ = np.concatenate((deltaQ, [charge_in_data - charge_in_ref]))
			if charge_in_ref > 1e-9:
				dQ_over_Q = np.concatenate((dQ_over_Q, [(charge_in_data - charge_in_ref)/charge_in_ref]))
				
		# print(dQ_over_Q)
		deltaQ_allTPCs[itpc] = deltaQ
		dQ_over_Q_allTPCs[itpc] = dQ_over_Q
	# print(deltaQ_allTPCs)
	if saveHDF5:
		with h5py.File(output_hdf5, 'w') as f:
			for itpc in deltaQ_allTPCs.keys():
				print(type(deltaQ_allTPCs[itpc]))
				f.create_dataset(f'{itpc}/deltaQ', data=deltaQ_allTPCs[itpc])
				f.create_dataset(f'{itpc}/dQ_over_Q', data=dQ_over_Q_allTPCs[itpc])

	return deltaQ_allTPCs, dQ_over_Q_allTPCs

def load_deltaQ_fromHDF5(hdf5_file=''):
	"""
		Load deltaQ and dQ_over_Q from an HDF5 file.
		Args:
			hdf5_file (str): Path to the HDF5 file.
		Returns:
			deltaQ_allTPCs (dict): Dictionary of deltaQ arrays per TPC.
			dQ_over_Q_allTPCs (dict): Dictionary of dQ_over_Q arrays per TPC.
	"""
	deltaQ_allTPCs = {}
	dQ_over_Q_allTPCs = {}
	with h5py.File(hdf5_file, 'r') as f:
		for itpc in f.keys():
			deltaQ_allTPCs[itpc] = f[f'{itpc}/deltaQ'][:]
			dQ_over_Q_allTPCs[itpc] = f[f'{itpc}/dQ_over_Q'][:]
	return deltaQ_allTPCs, dQ_over_Q_allTPCs

def overlay_hists_deltaQ(*deltaQ_list, title, xlabel, ylabel, output_file='', cut=None):
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
		## concatenate the accumulated charge from all TPCs
		all_tpcs_deltaQ = np.array([], dtype=np.float32)
		for itpc in deltaQ.keys():
			all_tpcs_deltaQ = np.concatenate((all_tpcs_deltaQ, deltaQ[itpc]), axis=0)
		mask = all_tpcs_deltaQ < cut if cut is not None else np.ones_like(all_tpcs_deltaQ, dtype=bool)
	
		## plot the distribution of all accumulated charges
		# plt.hist(deltaQ, bins=100, histtype='step', color=color, label=label)
		plt.hist(all_tpcs_deltaQ[mask], bins=100, histtype='step', color=color, label=label)
		# ax[1].hist(all_tpcs_deltaQ, bins=100, histtype='step', color=color, label=label)
		# plt.hist(deltaQ, bins=100, range=(-1000, 1000), histtype='step', color=color, label=label)
	plt.xlabel(xlabel, fontsize=18)
	plt.ylabel(ylabel, fontsize=18)
	plt.yscale('log')
	plt.xticks(fontsize=18)
	plt.yticks(fontsize=18)
	plt.title(title, fontsize=18)
	plt.grid(True)
	plt.legend(fontsize=14)
	plt.tight_layout()
	plt.savefig(output_file)
	plt.close()

if __name__=='__main__':
	## -------------------------------------------------
	## Example usage--- IGNORE --- TESTING PURPOSES ONLY
	## -------------------------------------------------
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
	deltaQ_6x6_2x2x2 = get_deltaQ(npz_data=path_to_6x6_2x2x2, npz_ref=path_to_ref)

	all_tpcs_10x10 = np.array([], dtype=np.float32)
	for itpc in deltaQ_10x10_4x4x2.keys():
		all_tpcs_10x10 = np.concatenate((all_tpcs_10x10, deltaQ_10x10_4x4x2[itpc]['accumulated_charge']), axis=0)
	print(deltaQ_10x10_4x4x2.keys())
	print(deltaQ_8x8_2x2x2.keys())
	print(deltaQ_10x10_4x4x2['tpc0']['accumulated_charge'])
	print(deltaQ_8x8_2x2x2['tpc0']['accumulated_charge'])
	# sys.exit()
	plt.figure(figsize=(10,6))
	plt.hist(all_tpcs_10x10, bins=100, histtype='step', color='blue', label='10x10 4x4x2')
	# plt.hist(deltaQ_10x10_4x4x2['tpc0']['accumulated_charge'], bins=100, histtype='step', color='blue', label='10x10 4x4x2')
	# plt.hist(deltaQ_8x8_2x2x2['tpc0']['accumulated_charge'], bins=100, histtype='step', color='red', label='8x8 2x2x2')
	# plt.hist(deltaQ_6x6_2x2x2['tpc0']['accumulated_charge'], bins=100, histtype='step', color='green', label='6x6 2x2x2')
	plt.xlabel('Delta Q [ke-]')
	plt.ylabel('Counts')
	plt.yscale('log')
	plt.title('Delta Q Distribution between 10x10_after and 10x10_before')
	plt.legend()
	plt.grid()
	plt.savefig('test.png')
	plt.close()
import os, sys, json
import numpy as np
import numpy.lib.recfunctions as rfn
import matplotlib.pyplot as plt
import h5py

def load_npz(input_file='', cut_on_Q=1): #ke-
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
	# all_tpc_data_below_thr = {}
	for itpc in range(70):
		# if itpc > 7:
		# 	continue
		tpc_key = f'tpc{itpc}'
		tmp_tpc_data = {
			'pixels_locations': np.array([], dtype=np.int32),
			'accumulated_charge': np.array([], dtype=np.float32)
		}

		for i, key in enumerate(tpcs_keys[tpc_key]):
			acc_charge = np.sum(data[key], axis=-1).reshape(-1) # sum over the last axis to get the accumulated charge per pixel
			mask = acc_charge >= cut_on_Q
			# acc_charge = acc_charge[mask]
			if i==0:
				tmp_tpc_data['accumulated_charge'] = acc_charge[mask]
				tmp_tpc_data['pixels_locations'] = data[f'{key}_location'][mask]
			else:
				tmp_tpc_data['pixels_locations'] = np.concatenate((tmp_tpc_data['pixels_locations'], data[f'{key}_location'][mask]), axis=0)
				tmp_tpc_data['accumulated_charge'] = np.concatenate((tmp_tpc_data['accumulated_charge'], acc_charge[mask]), axis=0)
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
	# print(len(combined), ' combined pixel locations.', len(pix_locs), ' unique pixel locations.')
	# print(len(np.unique(data_tpc['pixels_locations'])), ' data pixel locations.', len(np.unique(ref_data_tpc['pixels_locations'])), ' ref pixel locations.')
	return pix_locs

def npz2hdf5(npz_data, npz_ref, saveHDF5=False, output_hdf5=''):
	data = load_npz(input_file=npz_data, cut_on_Q=1e-12)
	ref_data = load_npz(input_file=npz_ref, cut_on_Q=1e-12)

	Ntpcs = 70
	# deltaQ_allTPCs = {f'tpc{i}': None for i in range(Ntpcs)}  # Only first 3 TPCs for now
	# dQ_over_Q_allTPCs = {f'tpc{i}': None for i in range(Ntpcs)}  # Only first 3 TPCs for now
	# saving Q_allTPCs, Qref_allTPCs, pixLocs_allTPCs, pixLocs_ref_allTPCs
	## all charges are above the threshold
	Q_allTPCs = {f'tpc{i}': None for i in range(Ntpcs)}
	Qref_allTPCs = {f'tpc{i}': None for i in range(Ntpcs)}
	pixLocs_allTPCs = {f'tpc{i}': None for i in range(Ntpcs)}
	pixLocs_ref_allTPCs = {f'tpc{i}': None for i in range(Ntpcs)}
	# Npix = 0 # total number of pixels
	# Npix_below_thr = 0 # total number of pixels below threshold
	for itpc in Q_allTPCs.keys():
		print('Processing ', itpc)
		data_tpc = data[itpc]
		ref_data_tpc = ref_data[itpc]
		pix_locs_array = get_all_pixels_locations(data_tpc, ref_data_tpc)
		# print(pix_locs_array.shape)
		# deltaQ = np.array([], dtype=np.float32)
		# dQ_over_Q = np.array([], dtype=np.float32)
		# Saving Q, Qref, pixLocs, pixLocs_ref from each tpc
		Q_tpc = np.array([], dtype=np.float32)
		Qref_tpc =  np.array([], dtype=np.float32)
		pixLocs_tpc = np.array([], dtype=np.int32)
		pixLocs_ref_tpc = np.array([], dtype=np.int32)
		# Npix += len(pix_locs_array) # increment the total number of pixels
		# print(Npix, ' total pixels so far.')
		for pix_loc in pix_locs_array:
			locs_in_data = np.where(np.all(data_tpc['pixels_locations']==pix_loc, axis=1))[0]
			locs_in_ref = np.where(np.all(ref_data_tpc['pixels_locations']==pix_loc, axis=1))[0]
			# print(len(locs_in_data), ' locations in data for pixel ', pix_loc, len(locs_in_ref), ' locations in ref.')
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
			# print(locs_in_data, ' charge in data for pixel ', pix_loc, ': ', charge_in_data, ' ke-.', locs_in_ref, ' charge in ref: ', charge_in_ref, ' ke-.')
			# if charge_in_ref < cut_onQ:
			# 	Npix_below_thr += 1 # increment the number of pixels below threshold
			# else: # only consider pixels where the reference charge is above the threshold
			if len(Q_tpc)==0:
				# deltaQ = np.array([charge_in_data - charge_in_ref], dtype=np.float32)
				# dQ_over_Q = np.array([(charge_in_data - charge_in_ref)/charge_in_ref], dtype=np.float32)
				Q_tpc = np.array([charge_in_data], dtype=np.float32)
				Qref_tpc = np.array([charge_in_ref], dtype=np.float32)
				pixLocs_tpc = np.array([pix_loc], dtype=np.int32)
				pixLocs_ref_tpc = np.array([pix_loc], dtype=np.int32)
			else:
				# deltaQ = np.concatenate((deltaQ, [charge_in_data - charge_in_ref]))
				# dQ_over_Q = np.concatenate((dQ_over_Q, [(charge_in_data - charge_in_ref)/charge_in_ref]))
				Q_tpc = np.concatenate((Q_tpc, [charge_in_data]))
				Qref_tpc = np.concatenate((Qref_tpc, [charge_in_ref]))
				pixLocs_tpc = np.concatenate((pixLocs_tpc, [pix_loc]))
				pixLocs_ref_tpc = np.concatenate((pixLocs_ref_tpc, [pix_loc]))
		# print(Q_tpc, ' Q_tpc shape: ', Q_tpc.shape)
		# saving Q_allTPCs, Qref_allTPCs, pixLocs_allTPCs, pixLocs_ref_allTPCs
		Q_allTPCs[itpc] = Q_tpc
		Qref_allTPCs[itpc] = Qref_tpc
		pixLocs_allTPCs[itpc] = pixLocs_tpc
		pixLocs_ref_allTPCs[itpc] = pixLocs_ref_tpc
	# print(deltaQ_allTPCs)
	if saveHDF5:
		with h5py.File(output_hdf5, 'w') as f:
			for itpc in Q_allTPCs.keys():
				print(type(Q_allTPCs[itpc]))
				# f.create_dataset(f'{itpc}/Q', data=Q_allTPCs[itpc])
				# f.create_dataset(f'{itpc}/Qref', data=Qref_allTPCs[itpc])
				# saving Q_allTPCs, Qref_allTPCs, pixLocs_allTPCs, pixLocs_ref_allTPCs
				f.create_dataset(f'{itpc}/Q', data=Q_allTPCs[itpc])
				f.create_dataset(f'{itpc}/Q_ref', data=Qref_allTPCs[itpc])
				f.create_dataset(f'{itpc}/pixLocs', data=pixLocs_allTPCs[itpc])
				f.create_dataset(f'{itpc}/pixLocs_ref', data=pixLocs_ref_allTPCs[itpc])

	# return Q_allTPCs, Qref_allTPCs

# def get_deltaQ_fromNPZ__(npz_data, npz_ref, saveHDF5=False, output_hdf5='', cut_onQ=1): # ke-
# 	data = load_npz(input_file=npz_data, cut_on_Q=1e-12)
# 	ref_data = load_npz(input_file=npz_ref, cut_on_Q=1e-12)

# 	Ntpcs = 7
# 	deltaQ_allTPCs = {f'tpc{i}': None for i in range(Ntpcs)}  # Only first 3 TPCs for now
# 	dQ_over_Q_allTPCs = {f'tpc{i}': None for i in range(Ntpcs)}  # Only first 3 TPCs for now
# 	# saving Q_allTPCs, Qref_allTPCs, pixLocs_allTPCs, pixLocs_ref_allTPCs
# 	## all charges are above the threshold
# 	Q_allTPCs = {f'tpc{i}': None for i in range(Ntpcs)}
# 	Qref_allTPCs = {f'tpc{i}': None for i in range(Ntpcs)}
# 	pixLocs_allTPCs = {f'tpc{i}': None for i in range(Ntpcs)}
# 	pixLocs_ref_allTPCs = {f'tpc{i}': None for i in range(Ntpcs)}
# 	Npix = 0 # total number of pixels
# 	Npix_below_thr = 0 # total number of pixels below threshold
# 	for itpc in deltaQ_allTPCs.keys():
# 		print('Processing ', itpc)
# 		data_tpc = data[itpc]
# 		ref_data_tpc = ref_data[itpc]
# 		pix_locs_array = get_all_pixels_locations(data_tpc, ref_data_tpc)
# 		# print(pix_locs_array.shape)
# 		deltaQ = np.array([], dtype=np.float32)
# 		dQ_over_Q = np.array([], dtype=np.float32)
# 		# Saving Q, Qref, pixLocs, pixLocs_ref from each tpc
# 		Q_tpc = np.array([], dtype=np.float32)
# 		Qref_tpc =  np.array([], dtype=np.float32)
# 		pixLocs_tpc = np.array([], dtype=np.int32)
# 		pixLocs_ref_tpc = np.array([], dtype=np.int32)
# 		Npix += len(pix_locs_array) # increment the total number of pixels
# 		# print(Npix, ' total pixels so far.')
# 		for pix_loc in pix_locs_array:
# 			locs_in_data = np.where(np.all(data_tpc['pixels_locations']==pix_loc, axis=1))[0]
# 			locs_in_ref = np.where(np.all(ref_data_tpc['pixels_locations']==pix_loc, axis=1))[0]
# 			# print(len(locs_in_data), ' locations in data for pixel ', pix_loc, len(locs_in_ref), ' locations in ref.')
# 			if len(locs_in_data)==0:
# 				tmp_charge = np.array([(pix_loc, 0.0)], dtype=data_tpc.dtype)
# 				data_tpc = rfn.stack_arrays((data_tpc, tmp_charge), usemask=False)
# 			if len(locs_in_ref)==0:
# 				tmp_charge = np.array([(pix_loc, 0.0)], dtype=ref_data_tpc.dtype)
# 				ref_data_tpc = rfn.stack_arrays((ref_data_tpc, tmp_charge), usemask=False)
		
# 			locs_in_data = np.where(np.all(data_tpc['pixels_locations']==pix_loc, axis=1))[0]
# 			locs_in_ref = np.where(np.all(ref_data_tpc['pixels_locations']==pix_loc, axis=1))[0]
# 			charge_in_ref = np.sum(ref_data_tpc['accumulated_charge'][locs_in_ref])
# 			charge_in_data = np.sum(data_tpc['accumulated_charge'][locs_in_data])
# 			print(locs_in_data, ' charge in data for pixel ', pix_loc, ': ', charge_in_data, ' ke-.', locs_in_ref, ' charge in ref: ', charge_in_ref, ' ke-.')
# 			if charge_in_ref < cut_onQ:
# 				Npix_below_thr += 1 # increment the number of pixels below threshold
# 			else: # only consider pixels where the reference charge is above the threshold
# 				if len(deltaQ)==0:
# 					deltaQ = np.array([charge_in_data - charge_in_ref], dtype=np.float32)
# 					dQ_over_Q = np.array([(charge_in_data - charge_in_ref)/charge_in_ref], dtype=np.float32)
# 					Q_tpc = np.array([charge_in_data], dtype=np.float32)
# 					Qref_tpc = np.array([charge_in_ref], dtype=np.float32)
# 					pixLocs_tpc = np.array([pix_loc], dtype=np.int32)
# 					pixLocs_ref_tpc = np.array([pix_loc], dtype=np.int32)
# 				else:
# 					deltaQ = np.concatenate((deltaQ, [charge_in_data - charge_in_ref]))
# 					dQ_over_Q = np.concatenate((dQ_over_Q, [(charge_in_data - charge_in_ref)/charge_in_ref]))
# 					Q_tpc = np.concatenate((Q_tpc, [charge_in_data]))
# 					Qref_tpc = np.concatenate((Qref_tpc, [charge_in_ref]))
# 					pixLocs_tpc = np.concatenate((pixLocs_tpc, [pix_loc]))
# 					pixLocs_ref_tpc = np.concatenate((pixLocs_ref_tpc, [pix_loc]))
# 		# print(Npix_below_thr, ' pixels below threshold so far.')
# 		# print(dQ_over_Q)
# 		deltaQ_allTPCs[itpc] = deltaQ
# 		dQ_over_Q_allTPCs[itpc] = dQ_over_Q
# 		# saving Q_allTPCs, Qref_allTPCs, pixLocs_allTPCs, pixLocs_ref_allTPCs
# 		Q_allTPCs[itpc] = Q_tpc
# 		Qref_allTPCs[itpc] = Qref_tpc
# 		pixLocs_allTPCs[itpc] = pixLocs_tpc
# 		pixLocs_ref_allTPCs[itpc] = pixLocs_ref_tpc
# 	# print(deltaQ_allTPCs)
# 	if saveHDF5:
# 		with h5py.File(output_hdf5, 'w') as f:
# 			for itpc in deltaQ_allTPCs.keys():
# 				print(type(deltaQ_allTPCs[itpc]))
# 				f.create_dataset(f'{itpc}/deltaQ', data=deltaQ_allTPCs[itpc])
# 				f.create_dataset(f'{itpc}/dQ_over_Q', data=dQ_over_Q_allTPCs[itpc])
# 				# saving Q_allTPCs, Qref_allTPCs, pixLocs_allTPCs, pixLocs_ref_allTPCs
# 				f.create_dataset(f'{itpc}/Q', data=Q_allTPCs[itpc])
# 				f.create_dataset(f'{itpc}/Q_ref', data=Qref_allTPCs[itpc])
# 				f.create_dataset(f'{itpc}/pixLocs', data=pixLocs_allTPCs[itpc])
# 				f.create_dataset(f'{itpc}/pixLocs_ref', data=pixLocs_ref_allTPCs[itpc])
# 			# total number of pixels
# 			f.create_dataset('Npix_total', data=Npix)
# 			f.create_dataset('Npix_below_threshold', data=Npix_below_thr)

# 	return deltaQ_allTPCs, dQ_over_Q_allTPCs

# def load_deltaQ_fromHDF5__(hdf5_file=''):
# 	"""
# 		Load deltaQ and dQ_over_Q from an HDF5 file.
# 		Args:
# 			hdf5_file (str): Path to the HDF5 file.
# 		Returns:
# 			deltaQ_allTPCs (dict): Dictionary of deltaQ arrays per TPC.
# 			dQ_over_Q_allTPCs (dict): Dictionary of dQ_over_Q arrays per TPC.
# 	"""
# 	deltaQ_allTPCs = {}
# 	dQ_over_Q_allTPCs = {}
# 	with h5py.File(hdf5_file, 'r') as f:
# 		tpc_keys = [key for key in f.keys() if key.startswith('tpc')]
# 		for itpc in tpc_keys:
# 			deltaQ_allTPCs[itpc] = f[f'{itpc}/deltaQ'][:]
# 			dQ_over_Q_allTPCs[itpc] = f[f'{itpc}/dQ_over_Q'][:]
# 			# loc_minusone = np.where(dQ_over_Q_allTPCs[itpc]==-1.0)[0]
# 			# print('--------------------------------')
# 			# print(f[f'{itpc}/Q'][loc_minusone], '\t', f[f'{itpc}/Q_ref'][loc_minusone])
# 			# print(f[f'{itpc}/pixLocs'].shape)
# 			# sys.exit()
# 			# print(f[f'{itpc}/Q_ref'][loc_minusone])
# 	return deltaQ_allTPCs, dQ_over_Q_allTPCs

def load_Q_fromHDF5(hdf5_file='', cut_on_Qref=1): # ke-
	"""
		Load Q and Q_ref from an HDF5 file.
		Args:
			hdf5_file (str): Path to the HDF5 file.
		Returns:
			Q_allTPCs (dict): Dictionary of Q arrays per TPC.
			Qref_allTPCs (dict): Dictionary of Q_ref arrays per TPC.
	"""
	Q_allTPCs = {}
	Qref_allTPCs = {}
	deltaQ_allTPCs = {}
	dQ_over_Q_allTPCs = {}
	Npix_total = 0
	Npix_below_thr = 0
	with h5py.File(hdf5_file, 'r') as f:
		tpc_keys = [key for key in f.keys() if key.startswith('tpc')]
		for itpc in tpc_keys:
			all_Qref = f[f'{itpc}/Q_ref'][:]
			all_Q = f[f'{itpc}/Q'][:]
			Qref_allTPCs[itpc] = all_Qref
			Q_allTPCs[itpc] = all_Q
			# mask = all_Qref >= cut_on_Qref # only consider pixels where Q_ref is above the threshold
			# mask_nocharge = all_Qref == 0.0
			# Npix_total += len(all_Qref[~mask_nocharge]) # total number of pixels
			# # print(len(all_Qref[mask_nocharge]), len(all_Qref))
			# Npix_below_thr += np.sum(all_Qref[~mask_nocharge] < cut_on_Qref) # total number of pixels below threshold
			# # print(f'TPC {itpc}: {len(all_Qref)} total pixels, {np.sum(all_Qref < cut_on_Qref)} pixels below threshold of {cut_on_Qref} ke-.')
			# deltaQ_allTPCs[itpc] = (f[f'{itpc}/Q'][:][mask] - f[f'{itpc}/Q_ref'][:][mask])
			# dQ_over_Q_allTPCs[itpc] = (f[f'{itpc}/Q'][:][mask] - f[f'{itpc}/Q_ref'][:][mask]) / f[f'{itpc}/Q_ref'][:][mask]
			deltaQ_allTPCs[itpc] = []
			dQ_over_Q_allTPCs[itpc] = []
			mask_nocharge = all_Qref == 0.0
			Npix_total += len(all_Qref[~mask_nocharge]) # total number of pixels
			Npix_below_thr += np.sum(all_Qref[~mask_nocharge] < cut_on_Qref) # total number of pixels below threshold
			for i in range(len(all_Qref)):
				# charge_in_data = f[f'{itpc}/Q'][:][i]
				charge_in_data = all_Q[i]
				charge_in_ref = all_Qref[i]
				deltaQ_allTPCs[itpc].append(charge_in_data - charge_in_ref)
				if all_Qref[i] < cut_on_Qref:
					continue
				## calculating deltaQ after the cut on Q_ref results in a non-symmetric distribution
				# charge_in_data = f[f'{itpc}/Q'][:][i]
				# charge_in_ref = all_Qref[i]
				# deltaQ_allTPCs[itpc].append(charge_in_data - charge_in_ref)
				dQ_over_Q_allTPCs[itpc].append((charge_in_data - charge_in_ref) / charge_in_ref)
			deltaQ_allTPCs[itpc] = np.array(deltaQ_allTPCs[itpc], dtype=np.float32)
			dQ_over_Q_allTPCs[itpc] = np.array(dQ_over_Q_allTPCs[itpc], dtype=np.float32)
	plot_dist = False
	if plot_dist:
		title = hdf5_file.split('/')[-1].replace('.hdf5', '')
		list_Q = [(Q_allTPCs, 'Q', 'blue'), (Qref_allTPCs, r'$Q_{ref}$', 'orange')]
		output_file = hdf5_file.replace('.hdf5', '_Q_distribution.png')
		overlay_hists_deltaQ(*list_Q, title=title, xlabel='Accumulated Charge [ke-]', ylabel='Counts', output_file=output_file)
	return deltaQ_allTPCs, dQ_over_Q_allTPCs, Npix_total, Npix_below_thr
	# return Q_allTPCs, Qref_allTPCs
	

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
		## concatenate the accumulated charge from all TPCs
		all_tpcs_deltaQ = np.array([], dtype=np.float32)
		for itpc in deltaQ.keys():
			all_tpcs_deltaQ = np.concatenate((all_tpcs_deltaQ, deltaQ[itpc]), axis=0)
		# mask = all_tpcs_deltaQ < cut if cut is not None else np.ones_like(all_tpcs_deltaQ, dtype=bool)
	
		## plot the distribution of all accumulated charges
		# plt.hist(deltaQ, bins=100, histtype='step', color=color, label=label)
		plt.hist(all_tpcs_deltaQ, bins=100, histtype='step', color=color, label=label)
		# ax[1].hist(all_tpcs_deltaQ, bins=100, histtype='step', color=color, label=label)
		# plt.hist(deltaQ, bins=100, range=(-1000, 1000), histtype='step', color=color, label=label)
	plt.xlabel(xlabel, fontsize=18)
	plt.ylabel(ylabel, fontsize=18)
	plt.yscale('log')
	# plt.xlim([-2, 5])
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
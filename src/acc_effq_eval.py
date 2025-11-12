import os, sys, json
import numpy as np
import numpy.lib.recfunctions as rfn
import matplotlib.pyplot as plt
import h5py
import matplotlib.colors as mcolors

def load_npz(input_file='', cut_on_Q=1, getEffq=False): #ke-
	"""
		Accumulated charge per TPC per event.
		Args:
			input_file (str): Path to the npz file containing the current data.
		Output structure: 
			Dictionary of 70 TPCs, each TPC contains a structured array with fields:
				-- [('pixels_locations', '<i4', (3,)), ('accumulated_charge', '<f4')]
				-- Accumulated charge per pixel : shape (1340,)
				-- Pixel locations : shape (1340, 3)
		####
		Let's get the effq instead of charge at the anode.
	"""
	key = 'current'
	key_output = 'accumulated_charge'
	if getEffq:
		key = 'effq_tpc'
		key_output = 'effective_charge'
	data = np.load(input_file)
	# current_keys = [k for k in data.keys() if ('current' in k) and ('location' not in k)]
	# current_location_keys = [k for k in data.keys() if ('current' in k) and ('location' in k)]
	current_keys = [k for k in data.keys() if (key in k) and ('location' not in k)]
	current_location_keys = [k for k in data.keys() if (key in k) and ('location' in k)]
	# current_keys = [k for k in data.keys() if ('effq_fine_grain_tpc' in k) and ('location' not in k)]
	# current_location_keys = [k for k in data.keys() if ('effq_fine_grain_tpc' in k) and ('location' in k)]

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
		# if itpc ==7: ## skip tpc7 for now, many pixels 
		# 	continue
		tpc_key = f'tpc{itpc}'
		tmp_tpc_data = {
			'pixels_locations': np.array([], dtype=np.int32),
			key_output: np.array([], dtype=np.float32)
		}

		for i, key in enumerate(tpcs_keys[tpc_key]):
			print(key)
			acc_charge = None
			if getEffq:
				acc_charge = data[key][:, -1] ### Get the effective charge at a pixel location
				# acc_charge = np.sum(data[key], axis=(1,2,3)) ### Get the effective charge at a pixel location
				# print(acc_charge.shape)
				# # sys.exit()
			else:
				acc_charge = np.sum(data[key], axis=-1).reshape(-1) # sum over the last axis to get the accumulated charge per pixel
			print(f'acc charge : {data[key].shape}')
			mask = acc_charge >= cut_on_Q
			# acc_charge = acc_charge[mask]
			if i==0:
				tmp_tpc_data[key_output] = acc_charge[mask]
				tmp_tpc_data['pixels_locations'] = data[f'{key}_location'][mask]
			else:
				tmp_tpc_data['pixels_locations'] = np.concatenate((tmp_tpc_data['pixels_locations'], data[f'{key}_location'][mask]), axis=0)
				# tmp_tpc_data[key_output] = np.concatenate((tmp_tpc_data[key_output], acc_charge[mask]), axis=0)
				tmp_tpc_data[key_output] = np.concatenate((tmp_tpc_data[key_output], acc_charge[mask]), axis=0)
		# Create a numpy structured array with fields 'pixels_locations' and key_output
		tpc_data = np.zeros(len(tmp_tpc_data['pixels_locations']), dtype=[('pixels_locations', np.int32, (3,)), (key_output, np.float32)])
		## This try-except is to handle empty TPC data
		try:
			tpc_data['pixels_locations'] = tmp_tpc_data['pixels_locations']
			tpc_data[key_output] = tmp_tpc_data[key_output]
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

def locations_to_keys(locs):
    """Convert (x_pix, y_pix, time_tick) locations to unique integer keys."""
    # Adjust multipliers based on your actual ranges
    return (locs[:, 0].astype(np.int64) * 100000000 + 
            locs[:, 1].astype(np.int64) * 10000 + 
            locs[:, 2].astype(np.int64))

def npz2hdf5(npz_data, npz_ref, saveHDF5=False, output_hdf5='', getEffq=False):
	metric_name = 'accumulated_charge'
	if getEffq:
		metric_name = 'effective_charge'
	ref_data = load_npz(input_file=npz_ref, cut_on_Q=1e-12, getEffq=getEffq)
	data = load_npz(input_file=npz_data, cut_on_Q=1e-12, getEffq=getEffq)

	Ntpcs = 70
	all_tpc_data = {f'tpc{i}': None for i in range(Ntpcs)}
	# all_tpc_data = {'tpc7': None}
	# Npix = 0 # total number of pixels
	# Npix_below_thr = 0 # total number of pixels below threshold
	for itpc in all_tpc_data.keys():
		data_tpc = data[itpc]
		ref_data_tpc = ref_data[itpc]
		pix_locs_array = get_all_pixels_locations(data_tpc, ref_data_tpc)
		# print(pix_locs_array, data_tpc, ref_data_tpc)
		# continue
		# Convert all locations to keys
		pix_keys = locations_to_keys(pix_locs_array)
		data_keys = locations_to_keys(data_tpc['pixels_locations'])
		ref_keys = locations_to_keys(ref_data_tpc['pixels_locations'])
		# Create lookup dictionaries for O(1) access
		data_key_to_idx = {key: idx for idx, key in enumerate(data_keys)}
		ref_key_to_idx = {key: idx for idx, key in enumerate(ref_keys)}
		# Find which pixel locations are missing
		missing_in_data = np.array([key not in data_key_to_idx for key in pix_keys])
		missing_in_ref = np.array([key not in ref_key_to_idx for key in pix_keys])
		# Add missing entries FIRST
		if np.any(missing_in_data):
			missing_data_locs = pix_locs_array[missing_in_data]
			tmp_charge = np.array([(tuple(loc), 0.0) for loc in missing_data_locs], dtype=data_tpc.dtype)
			old_len = len(data_tpc)
			data_tpc = rfn.stack_arrays((data_tpc, tmp_charge), usemask=False)
			
			# Update the lookup dictionary with new indices
			missing_indices = np.where(missing_in_data)[0]
			for i, idx in enumerate(missing_indices):
				key = pix_keys[idx]
				data_key_to_idx[key] = old_len + i

		if np.any(missing_in_ref):
			missing_ref_locs = pix_locs_array[missing_in_ref]
			tmp_charge = np.array([(tuple(loc), 0.0) for loc in missing_ref_locs], dtype=ref_data_tpc.dtype)
			old_len = len(ref_data_tpc)
			ref_data_tpc = rfn.stack_arrays((ref_data_tpc, tmp_charge), usemask=False)
			
			# Update the lookup dictionary with new indices
			missing_indices = np.where(missing_in_ref)[0]
			for i, idx in enumerate(missing_indices):
				key = pix_keys[idx]
				ref_key_to_idx[key] = old_len + i
		# NOW extract charges - all locations should exist
		data_indices = np.array([data_key_to_idx[key] for key in pix_keys])
		ref_indices = np.array([ref_key_to_idx[key] for key in pix_keys])
		# tpc_array = np.zeros(len(Q_tpc), dtype=[('pixels_locations', np.int32, (3,)), ('pixels_locations_ref', np.int32, (3,)), (metric_name, np.float32), (f'{metric_name}_ref', np.float32)])
		try:
			print('Extracting charges for TPC ', itpc)
			charge_in_data = data_tpc[metric_name][data_indices]
			charge_in_ref = ref_data_tpc[metric_name][ref_indices]
			# Build output arrays
			Q_tpc = charge_in_data
			Qref_tpc = charge_in_ref
			pixLocs_tpc = pix_locs_array.copy()
			pixLocs_ref_tpc = pix_locs_array.copy()
			tpc_array = np.zeros(len(Q_tpc), dtype=[('pixels_locations', np.int32, (3,)), ('pixels_locations_ref', np.int32, (3,)), (metric_name, np.float32), (f'{metric_name}_ref', np.float32)])
			if len(pixLocs_tpc) == 0:
				tpc_array['pixels_locations'] = pixLocs_tpc.reshape(0,3)
				tpc_array['pixels_locations_ref'] = pixLocs_ref_tpc.reshape(0,3)
				tpc_array[metric_name] = Q_tpc[:]
				tpc_array[f'{metric_name}_ref'] = Qref_tpc[:]	
			else:
				tpc_array['pixels_locations'] = pixLocs_tpc[:]
				tpc_array['pixels_locations_ref'] = pixLocs_ref_tpc[:]
				tpc_array[metric_name] = Q_tpc[:]
				tpc_array[f'{metric_name}_ref'] = Qref_tpc[:]
			all_tpc_data[itpc] = tpc_array
		except:
			print('Error extracting charges for TPC ', itpc)
			print(data_tpc, ref_data_tpc, pix_locs_array)
			# all_tpc_data[itpc] = tpc_array
	###------------------------------- WORKING BUT SLOW -------------------------------
	# for itpc in all_tpc_data.keys():
	# 	print('Processing ', itpc)
	# 	data_tpc = data[itpc]
	# 	ref_data_tpc = ref_data[itpc]
	# 	pix_locs_array = get_all_pixels_locations(data_tpc, ref_data_tpc)
	# 	print('Total unique pixel locations in ', itpc, ': ', len(pix_locs_array))
	# 	# Saving Q, Qref, pixLocs, pixLocs_ref from each tpc
	# 	Q_tpc = np.array([], dtype=np.float32)
	# 	Qref_tpc =  np.array([], dtype=np.float32)
	# 	pixLocs_tpc = np.array([], dtype=np.int32)
	# 	pixLocs_ref_tpc = np.array([], dtype=np.int32)
	# 	# Npix += len(pix_locs_array) # increment the total number of pixels
	# 	# print(Npix, ' total pixels so far.')
	# 	for pix_loc in pix_locs_array:
	# 		# try:
	# 		# print(f'Processing pixel location: {pix_loc}')
	# 		locs_in_data = np.where(np.all(data_tpc['pixels_locations']==pix_loc, axis=1))[0]
	# 		locs_in_ref = np.where(np.all(ref_data_tpc['pixels_locations']==pix_loc, axis=1))[0]
	# 		if len(locs_in_data)==0:
	# 			tmp_charge = np.array([(pix_loc, 0.0)], dtype=data_tpc.dtype)
	# 			data_tpc = rfn.stack_arrays((data_tpc, tmp_charge), usemask=False)
	# 		if len(locs_in_ref)==0:
	# 			tmp_charge = np.array([(pix_loc, 0.0)], dtype=ref_data_tpc.dtype)
	# 			ref_data_tpc = rfn.stack_arrays((ref_data_tpc, tmp_charge), usemask=False)
		
	# 		locs_in_data = np.where(np.all(data_tpc['pixels_locations']==pix_loc, axis=1))[0]
	# 		locs_in_ref = np.where(np.all(ref_data_tpc['pixels_locations']==pix_loc, axis=1))[0]
			
	# 		charge_in_ref = np.sum(ref_data_tpc[metric_name][locs_in_ref])
	# 		charge_in_data = np.sum(data_tpc[metric_name][locs_in_data])

	# 		# print(locs_in_data, ' charge in data for pixel ', pix_loc, ': ', charge_in_data, ' ke-.', locs_in_ref, ' charge in ref: ', charge_in_ref, ' ke-.')r
	# 		# if charge_in_ref < cut_onQ:
	# 		# 	Npix_below_thr += 1 # increment the number of pixels below threshold
	# 		# else: # only consider pixels where the reference charge is above the threshold
	# 		if len(Q_tpc)==0:
	# 			Q_tpc = np.array([charge_in_data], dtype=np.float32)
	# 			Qref_tpc = np.array([charge_in_ref], dtype=np.float32)
	# 			pixLocs_tpc = np.array([pix_loc], dtype=np.int32)
	# 			pixLocs_ref_tpc = np.array([pix_loc], dtype=np.int32)
	# 		else:
	# 			Q_tpc = np.concatenate((Q_tpc, [charge_in_data]))
	# 			Qref_tpc = np.concatenate((Qref_tpc, [charge_in_ref]))
	# 			pixLocs_tpc = np.concatenate((pixLocs_tpc, [pix_loc]))
	# 			pixLocs_ref_tpc = np.concatenate((pixLocs_ref_tpc, [pix_loc]))
	# 		# except:
	# 		# 	print(f'Error processing pixel location: {pix_loc}')
	# 		# 	pass
		# --- ------------------------------- WORKING BUT SLOW -------------------------------

		# tpc_array = np.zeros(len(Q_tpc), dtype=[('pixels_locations', np.int32, (3,)), ('pixels_locations_ref', np.int32, (3,)), (metric_name, np.float32), (f'{metric_name}_ref', np.float32)])
		# if len(pixLocs_tpc) == 0:
		# 	tpc_array['pixels_locations'] = pixLocs_tpc.reshape(0,3)
		# 	tpc_array['pixels_locations_ref'] = pixLocs_ref_tpc.reshape(0,3)
		# 	tpc_array[metric_name] = Q_tpc[:]
		# 	tpc_array[f'{metric_name}_ref'] = Qref_tpc[:]	
		# else:
		# 	tpc_array['pixels_locations'] = pixLocs_tpc[:]
		# 	tpc_array['pixels_locations_ref'] = pixLocs_ref_tpc[:]
		# 	tpc_array[metric_name] = Q_tpc[:]
		# 	tpc_array[f'{metric_name}_ref'] = Qref_tpc[:]
		# all_tpc_data[itpc] = tpc_array

	if saveHDF5:
		with h5py.File(output_hdf5, 'w') as f:
			for itpc in all_tpc_data.keys():
				try:
					print(type(all_tpc_data[itpc]))
					f.create_dataset(f'{itpc}', data=all_tpc_data[itpc])
				except:
					print(itpc, all_tpc_data[itpc])


	# return Q_allTPCs, Qref_allTPCs


def load_Q_fromHDF5(hdf5_file='', cut_on_Qref=1, getEffq=False): # ke-
	"""
		Load Q and Q_ref from an HDF5 file.
		Args:
			hdf5_file (str): Path to the HDF5 file.
		Returns:
			Q_allTPCs (dict): Dictionary of Q arrays per TPC.
			Qref_allTPCs (dict): Dictionary of Q_ref arrays per TPC.
	"""
	metric_name = 'accumulated_charge'
	if getEffq:
		metric_name = 'effective_charge'
	Q_allTPCs = {}
	Qref_allTPCs = {}
	deltaQ_allTPCs = {}
	dQ_over_Q_allTPCs = {}
	high_dQ_over_Q_allTPCs = {f'tpc{i}': {'dQ_over_Q': [], 'pixel_locs': []} for i in range(70)}
	
	##--- Plot Q vs Qref ----
	cut_on_deltaQ = 2#2 # ke-
	Q_array = np.array([], dtype=np.float32)
	Qref_array = np.array([], dtype=np.float32)
	####
	Npix_total = 0
	Npix_below_thr = 0
	with h5py.File(hdf5_file, 'r') as f:
		tpc_keys = [key for key in f.keys() if key.startswith('tpc')]
		for itpc in tpc_keys:
			# print(f[f'{itpc}']['accumulated_charge_ref'])
			# all_Qref = f[f'{itpc}/Q_ref'][:]
			# all_Q = f[f'{itpc}/Q'][:]
			all_Qref = f[itpc][f'{metric_name}_ref'][:]
			all_Q = f[itpc][metric_name][:]
			all_pixlocs = f[itpc]['pixels_locations'][:]
			all_pixlocs_ref = f[itpc]['pixels_locations_ref'][:]
			Qref_allTPCs[itpc] = all_Qref
			Q_allTPCs[itpc] = all_Q
			deltaQ_allTPCs[itpc] = []
			dQ_over_Q_allTPCs[itpc] = []
			# high_dQ_over_Q_allTPCs[itpc] = []

			mask_nocharge = all_Qref == 0.0
			Npix_total += len(all_Qref[~mask_nocharge]) # total number of pixels
			Npix_below_thr += np.sum(all_Qref[~mask_nocharge] < cut_on_Qref) # total number of pixels below threshold
			for i in range(len(all_Qref)):
				# charge_in_data = f[f'{itpc}/Q'][:][i]
				charge_in_data = all_Q[i]
				charge_in_ref = all_Qref[i]
				deltaQ_ = charge_in_data - charge_in_ref
				deltaQ_allTPCs[itpc].append(deltaQ_)
				if np.abs(deltaQ_) > cut_on_deltaQ: # collect Q and Qref for Q distribution plot
					Q_array = np.concatenate((Q_array, np.array([charge_in_data], dtype=np.float32)), axis=0)
					Qref_array = np.concatenate((Qref_array, np.array([charge_in_ref], dtype=np.float32)), axis=0)

				# if all_Qref[i] < cut_on_Qref:
				# 	continue
				if charge_in_ref < cut_on_Qref:
					continue
				dQ_overQ = (charge_in_data - charge_in_ref) / charge_in_ref
				# dQ_over_Q_allTPCs[itpc].append((charge_in_data - charge_in_ref) / charge_in_ref)
				dQ_over_Q_allTPCs[itpc].append(dQ_overQ)
				if dQ_overQ >= 2: # 
				# print(f'pixel loc data: {all_pixlocs[i]} \t pixel loc ref: {all_pixlocs_ref[i]}')
				# high_dQ_over_Q_allTPCs[itpc].append(dQ_overQ)
					high_dQ_over_Q_allTPCs[itpc]['dQ_over_Q'].append(dQ_overQ)
					high_dQ_over_Q_allTPCs[itpc]['pixel_locs'].append(all_pixlocs[i])
			# print(deltaQ_allTPCs[itpc])
			deltaQ_allTPCs[itpc] = np.array(deltaQ_allTPCs[itpc], dtype=np.float32)
			dQ_over_Q_allTPCs[itpc] = np.array(dQ_over_Q_allTPCs[itpc], dtype=np.float32)
			# high_dQ_over_Q_allTPCs[itpc] = np.array(high_dQ_over_Q_allTPCs[itpc], dtype=np.float32)
			high_dQ_over_Q_allTPCs[itpc]['dQ_over_Q'] = np.array(high_dQ_over_Q_allTPCs[itpc]['dQ_over_Q'], dtype=np.float32)
			high_dQ_over_Q_allTPCs[itpc]['pixel_locs'] = np.array(high_dQ_over_Q_allTPCs[itpc]['pixel_locs'], dtype=np.int32)
	# sys.exit()
	plot_dist = False
	if plot_dist:
		xlabel = 'Accumulated Charge'
		if getEffq:
			xlabel = 'Effective Charge'
		title = hdf5_file.split('/')[-1].replace('.hdf5', '')
		list_Q = [(Q_allTPCs, 'Q, DT = 88 cm2/s', 'blue'), (Qref_allTPCs, r'$Q_{ref}$, DT = 8.8 cm2/s', 'orange')]
		output_file = hdf5_file.replace('.hdf5', '_Q_distribution.png')
		overlay_hists_deltaQ(*list_Q, title=title, xlabel=f'{xlabel} [ke-]', ylabel='Counts', output_file=output_file)
	
	plot_Q_vs_Qref = True
	if plot_Q_vs_Qref:
		print(Q_array, Qref_array)
		title = hdf5_file.split('/')[-1].replace('.hdf5', '')
		# Calculate the range of your data
		x_min, x_max = np.min(Q_array), np.max(Q_array)
		y_min, y_max = np.min(Qref_array), np.max(Qref_array)

		# Create bin edges with step size of 1
		x_bins = np.arange(x_min - 0.5, x_max + 1.5, 1)  # -0.5 to +0.5 around each integer
		y_bins = np.arange(y_min - 0.5, y_max + 1.5, 1)

		# Create custom colormap with white at the bottom
		colors = ['white'] + [plt.cm.viridis(i) for i in range(1, 256)]
		cmap = mcolors.LinearSegmentedColormap.from_list('viridis_white', colors, N=256)
		plt.figure(figsize=(10,8))
		plt.hist2d(Qref_array, Q_array, bins=[x_bins, y_bins], cmap=cmap)
		plt.colorbar(label='Counts')
		plt.xlabel(r'$Q_{ref}$ [ke-]', fontsize=20)
		plt.ylabel('Q [ke-]', fontsize=20)
		plt.xticks(fontsize=15)
		plt.yticks(fontsize=15)
		plt.xlim([x_min, x_max])
		plt.ylim([y_min, y_max])
		plt.title(f'Q vs Q_ref: {title}', fontsize=20)
		plt.grid(True)
		plt.tight_layout()
		output_file = hdf5_file.replace('.hdf5', '_Q_vs_Qref.png')
		plt.savefig(output_file)
		plt.close()
	# return deltaQ_allTPCs, dQ_over_Q_allTPCs, Npix_total, Npix_below_thr
	return deltaQ_allTPCs, dQ_over_Q_allTPCs, high_dQ_over_Q_allTPCs, Npix_total, Npix_below_thr
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
	print(len(deltaQ_list))
	plt.figure(figsize=(10, 6))
	for (deltaQ, label, color, linestyle) in deltaQ_list:
		## concatenate the accumulated charge from all TPCs
		all_tpcs_deltaQ = np.array([], dtype=np.float32)
		for itpc in deltaQ.keys():
			all_tpcs_deltaQ = np.concatenate((all_tpcs_deltaQ, deltaQ[itpc]), axis=0)
		# mask = all_tpcs_deltaQ < cut if cut is not None else np.ones_like(all_tpcs_deltaQ, dtype=bool)
	
		## plot the distribution of all accumulated charges
		# plt.hist(deltaQ, bins=100, histtype='step', color=color, label=label)
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


def pixel_map(pix_with_relativeCharge, output_file='pixel_map_dQ_over_Q.png'):
	"""
		2D plot of pixel locations where each point in the 2d plane represents a pixel location (x,y) and the color represents the dQ/Q value at that pixel location.
		Args:
			pix_with_relativeCharge (dict): Dictionary with keys 'pixel_locs' and 'dQ_over_Q'.
	"""
	all_pix_locs = np.array([], dtype=np.int32)
	all_dQ_over_Q = np.array([], dtype=np.float32)
	for itpc in pix_with_relativeCharge.keys():
		if len(pix_with_relativeCharge[itpc]['pixel_locs'])==0:
			continue
		if len(all_pix_locs)==0:
			all_pix_locs = pix_with_relativeCharge[itpc]['pixel_locs']
			all_dQ_over_Q = pix_with_relativeCharge[itpc]['dQ_over_Q']
		else:
			all_pix_locs = np.concatenate((all_pix_locs, pix_with_relativeCharge[itpc]['pixel_locs']), axis=0)
			all_dQ_over_Q = np.concatenate((all_dQ_over_Q, pix_with_relativeCharge[itpc]['dQ_over_Q']), axis=0)
	pix_with_relativeCharge = {
		'pixel_locs': all_pix_locs,
		'dQ_over_Q': all_dQ_over_Q
	}
	# print(pix_with_relativeCharge['pixel_locs'])
	x_locs = pix_with_relativeCharge['pixel_locs'][:,0]
	y_locs = pix_with_relativeCharge['pixel_locs'][:,1]
	relative_charges = pix_with_relativeCharge['dQ_over_Q']

	# Calculate the range of your data
	x_min, x_max = np.min(x_locs), np.max(x_locs)
	y_min, y_max = np.min(y_locs), np.max(y_locs)

	# Create bin edges with step size of 1
	x_bins = np.arange(x_min - 0.5, x_max + 1.5, 1)  # -0.5 to +0.5 around each integer
	y_bins = np.arange(y_min - 0.5, y_max + 1.5, 1)

	# Create custom colormap with white at the bottom
	colors = ['white'] + [plt.cm.viridis(i) for i in range(1, 256)]
	cmap = mcolors.LinearSegmentedColormap.from_list('viridis_white', colors, N=256)

	plt.figure(figsize=(10,8))
	plt.hist2d(x_locs, y_locs, weights=relative_charges, bins=[x_bins, y_bins], cmap=cmap)
	# plt.colorbar(sc, label='dQ/Q')
	plt.colorbar(label='dQ/Q')
	plt.xlabel('X Pixel Location', fontsize=20)

	plt.ylabel('Y Pixel Location', fontsize=20)
	plt.xticks(fontsize=15)
	plt.yticks(fontsize=15)
	plt.xlim([x_min, 200])
	plt.title('Pixel Map Colored by dQ/Q')
	plt.tight_layout()
	plt.grid(True, linestyle='--', alpha=0.5)
	plt.savefig(output_file, dpi=300)
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
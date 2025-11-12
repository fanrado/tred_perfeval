import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
import src.peakmem_eval as peakmem_eval
import src.runtime_eval as runtime_eval
import src.acc_effq_eval as acc_effq_eval
import numpy.lib.recfunctions as rfn
# input_dir = '/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/MEMORY_EVAL/no_dynBatchChunk/benchmark_plot'


def load_json(input_file=''):
	with open(input_file, 'r') as f:
		data = json.load(f)
	return data

def effq_accuracy_eval_with_diffusion_cap():
	'''
	    Plot the distribution of deltaQ = EffQ - EffQ_ref and dQ_over_Q = (EffQ - EffQ_ref)/EffQ_ref.
		A minimum value of the transverse spread is set set to 0.035 cm:
			sigma_T = max( sqrt(2*D_T*t_drift), 0.035 cm )
	'''
	## Accuracy of the effective charge calculation ----
	# root_path = '/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/ACC_EFFQ/preliminary/benchmark_plots'
	root_path = "/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/ACC_EFFQ/cap_on_diffspread"

	# initialize the variables to store the deltaQ and dQ_over_Q
	deltaQ_10x10_4x4x2, dQ_over_Q_10x10_4x4x2 = None, None
	deltaQ_8x8_2x2x2, dQ_over_Q_8x8_2x2x2 = None, None
	deltaQ_6x6_2x2x2, dQ_over_Q_6x6_2x2x2 = None, None
	readFrom_npz = False
	getEffq = True  # Set to True to evaluate the effective charge accuracy, False for accumulated charge accuracy
	cut_on_Q = 0.001 #0.01 # ke-
	if readFrom_npz:
		saveHDF5 = True

		## DIFFERENT DIFFUSION COEFF
		path_to_ref = '/'.join([root_path, '10x10_partitions_2x2x2_drifttime_nocut.npz'])
		path_to_10x10_4x4x2 = '/'.join([root_path, '10x10_partitions_4x4x2_drifttime_nocut.npz'])
		output_file_10x10_hdf5 = '/'.join([root_path, 'HDF5/EffectiveCharge_10x10_4x4x2.hdf5'])
		acc_effq_eval.npz2hdf5(npz_data=path_to_10x10_4x4x2, npz_ref=path_to_ref, saveHDF5=saveHDF5, output_hdf5=output_file_10x10_hdf5, getEffq=getEffq)

		# path_to_ref = '/'.join([root_path, '10x10_partitions_2x2x2_DT88cm2.npz'])
		# path_to_10x10_4x4x2_DT88cm2 = '/'.join([root_path, '10x10_partitions_4x4x2_DT88cm2.npz'])
		# output_file_10x10_DT88cm2_hdf5 = '/'.join([root_path, 'HDF5_coarse_grain_cut_0pt01ke_DT88cm2/EffectiveCharge_10x10_4x4x2_DT88cm2.hdf5'])
		# acc_effq_eval.npz2hdf5(npz_data=path_to_10x10_4x4x2_DT88cm2, npz_ref=path_to_ref, saveHDF5=saveHDF5, output_hdf5=output_file_10x10_DT88cm2_hdf5, getEffq=getEffq)

	else:

		hdf5_file_10x10 = '/'.join([root_path, 'HDF5/EffectiveCharge_10x10_4x4x2.hdf5'])
		# hdf5_file_10x10_DT88cm2 = '/'.join([root_path, 'HDF5_coarse_grain_cut_0pt01ke_DT88cm2/EffectiveCharge_10x10_4x4x2_DT88cm2.hdf5'])
		deltaQ_10x10_4x4x2, dQ_over_Q_10x10_4x4x2, high_dQ_over_Q_10x10, Npix_tot_10x10, Npix_belowthr_10x10 = acc_effq_eval.load_Q_fromHDF5(hdf5_file=hdf5_file_10x10, cut_on_Qref=cut_on_Q, getEffq=getEffq)

		# deltaQ_10x10_4x4x2_DT88cm2, dQ_over_Q_10x10_4x4x2_DT88cm2, high_dQ_over_Q_10x10_DT88cm2, Npix_tot_10x1_DT88cm2, Npix_belowthr_10x10_DT88cm2 = acc_effq_eval.load_Q_fromHDF5(hdf5_file=hdf5_file_10x10_DT88cm2, cut_on_Qref=cut_on_Q, getEffq=getEffq)

		## DIFFERENT DIFFUSION COEFF
		delta_Q_list = [(deltaQ_10x10_4x4x2, '(4,4,2) x (10,10)', 'blue')]
						# (deltaQ_8x8_2x2x2, '(2,2,2) x (8,8)', 'green'),
						# (deltaQ_6x6_2x2x2, '(2,2,2) x (6,6)', 'blue')]
		output_file = '/'.join([root_path, 'HDF5/deltaQ_overlay_10x10_8x8_6x6.png'])
		acc_effq_eval.overlay_hists_deltaQ(*delta_Q_list, title='Effective charge distributions wrt (4,4,2) x (10,10)', xlabel='Delta Q [ke-]', ylabel='Counts', output_file=output_file)

		dQ_over_Q_list = [(dQ_over_Q_10x10_4x4x2, '(4,4,2) x (10,10)', 'blue')]
						# (dQ_over_Q_8x8_2x2x2, '(2,2,2) x (8,8)', 'green'),
						# (dQ_over_Q_6x6_2x2x2, '(2,2,2) x (6,6)', 'blue'),]
						
		output_file = '/'.join([root_path, 'HDF5/dQ_over_Q_overlay_10x10_8x8_6x6.png'])
		# output_file = '/'.join([root_path, 'test.png'])
		acc_effq_eval.overlay_hists_deltaQ(*dQ_over_Q_list, title='Relative difference of the charges at each pixel', xlabel=r'$(Q-Q_{ref})/Q_{ref}$', ylabel='Counts', output_file=output_file)


def memory_evaluation():
	# Peak memory usage ----
	# input_dir = '/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/MEMORY_EVAL/preliminary/no_dynBatchChunk/benchmark_plot'
	input_dir = '/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/MEMORY_EVAL/benchmark_plot_nodynChunkBatch'
	
	peakmem_eval.benchmark_peak_memory_nodynamic_chunking_and_batching(input_path=input_dir)
	# input_dir = "/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/MEMORY_EVAL/benchmark_plot_dynChunkBatch/files_plot"
	# input_dir = '/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/MEMORY_EVAL/benchmark_plot_dynChunkBatch/files_plot'
	# #
	# input_dir = '/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/MEMORY_EVAL/benchmark_plot_dynChunkBatch_xyzlim'
	# peakmem_eval.benchmark_peak_memory_dynamic_chunking_and_batching(input_path=input_dir)

def runtime_evaluation():
	## Runtime evaluation ----
	## PIECHART RUNTIME for one event
	# input_dir = '/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/RUNTIME_EVAL/preliminary/benchmark_plot/to_use'
	input_dir = '/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/RUNTIME_EVAL/preliminary/benchmark_plot/new_benchmark_plot'
	# input_subdir = 'runon_wcgpu1'
	input_subdir = ''
	filename = 'runtime_batchsize8192_NBCHUNK100_NBCHUNKCONV50.json'
	input_file = '/'.join([input_dir, input_subdir, filename])
	data = load_json(input_file=input_file)
	runtime_majorOps = runtime_eval.get_runtime_majorOperations(data=data)
	runtime_eval.runtimeshare_majorOp(organized_data=runtime_majorOps, output_file='/'.join([input_dir, input_subdir, 'runtime_share_majorOps.png']))
	## Overlay of the runtime for different batch schemes
	input_dir = '/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/RUNTIME_EVAL/preliminary/benchmark_plot/new_benchmark_plot'
	runtime_eval.benchmark_runtime(input_path=input_dir)

def effq_accuracy_eval_cuton_drifttime():
	# root_path = "/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/ACC_EFFQ/cut_on_drifttime_30_100_nocut_event1003"
	root_path = "/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/ACC_EFFQ/November10_2025/CUT_on_TDRIFT"

	# initialize the variables to store the deltaQ and dQ_over_Q
	deltaQ_10x10_4x4x2, dQ_over_Q_10x10_4x4x2 = None, None
	# deltaQ_8x8_2x2x2, dQ_over_Q_8x8_2x2x2 = None, None
	# deltaQ_6x6_2x2x2, dQ_over_Q_6x6_2x2x2 = None, None
	readFrom_npz = False
	getEffq = True  # Set to True to evaluate the effective charge accuracy, False for accumulated charge accuracy
	cut_on_Q = 0.01 #0.01 # ke-
	if readFrom_npz:
		saveHDF5 = True
		##
		## SAME DIFFUSION COEFF
		path_to_ref = '/'.join([root_path, '2x2x2/2x2x2_tdrift10.npz'])
		path_to_10x10_4x4x2 = '/'.join([root_path, '4x4x2/4x4x2_tdrift10.npz'])
		output_file_10x10_hdf5 = '/'.join([root_path, 'HDF5/EffectiveCharge_10x10_4x4x2_tdrift10.hdf5'])
		acc_effq_eval.npz2hdf5(npz_data=path_to_10x10_4x4x2, npz_ref=path_to_ref, saveHDF5=saveHDF5, output_hdf5=output_file_10x10_hdf5, getEffq=getEffq)

		# path_to_ref = '/'.join([root_path, '10x10_partitions_2x2x2_drifttime_morethan100.npz'])
		# path_to_10x10_4x4x2_DT88cm2 = '/'.join([root_path, '10x10_partitions_4x4x2_drifttime_morethan100.npz'])
		# output_file_10x10_DT88cm2_hdf5 = '/'.join([root_path, 'HDF5/EffectiveCharge_10x10_4x4x2_drifttime_morethan100.hdf5'])
		# acc_effq_eval.npz2hdf5(npz_data=path_to_10x10_4x4x2_DT88cm2, npz_ref=path_to_ref, saveHDF5=saveHDF5, output_hdf5=output_file_10x10_DT88cm2_hdf5, getEffq=getEffq)

		# # no cut 
		# path_to_ref_nocut = '/'.join([root_path, '10x10_partitions_2x2x2_drifttime_nocut.npz'])
		# path_to_10x10_4x4x2_nocut = '/'.join([root_path, '10x10_partitions_4x4x2_drifttime_nocut.npz'])
		# output_file_10x10_nocut_hdf5 = '/'.join([root_path, 'HDF5/EffectiveCharge_10x10_4x4x2_nocut.hdf5'])
		# acc_effq_eval.npz2hdf5(npz_data=path_to_10x10_4x4x2_nocut, npz_ref=path_to_ref_nocut, saveHDF5=saveHDF5, output_hdf5=output_file_10x10_nocut_hdf5, getEffq=getEffq)

	else:
		hdf5_file_10x10_drifttime_10us = '/'.join([root_path, 'HDF5/EffectiveCharge_10x10_4x4x2_tdrift10.hdf5'])
		hdf5_file_10x10_drifttime_30us = '/'.join([root_path, 'HDF5/EffectiveCharge_10x10_4x4x2_tdrift30.hdf5'])
		hdf5_file_10x10_drifttime_20us = '/'.join([root_path, 'HDF5/EffectiveCharge_10x10_4x4x2_tdrift20.hdf5'])
		hdf5_file_10x10_drifttime_60us = '/'.join([root_path, 'HDF5/EffectiveCharge_10x10_4x4x2_tdrift60.hdf5'])

		deltaQ_10x10_4x4x2_drifttime_10us, dQ_over_Q_10x10_4x4x2_drifttime_10us, high_dQ_over_Q_10x10, Npix_tot_10x10, Npix_belowthr_10x10 = acc_effq_eval.load_Q_fromHDF5(hdf5_file=hdf5_file_10x10_drifttime_10us, cut_on_Qref=cut_on_Q, getEffq=getEffq)

		deltaQ_10x10_4x4x2_drifttime_30us, dQ_over_Q_10x10_4x4x2_drifttime_30us, high_dQ_over_Q_10x10_DT88cm2, Npix_tot_10x1_DT88cm2, Npix_belowthr_10x10_DT88cm2 = acc_effq_eval.load_Q_fromHDF5(hdf5_file=hdf5_file_10x10_drifttime_30us, cut_on_Qref=cut_on_Q, getEffq=getEffq)

		deltaQ_10x10_4x4x2_drifttime_20us, dQ_over_Q_10x10_4x4x2_drifttime_20us, _, _, _ = acc_effq_eval.load_Q_fromHDF5(hdf5_file=hdf5_file_10x10_drifttime_20us, cut_on_Qref=cut_on_Q, getEffq=getEffq)

		deltaQ_10x10_4x4x2_drifttime_60us, dQ_over_Q_10x10_4x4x2_drifttime_60us, high_dQ_over_Q_10x10_nocut, Npix_tot_10x10_nocut, Npix_belowthr_10x10_nocut = acc_effq_eval.load_Q_fromHDF5(hdf5_file=hdf5_file_10x10_drifttime_60us, cut_on_Qref=cut_on_Q, getEffq=getEffq)

		## SAME DIFFUSION COEFF
		delta_Q_list = [(deltaQ_10x10_4x4x2_drifttime_10us, '(4,4,2) x (10,10), drift time < 10 us', 'red', None),#, 8.8 cm2 Transversal diff coeff
				  		(deltaQ_10x10_4x4x2_drifttime_30us, '(4,4,2) x (10,10), drift time < 30 us', 'green', '--'),
						  (deltaQ_10x10_4x4x2_drifttime_60us, '(4,4,2) x (10,10), drift time < 60 us', 'purple', None),
						  (deltaQ_10x10_4x4x2_drifttime_20us, '(4,4,2) x (10,10), drift time < 20 us', 'blue', ':')]
						# (deltaQ_8x8_2x2x2, '(2,2,2) x (8,8)', 'green'),
						# (deltaQ_6x6_2x2x2, '(2,2,2) x (6,6)', 'blue')]
		output_file = '/'.join([root_path, 'HDF5/deltaQ_overlay_10x10_8x8_6x6.png'])
		acc_effq_eval.overlay_hists_deltaQ(*delta_Q_list, title='Effective charge distributions wrt (2,2,2) x (10,10)', xlabel='Delta Q [ke-]', ylabel='Counts', output_file=output_file)

		dQ_over_Q_list = [(dQ_over_Q_10x10_4x4x2_drifttime_10us, '(4,4,2) x (10,10), drift time < 10 us', 'red', None), #, 8.8 cm2 Transversal diff coeff
						(dQ_over_Q_10x10_4x4x2_drifttime_30us, '(4,4,2) x (10,10), drift time < 30 us', 'green', '--'),
						(dQ_over_Q_10x10_4x4x2_drifttime_60us, '(4,4,2) x (10,10), drift time < 60 us', 'purple', None),
						(dQ_over_Q_10x10_4x4x2_drifttime_20us, '(4,4,2) x (10,10), drift time < 20 us', 'blue', ':')]
						# # (dQ_over_Q_8x8_2x2x2, '(2,2,2) x (8,8)', 'green'),
						# (dQ_over_Q_6x6_2x2x2, '(2,2,2) x (6,6)', 'blue'),]
						
		output_file = '/'.join([root_path, 'HDF5/dQ_over_Q_overlay_10x10_8x8_6x6.png'])
		# output_file = '/'.join([root_path, 'test.png'])
		acc_effq_eval.overlay_hists_deltaQ(*dQ_over_Q_list, title='Relative difference of the charges at each pixel', xlabel=r'$(Q-Q_{ref})/Q_{ref}$', ylabel='Counts', output_file=output_file)

def effq_accuracy_eval_diffent_diffCoeff():
	'''
	    Run this code if you want to convert the .npz to hdf5 and use the hdf5 to evaluate the accuracy of the effective charge calculation.
		I don't wanna modify this for the test I am doing now.
	'''
	## Accuracy of the effective charge calculation ----
	# root_path = '/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/ACC_EFFQ/preliminary/benchmark_plots'
	root_path = "/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/ACC_EFFQ/effq_out_nt_10_DIFFcoeff"

	# initialize the variables to store the deltaQ and dQ_over_Q
	deltaQ_10x10_4x4x2, dQ_over_Q_10x10_4x4x2 = None, None
	deltaQ_8x8_2x2x2, dQ_over_Q_8x8_2x2x2 = None, None
	deltaQ_6x6_2x2x2, dQ_over_Q_6x6_2x2x2 = None, None
	readFrom_npz = False
	getEffq = True  # Set to True to evaluate the effective charge accuracy, False for accumulated charge accuracy
	cut_on_Q = 0.01 #0.01 # ke-
	if readFrom_npz:
		saveHDF5 = True

		## DIFFERENT DIFFUSION COEFF
		path_to_ref = '/'.join([root_path, '10x10_partitions_2x2x2.npz'])
		path_to_10x10_4x4x2 = '/'.join([root_path, '10x10_partitions_4x4x2.npz'])
		output_file_10x10_hdf5 = '/'.join([root_path, 'HDF5_coarse_grain_cut_0pt01ke_DT88cm2/EffectiveCharge_10x10_4x4x2.hdf5'])
		acc_effq_eval.npz2hdf5(npz_data=path_to_10x10_4x4x2, npz_ref=path_to_ref, saveHDF5=saveHDF5, output_hdf5=output_file_10x10_hdf5, getEffq=getEffq)

		path_to_ref = '/'.join([root_path, '10x10_partitions_2x2x2_DT88cm2.npz'])
		path_to_10x10_4x4x2_DT88cm2 = '/'.join([root_path, '10x10_partitions_4x4x2_DT88cm2.npz'])
		output_file_10x10_DT88cm2_hdf5 = '/'.join([root_path, 'HDF5_coarse_grain_cut_0pt01ke_DT88cm2/EffectiveCharge_10x10_4x4x2_DT88cm2.hdf5'])
		acc_effq_eval.npz2hdf5(npz_data=path_to_10x10_4x4x2_DT88cm2, npz_ref=path_to_ref, saveHDF5=saveHDF5, output_hdf5=output_file_10x10_DT88cm2_hdf5, getEffq=getEffq)

	else:

		hdf5_file_10x10 = '/'.join([root_path, 'HDF5_coarse_grain_cut_0pt01ke_DT88cm2/EffectiveCharge_10x10_4x4x2.hdf5'])
		hdf5_file_10x10_DT88cm2 = '/'.join([root_path, 'HDF5_coarse_grain_cut_0pt01ke_DT88cm2/EffectiveCharge_10x10_4x4x2_DT88cm2.hdf5'])
		deltaQ_10x10_4x4x2, dQ_over_Q_10x10_4x4x2, high_dQ_over_Q_10x10, Npix_tot_10x10, Npix_belowthr_10x10 = acc_effq_eval.load_Q_fromHDF5(hdf5_file=hdf5_file_10x10, cut_on_Qref=cut_on_Q, getEffq=getEffq)

		deltaQ_10x10_4x4x2_DT88cm2, dQ_over_Q_10x10_4x4x2_DT88cm2, high_dQ_over_Q_10x10_DT88cm2, Npix_tot_10x1_DT88cm2, Npix_belowthr_10x10_DT88cm2 = acc_effq_eval.load_Q_fromHDF5(hdf5_file=hdf5_file_10x10_DT88cm2, cut_on_Qref=cut_on_Q, getEffq=getEffq)

		## DIFFERENT DIFFUSION COEFF
		delta_Q_list = [(deltaQ_10x10_4x4x2, '(4,4,2) x (10,10), 8.8 cm2/us Transverse diff coeff', 'red'), #, 8.8 cm2 Transversal diff coeff
				  		(deltaQ_10x10_4x4x2_DT88cm2, '(4,4,2) x (10,10), 88 cm2/us Transverse diff coeff', 'green'),]
						# (deltaQ_8x8_2x2x2, '(2,2,2) x (8,8)', 'green'),
						# (deltaQ_6x6_2x2x2, '(2,2,2) x (6,6)', 'blue')]
		output_file = '/'.join([root_path, 'HDF5_coarse_grain_cut_0pt01ke_DT88cm2/deltaQ_overlay_10x10_8x8_6x6.png'])
		acc_effq_eval.overlay_hists_deltaQ(*delta_Q_list, title='Effective charge distributions wrt (2,2,2) x (10,10)', xlabel='Delta Q [ke-]', ylabel='Counts', output_file=output_file)

		dQ_over_Q_list = [(dQ_over_Q_10x10_4x4x2, '(4,4,2) x (10,10), 8.8 cm2/us Transverse diff coeff', 'red'), #, 8.8 cm2 Transversal diff coeff
						(dQ_over_Q_10x10_4x4x2_DT88cm2, '(4,4,2) x (10,10), 88 cm2/us Transverse diff coeff', 'green')]
						# (dQ_over_Q_8x8_2x2x2, '(2,2,2) x (8,8)', 'green'),
						# (dQ_over_Q_6x6_2x2x2, '(2,2,2) x (6,6)', 'blue'),]
						
		output_file = '/'.join([root_path, 'HDF5_coarse_grain_cut_0pt01ke_DT88cm2/dQ_over_Q_overlay_10x10_8x8_6x6.png'])
		# output_file = '/'.join([root_path, 'test.png'])
		acc_effq_eval.overlay_hists_deltaQ(*dQ_over_Q_list, title='Relative difference of the charges at each pixel', xlabel=r'$(Q-Q_{ref})/Q_{ref}$', ylabel='Counts', output_file=output_file)

def effq_accuracy_evaluation():
	root_path = "/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/ACC_EFFQ/November10_2025"

	# initialize the variables to store the deltaQ and dQ_over_Q
	deltaQ_10x10_4x4x2, dQ_over_Q_10x10_4x4x2 = None, None
	deltaQ_8x8_2x2x2, dQ_over_Q_8x8_2x2x2 = None, None
	deltaQ_6x6_2x2x2, dQ_over_Q_6x6_2x2x2 = None, None
	readFrom_npz = False
	getEffq = True  # Set to True to evaluate the effective charge accuracy, False for accumulated charge accuracy
	cut_on_Q = 0.01 #0.01 # ke-
	if readFrom_npz:
		saveHDF5 = True
		##
		## SAME DIFFUSION COEFF
		path_to_ref = '/'.join([root_path, '10x10_2x2x2_tdrift20.npz'])
		path_to_10x10_4x4x2 = '/'.join([root_path, '10x10_4x4x2_tdrift20.npz'])
		output_file_10x10_hdf5 = '/'.join([root_path, 'HDF5/EffectiveCharge_10x10_4x4x2_tdrift20.hdf5'])
		acc_effq_eval.npz2hdf5(npz_data=path_to_10x10_4x4x2, npz_ref=path_to_ref, saveHDF5=saveHDF5, output_hdf5=output_file_10x10_hdf5, getEffq=getEffq)

		path_to_ref = '/'.join([root_path, '10x10_2x2x2_tdrift20.npz'])
		path_to_8x8_2x2x2_tdrift20 = '/'.join([root_path, '8x8_2x2x2_tdrift20.npz'])
		output_file_8x8_tdrift20 = '/'.join([root_path, 'HDF5/EffectiveCharge_8x8_2x2x2_tdrift20.hdf5'])
		acc_effq_eval.npz2hdf5(npz_data=path_to_8x8_2x2x2_tdrift20, npz_ref=path_to_ref, saveHDF5=saveHDF5, output_hdf5=output_file_8x8_tdrift20, getEffq=getEffq)

		# no cut 
		path_to_ref_nocut = '/'.join([root_path, '10x10_2x2x2_tdrift20.npz'])
		path_to_6x6_2x2x2_tdrift20 = '/'.join([root_path, '6x6_2x2x2_tdrift20.npz'])
		output_file_6x6_2x2x2_tdrift20 = '/'.join([root_path, 'HDF5/EffectiveCharge_6x6_2x2x2_tdrift20.hdf5'])
		acc_effq_eval.npz2hdf5(npz_data=path_to_6x6_2x2x2_tdrift20, npz_ref=path_to_ref_nocut, saveHDF5=saveHDF5, output_hdf5=output_file_6x6_2x2x2_tdrift20, getEffq=getEffq)

	else:
		hdf5_file_10x10 = '/'.join([root_path, 'HDF5/EffectiveCharge_10x10_4x4x2_tdrift20.hdf5'])
		hdf5_file_8x8 = '/'.join([root_path, 'HDF5/EffectiveCharge_8x8_2x2x2_tdrift20.hdf5'])
		hdf5_file_6x6 = '/'.join([root_path, 'HDF5/EffectiveCharge_6x6_2x2x2_tdrift20.hdf5'])

		deltaQ_10x10_4x4x2, dQ_over_Q_10x10_4x4x2, high_dQ_over_Q_10x10, Npix_tot_10x10, Npix_belowthr_10x10 = acc_effq_eval.load_Q_fromHDF5(hdf5_file=hdf5_file_10x10, cut_on_Qref=cut_on_Q, getEffq=getEffq)

		deltaQ_8x8_2x2x2, dQ_over_Q_8x8_2x2x2, high_dQ_over_Q_10x10_DT88cm2, Npix_tot_10x1_DT88cm2, Npix_belowthr_10x10_DT88cm2 = acc_effq_eval.load_Q_fromHDF5(hdf5_file=hdf5_file_8x8, cut_on_Qref=cut_on_Q, getEffq=getEffq)

		deltaQ_6x6_2x2x2, dQ_over_Q_6x6_2x2x2, high_dQ_over_Q_10x10_nocut, Npix_tot_10x10_nocut, Npix_belowthr_10x10_nocut = acc_effq_eval.load_Q_fromHDF5(hdf5_file=hdf5_file_6x6, cut_on_Qref=cut_on_Q, getEffq=getEffq)

		## SAME DIFFUSION COEFF
		delta_Q_list = [(deltaQ_10x10_4x4x2, '(4,4,2) x (10,10)', 'red', None), #, 8.8 cm2 Transversal diff coeff
				  		(deltaQ_8x8_2x2x2, '(2,2,2) x (8,8)', 'green', '--'),
						  (deltaQ_6x6_2x2x2, '(2,2,2) x (6,6)', 'blue', ':')]
						# (deltaQ_8x8_2x2x2, '(2,2,2) x (8,8)', 'green'),
						# (deltaQ_6x6_2x2x2, '(2,2,2) x (6,6)', 'blue')]
		output_file = '/'.join([root_path, 'HDF5/deltaQ_overlay_10x10_8x8_6x6.png'])
		acc_effq_eval.overlay_hists_deltaQ(*delta_Q_list, title='Effective charge distributions wrt (2,2,2) x (10,10)', xlabel='Delta Q [ke-]', ylabel='Counts', output_file=output_file)

		dQ_over_Q_list = [(dQ_over_Q_10x10_4x4x2, '(4,4,2) x (10,10)', 'red', None), #, 8.8 cm2 Transversal diff coeff
						(dQ_over_Q_8x8_2x2x2, '(2,2,2) x (8,8)', 'green', '--'),
						(dQ_over_Q_6x6_2x2x2, '(2,2,2) x (6,6)', 'blue', ':')]
						# (dQ_over_Q_8x8_2x2x2, '(2,2,2) x (8,8)', 'green'),
						# (dQ_over_Q_6x6_2x2x2, '(2,2,2) x (6,6)', 'blue'),]
						
		output_file = '/'.join([root_path, 'HDF5/dQ_over_Q_overlay_10x10_8x8_6x6.png'])
		# output_file = '/'.join([root_path, 'test.png'])
		acc_effq_eval.overlay_hists_deltaQ(*dQ_over_Q_list, title='Relative difference of the charges at each pixel', xlabel=r'$(Q-Q_{ref})/Q_{ref}$', ylabel='Counts', output_file=output_file)


def fill_missing_data(data, ref):
	pix_locs_array = acc_effq_eval.get_all_pixels_locations(data, ref)
	pix_keys = acc_effq_eval.locations_to_keys(pix_locs_array)
	data_keys = acc_effq_eval.locations_to_keys(data['pixels_locations'])
	ref_keys = acc_effq_eval.locations_to_keys(ref['pixels_locations'])

	data_key_to_idx = {key: idx for idx, key in enumerate(data_keys)}
	ref_key_to_idx = {key: idx for idx, key in enumerate(ref_keys)}

	missing_in_data = np.array([key not in data_key_to_idx for key in pix_keys])
	missing_in_ref = np.array([key not in ref_key_to_idx for key in pix_keys])

	if np.any(missing_in_data):
		missing_data_locs = pix_locs_array[missing_in_data]
		tmp_charge = np.array([(tuple(loc), 0.0) for loc in missing_data_locs], dtype=data.dtype)
		old_len = len(data)
		data = rfn.stack_arrays((data, tmp_charge), usemask=False)
		
		missing_indices = np.where(missing_in_data)[0]
		for i, idx in enumerate(missing_indices):
			key = pix_keys[idx]
			data_key_to_idx[key] = old_len + i
	if np.any(missing_in_ref):
		missing_ref_locs = pix_locs_array[missing_in_ref]
		tmp_charge = np.array([(tuple(loc), 0.0) for loc in missing_ref_locs], dtype=ref.dtype)
		old_len = len(ref)
		ref = rfn.stack_arrays((ref, tmp_charge), usemask=False)
		
		missing_indices = np.where(missing_in_ref)[0]
		for i, idx in enumerate(missing_indices):
			key = pix_keys[idx]
			ref_key_to_idx[key] = old_len + i
	
	data_indices = np.array([data_key_to_idx[key] for key in pix_keys])
	ref_indices = np.array([ref_key_to_idx[key] for key in pix_keys])
	# return data, ref, data_key_to_idx, ref_key_to_idx, pix_locs_array
	# return {'data': data,
	# 	    'ref': ref,
	# 		'data_indices': data_indices,
	# 		'ref_indices': ref_indices,
	# 		'data_key_to_idx': data_key_to_idx,
	# 		'ref_key_to_idx': ref_key_to_idx,
	# 		'pix_locs_array': pix_locs_array}
	data_low_timetick = data
	ref_low_timetick = ref
	try:
		charge_in_data = data_low_timetick['effective_charge'][data_indices]
		charge_in_ref = ref_low_timetick['effective_charge'][ref_indices]
		
		pixLocs_tpc = pix_locs_array.copy()
		pixLocs_ref_tpc = pix_locs_array.copy()
		tpc_array = np.zeros(len(charge_in_data), dtype=[('pixels_locations', np.int32, (3,)), ('pixels_locations_ref', np.int32, (3,)), ('effective_charge', np.float32), ('effective_charge_ref', np.float32)])
		if len(pixLocs_tpc) == 0:
			tpc_array['pixels_locations'] = pixLocs_tpc.reshape(0,3)
			tpc_array['pixels_locations_ref'] = pixLocs_ref_tpc.reshape(0,3)
			tpc_array['effective_charge'] = charge_in_data[:]
			tpc_array['effective_charge_ref'] = charge_in_ref[:]
		else:
			tpc_array['pixels_locations'] = pixLocs_tpc[:]
			tpc_array['pixels_locations_ref'] = pixLocs_ref_tpc[:]
			tpc_array['effective_charge'] = charge_in_data[:]
			tpc_array['effective_charge_ref'] = charge_in_ref[:]
			return tpc_array
	except:
		pass
	return np.zeros(0, dtype=[('pixels_locations', np.int32, (3,)), ('pixels_locations_ref', np.int32, (3,)), ('effective_charge', np.float32), ('effective_charge_ref', np.float32)])

def separation_by_time():
	root_path = "/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/ACC_EFFQ/effq_out_nt_10"
	output_dir = "tests/effq_by_time"
	getEffq = True
	readFrom_npz = True
	cut_on_Q = 0.01 # ke-
	# path_to_ref = '/'.join([root_path, '10x10_partitions_2x2x2_DT88cm2.npz'])
	# path_to_data = '/'.join([root_path, '10x10_partitions_4x4x2_DT88cm2.npz'])
	path_to_ref = '/'.join([root_path, '10x10_partitions_2x2x2.npz'])
	path_to_data = '/'.join([root_path, '10x10_partitions_4x4x2.npz'])
	
	all_tpc_data_ref = acc_effq_eval.load_npz(input_file=path_to_ref, cut_on_Q=1e-12, getEffq=getEffq)
	all_tpc_data_    = acc_effq_eval.load_npz(input_file=path_to_data, cut_on_Q=1e-12, getEffq=getEffq)
	print(all_tpc_data_ref.keys())

	all_tpc_hightimetick = {f'{itpc}': None for itpc in all_tpc_data_ref.keys()}
	all_tpc_lowtimetick = {f'{itpc}': None for itpc in all_tpc_data_ref.keys()}

	all_tpc_deltaQ_low_timetick = {f'{itpc}': None for itpc in all_tpc_data_ref.keys()}
	all_tpc_deltaQ_high_timetick = {f'{itpc}': None for itpc in all_tpc_data_ref.keys()}
	cutonTimetick_low = 1000
	cutonTimetick_high = 1000
	for itpc in all_tpc_data_ref.keys():
		tpc_data_ref = all_tpc_data_ref[itpc]
		tpc_data_    = all_tpc_data_[itpc]

		mask_low_timetick_ref = tpc_data_ref['pixels_locations'][:, -1] < cutonTimetick_low
		mask_high_timetick_ref = tpc_data_ref['pixels_locations'][:, -1] >= cutonTimetick_high
		mask_low_timetick_ = tpc_data_['pixels_locations'][:, -1] < cutonTimetick_low
		mask_high_timetick_ = tpc_data_['pixels_locations'][:, -1] >= cutonTimetick_high

		ref_high_timetick = tpc_data_ref[mask_high_timetick_ref]
		ref_low_timetick = tpc_data_ref[mask_low_timetick_ref]
		data_high_timetick = tpc_data_[mask_high_timetick_]
		data_low_timetick = tpc_data_[mask_low_timetick_]

		low_time_tick_tpc = fill_missing_data(data=data_low_timetick, ref=ref_low_timetick)
		high_time_tick_tpc = fill_missing_data(data=data_high_timetick, ref=ref_high_timetick)
		all_tpc_lowtimetick[itpc] = low_time_tick_tpc if low_time_tick_tpc is not None else np.array([], dtype=tpc_data_.dtype)
		all_tpc_hightimetick[itpc] = high_time_tick_tpc if high_time_tick_tpc is not None else np.array([], dtype=tpc_data_.dtype)
		
		all_tpc_deltaQ_low_timetick[itpc] = all_tpc_lowtimetick[itpc]['effective_charge'] - all_tpc_lowtimetick[itpc]['effective_charge_ref']
		all_tpc_deltaQ_high_timetick[itpc] = all_tpc_hightimetick[itpc]['effective_charge'] - all_tpc_hightimetick[itpc]['effective_charge_ref']
	All_DeltaQ_lowtimetick = np.concatenate([all_tpc_deltaQ_low_timetick[itpc] for itpc in all_tpc_deltaQ_low_timetick.keys()])
	All_DeltaQ_hightimetick = np.concatenate([all_tpc_deltaQ_high_timetick[itpc] for itpc in all_tpc_deltaQ_high_timetick.keys()])
	plt.figure(figsize=(8,6))
	plt.hist(All_DeltaQ_lowtimetick, bins=100, histtype='step', color='blue', alpha=0.7, label=f'Time tick < {cutonTimetick_low}')
	plt.hist(All_DeltaQ_hightimetick, bins=100, histtype='step', color='red', alpha=0.7, label=f'Time tick >= {cutonTimetick_high}')
	plt.xlabel('Delta Q [ke-]')
	plt.ylabel('Counts')
	plt.title(r'Delta Q distribution, transverse diffusion coeff = 8.8 $cm^2/s$')
	plt.grid()
	plt.yscale('log')
	plt.tight_layout()
	plt.legend()
	plt.savefig('/'.join([output_dir, 'deltaQ_distribution_10x10_partitions_8pt8_diffcoeff.png']))
	# plt.savefig('/'.join([output_dir, 'deltaQ_distribution_10x10_partitions_DT88cm2_diffcoeff.png']))
	plt.close()
#--------------
	plotDistTimetick = False
	if plotDistTimetick:
		dist_timeticks = np.array([], dtype=np.int32)
		for itpc, tpcdata in all_tpc_data_.items():
			print(tpcdata.dtype)
			# print(tpcdata['pixels_locations'])
			dist_timeticks = np.concatenate((dist_timeticks, tpcdata['pixels_locations'][:, -1]))

		print(dist_timeticks)
		plt.figure(figsize=(8,6))
		plt.hist(dist_timeticks, bins=100, histtype='stepfilled', color='blue', alpha=0.7)
		plt.xlabel('Time ticks')
		plt.ylabel('Counts')
		plt.title('Distribution of the pixels along time ticks')
		plt.grid()
		plt.savefig('/'.join([output_dir, 'pixel_time_distribution.png']))
		plt.close()
	
if __name__ == '__main__':
	# separation_by_time()
	# effq_accuracy_evaluation()

	# effq_accuracy_eval_cuton_drifttime()
	# effq_accuracy_eval_cuton_loctime()
	# effq_accuracy_eval_diffent_diffCoeff()
	# effq_accuracy_eval_with_diffusion_cap()
	runtime_evaluation()
	# memory_evaluation()
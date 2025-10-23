import os, sys, json
# import numpy as np
# import matplotlib.pyplot as plt
import src.peakmem_eval as peakmem_eval
import src.runtime_eval as runtime_eval
import src.acc_effq_eval as acc_effq_eval
# input_dir = '/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/MEMORY_EVAL/no_dynBatchChunk/benchmark_plot'


def load_json(input_file=''):
	with open(input_file, 'r') as f:
		data = json.load(f)
	return data

if __name__=='__main__':
	# # Peak memory usage ----
	# peakmem_eval.benchmark_peak_memory_nodynamic_chunking_and_batching(input_path=input_dir)
	# peakmem_eval.benchmark_peak_memory_dynamic_chunking_and_batching(input_path=input_dir)
	# # --------------------------
	# ## Runtime evaluation ----
	# ## PIECHART RUNTIME for one event
	# input_dir = '/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/RUNTIME_EVAL/benchmark_plot/to_use'
	# # input_subdir = 'runon_wcgpu1'
	# input_subdir = ''
	# filename = 'runtime_batchsize8192_NBCHUNK100_NBCHUNKCONV50.json'
	# input_file = '/'.join([input_dir, input_subdir, filename])
	# data = load_json(input_file=input_file)
	# runtime_majorOps = runtime_eval.get_runtime_majorOperations(data=data)
	# runtime_eval.runtimeshare_majorOp(organized_data=runtime_majorOps, output_file='/'.join([input_dir, input_subdir, 'runtime_share_majorOps.png']))
	# ## Overlay of the runtime for different batch schemes
	# input_dir = '/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/RUNTIME_EVAL/benchmark_plot/to_use'
	# runtime_eval.benchmark_runtime(input_path=input_dir)
	# ## --------------------------

	## Accuracy of the effective charge calculation ----
	root_path = '/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/ACC_EFFQ/benchmark_plots'

	# initialize the variables to store the deltaQ and dQ_over_Q
	deltaQ_10x10_4x4x2, dQ_over_Q_10x10_4x4x2 = None, None
	deltaQ_8x8_2x2x2, dQ_over_Q_8x8_2x2x2 = None, None
	deltaQ_6x6_2x2x2, dQ_over_Q_6x6_2x2x2 = None, None
	readFrom_npz = False
	if readFrom_npz:
		path_to_ref = '/'.join([root_path, '10x10_partitions_2x2x2.npz'])
		path_to_10x10_4x4x2 = '/'.join([root_path, '10x10_partitions_4x4x2.npz'])
		path_to_8x8_2x2x2 = '/'.join([root_path, '8x8_partitions_2x2x2.npz'])
		path_to_6x6_2x2x2 = '/'.join([root_path, '6x6_partitions_2x2x2.npz'])

		output_file_10x10_hdf5 = '/'.join([root_path, 'HDF5/deltaQ_dQoverQ_10x10_4x4x2.hdf5'])
		output_file_8x8_hdf5 = '/'.join([root_path, 'HDF5/deltaQ_dQoverQ_8x8_2x2x2.hdf5'])
		output_file_6x6_hdf5 = '/'.join([root_path, 'HDF5/deltaQ_dQoverQ_6x6_2x2x2.hdf5'])
		saveHDF5 = True
		deltaQ_10x10_4x4x2, dQ_over_Q_10x10_4x4x2 = acc_effq_eval.get_deltaQ_fromNPZ(npz_data=path_to_10x10_4x4x2, npz_ref=path_to_ref, saveHDF5=saveHDF5, output_hdf5=output_file_10x10_hdf5)
		deltaQ_8x8_2x2x2, dQ_over_Q_8x8_2x2x2 = acc_effq_eval.get_deltaQ_fromNPZ(npz_data=path_to_8x8_2x2x2, npz_ref=path_to_ref, saveHDF5=saveHDF5, output_hdf5=output_file_8x8_hdf5)
		deltaQ_6x6_2x2x2, dQ_over_Q_6x6_2x2x2 = acc_effq_eval.get_deltaQ_fromNPZ(npz_data=path_to_6x6_2x2x2, npz_ref=path_to_ref, saveHDF5=saveHDF5, output_hdf5=output_file_6x6_hdf5)
	else:
		hdf5_file_10x10 = '/'.join([root_path, 'HDF5/deltaQ_dQoverQ_10x10_4x4x2.hdf5'])
		hdf5_file_8x8 = '/'.join([root_path, 'HDF5/deltaQ_dQoverQ_8x8_2x2x2.hdf5'])
		hdf5_file_6x6 = '/'.join([root_path, 'HDF5/deltaQ_dQoverQ_6x6_2x2x2.hdf5'])
		deltaQ_10x10_4x4x2, dQ_over_Q_10x10_4x4x2 = acc_effq_eval.load_deltaQ_fromHDF5(hdf5_file=hdf5_file_10x10)
		deltaQ_8x8_2x2x2, dQ_over_Q_8x8_2x2x2 = acc_effq_eval.load_deltaQ_fromHDF5(hdf5_file=hdf5_file_8x8)
		deltaQ_6x6_2x2x2, dQ_over_Q_6x6_2x2x2 = acc_effq_eval.load_deltaQ_fromHDF5(hdf5_file=hdf5_file_6x6)

	delta_Q_list = [(deltaQ_10x10_4x4x2, '(4,4,2) x (10,10)', 'red'),
					(deltaQ_8x8_2x2x2, '(2,2,2) x (8,8)', 'green'),
					(deltaQ_6x6_2x2x2, '(2,2,2) x (6,6)', 'blue')]
	output_file = '/'.join([root_path, 'deltaQ_overlay_10x10_8x8_6x6.png'])
	# output_file = '/'.join([root_path, 'test.png'])
	acc_effq_eval.overlay_hists_deltaQ(*delta_Q_list, title='Accumulated charge distributions wrt (2,2,2) x (10,10)', xlabel='Delta Q [ke-]', ylabel='Counts', output_file=output_file)

	dQ_over_Q_list = [(dQ_over_Q_10x10_4x4x2, '(4,4,2) x (10,10)', 'red'),
					(dQ_over_Q_8x8_2x2x2, '(2,2,2) x (8,8)', 'green'),
					(dQ_over_Q_6x6_2x2x2, '(2,2,2) x (6,6)', 'blue')]
	output_file = '/'.join([root_path, 'dQ_over_Q_overlay_10x10_8x8_6x6.png'])
	# output_file = '/'.join([root_path, 'test.png'])
	acc_effq_eval.overlay_hists_deltaQ(*dQ_over_Q_list, title='Relative difference of the charges at each pixel', xlabel=r'$(Q-Q_{ref})/Q_{ref}$', ylabel='Counts', output_file=output_file, cut=1e3)
	## --------------------------
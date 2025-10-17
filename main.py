import os, sys, json
# import numpy as np
# import matplotlib.pyplot as plt
import src.peakmem_eval as peakmem_eval
import src.runtime_eval as runtime_eval

input_dir = '/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/MEMORY_EVAL'
# input_subdir = 'runon_wcgpu1'
input_subdir = ''
filename = 'peak_memory_usage_batch2048_nbchunk100_nbchunkconv50.json'
input_file = '/'.join([input_dir, input_subdir, filename])

def load_json(input_file=''):
	with open(input_file, 'r') as f:
		data = json.load(f)
	return data

if __name__=='__main__':
	peakmem_eval.benchmark_peak_memory_nodynamic_chunking_and_batching(input_path=input_dir)
	# print(input_dir)
	# data = load_json(input_file=input_file)
	# organized_data = peakmem_eval.organize_peakmem_data(data=data)
	# output_file = '/'.join([input_dir, input_subdir, 'peakmem_vs_Nsegments.png'])
	# peakmem_eval.overlay_plots((organized_data['N_segments'], organized_data['peak_memory_perbatch'], f'Batch size: {organized_data['batch_size']}, nbchunk: {organized_data['nbchunk']}, nbchunk_conv: {organized_data['nbchunk_conv']}', 'red'), title='Peak memory vs N_segments', output_file=output_file, xlabel='N_segments', ylabel='Peak Memory (MB)')
	# organized_data = runtime_eval.get_runtime_majorOperations(data=data)
	# print(organized_data.keys())
	# runtime_perbatch = runtime_eval.get_runtime_perbatch(data=data)
	# runtime_eval.overlay_plots(((runtime_perbatch['N_segments'], runtime_perbatch['runtimes_perbatch'], f'Batch size: {runtime_perbatch["batch_size"]}, nbchunk: {runtime_perbatch["nbchunk"]}, nbchunk_conv: {runtime_perbatch["nbchunk_conv"]}', 'blue'),), title='Runtime vs N_segments', xlabel='N_segments', ylabel='Runtime (sec)', output_file='/'.join([input_dir, input_subdir, 'runtime_vs_Nsegments.png']))

	# data = load_json(input_file=input_file)
	# runtime_majorOps = runtime_eval.get_runtime_majorOperations(data=data)
	# runtime_eval.runtimeshare_majorOp(organized_data=runtime_majorOps, output_file='/'.join([input_dir, input_subdir, 'runtime_share_majorOps.png']))
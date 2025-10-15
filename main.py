import os, sys, json
# import numpy as np
# import matplotlib.pyplot as plt
import src.peakmem_eval as peakmem_eval

input_dir = '/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/MEMORY_EVAL'
input_subdir = 'runon_wcgpu1'
filename = 'peak_memory_usage_2.json'
input_file = '/'.join([input_dir, input_subdir, filename])

def load_json(input_file=''):
	with open(input_file, 'r') as f:
		data = json.load(f)
	return data

if __name__=='__main__':
	print(input_dir)
	data = load_json(input_file=input_file)
	organized_data = peakmem_eval.organize_peakmem_data(data=data)
	# peakmem_eval.plot_peakmem_vs_nsegs(
	# 	n_segments=organized_data['N_segments'],
	# 	peak_memory=organized_data['peak_memory_perbatch'],
	# 	title='Peak Memory vs Number of Segments'
	# )
	peakmem_eval.overlay_plots((organized_data['N_segments'], organized_data['peak_memory_perbatch'], f'Batch size: {organized_data['batch_size']}, nbchunk: {organized_data['nbchunk']}, nbchunk_conv: {organized_data['nbchunk_conv']}', 'red'), title='Peak memory vs N_segments')
import os, sys, json
import numpy as np

def load_npz(input_file=''):
	data = np.load(input_file)
	hits_keys = [k for k in data.keys() if ('hits' in k) and ('location' not in k)]
	for k in hits_keys:
		print(f'data[{k}] : {data[k]}')

if __name__=='__main__':
	path_to_file = '/'.join(['/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/ACC_EFFQ', '10x10_partitions.npz'])
	load_npz(path_to_file)

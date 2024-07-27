import argparse
import json
import logging
import sys
import glob
# import time
from tqdm import tqdm

# import torch

import numpy as np


from lib import config, train, utils, seeds
from lib.models import energy, mlp
from lib.plot import plot
from lib.data import mnist as data_mnist
from lib.exp.varyingdatapoints import run_exp as vd_run_exp, read_exp_data as vd_read_exp_data
from ipynb_utils import ipynb_main
from main import run_energy_model_mnist

def model_acc_full_dataset(N=10, epochs=10, exp=run_energy_model_mnist, model:str=None):
	# gen. seeds
	seeds.set_seed(2019)
	# _seeds = (int(_i) for _i in tuple(map(tuple, np.random.randint(3000, size=10))))
	_seeds = tuple(int(_i) for _i in tuple(np.random.randint(3000, size=10)))
	print(f'Seeds:[{_seeds}]')
	# run exp.s loop
	_log_refs = []
	for _seed in tqdm(_seeds):#check if it breaks
		# def. exp
		_cmd = f'python main.py --c_energy cross_entropy --seed {str(_seed)} --epochs {str(epochs)} --summary-writer{f" --energy {model}" if model else ""}'
		_log_refs.append(ipynb_main(_argv=_cmd.split(' ')[1:], exp=run_energy_model_mnist)['log'])

	#return log_ref.s
	return _log_refs




def plot_model_acc_full_dataset():
	# Read data from all logs
	# Aggregate (mean, std variation)
	# Plot
	pass






if __name__ == '__main__':
	model_acc_full_dataset(N=10, epochs=10, exp=run_energy_model_mnist, model=None)
	
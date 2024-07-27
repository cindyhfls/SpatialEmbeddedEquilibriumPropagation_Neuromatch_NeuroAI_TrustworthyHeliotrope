import argparse
import json
import logging
import sys
import glob
# import time

# import torch

from lib import config, train, utils, seeds
from lib.models import energy, mlp
from lib.plot import plot
from lib.data import mnist as data_mnist
from lib.exp.varyingdatapoints import run_exp as vd_run_exp, read_exp_data as vd_read_exp_data

def load_default_config(energy):
	"""
	Load default parameter configuration from file.

	Args:
		tasks: String with the energy name

	Returns:
		Dictionary of default parameters for the given energy
	"""
	if energy == "restr_hopfield":
		default_config = "etc/energy_restr_hopfield.json"
	elif energy == "cond_gaussian":
		default_config = "etc/energy_cond_gaussian.json"
	else:
		default_config = "etc/bp.json"

	with open(default_config) as config_json_file:
		cfg = json.load(config_json_file)

	return cfg


def parse_shell_args(args):
	"""
	Parse shell arguments for this script.

	Args:
		args: List of shell arguments

	Returns:
		Dictionary of shell arguments
	"""
	parser = argparse.ArgumentParser(
		description="Train an energy-based model on MNIST using Equilibrium Propagation."
	)

	parser.add_argument("--batch_size", type=int, default=argparse.SUPPRESS,
						help="Size of mini batches during training.")
	parser.add_argument("--c_energy", choices=["cross_entropy", "squared_error"],
						default=argparse.SUPPRESS, help="Supervised learning cost function.")
	parser.add_argument("--dimensions", type=int, nargs="+",
						default=argparse.SUPPRESS, help="Dimensions of the neural network.")
	parser.add_argument("--energy", choices=["cond_gaussian", "restr_hopfield", None],
						default=None, help="Type of energy-based model.")
	parser.add_argument("--epochs", type=int, default=argparse.SUPPRESS,
						help="Number of epochs to train.")
	parser.add_argument("--fast_ff_init", action='store_true', default=argparse.SUPPRESS,
						help="Flag to enable fast feedforward initialization.")
	parser.add_argument("--learning_rate", type=float, default=argparse.SUPPRESS,
						help="Learning rate of the optimizer.")
	parser.add_argument("--log_dir", type=str, default="",
						help="Subdirectory within ./log/ to store logs.")
	parser.add_argument("--nonlinearity", choices=["leaky_relu", "relu", "sigmoid", "tanh"],
						default=argparse.SUPPRESS, help="Nonlinearity between network layers.")
	parser.add_argument("--optimizer", choices=["adam", "adagrad", "sgd"],
						default=argparse.SUPPRESS, help="Optimizer used to train the model.")
	parser.add_argument("--seed", type=int, default=argparse.SUPPRESS,
						help="Random seed for pytorch")
	parser.add_argument("--early-stopping", action='store_true',
						help="Toogle early stopping")
	parser.add_argument("--summary-writer", action='store_true',
						help="Toggle SummaryWriter")

	return vars(parser.parse_args(args))


def run_energy_model_mnist(cfg):
	"""
	Main script.

	Args:
		cfg: Dictionary defining parameters of the run
	"""
	logging.info(f"Device:\n{config.device}")

	# Initialize seed if specified (might slow down the model)
	if cfg['seed'] is not None:
		seeds.set_seed(cfg['seed'])

	# Create the cost function to be optimized by the model
	c_energy = utils.create_cost(cfg['c_energy'], cfg["energy"], cfg['beta'])

	# Create activation functions for every layer as a list
	phi = utils.create_activations(cfg['nonlinearity'], len(cfg['dimensions']))

	# Initialize energy based model
	if cfg["energy"] == "restr_hopfield":
		model = energy.RestrictedHopfield(
			cfg['dimensions'], c_energy, cfg['batch_size'], phi).to(config.device)
	elif cfg["energy"] == "cond_gaussian":
		model = energy.ConditionalGaussian(
			cfg['dimensions'], c_energy, cfg['batch_size'], phi).to(config.device)
	else:
		model = mlp.MLP(cfg['dimensions'], cfg['batch_size'], phi).to(config.device)

	# Define optimizer (may include l2 regularization via weight_decay)
	w_optimizer = utils.create_optimizer(model, cfg['optimizer'],  lr=cfg['learning_rate'])

	# Create torch data loaders with the MNIST data set
	train_dataloader, val_dataloader, test_dataloader = data_mnist.create_mnist_loaders(cfg['batch_size'])

	logging.info("Start training with parametrization:\n{}".format(
		json.dumps(cfg, indent=4, sort_keys=True)))

	writer = config.setup_writer(cfg['summary_writer'])
	train.run_model_training(cfg, model, cost=c_energy, optimizer=w_optimizer, train_dataloader=train_dataloader, val_dataloader=val_dataloader, test_dataloader=test_dataloader, writer=writer)
	writer.flush()
	writer.close()

def default_main(exp):
	# Parse shell arguments as input configuration
	user_config = parse_shell_args(sys.argv[1:])

	# Load default parameter configuration from file for the specified energy-based model
	cfg = load_default_config(user_config["energy"])

	# Overwrite default parameters with user configuration where applicable
	cfg.update(user_config)

	# Setup global logger and logging directory
	config.setup_logging(cfg["energy"] if cfg["energy"] else "bp" + "_" + cfg["c_energy"] + "_" + cfg["dataset"],
						dir=cfg['log_dir'])
	logging.info(f"Cmd: python {' '.join(sys.argv)}")
	logging.info(f"Device:\n{config.device}")
	# Run the script using the created parameter configuration
	_result = exp(cfg)
	# Close logging
	# logging.info('log file is open')
	_log_file_name = config.get_log_name()
	# https://stackoverflow.com/a/61457520/8612123
	logger = logging.getLogger()
	while logger.hasHandlers():
		logger.removeHandler(logger.handlers[0])
	logging.shutdown()
	# Return name for log to re-use in plotting function
	return {'log':_log_file_name, 'result':_result}




def plot_single(file_glob:str, _show:bool=False, _save:bool=True):
	files = glob.glob(f'events*{file_glob}*', root_dir='log/')
	if len(files)>1:
		raise ValueError(f'More than one file had been found:{files}\nExpected to find only one, please refine the `file_glob` argument.')
	elif len(files)==0:
		raise ValueError('No event file was found, please make sure an event file including the content of the `file_glob` argument exists in `./log`.')
	ea = utils.load_tensorboard_data('log/'+files[0])
	a = utils.extract_sacalars_from_tensorboard_ea(ea)
	a= {key:tuple(df.value.values) for key, df in a.items()}
	plot.plot_single_model_train_metrics(a, _show, _save)


if __name__ == '__main__':
	# Train a single model
	# default_main(run_energy_model_mnist)
	# Demo for plot of a single training run (all captured metrics)
	# plot_single()
	# Run the varying datapoints experiment
	# default_main(vd_run_exp)
	# Visualize results of vd_run_exp(cfg)
	# vd_read_exp_data(file_glob='20240718_1713_bp_cross_entropy_mnist_N', scalar_tag='test_loss')
	# vd_read_exp_data(file_glob='20240718_1911_cond_gaussian_N', scalar_tag='test_acc')
	vd_read_exp_data(file_glob='20240718_1921_restr_hopfield_N', scalar_tag='test_E')
	
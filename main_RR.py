import argparse
import json
import logging
import sys
from functools import partial

import torch

from lib import config, train, utils, seeds
from lib.models import energy, mlp
from lib.data import mnist as data

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
	mnist_train, mnist_val, mnist_test = data.create_mnist_loaders(cfg['batch_size'])

	logging.info("Start training with parametrization:\n{}".format(
		json.dumps(cfg, indent=4, sort_keys=True)))

	if cfg["energy"]:
		train_dict = {'w_optimizer': w_optimizer}
		test_dict = {'dynamics': cfg['dynamics'], 'fast_init': cfg["fast_ff_init"]}
		train_model = partial(train.train, **train_dict, **test_dict)
		test_model = partial(train.test, **test_dict)
		legend = 'mean_E'
	else:
		train_dict = {'optimizer': w_optimizer}
		test_dict = {'criterion': c_energy}
		train_model = partial(train.train_backprop, **train_dict, **test_dict)
		test_model = partial(train.test_backprop, **test_dict)
		legend = 'mean_loss'

	# record the validation accuracy of each epoch for early stopping
	PATIENCE = 2
	wait = 0
	best_val_acc = 0.0

	for epoch in range(1, cfg['epochs'] + 1):
		# Training
		train_model(model, mnist_train)

		# Validation
		val_acc, val_energy = test_model(model, mnist_val)

		# Logging
		logging.info(
			"epoch: {} \t VAL val_acc: {:.4f} \t {}: {:.4f}".format(
				epoch, val_acc, legend, val_energy)
		)

		# Testing
		test_acc, test_energy = test_model(model, mnist_test)

		# Logging
		logging.info(
			"epoch: {} \t TEST test_acc: {:.4f} \t {}: {:.4f}".format(
				epoch, test_acc, legend, test_energy)
		)

		# early stopping
		if val_acc > best_val_acc:
			best_val_acc = val_acc
			wait = 0
		else:
			wait += 1
			if wait >= PATIENCE:
				logging.info(f'Early stopping at epoch {epoch}')
				break


if __name__ == '__main__':
	# Parse shell arguments as input configuration
	user_config = parse_shell_args(sys.argv[1:])

	# Load default parameter configuration from file for the specified energy-based model
	cfg = load_default_config(user_config["energy"])

	# Overwrite default parameters with user configuration where applicable
	cfg.update(user_config)

	# Setup global logger and logging directory
	config.setup_logging(cfg["energy"] if cfg["energy"] else "bp" + "_" + cfg["c_energy"] + "_" + cfg["dataset"],
						dir=cfg['log_dir'])

	# Run the script using the created paramter configuration
	run_energy_model_mnist(cfg)
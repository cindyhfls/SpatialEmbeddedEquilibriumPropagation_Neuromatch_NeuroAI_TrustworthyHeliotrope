

from lib import seeds, utils, config
from lib.data import mnist as data_mnist
from lib.models import energy, mlp


def exp_init(cfg):
	# Initialize seed if specified (might slow down the model)
	if cfg['seed'] is not None:
		seeds.set_seed(cfg['seed'])

	# Create the cost function to be optimized by the model
	cost = utils.create_cost(cfg['c_energy'], cfg["energy"], cfg['beta'])

	# Create activation functions for every layer as a list
	phi = utils.create_activations(cfg['nonlinearity'], len(cfg['dimensions']))

	# Create torch data loaders with the MNIST data set
	train_dataloader, val_dataloader, test_dataloader = data_mnist.create_mnist_loaders(cfg['batch_size'])

	return cost, phi, train_dataloader, val_dataloader, test_dataloader

def set_model_N_optim(cfg, cost, phi):
	# Initialize energy based model
	if cfg["energy"] == "restr_hopfield":
		model = energy.RestrictedHopfield(
			cfg['dimensions'], cost, cfg['batch_size'], phi).to(config.device)
	elif cfg["energy"] == "cond_gaussian":
		model = energy.ConditionalGaussian(
			cfg['dimensions'], cost, cfg['batch_size'], phi).to(config.device)
	else:
		model = mlp.MLP(cfg['dimensions'], cfg['batch_size'], phi).to(config.device)

	# Define optimizer (may include l2 regularization via weight_decay)
	w_optimizer = utils.create_optimizer(model, cfg['optimizer'],  lr=cfg['learning_rate'])

	return model, w_optimizer
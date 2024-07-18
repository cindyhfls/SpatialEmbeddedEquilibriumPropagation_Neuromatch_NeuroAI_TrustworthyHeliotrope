
import logging
import json

import glob
from collections import OrderedDict



from lib import seeds, utils, config, train
from lib.data import mnist as data_mnist, utils as data_utils
from lib.models import energy, mlp
from lib.plot.plot import plot_varying_datapoints

def run_exp(cfg):

	training_points = (10, 100, 1000, 10000)

	# Initialize seed if specified (might slow down the model)
	if cfg['seed'] is not None:
		seeds.set_seed(cfg['seed'])

	# Create the cost function to be optimized by the model
	c_energy = utils.create_cost(cfg['c_energy'], cfg["energy"], cfg['beta'])

	# Create activation functions for every layer as a list
	phi = utils.create_activations(cfg['nonlinearity'], len(cfg['dimensions']))

	# Create torch data loaders with the MNIST data set
	train_dataloader, val_dataloader, test_dataloader = data_mnist.create_mnist_loaders(cfg['batch_size'])

	logging.info("Start training with parametrization:\n{}".format(
		json.dumps(cfg, indent=4, sort_keys=True)))

	for N_train_data in training_points:
		logging.info(f"[ Running training of model for: N={N_train_data} ]")
		# Initialize energy based model
		if cfg["energy"] == "restr_hopfield":
			model = energy.RestrictedHopfield(
				cfg['dimensions'], c_energy, cfg['batch_size'], phi).to(config.device)
		elif cfg["energy"] == "cond_gaussian":
			model = energy.ConditionalGaussian(
				cfg['dimensions'], c_energy, cfg['batch_size'], phi).to(config.device)
		else:
			model = mlp.MLP(cfg['dimensions'], cfg['batch_size'], phi).to(config.device)

		sampled_train_loader, sampled_val_loader = data_utils.get_random_sample_train_val(train_dataloader.dataset, val_dataloader.dataset, cfg['batch_size'], N_train_data)

		# Define optimizer (may include l2 regularization via weight_decay)
		w_optimizer = utils.create_optimizer(model, cfg['optimizer'],  lr=cfg['learning_rate'])

		# Update the train function call to get training costs
		writer = config.setup_writer(cfg['summary_writer'], suffix=f'_N{N_train_data}')
		train.run_model_training(cfg, model, cost=c_energy, optimizer=w_optimizer, train_dataloader=sampled_train_loader, val_dataloader=sampled_val_loader, test_dataloader=test_dataloader, writer=writer)
		writer.flush()
		writer.close()

def read_exp_data(file_glob, scalar_tag):
	training_points = (10, 100, 1000, 10000)
	N_dict = OrderedDict()
	for _point in training_points:
		N_dict[f'{_point}']=None
	files = glob.glob(f'*{file_glob}*', root_dir='log/')
	for file in files:
		N = file.split('N')[-1]
		N_dict[f'{N}']=f'log/{file}'
	for N_train_data, file in N_dict.items():
		ea = utils.load_tensorboard_data(file)
		a = utils.extract_sacalars_from_tensorboard_ea(ea)
		N_dict[f'{N_train_data}'] = {key:tuple(df.value.values) for key, df in a.items()}
	# plot_data = {key:_dict['test_loss'] if 'test_loss' in _dict.keys() else _dict['test_E'] for key, _dict in N_dict.items()}
	plot_data = {key:_dict[scalar_tag] for key, _dict in N_dict.items()}
	print(plot_data)
	label_dict = {'test_loss': ('Test cost', 'log'),'test_acc':('Test accuracy','linear'), 'test_E':('Test E','symlog')}
	plot_varying_datapoints(plot_data, fig_name='BP_vd_N', label=label_dict[scalar_tag][0], yscale=label_dict[scalar_tag][1])

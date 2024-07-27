
import logging
import json

import glob
from collections import OrderedDict



from lib import utils, config, train
from lib.data import utils as data_utils
from lib.exp.utils import exp_init, set_model_N_optim
from lib.plot.plot import plot_varying_datapoints, plot_varying_datapoints_all_models

def run_exp(cfg):

	training_points = (10, 100, 1000, 10000)

	c_energy, phi, train_dataloader, val_dataloader, test_dataloader = exp_init(cfg)

	logging.info("Start training with parametrization:\n{}".format(
		json.dumps(cfg, indent=4, sort_keys=True)))

	for N_train_data in training_points:
		logging.info(f"[ Running training of model for: N={N_train_data} ]")

		model, w_optimizer = set_model_N_optim(cfg, c_energy, phi)
		
		sampled_train_loader, sampled_val_loader = data_utils.get_random_sample_train_val(train_dataloader.dataset, val_dataloader.dataset, cfg['batch_size'], N_train_data)

		# Update the train function call to get training costs
		writer = config.setup_writer(cfg['summary_writer'], suffix=f'_N{N_train_data}')
		train.run_model_training(cfg, model, cost=c_energy, optimizer=w_optimizer, train_dataloader=sampled_train_loader, val_dataloader=sampled_val_loader, test_dataloader=test_dataloader, writer=writer)
		writer.flush()
		writer.close()

def read_exp_data(file_glob:str, scalar_tag:str):
	training_points = (10, 100, 1000, 10000)
	N_dict = OrderedDict()
	for _point in training_points:
		N_dict[f'{_point}']=None
	files = glob.glob(f'events*{file_glob}*', root_dir='log/')
	for file in files:
		N = file.split('N')[-1]
		N_dict[f'{N}']=f'log/{file}'
	for N_train_data, file in N_dict.items():
		ea = utils.load_tensorboard_data(file)
		a = utils.extract_sacalars_from_tensorboard_ea(ea)
		N_dict[f'{N_train_data}'] = {key:tuple(df.value.values) for key, df in a.items()}
	# _data = {key:_dict['test_loss'] if 'test_loss' in _dict.keys() else _dict['test_E'] for key, _dict in N_dict.items()}
	_data = {key:_dict[scalar_tag] for key, _dict in N_dict.items()}
	return _data

def plot_exp_data(plot_data:dict, scalar_tag:str, _show:bool=False, _save:bool=True, _fig_name:str='./log/BP_vd_N.pdf'):
	print(plot_data)
	label_dict = {'test_loss': ('Test cost', 'log'),'test_acc':('Test accuracy','linear'), 'test_E':('Test E','symlog')}
	plot_varying_datapoints(plot_data, fig_name=_fig_name, label=label_dict[scalar_tag][0], yscale=label_dict[scalar_tag][1], _show=_show, _save=_save)

def read_N_plot_exp_data(file_glob:str, scalar_tag:str, _show:bool=False, _save:bool=True, _fig_name:str='./log/BP_vd_N.pdf'):
	plot_exp_data(read_exp_data(file_glob, scalar_tag), scalar_tag, _show=False, _save=True, _fig_name='./log/BP_vd_N.pdf')


def read_multi_exp_data(file_globs:tuple[str]|list[str], scalar_tag:str, names:tuple[str]|list[str]):
	_res = {}
	for _file_glob, _name in zip(file_globs, names):
		_res[_name] = read_exp_data(_file_glob, scalar_tag)
	return _res

def read_N_plot_multi_exp_data(file_globs:tuple[str]|list[str], scalar_tag:str, names:tuple[str]|list[str], _show:bool=False, _save:bool=True, _fig_name:str='./log/BP_vd_N.pdf', plot_fct=plot_varying_datapoints_all_models):
	_plot_data = read_multi_exp_data(file_globs, scalar_tag, names)
	print(_plot_data)
	plot_fct(_plot_data, fig_name=_fig_name, label='Test accuracy', yscale='linear', _show=_show, _save=_save)



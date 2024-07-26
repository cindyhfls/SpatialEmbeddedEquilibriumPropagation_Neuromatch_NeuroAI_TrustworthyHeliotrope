
import logging
import torch.nn.functional as F
import json

from lib import config, train
from lib.rsa.rsa import sample_images, extract_features, calc_rdms
from lib.plot.rsa import plot_label_rdm, plot_maps
from lib.exp.utils import exp_init, set_model_N_optim

def run_exp(cfg):

	c_energy, phi, train_dataloader, val_dataloader, test_dataloader = exp_init(cfg)

	logging.info("Start training with parametrization:\n{}".format(
		json.dumps(cfg, indent=4, sort_keys=True)))


	# logging.info(f"[ Running training of model for: N={N_train_data} ]")

	model, w_optimizer = set_model_N_optim(cfg, c_energy, phi)

	# Update the train function call to get training costs
	# writer = config.setup_writer(cfg['summary_writer'], suffix=f'_N{N_train_data}')
	# train.run_model_training(cfg, model, cost=c_energy, optimizer=w_optimizer, train_dataloader=train_dataloader, val_dataloader=val_dataloader, test_dataloader=test_dataloader, writer=writer)
	train.run_model_training(cfg, model, cost=c_energy, optimizer=w_optimizer, train_dataloader=train_dataloader, val_dataloader=val_dataloader, test_dataloader=test_dataloader)
	# writer.flush()
	# writer.close()


	# Grab 5 test images from each category and visualize them
	imgs, targets = sample_images(test_dataloader, n=5, plot = False)

	# return_layers = model.layers.keys
	return_layers = None
	# features_model_imgs = extract_features(model, imgs, return_layers, plot = 'rolled') #comment this line if Graphviz installation was unsuccessful for you
	features_model_imgs = extract_features(model, imgs.view(imgs.size(0), -1), return_layers)

	# RDMs for the labels (see the category structures)

	one_hot_labels = {"labels": F.one_hot(targets, num_classes=10) }
	label_rdms, label_rdms_dict = calc_rdms(one_hot_labels, method='euclidean')


	rdms, rdms_dict = calc_rdms(features_model_imgs)

	return label_rdms_dict, rdms_dict

def plot_exp(label_rdms_dict, rdms_dict):
	plot_label_rdm(label_rdms_dict)
	plot_maps(rdms_dict, "Standard Model with Standard Images")
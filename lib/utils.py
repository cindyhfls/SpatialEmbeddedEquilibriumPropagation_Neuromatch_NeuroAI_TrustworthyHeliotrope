import torch
import pandas as pd

from lib import cost

from tensorboard.backend.event_processing import event_accumulator


def create_activations(name, n_layers):
	"""
	Create  activation functions for every layer of the network.

	Args:
		name: Name of the activation function
		n_layers: Number of layers

	Returns:
		List of activation functions for every layer
	"""
	if name == 'relu':
		# phi_l = torch.relu
		phi_l = torch.nn.functional.relu
	elif name == "leaky_relu":
		def phi_l(x): torch.nn.functional.leaky_relu(x, negative_slope=0.05)
	elif name == 'softplus':
		phi_l = torch.nn.functional.softplus
	elif name == 'sigmoid':
		phi_l = torch.sigmoid
	elif name == 'hard_sigmoid':
		def phi_l(x): torch.clamp(x, min=0, max=1)
	else:
		raise ValueError(f'Nonlinearity \"{name}\" not defined.')

	return [lambda x: x] + [phi_l] * (n_layers - 1)


def create_cost(name, model_type, beta):
	"""
	Create a supervised learning cost function used to nudge
	the network towards a desired state during training.

	Args:
		name: Name of the cost function
		model_type: See argv 'energy', if None cost for mlp
		beta: Scalar weighting factor of the cost function

	Returns:
		CEnergy object
	"""
	if name == "squared_error":
		if model_type:
			return cost.SquaredError(beta)
		else:
			return torch.nn.functional.mse_loss
	elif name == "cross_entropy":
		if model_type:
			return cost.CrossEntropy(beta)
		else:
			return torch.nn.functional.cross_entropy
	else:
		raise ValueError("Cost function \"{}\" not defined".format(name))


def create_optimizer(model, name, **kwargs):
	"""
	Create optimizer for the given model.

	Args:
		model: nn.Module whose parameters will be optimized
		name: Name of the optimizer to be used

	Returns:
		torch.optim.Optimizer instance for the given model
	"""
	if name == "adagrad":
		return torch.optim.Adagrad(model.parameters(), **kwargs)
	elif name == "adam":
		return torch.optim.Adam(model.parameters(), **kwargs)
	elif name == "sgd":
		return torch.optim.SGD(model.parameters(), **kwargs)
	else:
		raise ValueError("Optimizer \"{}\" undefined".format(name))

def load_tensorboard_data(file:str):

	ea = event_accumulator.EventAccumulator(file,
		size_guidance={
			event_accumulator.COMPRESSED_HISTOGRAMS: 500,
			event_accumulator.IMAGES: 4,
			event_accumulator.AUDIO: 4,
			event_accumulator.SCALARS: 0,
			event_accumulator.HISTOGRAMS: 1,
		})

	ea.Reload() # loads events from file
	return ea

def get_scalars_from_tensorboad_ea(ea, scalar_tag):
	# ea.Tags()
	return ea.Scalars(scalar_tag)

def extract_sacalars_from_tensorboard_ea(ea):
	return {k: pd.DataFrame(ea.Scalars(k)) for k in ea.Tags()['scalars']}


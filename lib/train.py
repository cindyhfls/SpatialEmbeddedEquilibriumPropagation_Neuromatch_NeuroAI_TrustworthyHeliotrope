# MIT License

# Copyright (c) 2020 Simon Schug, João Sacramento

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from functools import partial
import time
import torch
import logging


from lib import config

from lib.models.energy import EnergyBasedModel

from lib.cost import CrossEntropy, SquaredError


def predict_batch(model, x_batch, dynamics, fast_init):
    """
    Compute the softmax prediction probabilities for a given data batch.

    Args:
        model: EnergyBasedModel
        x_batch: Batch of input tensors
        dynamics: Dictionary containing the keyword arguments
            for the relaxation dynamics on u
        fast_init: Boolean to specify if fast feedforward initilization
            is used for the prediction

    Returns:
        Softmax classification probabilities for the given data batch
    """
    # Initialize the neural state variables
    model.reset_state()

    # Clamp the input to the test sample, and remove nudging from ouput
    model.clamp_layer(0, x_batch.view(-1, model.dimensions[0]))
    model.set_C_target(None)

    # Generate the prediction
    if fast_init:
        model.fast_init()
    else:
        model.u_relax(**dynamics)

    return torch.nn.functional.softmax(model.u[-1].detach(), dim=1)


def test(model, test_loader, dynamics, fast_init):
    """
    Evaluate prediction accuracy of an energy-based model on a given test set.

    Args:
        model: EnergyBasedModel
        test_loader: Dataloader containing the test dataset
        dynamics: Dictionary containing the keyword arguments
            for the relaxation dynamics on u
        fast_init: Boolean to specify if fast feedforward initilization
            is used for the prediction

    Returns:
        Test accuracy
        Mean energy of the model per batch
    """
    test_E, correct, total = 0.0, 0.0, 0.0

    for x_batch, y_batch in test_loader:
        # Prepare the new batch
        x_batch, y_batch = x_batch.to(config.device), y_batch.to(config.device)

        # Extract prediction as the output unit with the strongest activity
        output = predict_batch(model, x_batch, dynamics, fast_init)
        prediction = torch.argmax(output, 1)

        with torch.no_grad():
            # Compute test batch accuracy, energy and store number of seen batches
            correct += float(torch.sum(prediction == y_batch.argmax(dim=1)))
            test_E += float(torch.sum(model.E))
            total += x_batch.size(0)

    return correct / total, test_E / total


def train(model, train_loader, dynamics, w_optimizer, fast_init):
    """
    Use equilibrium propagation to train an energy-based model.

    Args:
        model: EnergyBasedModel
        train_loader: Dataloader containing the training dataset
        dynamics: Dictionary containing the keyword arguments
            for the relaxation dynamics on u
        w_optimizer: torch.optim.Optimizer object for the model parameters
        fast_init: Boolean to specify if fast feedforward initilization
            is used for the prediction
    """
    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        x_batch, y_batch = x_batch.to(config.device), y_batch.to(config.device)

        # Reinitialize the neural state variables
        model.reset_state()

        # Clamp the input to the training sample
        model.clamp_layer(0, x_batch.view(-1, model.dimensions[0]))

        # Free phase
        if fast_init:
            # Skip the free phase using fast feed-forward initialization instead
            model.fast_init()
            free_grads = [torch.zeros_like(p) for p in model.parameters()]
        else:
            # Run free phase until settled to fixed point and collect the free phase derivates
            model.set_C_target(None)
            dE = model.u_relax(**dynamics)
            free_grads = model.w_get_gradients()

        # Run nudged phase until settled to fixed point and collect the nudged phase derivates
        model.set_C_target(y_batch)
        dE = model.u_relax(**dynamics)
        nudged_grads = model.w_get_gradients()

        # Optimize the parameters using the contrastive Hebbian style update
        model.w_optimize(free_grads, nudged_grads, w_optimizer)

        # Logging key statistics
        if not (len(train_loader)>10) or batch_idx % (len(train_loader) // 10) == 0:

            # Extract prediction as the output unit with the strongest activity
            output = predict_batch(model, x_batch, dynamics, fast_init)
            prediction = torch.argmax(output, 1)

            # Log energy and batch accuracy
            batch_acc = float(torch.sum(prediction == y_batch.argmax(dim=1))) / x_batch.size(0)
            logging.info('{:.0f}%:\tE: {:.2f}\tdE {:.2f}\tbatch_acc {:.4f}'.format(
                100. * batch_idx / len(train_loader), torch.mean(model.E), dE, batch_acc))
        
        with torch.no_grad():
            # Compute train batch accuracy, energy and store number of seen batches
            correct += float(torch.sum(prediction == y_batch.argmax(dim=1)))
            train_E += float(torch.sum(model.E))
            total += x_batch.size(0)

    return correct / total, train_E / total

def train_backprop(model, train_loader, criterion, optimizer):
    """
    Train a model using backpropagation.

    Args:
        model: A torch.nn.Module neural network model.
        train_loader: DataLoader containing the training dataset.
        criterion: Loss function used to compute the model loss.
        optimizer: torch.optim.Optimizer object for updating model parameters.
    """
    model.train()  # Set the model to training mode
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(config.device), targets.to(config.device)
        inputs = inputs.view(inputs.size(0), -1)

        
        # Forward pass
        outputs = model(inputs)
        
        # Compute loss
        loss = criterion(outputs, targets.float())
        
        # Backward pass and optimize
        optimizer.zero_grad()  # Clear existing gradients
        loss.backward()  # Backpropagate the error
        optimizer.step()  # Update model parameters
        
        # Statistics
        total_loss += loss.item()
        predicted = torch.argmax(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == torch.argmax(targets,1)).sum().item()
        
        # Log every 10th of the dataset
        if not (len(train_loader)>10) or batch_idx % (len(train_loader) // 10) == 0:
            logging.info('[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(inputs), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    
    # Calculate accuracy and average loss
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(train_loader)

    # Log after each epoch
    logging.info('Epoch Finished: Avg. Loss: {:.4f}, Accuracy: {:.2f}%'.format(
        avg_loss, accuracy))
    
    return accuracy, avg_loss

def test_backprop(model, test_loader, criterion):
    """
    Evaluate prediction accuracy of a model on a given test set.

    Args:
        model: A torch.nn.Module neural network model.
        test_loader: DataLoader containing the test dataset.
        criterion: Loss function used to compute the model loss.

    Returns:
        Test accuracy
        Mean loss of the model per batch
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(config.device), targets.to(config.device)
            inputs = inputs.view(inputs.size(0), -1)
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            loss = criterion(outputs, targets.float())
            
            # Statistics
            total_loss += loss.item()
            predicted = torch.argmax(outputs.data, 1)
            targetmax = torch.argmax(targets.data, 1)
            total += targets.size(0)
            correct += (predicted == targetmax).sum().item()

    # Calculate accuracy and average loss
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(test_loader)

    logging.info('Test Set: Avg. Loss: {:.4f}, Accuracy: {:.2f}%'.format(
        avg_loss, accuracy))

    return accuracy, avg_loss


def run_model_training(cfg, model:torch.nn.Module, cost, optimizer:torch.optim.Optimizer, train_dataloader:torch.utils.data.DataLoader, val_dataloader:torch.utils.data.DataLoader, test_dataloader:torch.utils.data.DataLoader, writer:torch.utils.tensorboard.SummaryWriter|config.dummySummaryWriter=config.dummySummaryWriter()):
	"""Run a training for the given arguments.

	Args:
		cfg (_type_): _description_
		model (torch.nn.Module): Model
		cost (CEnergy | torch.nn.functional.loss): Cost function
		optimizer (torch.optim.Optimizer): Optimizer
		train_dataloader (torch.utils.data.DataLoader): Train set DataLoader
		val_dataloader (torch.utils.data.DataLoader): Validation set DataLoader
		test_dataloader (torch.utils.data.DataLoader): Test set DataLoader
		writer (torch.utils.tensorboard.SummaryWriter|dummySummaryWriter): Tensorboard SummaryWriter or a dummy if no writer is required
	"""

	if cfg["energy"]:
		train_dict = {'w_optimizer': optimizer}
		test_dict = {'dynamics': cfg['dynamics'], 'fast_init': cfg["fast_ff_init"]}
		train_model = partial(train, **train_dict, **test_dict)
		test_model = partial(test, **test_dict)
		legend = 'E'
	else:
		train_dict = {'optimizer': optimizer}
		test_dict = {'criterion': cost}
		train_model = partial(train_backprop, **train_dict, **test_dict)
		test_model = partial(test_backprop, **test_dict)
		legend = 'loss'

	# record the validation accuracy of each epoch for early stopping
	PATIENCE = 2
	wait = 0
	best_val_acc = 0.0

	for epoch in range(1, cfg['epochs'] + 1):
		# Training
		train_acc, train_energy = train_model(model, train_dataloader)
		# Summary writer
		writer.add_scalar('train_acc', train_acc, epoch, time.time())
		writer.add_scalar(f'train_{legend}', train_energy, epoch, time.time())

		# Validation
		val_acc, val_energy = test_model(model, val_dataloader)

		# Logging
		logging.info(
			"epoch: {} \t VAL val_acc: {:.4f} \t mean_{}: {:.4f}".format(
				epoch, val_acc, legend, val_energy)
		)
		# Summary writer
		writer.add_scalar('val_acc', val_acc, epoch, time.time())
		writer.add_scalar(f'val_{legend}', val_energy, epoch, time.time())

		# Testing
		test_acc, test_energy = test_model(model, test_dataloader)

		# Logging
		logging.info(
			"epoch: {} \t TEST test_acc: {:.4f} \t {}: {:.4f}".format(
				epoch, test_acc, legend, test_energy)
		)
		# Summary writer
		writer.add_scalar('test_acc', test_acc, epoch, time.time())
		writer.add_scalar(f'test_{legend}', test_energy, epoch, time.time())

		# early stopping
		if cfg['early_stopping']:
			if val_acc > best_val_acc:
				best_val_acc = val_acc
				wait = 0
			else:
				wait += 1
				if wait >= PATIENCE:
					logging.info(f'Early stopping at epoch {epoch}')
					break

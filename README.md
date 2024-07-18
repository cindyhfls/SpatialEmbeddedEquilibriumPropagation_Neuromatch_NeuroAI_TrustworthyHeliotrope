# SpatialEmbeddedEquilibriumPropagation_Neuromatch_NeuroAI_TrustworthyHeliotrope
Code for Trustworthy Heliotrope group "Equilibrium" team



```python
# Run MLP with BP
python main_RR.py --c_energy cross_entropy --seed 2019 --epochs 2
# Run 'cond_gaussian' energy model
python main_RR.py --energy cond_gaussian --c_energy cross_entropy --seed 2019 --epochs 2
```

---
From equilibrium-propagation-master [TO BE ADPATED SOON]


## Usage
You can run the models using the `run_energy_model_mnist.py` script which provides the following options:
```
python run_energy_model_mnist.py -h
usage: run_energy_model_mnist.py [-h] [--batch_size BATCH_SIZE]
                                 [--c_energy {cross_entropy,squared_error}]
                                 [--dimensions DIMENSIONS [DIMENSIONS ...]]
                                 [--energy {cond_gaussian,restr_hopfield}]
                                 [--epochs EPOCHS] [--fast_ff_init]
                                 [--learning_rate LEARNING_RATE]
                                 [--log_dir LOG_DIR]
                                 [--nonlinearity {leaky_relu,relu,sigmoid,tanh}]
                                 [--optimizer {adam,adagrad,sgd}]
                                 [--seed SEED]

Train an energy-based model on MNIST using Equilibrium Propagation.

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        Size of mini batches during training.
  --c_energy {cross_entropy,squared_error}
                        Supervised learning cost function.
  --dimensions DIMENSIONS [DIMENSIONS ...]
                        Dimensions of the neural network.
  --energy {cond_gaussian,restr_hopfield}
                        Type of energy-based model.
  --epochs EPOCHS       Number of epochs to train.
  --fast_ff_init        Flag to enable fast feedforward initialization.
  --learning_rate LEARNING_RATE
                        Learning rate of the optimizer.
  --log_dir LOG_DIR     Subdirectory within ./log/ to store logs.
  --nonlinearity {leaky_relu,relu,sigmoid,tanh}
                        Nonlinearity between network layers.
  --optimizer {adam,adagrad,sgd}
                        Optimizer used to train the model.
  --seed SEED           Random seed for pytorch
```

The default configurations for unspecified parameters are stored in `/etc/`.

## Documentation
[Documentation](https://smonsays.github.io/equilibrium-propagation/) is auto-generated from docstrings using `pdoc3 . --html --force --output-dir docs`.

## Results
Two demo runs for the conditional Gaussian and the restricted Hopfield model using the default configuration can be found in the `/log/` directory. They can be reproduced with:
```bash
#!/bin/bash
python run_energy_model_mnist.py --energy cond_gaussian --c_energy cross_entropy --seed 2019
python run_energy_model_mnist.py --energy restr_hopfield --c_energy squared_error --seed 2019
```

## Dependencies
```
python 3.6
pytorch 1.1.0
torchvision 0.3.0
```
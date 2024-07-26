# SpatialEmbeddedEquilibriumPropagation_Neuromatch_NeuroAI_TrustworthyHeliotrope
Code for Trustworthy Heliotrope group "Equilibrium" team


## Choose the experiment or data visualization

For now, it is a switch by hand in `main.py`, uncomment the function you want to run, and comment the others:
```python
## Train a single model
default_main(run_energy_model_mnist)
## Demo for plot of a single training run (all captured metrics): (adjust the argument according to the generated data)
# plot_single()
## Run the varying datapoints experiment:
# default_main(vd_run_exp)
## Visualize results of vd_run_exp(cfg): (adjust the argument according to the generated data)
# vd_read_exp_data(file_glob='20240718_1713_bp_cross_entropy_mnist_N')
```

## Usage
You can run the models using the `run_energy_model_mnist.py` script which provides the following options:
```
python main.py -h
usage: main.py [-h] [--batch_size BATCH_SIZE] [--c_energy {cross_entropy,squared_error}]
               [--dimensions DIMENSIONS [DIMENSIONS ...]] [--energy {cond_gaussian,restr_hopfield,None}] [--epochs EPOCHS]
               [--fast_ff_init] [--learning_rate LEARNING_RATE] [--log_dir LOG_DIR]
               [--nonlinearity {leaky_relu,relu,sigmoid,tanh}] [--optimizer {adam,adagrad,sgd}] [--seed SEED]
               [--early-stopping] [--summary-writer]

Train an energy-based model on MNIST using Equilibrium Propagation.

options:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        Size of mini batches during training.
  --c_energy {cross_entropy,squared_error}
                        Supervised learning cost function.
  --dimensions DIMENSIONS [DIMENSIONS ...]
                        Dimensions of the neural network.
  --energy {cond_gaussian,restr_hopfield,None}
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
  --early-stopping      Toogle early stopping
  --summary-writer      Toggle SummaryWriter
```

## Usage Example

```python
# Run MLP with BP
python main.py --c_energy cross_entropy --seed 2019 --epochs 2
# Run 'cond_gaussian' energy model
python main.py --energy cond_gaussian --c_energy cross_entropy --seed 2019 --epochs 2
```

## Dosctings: Documenting a function

To make use of auto-docstring in VSCode, you can install the extension: `autoDocstring - Python Docstring Generator` by Nils Werner.

It makes use of the `docs_template.mustache` file, to generate a default docstring after you type `"""` under a function name. If you have typed your arguments like so `def get_random_sample_dataloader(dataset:torchvision.datasets, batch_size:int, M:int):`, it will automatically capture the arguments and their type.






The default configurations for unspecified parameters are stored in `/etc/`.

# From equilibrium-propagation-master [TO BE ADPATED SOON]

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


TODO: Have a look at base model for RSA in tuto!



import torch

import lib.config as config

class MLP(torch.nn.Module):
    def __init__(self, dimensions, batch_size, phi):
            super(MLP, self).__init__() 
            self.dimensions = dimensions
            self.batch_size = batch_size
            self.phi = phi
            self.layers = torch.nn.ModuleList(
                torch.nn.Linear(dim1, dim2) for dim1, dim2 in zip(self.dimensions[:-1], self.dimensions[1:])
            ).to(config.device)
    
    def forward(self,x):
        for i, layer in enumerate(self.layers):
          x = layer(x)
          x = self.phi[i](x)
        return x

    def fast_init(self):
            raise NotImplementedError("Fast initialization not possible for the Hopfield model.")

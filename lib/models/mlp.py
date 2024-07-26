import torch

import lib.config as config

class MLP(torch.nn.Module):
	def __init__(self, dimensions, batch_size, phi):
			super(MLP, self).__init__() 
			self.dimensions = dimensions
			self.batch_size = batch_size
			self.phi = phi
			# self.layers = torch.nn.ModuleList(
			# 	torch.nn.Linear(dim1, dim2) for dim1, dim2 in zip(self.dimensions[:-1], self.dimensions[1:])
			# ).to(config.device)
			# self.layers = torch.nn.ModuleDict(
				# {f"fc{i}": torch.nn.Linear(dim1, dim2) for i, (dim1, dim2) in enumerate(zip(self.dimensions[:-1], self.dimensions[1:]))}
			# ).to(config.device)

			self.fc1 = torch.nn.Linear(self.dimensions[:-1][0], self.dimensions[1:][0]).to(config.device)
			self.fc2 = torch.nn.Linear(self.dimensions[:-1][1], self.dimensions[1:][1]).to(config.device)
			self.layers = (self.fc1,self.fc2)
	
	
	def forward(self,x):
		# x = x.view(x.size(0), -1) # tl compat
		# for i, layer in enumerate(self.layers.values()):
			# x = layer(x)
			# x = self.phi[i](x)
		for i, layer in enumerate(self.layers):
			x = layer(x)
			x = self.phi[i](x)
		return x

	def fast_init(self):
			raise NotImplementedError("Fast initialization not possible for the mlp model.")

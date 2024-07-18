
class MLP(torch.nn.Module):
    def __init__(self, dimensions, criterion, batch_size, phi):
            super(MLP, self).__init__(dimensions, criterion, batch_size, phi) 
            self.dimensions = dimensions
            self.criterion = criterion
            self.batch_size = batch_size
            self.phi = phi
        self.layers = torch.nn.ModuleList(
            torch.nn.Linear(dim1, dim2) for dim1, dim2 in zip(self.dimensions[:-1], self.dimensions[1:])
        ).to(config.device)
    def forward(self,x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # Apply activation function to all but the output layer
                x = self.phi(x)
        return x

        def fast_init(self):
                raise NotImplementedError("Fast initialization not possible for the Hopfield model.")

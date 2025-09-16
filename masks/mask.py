import torch.nn as nn
import torch



class Mask(nn.Module):
    def __init__(self, model, random_seed=12, initial_type='uniform'):
        super(Mask, self).__init__()
        self.layers = []
        for layer in model.children():
            if isinstance(layer, nn.Linear):
                self.layers.append('Linear')
            elif isinstance(layer, nn.ReLU):
                self.layers.append('ReLU')
            elif isinstance(layer, nn.BatchNorm2d):
                self.layers.append('BatchNorm2d')
            elif isinstance(layer, nn.Dropout):
                self.layers.append('Dropout')
                
        self.model_parameters = list(model.parameters())
        
        torch.manual_seed(random_seed)  # Set the random seed for reproducibility
        self.initialize_mask(model, initial_type)


    def initialize_mask(self, model, initial_type):
        if initial_type == 'uniform':
            self.mask_parameters = nn.ParameterList(nn.Parameter(
                torch.empty(p.size()).uniform_(-1/torch.sqrt(torch.tensor(p.size(0), 
                                                                        dtype=torch.float32)), 
                                            1/torch.sqrt(torch.tensor(p.size(0), dtype=torch.float32))), 
                requires_grad=True) for p in model.parameters())
        elif initial_type == 'zeros':
            self.mask_parameters = nn.ParameterList(nn.Parameter(torch.zeros_like(p), requires_grad=True) for p in model.parameters())
        elif initial_type == 'ones':
            self.mask_parameters = nn.ParameterList(nn.Parameter(torch.ones_like(p), requires_grad=True) for p in model.parameters())
        
    
    
    def forward(self, x):
        x = x.reshape(x.size(0), -1) # torch.Size([1, 1, 28, 28]) to torch.Size([1, 784]) @ [784,10] = [1,10]; 

        for i in range(0, len(self.model_parameters), 2):
            W = self.model_parameters[i]
            b = self.model_parameters[i+1]
            
            masked_W = W * torch.sigmoid(self.mask_parameters[i])
            masked_b = b * torch.sigmoid(self.mask_parameters[i+1])

            x = nn.functional.linear(x, masked_W, masked_b)
            if i!=len(self.model_parameters)-2:
                if self.layers[i+1] == 'ReLU':
                    x = nn.functional.relu(x)
                if self.layers[i+1] == 'BatchNorm2d':
                    x = nn.functional.batch_norm(x, masked_W, masked_b)
                if self.layers[i+1] == 'Dropout':
                    x = nn.functional.dropout(x, masked_W, masked_b)

      
        return x
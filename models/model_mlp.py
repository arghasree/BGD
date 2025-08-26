import torch.nn as nn
import torch
import copy
import math

class MLP(nn.Module):
    def __init__(self, num_classes, input_size):
        super(MLP, self).__init__()
        
        self.lin1 =  nn.Linear(torch.prod(torch.tensor(input_size)), 12)
        self.act1 = nn.ReLU()
        self.lin2 =  nn.Linear(12, 8)
        self.act2 = nn.ReLU()
        self.lin3 = nn.Linear(8, num_classes)
        
    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        self.output = []
        x = self.lin1(x)
        self.output.append(x)
        
        x = self.act1(x)
        self.output.append(x)
        
        x = self.lin2(x)
        self.output.append(x)
        
        x = self.act2(x)
        self.output.append(x)
        
        x = self.lin3(x)
        self.output.append(x)
        
        return x
        
    def set_weights(self, changed_weights):
        """
        Args:
            changed_weights (list(list)): [[k,a,*]]
            k: changing value
            a: name of layer
            *: indices

        Returns:
            AlexNet: new model with changes in changed_weights
        """
        model_weights = self.get_weights()

        for change in changed_weights:
            changed_value = change[1]
            change_index = change[0]

            with torch.no_grad():
                if 'lin' in change_index[0]:
                    # fc layers, e.g. 10 X 4096
                    name, row_index, column_index = change_index
                    model_weights[name]['weights'][row_index][column_index] = torch.tensor(changed_value)
                else:
                    # conv layers
                    name, node_index, channel_index, row_index, column_index = change_index
                    model_weights[name]['weights'][node_index][channel_index][row_index][column_index] = torch.tensor(changed_value)

        param = self.state_dict()
        for name in param:
            param[name] = model_weights[name]['weights']

        return self.load_state_dict(param)

    def get_weights(self, batch_norm_weights=False):
        """
        keys: 'name'
        value: dict 
            |keys: 'weights', 'check'
            |check = 0,for biases, and batch norm layers
        
        Example:
            To get weights for layer 'fc.1.weight':
                if model.get_weights()['fc.1.weight]['check']
                    w_fc1 = model.get_weights()['fc.1.weight]['weights']

        Args:
            batch_norm_weights (bool, optional): If batch normalization layers are present in the model architecture. 
            Defaults to False.

        Returns:
            dict: Weights are returned in form of dictionary
        """
        
        model_weights = {}
        param = self.state_dict()

        for name in param:
            model_weights[name]={}
            model_weights[name]['weights']=param[name]
            model_weights[name]['check'] = 0
            if 'classifier' in name and 'weight' in name:
                model_weights[name]['check'] = 1
            if not batch_norm_weights and 'weight' in name:
                model_weights[name]['check'] = 1
            if batch_norm_weights and 'weight' in name:
                if '0.weight' in name:
                    model_weights[name]['check'] = 1
                
            
        return model_weights # weights has all the weights including 0weights, 1weights, fc weights


    def get_activation(self, input_image):
        """
        Getting the activation of each layer
        activations from: 5 conv blocks, 2 fc blocks, class activation layer
        """
        relu_output = None
        linear_output = None
        index_count = 0
        activations = [[] for i in range(8)]  # 7 ReLU and 1 output activation
        x = input_image
        device = "cuda" if torch.cuda.is_available() else "cpu"

        with torch.no_grad():

            # For all units including 5 conv blocks and 2 fc blocks
            for module_pos, module in enumerate(get_children(self, [])):
                if module_pos == 13:
                    x = x.reshape(x.size(0), -1).to(device)
                x = module(x).to(device)  # main lOC

                if isinstance(module, nn.ReLU):  # if ReLU then save activation
                    relu_output = x[0]
                    # print(f'ReLU output size: {relu_output.size()}')
                    for i in range(relu_output.size()[0]):
                        activations[index_count].append(
                            torch.sum(relu_output[i]).item())
                    index_count += 1

                # For the last class activation layer it is nn.Linear
                if module_pos == 18:
                    linear_output = x[0]
                    # print(f'Class activation output size: {linear_output.size()}')
                    for i in range(linear_output.size()[0]):
                        activations[index_count].append(
                            torch.sum(linear_output[i]).item())

        return activations


def get_children(model, lis):
    children = list(model.children())
    for child in children:
        if not isinstance(child, nn.Sequential):
            lis.append(child)
        get_children(child, lis)
    return lis

        
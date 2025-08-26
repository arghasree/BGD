import torch
import torch.nn as nn
import copy
import math


class AlexNet(nn.Module):
    def __init__(self, input_size, input_channel, num_classes=10):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 32, kernel_size=5, stride=1, padding=1)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.act2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 96, kernel_size=3, padding=1)
        self.act3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(96, 64, kernel_size=3, padding=1)
        self.act4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.act5 = nn.ReLU(inplace=True)
        self.pool2= nn.MaxPool2d(kernel_size=2, stride=1)
    
        output_size = math.ceil((input_size-5+2+1-3+2-2)/2+1-3+2+1-3+2+1-2+1)
        input = 32 * output_size**2
        
        self.dropout1 = nn.Dropout()
        self.lin1 = nn.Linear(input, 2048)
        self.act6 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout()
        self.lin2 = nn.Linear(2048, 1024)
        self.act7 = nn.ReLU(inplace=True)
        self.lin3 = nn.Linear(1024, num_classes)


    def forward(self, x):
        self.output = []
        x = self.conv1(x)
        self.output.append(x)
        x = self.act1(x)
        self.output.append(x)
        x = self.conv2(x) 
        self.output.append(x)
        x = self.act2(x) 
        self.output.append(x)
        x = self.pool1(x)
        self.output.append(x)
        x = self.conv3(x)
        self.output.append(x)
        x = self.act3(x)
        self.output.append(x)
        x = self.conv4(x)
        self.output.append(x)
        x = self.act4(x)
        self.output.append(x)
        x = self.conv5(x)
        self.output.append(x)
        x = self.act5(x) 
        self.output.append(x)
        x = self.pool2(x)
        self.output.append(x)
        
        x = x.reshape(x.size(0), -1)
        x = self.dropout1(x)
        self.output.append(x)
        x = self.lin1(x)
        self.output.append(x)
        x = self.act6(x)
        self.output.append(x)
        x = self.dropout2(x)
        self.output.append(x)
        x = self.lin2(x)
        self.output.append(x)
        x = self.act7(x)
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
            if 'lin' in name and 'weight' in name:
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

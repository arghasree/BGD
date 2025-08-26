import numpy as np
import torch
from model_trainers.trainer import *
from dataloaders.custom_dataloader import *
import copy


class Search:
    def __init__(self,
                 dataset,
                 sample,
                 gradients,
                 curvature,
                 check_curve_info,
                 lower_bound,
                 normalize):
        
        self.dataset = dataset
        self.sample = sample
        self.image, self.label = next(iter(sample))
        # self.image = sample[0]; self.label = sample[1];
        self.gradients = gradients
        self.curvature=curvature
        self.lower_bound=lower_bound
        self.check_curve_info=check_curve_info
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.probability_matrix = self.initial_activation_gradient(normalize=normalize)
        
        
    def refresh(self):
        self.model.load_state_dict(self.main_model_weights)
    
    
    def initial_activation_gradient(self, normalize=False):
        """Leveraging gradient information (calculated on sample) 
        to calculate the initial probability matrix.

        Returns:
            list(tensor): probability_matrix
        """
        probability_matrix = {}
        # print(f'Normalizing gradients per layer?: {normalize}')
        
        if normalize:
            # If gradient is negative, we don't care, we take the absolute value of it
            for k,v in self.gradients.items():
                probability_matrix[k]={}
                total_gradient_per_layer = sum([torch.sum(torch.abs(i)) for i in self.gradients[k]['weights']])
                probability_matrix[k]['weights'] = torch.div(torch.abs(self.gradients[k]['weights']), total_gradient_per_layer)
                probability_matrix[k]['check']=0
                if self.gradients[k]['check']==1:
                    probability_matrix[k]['check']=1
        else:
            # Calculate total gradient for all layers
            total_gradient = 0
            for k, v in self.gradients.items():
                if self.gradients[k]['check'] == 1:
                    # total_gradient += sum([torch.sum(torch.abs(i)) for i in self.gradients[k]['weights']])
                    total_gradient += sum([torch.sum(torch.abs(i)) for i in self.gradients[k]['weights']])
                    
            # Define the probability_matrix
            for k, v in self.gradients.items():
                probability_matrix[k] = {}
                probability_matrix[k]['check']=0
                if self.gradients[k]['check'] == 1:
                    probability_matrix[k]['check']=1
                    probability_matrix[k]['weights'] = torch.div(torch.abs(self.gradients[k]['weights']), total_gradient)
                    
                    
                    # # second = torch.div(self.gradients[k]['weights'], torch.pow(self.gradients[k]['weights'],2))
                    # probability_matrix[k]['weights'] = torch.div(self.gradients[k]['weights'], total_gradient)
                    # probability_matrix[k]['weights'] = ( -1*torch.min(probability_matrix[k]['weights'])) + probability_matrix[k]['weights']
                    
        
        return probability_matrix
    
    
    def where_to_change_gradient(self, layer_name=None):
        
        chosen_indices=[]; exceed=0
        if layer_name is not None:
            weight_indices,_ = self.select_parameter(layer_name, chosen_indices, exceed)
            weight_indices=chosen_indices[0]
        else:
            layer_wise_prob = {}
            weight_indices_dict = {}
            for k,v in self.probability_matrix.items():
                chosen_indices=[]
                if self.probability_matrix[k]['check'] == 1:
                    weight_indices_dict[k], layer_wise_prob[k] = self.select_parameter(k,chosen_indices, exceed)
                    weight_indices_dict[k]=chosen_indices[0]
                
                
            layer_wise_prob_array = np.array([prob for _,prob in layer_wise_prob.items()])
            which_layer_index = choose_from(layer_wise_prob_array)
            layer_name = list(weight_indices_dict.keys())[which_layer_index]
            
            weight_indices = weight_indices_dict[layer_name]
            
        # Set probability of chosen elements to 0
        self.set_prob_chosen_zero(layer_name, weight_indices)
            
        # print(f'\nLayer chosen = {layer_name}; layer size: {self.probability_matrix[layer_name]["weights"].size()}\n----\n')
            
        return [layer_name, weight_indices]
    
    
    def select_parameter(self, layer_name, chosen_indices, exceed):
        # print(list(self.probability_matrix.keys()))
        
        prob_mtrx_layer = self.probability_matrix[layer_name]['weights']
        
        # print(f'Choosing from layer {layer_name}, layer size: {prob_mtrx_layer.size()}')
        
        # flatten the layer and choose index, based on probability 
        flattened_layer = torch.flatten(prob_mtrx_layer)
        chosen_index_flattened = choose_from(flattened_layer)
        
                
        # Find 1. Index reshaped into prob_mtrx_layer size, 2. probability of choosing the index
        chosen_index = np.unravel_index(chosen_index_flattened, prob_mtrx_layer.size())
        
        probability = np.take(prob_mtrx_layer.cpu().detach().numpy(), np.ravel_multi_index(
                        np.rollaxis(np.array(chosen_index), -1, 0), prob_mtrx_layer.shape))
        
        if self.check_curve_info:
            # The curvature of the overall loss function at chosen index has to be low. 
            # "How low?" Anything below the 90th percentile is good 
            value_at_index  =  self.curvature[layer_name]['weights'][tuple(chosen_index)]
            if abs(value_at_index) > abs(self.lower_bound) and exceed<5:
                self.set_prob_chosen_zero(layer_name, chosen_index)
                exceed+=1
                chosen_indices.append(self.select_parameter(layer_name, chosen_indices, exceed)[0])
                
        if chosen_indices==[]:
            chosen_indices.append(list(chosen_index))
            
        return [list(chosen_index)], probability
                
    
    def set_prob_chosen_zero(self, layer_name, weight_indices):
        self.probability_matrix[layer_name]['weights'][tuple(weight_indices)] = 0
        
    def calculate(self, layer_name, weight_indices):
        curvature_at_index = extract_2nd_order_info(self.sample, self.gradients, [layer_name, weight_indices], self.model, self.criterion, self.optimizer)
        dWeight = self.gradients[layer_name]['weights'][tuple(weight_indices)]/curvature_at_index
        
        # Calculate \delta Loss = \delta \theta . training_gradients
        dLoss_estimate = []
        for name in self.training_gradients.keys():
            dLoss_estimate += torch.flatten(torch.abs(self.training_gradients[name]['weights'] * dWeight))
            
        for name in self.training_gradients.keys():
            dLoss_estimate += torch.flatten(torch.abs(self.training_gradients[name]['weights'] * self.gradients[name]['weights']))
         
        
        dLoss_estimate = torch.tensor([i.item() for i in dLoss_estimate])
        lower_bound=torch.sort(dLoss_estimate)[0][int(0.25 * len(dLoss_estimate))]  # At 25th percentile the curvature value is very low.

        # If the gradient value is updated for this parameter, then the change in training loss will be very low.
        
        
        value_at_index = (self.training_gradients[layer_name]['weights'] * dWeight)[tuple(weight_indices)]
        
        return lower_bound, value_at_index
        
        
                
   
def choose_from(array):
    try:
        array = array.cpu().detach().numpy()
    except:
        array = array
    try:
        norm_ = array/np.sum(array)
        norm_ /= np.sum(norm_)
        index = np.random.choice(len(norm_),p=norm_)
    except:
        index = np.random.choice(len(array))
    return index 
            
            

    
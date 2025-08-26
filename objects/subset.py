import copy
import math
import numpy as np
import torch
from dataloaders.custom_dataloader import *
from objects.search import *


class Subset:
    def __init__(self,
                 subset_len,
                 parameter_subset,
                 layer_name,
                 dataset,
                 model,
                 main_model_weights,
                 gradient,
                 learning_rate,
                 sample,
                 sample_id,
                 testing_data, 
                 training_data, 
                 criterion,
                 optimizer,
                 Search_obj):
        self.sample_id = sample_id
        self.subset_len = subset_len
        self.dataset = dataset
        self.model = model
        self.main_model_weights = main_model_weights
        self.refresh()  # Original model load at initialization
        self.gradient = gradient
        self.sample = sample
        self.testing_data = testing_data
        self.training_data = training_data
        self.criterion = criterion
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.Search_obj=Search_obj
        self.layer_name=layer_name
        self.parameter_subset=parameter_subset
        # self.image = sample[0]; self.lable = sample[1]
        self.image, self.lable = next(iter(self.sample))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        """parameter_subset is a list of [[layer_name, weight_indices], ... ]
        layer_name = is the name of the layer
        weight_indices = _list_; List of integer indices
        """
        
    def __len__(self):
        return len(self.parameter_subset)
    
    
    def refresh(self):
        self.model.load_state_dict(self.main_model_weights)
        
    def load_parameter_subset(self):
        self.select_initial_subset(self.Search_obj, self.layer_name)

    def check_class_activation(self, wrong_class_name, correct_class_name, classes, swap = True):
        """
        For a subset S, load the model with changed parameters in S, 
        and compute the correct class activation, wrong class activation
        Args:
            wrong_class_name (str): Wrong class name
            correct_class_name (str): Correct class name
            classes (list): List of class names

        Returns:
            (float, float): (correct_class_activation, wrong_class_activation)
        """
        
        # Calculate (correct_class_activation, wrong_class_activation)
        activation_last_layer = self.model.get_activation(self.image)[-1]
        wrong_class_index = classes.index(wrong_class_name)
        correct_class_index = classes.index(correct_class_name)
        correct_class_activation = activation_last_layer[correct_class_index]
        wrong_class_activation = activation_last_layer[wrong_class_index]
        print(f'The wrong class activation = {wrong_class_activation}\n'
              f'The correct class activation = {correct_class_activation}')

        return correct_class_activation, wrong_class_activation

    def deduce_model(self, model_type='MLP', first=False):
        """
        Returns a list of x parameters with their respective changed values
        Returns:
            alpha_values: [[[a,indx1,...], value],.....]
            a = str : layer name
            indx1 = int : index 1 for change
        Example:
            changed_model = subset.deduce_model()
        """
        alpha_values = []
        self.refresh()
        
        for i in self.parameter_subset:
            """
            self.parameter_subset has all the positions to fix, [[layer_name, weight_indices], ... ] 
            i = [layer_name, weight_indices]
            """
            layer_name = i[0]
            weight_indices = np.array(i[1])
            
            gradient_mtrx = self.gradient[layer_name]['weights'].cpu().data.numpy()
            
            grad_value = np.take(gradient_mtrx, 
                                np.ravel_multi_index(np.rollaxis(weight_indices, -1, 0), gradient_mtrx.shape))
            
            model_weights = self.model.get_weights()
            
            model_weights_mtrx = model_weights[layer_name]['weights'].cpu().data.numpy()
            
            weight_value = np.take(model_weights_mtrx,
                                              np.ravel_multi_index(np.rollaxis(weight_indices, -1, 0), model_weights_mtrx.shape))

            if first:       
                value = weight_value - 0.01 * grad_value
            else:
                value = weight_value - grad_value/(grad_value*grad_value)
                
                
        
            alpha_values.append([[layer_name]+list(weight_indices), value])
            
        self.model.set_weights(alpha_values)

        return alpha_values
    
    
    # Returns the loss on the training data.
    def fitness(self, test_data_check=False):
        """For a subset S, load the model with changed parameters in S.
        Computes the training_loss, testing_loss, sample_loss for a subset.
        Args:
            test_data_check (bool, optional): Check if test data performance is necessary. Defaults to False.
        Returns:
            (float, float, float): (training_loss, testing_loss, sample_loss)
        """
        self.model = self.model.to(self.device)
        training_loss = testing_loss = sample_loss = None
        
        with torch.no_grad():
            if self.testing_data is not None and test_data_check:
                for i, x_batch in enumerate(self.testing_data):
                    images, labels = x_batch
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    y_pred = self.model(images)
                    testing_loss = self.criterion(y_pred.to(self.device), labels).to(self.device)

                
            
            if self.sample is not None:
                for i, x_batch in enumerate(self.sample):
                    images, labels = x_batch
                    # images = self.sample[0]; labels = self.sample[1]
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    y_pred = self.model(images)
                    sample_loss = self.criterion(y_pred.to(self.device), labels).to(self.device)

     

            if self.training_data is not None:
                training_loss = count = 0
                for i, x_batch in enumerate(self.training_data):
                    images, labels = x_batch
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    y_pred = self.model(images)
                    count += images.shape[0]

                training_loss += self.criterion(y_pred, labels).to(self.device)
                training_loss = training_loss / count


        return training_loss, testing_loss, sample_loss
    
    
    def calculate_score(self, training_acc_check=False, test_acc_check=False):
        """Measures accuracy for each set
        Args:
            training_acc_check (bool, optional): Check if training set performance is required. Defaults to False.
            test_acc_check (bool, optional): Check if test set performance is required. Defaults to False.
        Returns:
            (float, float, float): (training_score, testing_score, sample_score)
        """
        self.model = self.model.to(self.device)
        training_score = testing_score = sample_score = None
        
        # print(f"\nScores ---------------->")
        with torch.no_grad():
            # For Training set
            if training_acc_check:
                y_true, y_pred = predict(self.model, self.training_data)
                training_score = evaluate(y_true.to(self.device), y_pred.to(self.device))

            # For Testing set
            if test_acc_check:
                y_true, y_pred = predict(self.model, self.testing_data)
                testing_score = evaluate(y_true.to(self.device), y_pred.to(self.device))

            # For Sample
            y_true, y_pred = predict(self.model, self.sample)
            sample_score = evaluate(y_true.to(self.device), y_pred.to(self.device))


        return training_score, testing_score, sample_score
    
    
    def select_initial_subset(self, Search_obj, layer_name):
        for i in range(self.subset_len):
            self.parameter_subset.append(Search_obj.where_to_change_gradient(layer_name=layer_name))
                                         
  

    def second_order_fisher(self, layer_name, weight_indices, size_layer):
        
        indx=0
        for name in self.gradient.keys():
            if self.gradient[name]['check']==1:
                if name == layer_name:
                    break
                else:
                    indx+=1
                    
        
        all = [dic['weights']for dic in self.gradient.values() if dic['check']==1]
        before = self.calc_index(all, indx, weight_indices, size_layer).item()
        
        all = torch.cat([torch.flatten(dic['weights']) for _,dic in self.gradient.items() if dic['check']==1])
        flat_grad_matrx = torch.flatten(all)
        
        outer = torch.outer(flat_grad_matrx,flat_grad_matrx.T)
        
        scnd_order_grad = torch.sort(torch.abs(outer[before]))[0][-10:]
        scnd_order_grad = torch.mean(scnd_order_grad)

        # scnd_order_grad = torch.sum(scnd_order_grad)/len(scnd_order_grad)

        return scnd_order_grad
    
    def calc_index(self, array, i, weight_indices, size_layer):
        tot = 0
        for j in range(i):
            tot+=torch.prod(torch.tensor(array[j].size()))
            
        return tot+np.ravel_multi_index(weight_indices, size_layer)
    
    

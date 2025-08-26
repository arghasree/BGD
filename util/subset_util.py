import copy
import math
import numpy as np
import torch
from model_trainers.trainer import *
from dataloaders.custom_dataloader import *
import copy
import random
from objects.subset import Subset
from objects.search import *

def subset_reduction(Subset_t, gradient, sample, model, main_model_weights, dataset, model_type, first=False, learning_rate=0.001):
    """
    Last step to the algorithm to filter out unwanted parameters
    """
    alphas_index = 0
    completed = False
    while not completed:
        
        prev_subset = copy.deepcopy(Subset_t.parameter_subset)  # Preserve a copy if removing the parameter does not work
    
        subset_1_less = prev_subset[:alphas_index] + prev_subset[alphas_index+1:]
        
        # New subset object  
        Subset_t.parameter_subset = subset_1_less
        Subset_t.t = 'final'
            
        # Subset is loaded, the model is prepared at this setup
        Subset_t.deduce_model(model_type, first) 
        # Now the model is loaded into Subset_t.model
            
        # Calculate sample acc 
        _, _, sample_accuracy = Subset_t.calculate_score(training_acc_check=False, test_acc_check=False)
        # If the sample is still correctly predicted, then remove the parameter from subset
        if sample_accuracy == 0.0:
            alphas_index += 1
            Subset_t.parameter_subset = prev_subset
            
        if alphas_index == len(Subset_t):
            completed = True
                
    """
    As this is the last step, probabilities of parameters are not updated any more
    """

    print(f'Final subset length : {len(Subset_t)}')

    return Subset_t
    
    
def replace_subset(Subset_t, Search_t, layer_name, model, main_model_weights, model_type, first, learning_rate=0.001, multiple=None):
    """
    Replace or keep each paramter in the subset 
    Args:
        Subset_t (Subset)
        layer_name (str)
        gradients (dict)
        sample (dataloader)
        training_set (dataloader): 
        criterion (Loss func)
        training_subset (dataloader)
        multiple (int or None): None for no training loss; 
                                1 or more for different \lamda values

    Returns:
        Subset: Replaced subset
    """
    if multiple is None:
        multiple = 0
    
    # Calculating initial loss for start of iteration subset's fitness
    training_loss, _, sample_loss = Subset_t.fitness(test_data_check=False)
    sample_loss_inital = sample_loss + multiple * training_loss
         
    replace_itr = len(Subset_t)
        
    for i in range(replace_itr):
        # print(f'\n\t\tReplace iteration {i}...')
        
        # Replace sub[i]
        param_at_i_ = Search_t.where_to_change_gradient(layer_name=layer_name)
        
        sub = copy.deepcopy(Subset_t.parameter_subset)
        new_subset = sub[:i]+[param_at_i_]+sub[i+1:]
        
        assert len(new_subset)==len(sub)
        
        # Make new subset object
        Subset_t.parameter_subset = new_subset
        
        # Load model with changes in paramaters 
        Subset_t.deduce_model(model_type, first)
        
        # Compute performance for sample and training (subset)
        training_loss, _, sample_loss = Subset_t.fitness(test_data_check=False)
        
        new_loss = sample_loss + multiple * training_loss
        
        # If new loss is lesser, subset_{t+1} is better than subset_{t}
        if sample_loss_inital > new_loss:
            sample_loss_inital = new_loss
        else:
            Subset_t.parameter_subset = sub
    return Subset_t


def exploration_exploitation(choice_probability, Subset_t, Search_t, layer_name, model, main_model_weights, model_type, first, multiple, learning_rate=0.001):
    """Chooses to expand the subset or replace the subset with better parameters
    Args:
        choice_probability (list): [probability for replace, probability for expanding subset]
        Subset_t (Subset): Subset at t
        sample_loss (): _description_
        Search_t (Search): Search space at t
        layer_name (str): Layer for replace/expansion
        model (AlexNet)

    Returns:
        Subset: new subset after expansion or replacement
    """
    choice = ['replace', 'explore']
    # print(f'Actions: Replace/Expand; Corresponding probability: {"/".join(map(str, choice_probability))}')
    option = choice[choose_from(choice_probability)]
    
    if option == 'replace':
        Subset_t_plus1 = replace_subset(Subset_t, Search_t, layer_name, model, main_model_weights, model_type, first, learning_rate, multiple)

    else:
        Subset_t_plus1 = expand(Search_t, Subset_t, layer_name, model_type, first, 10)

    return Subset_t_plus1
    

def expand(Search_t, Subset_t, layer_name, model_type, first, iterations = 10):
    extra_ = []
    for i in range(iterations):
        extra_.append(Search_t.where_to_change_gradient(layer_name))
        
    Subset_t.parameter_subset = Subset_t.parameter_subset+extra_
    
    Subset_t.deduce_model(model_type, first)
    return Subset_t


def subset_sel(model, 
               main_model_weights,
               sample, sample_id,
               choice_probability,
               testing_data,
               training_data, 
               training_subset,
               criterion, optimizer, 
               gradients,
               curvature, 
               lower_bound,
               args):
    """Generates parameter subset based on iterations

    Args:
        model (AlexNet)
        main_model_weights:weights for main model
        sample (dataloader): incorrectly predicted sample
        choice_probability (list): list containing probability for replace/expand
        testing_data (dataloader): test set
        training_data (dataloader): training set/subset
        training_subset (dataloader): training_subset 
        criterion (loss func)
        gradients (dict): dictionary of parameters with name
        args: Other arguments
    Returns:
        Subset type object
    """
    # Initialize search space
    Search_t = Search(args.dataset, 
                      sample, gradients, 
                      curvature, 
                      args.check_curve_info,
                      lower_bound,
                      args.normalize)
    
    correctly_predicted = False
    
    sample_loss_list = []  # to keep track of sample loss over iterations
    
    if args.training_subset_check:
        training_data_ = training_subset
    else:
        training_data_ = training_data
    
    for i in range(-1,args.iterations):
        if i == -1:
            print("\t\t\t\tMain model:")
            
            # This is the subset for main model to check it's fitness; No paramters are included in this subset
            Subset_t = Subset(subset_len=0,
                                parameter_subset=[],
                                layer_name=None,
                                dataset=None,
                                model=model,
                                main_model_weights=main_model_weights,
                                gradient=gradients,
                                learning_rate=args.learning_rate,
                                sample=sample,
                                sample_id=sample_id,
                                testing_data=testing_data,
                                training_data=training_data_,
                                criterion=criterion, 
                                optimizer=optimizer,
                                Search_obj=None)  
        elif i == 0:
            print(f'\t\t\t\tSubset {i+1}')
            
            # This the initial subset object 
            Subset_t.subset_len=5
            Subset_t.Search_obj=Search_t
            Subset_t.load_parameter_subset()
            if Subset_t.parameter_subset==[]:
                correctly_predicted = False
                break
            
        else:
            print(f'\t\t\t\tSubset {i+1}')
            
            # Replace/Expand
            Subset_t = exploration_exploitation(choice_probability, Subset_t, Search_t, args.layer_name, model, main_model_weights, args.model_type, args.first, args.multiple, args.learning_rate)
            """
             ------   Learning from the interaction between the model and parameter space ------ 
            If the rate of change is less than threshold, more parameters are needed to correctly predict the sample
            """
            if i > 5:
                rate = abs((sample_loss_list[-2]-sample_loss_list[-1])/sample_loss_list[0]) 
                
                if rate < args.threshold:
                    # print(f"Exploration rate increased")
                    choice_probability[-1] += 0.01 * rate.item()
                    choice_probability[0] -= 0.01 * rate.item()
        
        Subset_t.deduce_model(args.model_type, args.first)
        sample_loss_list.append(Subset_t.fitness()[2])
        if Subset_t.calculate_score(training_acc_check=False)[2] == 1.0:
            correctly_predicted = True
            break
    print(Subset_t.parameter_subset)
    training_score, testing_score, sample_score = Subset_t.calculate_score(training_acc_check=True, test_acc_check=args.test_acc_check)
    if args.training_subset_check:
        # print("\nSubset Fitness Check on ENTIRE training set ")
        Subset_t.training_data = training_data
        Subset_t.fitness()
        
    if correctly_predicted:
        # print("Sample predicted correctly\n\n\t\t\t----- Reduction Algorithm -----")
        Subset_t = subset_reduction(Subset_t, gradients, sample, model, main_model_weights, args.dataset, args.model_type, args.first, args.learning_rate)
        alpha_values = Subset_t.deduce_model(args.model_type, args.first)
        training_score, testing_score, sample_score = Subset_t.calculate_score(training_acc_check=True, test_acc_check=args.test_acc_check)
        training_loss, testing_loss, sample_loss = Subset_t.fitness()
        if not args.test_acc_check:
            print(f'FINAL MODEL -- Sample id: {Subset_t.sample_id}---> Training: Loss = {training_loss}, Accuracy = {training_score}'
              f'\tSample: Loss = {sample_loss}, Accuracy = {sample_score}')
        else:
            print(f'FINAL MODEL -- Sample id: {Subset_t.sample_id}---> Training: Loss = {training_loss}, Accuracy = {training_score}'
              f'\tSample: Loss = {sample_loss}, Accuracy = {sample_score}'
              f'\tTest: Loss = None, Accuracy = {testing_score}')

        return alpha_values
    
    else:
        return None



import copy
import time
import numpy as np
from model_trainers.trainer import *
from dataloaders.custom_dataloader import *
import dataloaders.dataloading as dataloading
import random
from util.subset_util import *
from torch.func import functional_call, vmap, hessian
from pathlib import Path
from objects.sample import Test_Sample
import matplotlib.pyplot as plt
import os
# import seaborn as sns


def parameter_search(model, best_model_wts, train_loader, test_loader, criterion, optimizer, scheduler, args):
    """
    For each sample:
                1. Relaod main model (or not)
                2. Extract sample
                3. Extract gradient based on sample
                4. Extract training subset
                    4.1. Extract parameter space
                5. Track number of correctly predicted samples
                6. Track Retrained model 
    """

    # Loading the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    training_subset = samples = None  # Training subset selection
    classes = dataloading.class_name(
        directory=args.directory, dataset=args.dataset)

    samples = os.listdir(
        f'{args.directory}all_samples/{args.model_type}/{args.dataset}')

    correctly_predicted = 0
    i = 0
    dLoss_estimate = []
    alpha_values = None
    for filename in samples:
        sample_info = os.path.join(
            f'{args.directory}all_samples/{args.model_type}/{args.dataset}', filename)
        print(
            f"\n\t\t############## Example : {i + 1}/{len(samples)} ##############")

        # Reload main model
        if args.reloaded:
            model.load_state_dict(best_model_wts)
        else:
            if i > 0 and alpha_values is not None:
                model.set_weights(alpha_values)
                del best_model_wts
                best_model_wts = copy.deepcopy(model.state_dict())

        sample_obj = torch.load(sample_info)

        classes = sample_obj.classes
        wrong_class_name = sample_obj.wrong_class_name
        correct_class_name = sample_obj.correct_class_name
        sample = sample_obj.sample

        print(
            f'Predicted Class Name = {wrong_class_name}, Correct Class Name = {correct_class_name}\n')

        training_gradients = extract_gradients(
            sample, train_loader, model, criterion, optimizer, batch_norm_weights=False, full=True)

        correct_pred, training_subset, alpha_values = sample_operation(sample, i, training_gradients,
                                                                       training_subset, train_loader, test_loader,
                                                                       criterion, optimizer,
                                                                       model, best_model_wts,
                                                                       classes, wrong_class_name, correct_class_name,
                                                                       args)

        if correct_pred:
            correctly_predicted += 1

        del sample_obj
        i += 1
    # visualize(dLoss_estimate)

    # if args.save_final_model:
    #     directory_to_save = f'{args.directory}models/no_reload/{args.model_type}/{args.dataset}/final_model.pth'
    #     torch.save(model.state_dict(), directory_to_save)
    # print(f'Correctly predicted = {correctly_predicted} out of {len(samples)}')


def parameter_search_imbalanced(model, best_model_wts, train_loader,
                                test_loader, criterion, optimizer, 
                                incorrect_loader, 
                                args):
    """
    For each sample:
                1. Relaod main model (or not)
                2. Extract sample
                3. Extract gradient based on sample
                4. Extract training subset
                    4.1. Extract parameter space
                5. Track number of correctly predicted samples
                6. Track Retrained model 
    """

    # Loading the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    correctly_predicted = 0; i = 0
    dLoss_estimate = []
    
    training_subset = None  # Training subset selection
    alpha_values = None
    avg_state_dict = best_model_wts
    
    for sample_indx, sample in enumerate(incorrect_loader):
        if i==10:
            torch.save(model.state_dict(), f'{args.directory}models/{args.model_type}/model_weights_copy_10.pth')
            break
        # sample = list(test_loader)[sample_indx] 
        image, label = sample[0].to(device), sample[1].to(device)
        output = model(image)
        _, predicted = torch.max(output, 1)
        correct_pred = (predicted == label).item()
        
        print(f"\n\t\t############## Example : {i+1}/{len(incorrect_loader)} ##############")

        if correct_pred:
            print(f"Sample already correct")
            i+=1
            continue
        
        # Reload main model
        if args.reloaded:
            model.load_state_dict(best_model_wts)
        else:
            if i>0 and args.copy_weight_reinit:
                print("Process - Copy weight reinit")
                model.set_weights(alpha_values)
                state_dict_list = [avg_state_dict, copy.deepcopy(model.state_dict())]
                avg_state_dict = make_average(state_dict_list)
                model.load_state_dict(avg_state_dict)
            else:
                if i > 0 and alpha_values is not None:
                    model.set_weights(alpha_values)
                    del best_model_wts
                    best_model_wts = copy.deepcopy(model.state_dict())
        sample = [(sample[0], sample[1])]

        training_gradients = extract_gradients(sample, train_loader, model, criterion, optimizer, batch_norm_weights=False, full=True)

        correct_pred, training_subset, alpha_values = sample_operation(sample, i, training_gradients,
                                                                       training_subset, train_loader, test_loader,
                                                                       criterion, optimizer,
                                                                       model, best_model_wts,
                                                                       args)

        if correct_pred:
            correctly_predicted += 1
            # incorrect_loader_performance(model, test_loader, incorrect_sample_indices)
            
        i += 1
    # visualize(dLoss_estimate)

    # if args.save_final_model:
    #     directory_to_save = f'{args.directory}models/no_reload/{args.model_type}/{args.dataset}/final_model.pth'
    #     torch.save(model.state_dict(), directory_to_save)
    # print(f'Correctly predicted = {correctly_predicted} out of {len(samples)}')
    


def make_average(state_dict_list):
    state_dict_avg = {}
    for k, v in state_dict_list[0].items():
        state_dict_avg[k] = torch.mean(torch.stack([state_dict_list[i][k] for i in range(len(state_dict_list))]), dim=0)
        # print(state_dict_avg[k].shape, state_dict_list[0][k].shape) # these should match; they match 
    
    # To check if the weights are different - They are
    # max_diff = {}
    # for k in state_dict_list[0].keys():
    #     max_diff[k] = torch.max(torch.abs(state_dict_list[0][k] - state_dict_avg[k]))
    #     print(f"Max difference in layer {k}: {max_diff[k].item()}")
    return state_dict_avg

def incorrect_loader_performance(model, test_loader, incorrect_sample_indices):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    acc=0
    for sample_indx in incorrect_sample_indices:
        image = list(test_loader)[sample_indx][0].to(device)
        y_true = list(test_loader)[sample_indx][1].to(device)
        
        y_pred = model(image).to(device)
        y_pred = torch.argmax(y_pred, dim=1)
        if y_pred==y_true:
            acc+=1
    print(f'Total accuracy for the incorrectly predicted sample = {(acc*100)/len(incorrect_sample_indices)}')
    
        


def get_incorrect_samples(model_type, directory, dataset, test_samples_check):
    """
    Depending on the test case, all dataset case, this function outputs a list of dataloader type samples
    """
    samples = []
    if not test_samples_check:
        directory_to_save = f'{directory}all_samples/{model_type}/{dataset}/'
    else:
        ####### Get samples if there are no samples to test on #######
        directory_to_save = f'{directory}samples/{model_type}/{dataset}/'
    subset_for_test = get_samples(directory_to_save)

    print(
        f"Iterating through {len(subset_for_test)} incorrectly predicted samples")

    for i, sample_obj in enumerate(subset_for_test):
        classes = sample_obj.classes
        wrong_class_name = sample_obj.wrong_class_name
        correct_class_name = sample_obj.correct_class_name
        sample = sample_obj.sample

        meta_data = [sample, wrong_class_name, correct_class_name]
        samples.append(meta_data)

    return samples


def retrain(model, best_model_wts, training_data, sample, epochs, criterion, optimizer, scheduler):
    """Calculates Sample/Training Loss for the retrained model
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("############### Retrained Model ###############")

    # Load main model
    retrained_model = copy.deepcopy(model)
    retrained_model.load_state_dict(best_model_wts)

    retrained_model, _, _ = train_early_stopping(
        sample, retrained_model, epochs, criterion, optimizer, scheduler)

    # Calculate sample/training loss
    print(f'Sample Loss = {loss(retrained_model, sample, criterion)}')
    print(f'Training Loss = {loss(retrained_model, training_data, criterion)}')

    # Calculate sample/training accuracy
    y_true, y_hat = predict(retrained_model, sample)
    print(f'Sample Accuracy = {evaluate(y_true.to(device), y_hat.to(device))}')

    y_true, y_hat = predict(retrained_model, training_data)
    print(
        f'Training Accuracy = {evaluate(y_true.to(device), y_hat.to(device))}')


def loss(model, data, criterion):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    with torch.no_grad():
        testing_loss = count = 0
        for i, x_batch in enumerate(data):
            images, labels = x_batch
            images = images.to(device)
            labels = labels.to(device)

            y_pred = model(images)
            count += images.shape[0]
            testing_loss += criterion(y_pred.to(device), labels).to(device)
    testing_loss /= count
    return testing_loss.item()


def get_test_samples(classes, incorrect_pred_img, incorrect_pred_labels, incorrect_pred_classes, directory):
    samples_idx = np.random.randint(0, len(incorrect_pred_img), 5)

    for i in samples_idx:
        image = [incorrect_pred_img[i]]
        label = [incorrect_pred_labels[i]]
        wrong_class_name = incorrect_pred_classes[i]
        correct_class_name = classes[label[0]]
        sample = data_loader(images=image, labels=label,
                             batch_size=1, shuffle=False)
        sample_name = f'{directory}sample_{i}.pth'

        torch.save(Test_Sample(classes, wrong_class_name,
                   correct_class_name, sample_name, sample), sample_name)


def get_samples(directory):
    directory = Path(directory)
    # List all files in the directory
    files = [f for f in directory.iterdir() if f.is_file()]

    samples_to_return = []
    print(f"Sample files for test in directory: '{directory}'")
    for file in files:
        if 'sample' in file.name:
            samples_to_return.append(torch.load(file))

    return samples_to_return


def sample_operation(sample, sample_id, training_gradients,
                     training_subset, training_data, testing_data,
                     criterion, optimizer,
                     model, main_model_weights,
                     args):
    image, label = next(iter(sample))
    

    # Extracting gradient for sample
    gradients = extract_gradients(sample, training_data, 
                                  model, criterion, optimizer, 
                                  batch_norm_weights=False, full=False)
    
    lower_bound, curvature = None, None
    if args.check_curve_info:
        # This is a heavy operation, breaks for 'AlexNet'
        lower_bound, curvature = analyse_training_gradients(training_gradients, 
                                                            gradients, args.p/100, 
                                                            viz=False)
    

    correctly_predicted = False

    ################################### Extract training subset ###################################
    if args.training_subset_check:
        if args.choice_of_training != 'same class' and training_subset is None:
            training_img, training_labels = get_sample(
                training_data, label=None, number=10, type=args.choice_of_training)
            training_subset = data_loader(
                images=training_img, labels=training_labels, batch_size=10, shuffle=True)

        if args.choice_of_training == 'same class' and training_subset is None:
            training_img, training_labels = get_sample(
                training_data, label=label, number=10, type=args.choice_of_training)
            training_subset = data_loader(
                images=training_img, labels=training_labels, batch_size=10, shuffle=True)
    else:
        training_subset = None
    ################################### Extract subset ###################################

    choice_probability = [0.7, 0.3]  # Replace/Expand
    # choice_probability = [0.5, 0.5]
    
    alpha_values = subset_sel(model=model,
                              main_model_weights=main_model_weights,
                              sample=sample, sample_id=sample_id,
                              choice_probability=choice_probability,
                              testing_data=testing_data,
                              training_data=training_data,
                              training_subset=training_subset,
                              criterion=criterion,
                              optimizer=optimizer,
                              gradients=gradients,
                              curvature=curvature,
                              lower_bound=lower_bound,
                              args=args)

    if alpha_values is not None:
        correctly_predicted = True

    return correctly_predicted, training_subset, alpha_values


def imshow(sample, directory, name):
    img, _ = next(iter(sample))
    img = img / 2 + 0.5
    npimg = img.cpu().numpy().reshape(
        (img.shape[1], img.shape[2], img.shape[3]))
    fig = plt.figure(figsize=(1, 1))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    filename = f'{directory}{name}.png'
    plt.savefig(filename)


def analyse_training_gradients(training_gradients, gradients, p, viz=False):
    curvature = {}
    dLoss_estimate = []
    for name in training_gradients.keys():
        curvature[name] = {}
        curvature[name]['check'] = 0
        if training_gradients[name]['check'] == 1:
            curvature[name]['check'] = 1
            curvature[name]['weights'] = torch.abs(
                training_gradients[name]['weights'] * gradients[name]['weights'])
            dLoss_estimate += torch.flatten(curvature[name]['weights'])

    dLoss_estimate = torch.tensor([i.item() for i in dLoss_estimate])
    dLoss_estimate_descending = torch.sort(dLoss_estimate)[0]
    percentile99d = dLoss_estimate_descending[int(
        len(dLoss_estimate_descending)*p)]

    return percentile99d, curvature


import torch
import torch.nn as nn
from torch.utils import data
from torch import optim
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy
from dataloaders.dataloading import *
import copy

def extract_2nd_order_info(train_set, gradients, index, model, criterion, optimizer):
    """
    For a certain index, returns the curvature.
    Defined by: sum_of_all_training_data (gradient at index * gradient of all parameters )/# of training data
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    criterion = criterion.to(device)
    layer_name=index[0]; index=index[1];
    param_at_index = torch.abs(gradients[layer_name]['weights'][tuple(index)])
    flat_grad_list=None
    for i, x_batch in enumerate(train_set):
        inner=[]
        images, labels = x_batch
        images = images.to(device)
        labels = labels.to(device)

        y_pred = model(images)
        loss = criterion(y_pred.to(device), labels).to(device)
        optimizer.zero_grad()
        loss.backward()
        with torch.no_grad():
            for name, param in model.named_parameters():
                grad = param_at_index*torch.flatten(torch.abs(param.grad))
                inner+=grad
        if flat_grad_list is None:
            flat_grad_list=torch.zeros_like(torch.tensor(inner))
        flat_grad_list+=torch.tensor(inner)

    flat_grad_list/=i+1
    curvature_at_index = torch.mean(torch.tensor(torch.sort(flat_grad_list, descending=True)[0][0:10]))
    
    return curvature_at_index 


def extract_gradients(sample, train_set, model, criterion, optimizer, batch_norm_weights=False, full=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    criterion = criterion.to(device)
    
    """Initializing the gradient dictionary
    If layer is f.c. or conv, then layer can be sampled from 
    """
    gradients={}
    with torch.no_grad():
        for name, param in model.named_parameters():
            gradients[name]={}
            gradients[name]['weights']=torch.zeros_like(param)
            gradients[name]['check'] = 0
            if 'fc' in name and 'weight' in name:
                gradients[name]['check'] = 1
            if not batch_norm_weights and 'weight' in name:
                gradients[name]['check'] = 1
            if batch_norm_weights and 'weight' in name:
                if '0.weight' in name:
                    gradients[name]['check'] = 1
                            
    if full:
        for i, x_batch in enumerate(train_set):
            images, labels = x_batch
            images = images.to(device)
            labels = labels.to(device)

            y_pred = model(images)
            loss = criterion(y_pred.to(device), labels).to(device)
            optimizer.zero_grad()
            loss.backward()
            with torch.no_grad():
                for name, param in model.named_parameters():
                    gradients[name]['weights']+=param.grad
        with torch.no_grad():
            for name, param in model.named_parameters():
                gradients[name]['weights']/=i+1
    else:
        images, labels = next(iter(sample))
        # images = sample[0]; labels = sample[1]
        images = images.to(device)
        labels = labels.to(device)

        y_pred = model(images)
        loss = criterion(y_pred.to(device), labels).to(device)
        optimizer.zero_grad()
        loss.backward()
        with torch.no_grad():
            for name, param in model.named_parameters():
                gradients[name]['weights']=param.grad
                    
           

    return gradients



def load_trained(validation_data, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    y_true, y_hat = predict(model, validation_data)
    accuracy = evaluate(y_true.to(device), y_hat.to(device))
    best_model_wts = copy.deepcopy(model.state_dict())

    return accuracy, best_model_wts


def train_with_l1(sample, model, epochs, criterion, optimizer, scheduler):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    criterion = criterion.to(device)

    for epoch in range(epochs):
        model.train()
        image, label = next(iter(sample))
        image = image.to(device)
        label = label.to(device)
          
        y_pred = model(image)
        
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        l1_lambda = 0.01  # L1 regularization strength
        loss = criterion(y_pred.to(device), label).to(device) + l1_lambda * l1_norm
        optimizer.zero_grad()
        loss.backward(retain_graph = True)
        optimizer.step()
        scheduler.step()

        # Validation data
        y_true, y_hat = predict(model, sample)
        acc = evaluate(y_true.to(device), y_hat.to(device))
        if acc==1.0:
            print(f"Accuracy of sample is {acc} at epoch {epoch}")
            break


    best_model_wts = model.state_dict()
    print("")

    return best_model_wts


def train(training_data, validation_data, model, epochs, criterion, optimizer, scheduler):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    criterion = criterion.to(device)

    loss_over_epoch = []
    total_acc = []
    loss = 0
    for epoch in range(epochs):
        print(f'\n------------------------ epoch = {epoch + 1} -----------------------------')
        model.train()
        for i, x_batch in enumerate(training_data):
            images, labels = x_batch
            images = images.to(device)
            labels = labels.to(device)
          
            y_pred = model(images)
            loss = criterion(y_pred.to(device), labels).to(device)
            optimizer.zero_grad()
            loss.backward(retain_graph = True)
            optimizer.step()

        scheduler.step()

        # Validation data
        if validation_data is not None:
            y_true, y_hat = predict(model, validation_data)
        else:
            print('No validation data found. Checking training data accuracy instead .. ')
            y_true, y_hat = predict(model, training_data)
        acc = evaluate(y_true.to(device), y_hat.to(device))
        print(f"Accuracy of val data is {acc}")

        loss_over_epoch.append(loss.item())
        total_acc.append(acc)

    best_model_wts = copy.deepcopy(model.state_dict())
    layer_weights_epochs = layer_information_(best_model_wts)[0]
    print("")

    return loss_over_epoch, total_acc, model, layer_weights_epochs, best_model_wts


def train_early_stopping(training_data, model, epochs, criterion, optimizer, scheduler):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    criterion = criterion.to(device)

    layer_weights_epochs = []
    for epoch in range(epochs):
        print(f'\n------------------------ epoch = {epoch + 1} -----------------------------')
        model.train()
        for i, x_batch in enumerate(training_data):
            print(f'processing batch no. {i + 1} ', end="")

            images, labels = x_batch
            images = images.to(device)
            labels = labels.to(device)

            y_pred = model(images)
            loss = criterion(y_pred.to(device), labels).to(device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validation data
        y_true, y_hat = predict(model, training_data)

        acc = evaluate(y_true.to(device), y_hat.to(device))

        if acc == 1.0:
            print(f'Correctly predicted at {epoch + 1}')
            break
    best_model_wts = copy.deepcopy(model.state_dict())
    layer_weights_epochs = layer_information_(best_model_wts)[0]

    return model, best_model_wts, layer_weights_epochs


def predict(model, validation_data):
    """
    Output <- Model(Input)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    y_true = []
    y_pred = []
    for val_images, val_labels in validation_data:
        val_images = val_images.to(device)
        val_labels = val_labels.to(device)

        output = model(val_images)
        y_hat = torch.argmax(output, dim=1)

        y_true.append(val_labels)
        y_pred.append(y_hat)

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    return y_true, y_pred


def evaluate(y_true, y):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    y_true = y_true.to(device)
    y = y.to(device)

    accuracy = len(y_true[y_true == y]) / len(y_true)
    
    return accuracy


def predict_test(model, data, directory, dataset):
    """
    Predicts and outputs the images and images that were wrongly predicted
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    incorrect_pred_img = []; incorrect_pred_labels = []; incorrect_pred_classes = []
    incorrect=total=0
    for images, labels in data:

        images = images.to(device)
        labels = labels.to(device)

        output = model(images)
        y_hat = torch.argmax(output, dim=1)

        indices = [i for i in range(len(labels)) if labels[i] != y_hat[i]]
        incorrect +=len(indices)
        total+=len(labels)
        

        for i in indices:
            incorrect_pred_img.append(images[i])
            incorrect_pred_labels.append(labels[i].item())

            # For tracking the incorrectly predicted items, return the list of the incorrect labels as well
            classes = class_name(directory=directory, dataset=dataset)
            incorrect_pred_classes.append(classes[y_hat[i].item()])

    print(f'{incorrect} out of {total} samples are incorrect')
    
    return incorrect_pred_img, incorrect_pred_labels, incorrect_pred_classes


def layer_information_(best_model_wts, layer_name="Null"):
    layers = []
    names = []
    clone = copy.deepcopy(best_model_wts)
    for param_tensor in clone:
        # layers.append(model.state_dict()[param_tensor].clone())
        names.append(param_tensor)
        layers.append(clone[param_tensor].detach().cpu().numpy())

    if layer_name != "Null":
        return layers[names.index(layer_name)]
    else:
        return layers, names

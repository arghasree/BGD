import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_trainers.trainer import predict, evaluate
from dataloaders.dataloading import *
from util.search_util import *
from dataloaders.custom_dataloader import *
from models.model_mlp import *
from models.model_small import *
from dataloaders.get_all_samples import *
from dataloaders.imbalanced_dataloader import *
from masks.mask import *

import torch.nn as nn
from torch import optim
import os
import time
import argparse
import wandb
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the loss function
def loss_fn(output_model, target, Mask, lambda_reg=0.01):
    model_loss = nn.CrossEntropyLoss()(output_model, target)
    # Encourage mask values to be close to 1 (after sigmoid)
    # This penalizes deviations from 1
    mask_loss = sum((torch.sigmoid(p) - 1.0).abs().mean() for p in Mask.mask_parameters)
    # print(f"Model loss is {model_loss}, Regularization loss is {mask_loss}")
    return model_loss + lambda_reg * mask_loss # penalize deviations from 1, encourage masks to be close to 1


def get_accuracies(loaders, model):
    """Get accuracies for train, test, and incorrect loaders"""
    accuracies = []
    for loader in loaders:
        y_true = []
        y_hat = []
        for img, label in loader:
            img = img.to(device)
            label = label.to(device)
            output = model(img)
            y_pred = torch.argmax(output, dim=1)
            y_true.extend(label.cpu().numpy().tolist())
            y_hat.extend(y_pred.cpu().numpy().tolist())
        
        y_true = np.array(y_true)
        y_hat = np.array(y_hat)
        acc = np.mean(y_true == y_hat)
        accuracies.append(acc)
    
    return accuracies[0], accuracies[1], accuracies[2]  # train, test, incorrect


def loader_performance(loaders, model, epoch):  # Always assumes you are passing test and incorrect loader in that order
    names = ['train_loader','test_loader', 'incorrect_loader']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for loader, name in zip(loaders, names):
        y_true = []
        y_hat = []
        for img, label in loader:
            img = img.to(device)
            label = label.to(device)

            output = model(img)
            y_pred = torch.argmax(output, dim=1)
            # print(y_pred)
            # return None
             # Convert tensors to lists for comparison
            y_true.extend(label.cpu().numpy().tolist())
            y_hat.extend(y_pred.cpu().numpy().tolist())
        
        # Convert lists to numpy arrays for comparison
        y_true = np.array(y_true)
        y_hat = np.array(y_hat)
        
        # Calculate accuracy
        acc = np.mean(y_true == y_hat)
        if name == 'incorrect_loader':
            c=np.sum(y_true == y_hat)
            print(f'Number of samples correctly predicted is {c}/{len(loader)}')
        print(f"Accuracy of {name} is {acc} at epoch {epoch}")


def visualize_mask(Mask):
    from matplotlib import pyplot as plt
    plt.figure(figsize=(10, 6))
    for i,p in enumerate(Mask):
        plt.imshow(p.cpu().detach().numpy())
        # plt.show()
        print(f'Parameter {i}')
        print(p)
        print(torch.sigmoid(p))
        print('-'*100)
        break
    print(f"Mask values:")
    for i, p in enumerate(Mask):
        print(f"  Parameter {i}: mean={torch.mean(torch.sigmoid(p)).item():.4f}, std={torch.std(torch.sigmoid(p)).item():.4f}")


def train(loaders, model, Mask, epochs, criterion, optimizer, args):
    sample = loaders[-1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # model.train()
    Mask.train()
    acc=0
    loss_=[]

    for epoch in range(epochs):
        image, label = sample[0], sample[1]
        image = image.to(device)
        label = label.to(device)
        optimizer.zero_grad()

        
        y_pred_mask = Mask(image)
        y_hat_mask = torch.argmax(y_pred_mask)
        acc = evaluate(label, y_hat_mask)
    
        
        if acc!=1.0:
            label_one_hot = torch.nn.functional.one_hot(label, num_classes=10).float()[0].reshape(1, -1)
            loss = criterion(y_pred_mask, label_one_hot, Mask, args.lambda_reg)

            loss_.append(loss)

            loss.backward()
            optimizer.step()


        else:
            best_model_wts = model.state_dict()
            return best_model_wts, Mask, loss_, True


    best_model_wts = model.state_dict()


    return best_model_wts, Mask, loss_, False


def bar_plot(incorrect_loader, mask, mask_id, args):
    directory = args.directory
    save={}
    for sample in incorrect_loader:
        image, label = sample[0], sample[1]
        y_pred_mask = mask(image)
        y_hat_mask = torch.argmax(y_pred_mask)
        acc = evaluate(label, y_hat_mask)
        label=label.item()
        if acc==1.0:
            if label in save.keys():
                save[label]+=1
            else:
                save[label]=1
            wandb.log({f"mask_{mask_id}/label_{label}": save[label]})

            
    # plot the histogram of the correctly predicted samples
    # plt.figure(figsize=(10, 6))
    # plt.bar(range(10), save)
    # plt.xlabel('Label')
    # plt.ylabel('Number of samples')
    # plt.title(f'Plot of correctly predicted samples by mask {mask_id}/class {mask_id}')
    # plt.xticks(range(10), ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    
    # plt.show()
    # if not os.path.exists(f'{directory}masks/plots/{args.model_type}/{args.dataset}/'):
    #     os.makedirs(f'{directory}masks/plots/{args.model_type}/{args.dataset}/')
    # else:
    #     plt.savefig(f'{directory}masks/plots/{args.model_type}/{args.dataset}/mask_{mask_id}_{args.mask_initial_type}_{args.lambda_reg}.png')


def check_if_model_changed(original_model_params, model):
    model_changed = False
    for orig, curr in zip(original_model_params, model.parameters()):
        if not torch.allclose(orig, curr, atol=1e-6):
            model_changed = True
            break
    print(f"Model parameters changed: {model_changed}")
    
    
def rn_masks(loaders, model, model_directory, args):
    # VERSION: Only optimize mask parameters, not model parameters
    print("Running corrected version that only optimizes mask parameters...")
    incorrect_loader = loaders[-1]
    train_loader, test_loader = loaders[:-1]
    total = len(incorrect_loader)

    model.load_state_dict(torch.load(model_directory, map_location=torch.device(device)))
    # Store original model parameters to verify they don't change
    original_model_params = [p.clone().detach() for p in model.parameters()]
    masks = []
    for mask_id in range(10):  # Reduced to 3 for testing
        mask = Mask(model, random_seed=mask_id, initial_type=args.mask_initial_type)
        optimizer = torch.optim.SGD(mask.mask_parameters, lr=args.mask_lr)
        masks.append(mask)
        
    # Track training progress
    sample_count = 0
    total_correct = {i: 0 for i in range(10)}
    save_img_per_class = {}; save_img = False
    
    # Test on first few samples only for demonstration
    for sample in incorrect_loader:
        _, label = sample[0], sample[1]
        
        if save_img:
            if label.item() not in save_img_per_class.keys():
                # Prepare image for wandb logging
                img = sample[0].squeeze().cpu().numpy()
                img = (img - img.min()) / (img.max() - img.min())
            
                save_img_per_class[label.item()] = img
                wandb.log({
                    f"sample/label_{label.item()}": wandb.Image(img)
                })
        
        mask = masks[label]
        _,_,loss_, correct_prediction = train(loaders+[sample], 
            model, 
            mask, 
            epochs=100, criterion=loss_fn, optimizer=optimizer, args=args)
        
        sample_count += 1
        label=label.item()
        if correct_prediction:
            total_correct[label] += 1
        wandb.log({f"mask_{label}/correct_prediction": total_correct[label]
        })
            
        # Log sample-level metrics
        # wandb.log({
        #     "sample/sample_id": sample_count,
        #     "sample/label": label.item(),
        #     "sample/total_accuracy": total_correct
        # })
                
        # Verify model parameters haven't changed
        # check_if_model_changed(original_model_params, model)

        # Check mask values to see if they're different
        # visualize_mask(mask.mask_parameters)

    # Initialize average mask parameters
    avg_mask_params = []
    for param in masks[0].mask_parameters:
        avg_mask_params.append(torch.zeros_like(param))
    
    for mask_id in range(10):
        print(f'Mask {mask_id}/Class {mask_id}')
        mask = masks[mask_id]
      
        
        # Update running average of mask parameters
        for i, param in enumerate(mask.mask_parameters):
            avg_mask_params[i] = (avg_mask_params[i] * mask_id + param) / (mask_id + 1)
        
        # Calculate mask statistics
        mask_mean = np.mean([torch.mean(torch.sigmoid(p)).item() for p in mask.mask_parameters])
        mask_std = np.mean([torch.std(torch.sigmoid(p)).item() for p in mask.mask_parameters])
        
        # Get accuracies
        train_acc, test_acc, incorrect_acc = get_accuracies([train_loader, test_loader, incorrect_loader], mask)
        # mask_accuracies.append({"train": train_acc, "test": test_acc, "incorrect": incorrect_acc})
        
        # Log mask metrics
        wandb.log({
            f"mask_/train_accuracy": train_acc,
            f"mask_/test_accuracy": test_acc,
            f"mask_/incorrect_accuracy": incorrect_acc,
            f"mask_/mask_mean": mask_mean,
            f"mask_/mask_std": mask_std
        })
        
        wandb.log({
            f"mask_{mask_id}/train_accuracy": train_acc,
            f"mask_{mask_id}/test_accuracy": test_acc,
            f"mask_{mask_id}/incorrect_accuracy": incorrect_acc,
        })
        bar_plot(incorrect_loader, mask, mask_id, args)
        print('*'*100)
        
        mask_avg = Mask(model, random_seed=mask_id, initial_type=args.mask_initial_type)
        mask_avg.mask_parameters = nn.ParameterList(nn.Parameter(p, requires_grad=True) for p in avg_mask_params)
        
        train_acc, test_acc, incorrect_acc = get_accuracies([train_loader, test_loader, incorrect_loader], mask_avg)
        wandb.log({
            f"avg_mask_/train_accuracy": train_acc,
            f"avg_mask_/test_accuracy": test_acc,
            f"avg_mask_/incorrect_accuracy": incorrect_acc,
        })

    wandb.finish()
    
    # save the average mask parameters
    torch.save(avg_mask_params, f'{args.directory}masks/avg_mask_params.pth')



def run(args):
    # Initialize wandb
    wandb.init(
        project="mask-regularization",
        name=f"{args.dataset}_{args.model_type}_lambda{args.lambda_reg}_{args.mask_initial_type}",
        config={
            "dataset": args.dataset,
            "model_type": args.model_type,
            "batch_size": args.batch_size,
            "learning_rate": args.mask_lr,
            "lambda_reg": args.lambda_reg,
            "mask_initial_type": args.mask_initial_type,
            "num_masks": 10,
            "device": device
        }
    )
    
    print('Device =', device)
    dataload = {
        'CIFAR10': data_loader_CIFAR10,
        'MNIST': data_loader_MNIST,
        'MED': data_loader_MED
    }
    small_data = {
        'MED': True,
        'CIFAR10': False,
        'MNIST': False
    }
    load_model = {
            'CIFAR10': args.directory + f'models/{args.model_type}/model_weights_cifar_diff.pth',
            'MNIST': args.directory + f'models/{args.model_type}/model_weights_mnist.pth',
            'MED': args.directory + f'models/{args.model_type}/model_weights_med.pth'
        }
    train_loader, valid_loader, test_loader = dataload[args.dataset](directory=args.directory,
                                                                         batch_size=args.batch_size,
                                                                         random_seed=args.random_seed,
                                                                         small=small_data[args.dataset],
                                                                         shuffle=False)
    
    image, _ = next(iter(train_loader))
    if args.model_type == 'AlexNet':
        model = AlexNet(input_size=image.size()[2], input_channel=image.size()[
                        1], num_classes=args.num_classes)
    elif args.model_type == 'MLP':
        model = MLP(num_classes=args.num_classes, input_size=image.size()[1:])
    model = model.to(device)
    model.load_state_dict(torch.load(load_model[args.dataset], map_location=torch.device(device))) # assumes model is already trained and saved
    model_directory = load_model[args.dataset]
    incorrect_loader_path = f'{args.directory}dataloaders/{args.model_type}/{args.dataset}_incorrect_loader.pth'
    if os.path.exists(incorrect_loader_path):
        # incorrect_loader = torch.load(incorrect_loader_path, weights_only=False)  # for compute canada compatibility
        incorrect_loader = torch.load(incorrect_loader_path, map_location=torch.device(device))
    else:
        incorrect_loader = get_incorrect_loader(model=model, test_loader=test_loader)
        torch.save(incorrect_loader, incorrect_loader_path)
        
        
    rn_masks(loaders=[train_loader, test_loader, incorrect_loader], model=model, model_directory=model_directory, args=args)

def boolean(x):
    if x == 'True':
        return True
    else:
        return False

def print_params(args):
    print(f'Directory = {args.directory}')
    print(f'Dataset = {args.dataset}')
    print(f'Batch size = {args.batch_size}')
    print(f'Random seed = {args.random_seed}')
    print(f'Number of classes = {args.num_classes}')
    print(f'Epochs = {args.epochs}')
    print(f'Mask learning rate = {args.mask_lr}')
    print(f'Model type = {args.model_type}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Settings to run AlexNet model')
    # parser.add_argument('--directory', default='/home/arghasre/scratch/XAI/', help='directory')
    parser.add_argument('--directory', default='/Users/arghasreebanerjee/PycharmProjects/BGD/', help='directory')
    parser.add_argument('--dataset', default='MNIST', type=str, help='Which dataset?')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--random_seed', default=1, type=int, help='Random seed')
    parser.add_argument('--num_classes', default=10, type=int, help='10 for cifar10')
    parser.add_argument('--epochs', default=100, type=int, help='epochs')
    parser.add_argument('--mask_lr', default=0.1, type=float, help='Learning rate') # experiment with this
    parser.add_argument('--model_type', default='MLP', type=str, help='AlexNet or MLP or LSTM')
    parser.add_argument('--lambda_reg', default=0.01, type=float, help='Lambda regularization') # experiment with this
    parser.add_argument('--mask_initial_type', default='uniform', type=str, help='uniform, zeros, ones')
    

    args = parser.parse_args()


    # print_params(args)
    run(args)


# based on the activation of model and mask
# start with a raandom out of 10 and then see 
# one for each class 

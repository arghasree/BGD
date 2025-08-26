import argparse
from model_trainers.trainer import *
from dataloaders.dataloading import *
import torch.nn as nn
from torch import optim
from util.search_util import *
from dataloaders.custom_dataloader import *
from models.model_mlp import *
from models.model_small import *
import os
import time
# from dataloaders.get_all_samples import *
# from dataloaders.imbalanced_dataloader import *


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
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

    if args.imbalanced:
        train_loader, test_loader = imbalanced_data_loader_CIFAR10(imbanlance_rate=args.imbalanced_rate,
                                                                   directory=args.directory,
                                                                   dataset=args.dataset,
                                                                   batch_size=args.batch_size,
                                                                   random_seed=args.random_seed,
                                                                   small=small_data[args.dataset],
                                                                   shuffle=False)
    else:
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
    
    # print the model layer names 
    # print(model)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(
    ), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=args.lr_decay)

    if args.imbalanced:
        load_model = {
            'CIFAR10': args.directory + f'models/{args.model_type}/model_weights_cifar_imbalanced.pth',
            'MNIST': args.directory + f'models/{args.model_type}/model_weights_mnist_imbalanced.pth'
        }
    else:
        load_model = {
            'CIFAR10': args.directory + f'models/{args.model_type}/model_weights_cifar_diff.pth',
            'MNIST': args.directory + f'models/{args.model_type}/model_weights_mnist.pth',
            'MED': args.directory + f'models/{args.model_type}/model_weights_med.pth'
        }

    if os.path.exists(load_model[args.dataset]):
        model.load_state_dict(torch.load(
            load_model[args.dataset], map_location=torch.device(device)))
        accuracy, best_model_wts = load_trained(train_loader, model)
    else:
        _, _, model, _, best_model_wts = train(train_loader, None, model,  # validation_data = None
                                               args.epochs,
                                               criterion, optimizer, scheduler)
        torch.save(model.state_dict(), load_model[args.dataset])

    y_true, y_hat = predict(model, test_loader)
    print('Testing performance of main model =', evaluate(y_true.to(device), y_hat.to(device)))


    incorrect_loader_path = f'{args.directory}dataloaders/{args.model_type}/{args.dataset}_incorrect_loader.pth'
    if os.path.exists(incorrect_loader_path):
        incorrect_loader = torch.load(incorrect_loader_path, weights_only=False)
    else:
        incorrect_loader = get_incorrect_loader(model=model, test_loader=test_loader)
        torch.save(incorrect_loader, incorrect_loader_path)
    
    if os.path.exists(f'{args.directory}models/{args.model_type}/model_weights_copy_10.pth'):
        changed_model = MLP(num_classes=args.num_classes, input_size=image.size()[1:])
        changed_model = changed_model.to(device)
        print(f'Loading changed model from {args.directory}models/{args.model_type}/model_weights_copy_10.pth')
        changed_model.load_state_dict(torch.load(f'{args.directory}models/{args.model_type}/model_weights_copy_10.pth', map_location=torch.device(device)))
        y_true, y_hat = predict(changed_model, incorrect_loader)
        print('Testing performance of changed model =', evaluate(y_true.to(device), y_hat.to(device)))
        
        # checking for the first 10 samples
        c=0
        for i, sample in enumerate(incorrect_loader):
            if i==10:
                break
            image, label = sample[0].to(device), sample[1].to(device)
            output = changed_model(image)
            _, predicted = torch.max(output, 1)
            correct_pred = (predicted == label).item()
            if correct_pred:
                print(f'Correctly predicted -> {i}')
            
    
    # # Make a loader based on only the incorrect samples
    # dataset = [list(test_loader)[sample_indx] for sample_indx in incorrect_sample_indices]
    # incorrect_data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    parameter_search_imbalanced(model=model,
                                best_model_wts=best_model_wts,
                                train_loader=train_loader,
                                test_loader=test_loader,
                                criterion=criterion,
                                optimizer=optimizer,
                                incorrect_loader=incorrect_loader,
                                args=args)
    print(f'Time taken = {time.time()-start}')


def print_params(args):
    print('\nDataset =', args.dataset)
    print('Directory =', args.directory)
    print('Batch Size =', args.batch_size)
    print(f'Long Tailed Distributions? {args.imbalanced}')
    print(f'Imbalanced Rate = {args.imbalanced_rate}')
    print('Random seed =', args.random_seed)
    print('Number of classes =', args.num_classes)
    print('Epochs =', args.epochs)
    print('Learning Rate =', args.learning_rate, '; Learning rate decay =',
          args.lr_decay, '; Weight decay =', args.weight_decay)
    print('Reload main model =', args.reloaded)
    print('Choose Sample from pre-downloaded samples? =', args.test_samples_check)
    print(f'Training set accuracy check? = {args.training_acc_check}')
    print(f'Testing set accuracy check? = {args.test_acc_check}')
    print(f'Layer for sampling parameters = {args.layer_name}')
    print(
        f'Update by first order derivative? (If False, update happens by 2nd order)  = {args.first}')
    print(
        f'Lamda hyperparameter for training loss regularization term = {args.multiple}')
    print(f'Iterations for subset selection = {args.iterations}')
    print(f'Normalize the gradients per layer? = {args.normalize}')
    print(f'Model Type = {args.model_type}')
    print(f'Threshold for increasing explore probability = {args.threshold}')
    print(f'Percentile for lower bound = {args.p}')
    print(
        f'Check curve information while sampling gradients? = {args.check_curve_info}')
    print(f'Copy weight and reinitialize? = {args.copy_weight_reinit}')


def boolean(x):
    if x == 'True':
        return True
    else:
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Settings to run AlexNet model')
    parser.add_argument(
        '--directory', default='/home/arghasre/scratch/XAI/', help='directory')
    parser.add_argument('--dataset', default='CIFAR10',
                        type=str, help='Which dataset?')
    parser.add_argument('--imbalanced', default='True',
                        help='Long Tailed Distribution or Balanced classes')
    parser.add_argument('--imbalanced_rate', default=0.01,
                        type=float, help='Imbalanced rate')
    parser.add_argument('--batch_size', default=64,
                        type=int, help='Batch size')
    parser.add_argument('--random_seed', default=1,
                        type=int, help='Random seed')
    parser.add_argument('--num_classes', default=10,
                        type=int, help='10 for cifar10')
    parser.add_argument('--epochs', default=100, type=int, help='epochs')
    parser.add_argument('--learning_rate', default=0.001,
                        type=float, help='Learning rate')
    parser.add_argument('--lr_decay', default=0.997,
                        type=float, help='Learning rate decay')
    parser.add_argument('--weight_decay', default=0.0,
                        type=float, help='Weight decay')
    parser.add_argument('--reloaded', default='False',
                        help='Reload the main model?')
    parser.add_argument('--choice_of_training', default='random',
                        type=str, choices=['random', 'same class', 'class'])
    parser.add_argument('--training_subset_check', default='False',
                        help='Training loss regularization term?')
    parser.add_argument('--training_acc_check', default='True',
                        help='Training set accuracy check?')
    parser.add_argument('--test_samples_check', default='True',
                        help='Choose Sample from pre-downloaded samples')
    parser.add_argument('--test_acc_check', default='True',
                        help='Testing set accuracy check?')
    parser.add_argument('--layer_name', default=None, type=str,
                        help='Which layer to sample parameters from?')
    parser.add_argument('--first', default='True',
                        help='First order derivative or Second order')
    parser.add_argument('--multiple', default=None, type=int,
                        help='Lamda hyperparameter for training loss regularization term')
    parser.add_argument('--iterations', default=20, type=int,
                        help='Iterations for subset selection')
    parser.add_argument('--normalize', default='False',
                        help='Normalize the gradients per layer')
    parser.add_argument('--model_type', default='MLP',
                        type=str, help='AlexNet or MLP or LSTM')
    parser.add_argument('--threshold', default=0.01, type=float,
                        help='threshold for increasing explore probability')
    parser.add_argument('--n_Param_update', default=1, type=int,
                        help='threshold for increasing explore probability')
    parser.add_argument('--start_index', default=300,
                        type=int, help='Starting index')
    parser.add_argument('--p', default=98, type=int,
                        help='Percentile for lower bound')
    parser.add_argument('--check_curve_info', default='False',
                        help='Check curve information while sampling gradients?')
    parser.add_argument('--copy_weight_reinit', default='False',
                        help='Copy weight and reinitialize?')

    args = parser.parse_args()

    args.reloaded = boolean(args.reloaded)
    args.training_subset_check = boolean(args.training_subset_check)
    args.training_acc_check = boolean(args.training_acc_check)
    args.test_samples_check = boolean(args.test_samples_check)
    args.test_acc_check = boolean(args.test_acc_check)
    args.first = boolean(args.first)
    args.normalize = boolean(args.normalize)
    args.imbalanced = boolean(args.imbalanced)
    args.check_curve_info = boolean(args.check_curve_info)
    args.copy_weight_reinit = boolean(args.copy_weight_reinit)

    print_params(args)
    run(args)

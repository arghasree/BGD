import argparse
from model_trainers.trainer import *
from dataloaders.dataloading import *
import torch.nn as nn
from torch import optim
from util.search_util import *
from dataloaders.custom_dataloader import *
from models.model_small import *
import os
import time


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('\nDevice =',device)
    print('Dataset =', args.dataset)
    dataload = {
        'CIFAR10': data_loader_CIFAR10,
        'MNIST': data_loader_MNIST
    }
    
    train_loader, valid_loader, test_loader = dataload[args.dataset](directory=args.directory, 
                                                                     batch_size=args.batch_size, 
                                                                     random_seed=args.random_seed, 
                                                                     small=False, 
                                                                     shuffle=False)
    
    image, _ = next(iter(train_loader))
    model = AlexNet(input_size=image.size()[2], input_channel=image.size()[1], num_classes=args.num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=args.lr_decay)
    
    load_model = {
        'CIFAR10': args.directory + 'models/model_weights_cifar_diff.pth',
        'MNIST': args.directory + 'models/model_weights_mnist.pth'
    }
    if os.path.exists(load_model[args.dataset]):
        model.load_state_dict(torch.load(load_model[args.dataset], map_location=torch.device(device)))
        accuracy, best_model_wts = load_trained(valid_loader, model)
    else:
        loss, accuracy, model, before_weights_epochs, best_model_wts = train(train_loader, valid_loader, model,
                                                                            args.epochs,
                                                                            criterion, optimizer, scheduler)
        torch.save(model.state_dict(), load_model[args.dataset])


    
    print("\n--Testing--")
    y_true, y_hat = predict(model, test_loader)
    print('Testing performance of main model =', evaluate(y_true.to(device), y_hat.to(device)))
    
    start = time.time()
    parameter_search(dataset=args.dataset,
                     model=model, 
                     best_model_wts=best_model_wts, 
                     training_data=train_loader, 
                     test_loader=test_loader, 
                     choice_of_training=args.choice_of_training, 
                     directory=args.directory, 
                     criterion=criterion, 
                     optimizer=optimizer, 
                     scheduler=scheduler,
                     epochs=args.epochs, 
                     learning_rate=args.lr,
                     reloaded=args.reloaded, 
                     iterations=args.iterations, 
                     multiple=args.multiple, 
                     layer_name=args.layer_name, 
                     training_acc_check=args.training_acc_check, 
                     training_subset_check=args.training_subset_check, 
                     test_acc_check=args.test_acc_check, 
                     swap=args.swap,
                     threshold=0.01,
                     test_samples_check=args.test_samples_check,
                     normalize=args.normalize)
    print(f'Time taken = {time.time()-start}')
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Settings to run AlexNet model')
    parser.add_argument('--directory', default='/local/scratch1/arghasre/XAI_code/', help='directory')
    parser.add_argument('--dataset', default='MNIST', type=str, help='Which dataset?')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--random_seed', default=1, type=int, help='Random seed')
    parser.add_argument('--num_classes', default=10, type=int, help='10 for cifar10')
    parser.add_argument('--epochs', default=100, type=int, help='epochs')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--lr_decay', default=0.997, type=float, help='Learning rate decay')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='Weight decay')
    parser.add_argument('--reloaded', default=True, type=bool, help='Reload the main model?')
    parser.add_argument('--choice_of_training', default='random', type=str, choices=['random','same class','class'])
    parser.add_argument('--training_subset_check', default=False, type=bool, help='Training loss regularization term?')
    parser.add_argument('--training_acc_check', default=True, type=bool, help='Training set accuracy check?')
    parser.add_argument('--test_samples_check', default=False, type=bool, help='Choose Sample from pre-downloaded samples')
    parser.add_argument('--test_acc_check', default=False, type=bool, help='Testing set accuracy check?')
    parser.add_argument('--layer_name', default=None, type=str, help='Which layer to sample parameters from?')
    parser.add_argument('--swap', default=False, type=bool, help='Parameter value swap with transpose element?')
    parser.add_argument('--multiple', default=None, type=int, help='Lamda hyperparameter for training loss regularization term')
    parser.add_argument('--iterations', default=20, type=int, help='Iterations for subset selection')
    parser.add_argument('--normalize', default=False, type=bool, help='Normalize the gradients per layer')
    

    args = parser.parse_args()
    run(args)

from model_trainers.trainer import *
from dataloaders.custom_dataloader import *
import dataloaders.dataloading as dataloading
from util.subset_util import *
from objects.sample import Test_Sample
import matplotlib.pyplot as plt
import os

def delete(directory_pth, directory_png):
    """
    Deletes the files(both pth and png files) having indices in list <list_of_indices>
    """
    list_of_indices = [22,25,27,28,30,31,35,40,43,44,62,97,102,113,142,153,176,179,180,190,191,202,205,207,210,215,219,220,231,235,242,245,246,
                       247,251,265,268,318,328,386,398,403,405,413,425,427,433,443,473,484,503,504,505,507,510,529,532,551,593,600,601,610,612,648,650,655,669,683,692,695,699,775,
                       779,786,798,799,835,869,876,885,922,923,950,959,969,975,976,1009,1030]
    list_of_sample_images = [f'sample_{i}.png' for i in list_of_indices] 
    list_of_sample_pth = [f'sample_{i}.pth' for i in list_of_indices]
    
    for i in range(len(list_of_indices)):
        # os.remove(f'{directory_png}{list_of_sample_images[i]}')
        os.remove(f'{directory_pth}{list_of_sample_pth[i]}')
        
        

def central(args, model, test_loader):
    """
    Creates sample objects for all incorrectly predicted samples
    """
    incorrect_pred_samples = predict_test(model, test_loader, args.directory, args.dataset)
    classes = dataloading.class_name(directory=args.directory, dataset=args.dataset)
    incorrect_pred_img = incorrect_pred_samples[0]; incorrect_pred_labels = incorrect_pred_samples[1]; incorrect_pred_classes = incorrect_pred_samples[2];
    get_test_samples(classes, incorrect_pred_img, incorrect_pred_labels, incorrect_pred_classes, args)
    


def get_test_samples(classes, incorrect_pred_img, incorrect_pred_labels, incorrect_pred_classes, args):
    """
    Creates sample objects 
    """
    for i in range(len(incorrect_pred_img)):
        image = [incorrect_pred_img[i]]
        label = [incorrect_pred_labels[i]]
        wrong_class_name = incorrect_pred_classes[i]
        correct_class_name = classes[label[0]]
        sample = data_loader(images=image, labels=label, batch_size=1, shuffle=False)
        directory_to_save = f'{args.directory}all_samples/{args.model_type}/{args.dataset}/'
        sample_name = f'{directory_to_save}sample_{i}.pth'
        
        # directory_to_save_images = f'{args.directory}all_samples/{args.model_type}/{args.dataset}/images/sample_{i}'
        # imshow(sample, directory_to_save_images)
        
        torch.save(Test_Sample(classes, wrong_class_name, correct_class_name, sample_name, sample), sample_name)
        

def imshow(sample, directory):
    """
    Creates image files to save for those incorrectly predicted samples
    """
    img,_ = next(iter(sample))
    img = img / 2 + 0.5
    npimg = img.cpu().numpy().reshape((img.shape[1], img.shape[2], img.shape[3]))
    fig = plt.figure(figsize=(1, 1))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(f'{directory}.png')


def get_incorrect_samples(test_loader, model):
    """Calculates the indices to the samples that are incorrectly predicted by the model
    Returns:
        incorrect_sample_indices:  <list>
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    incorrect_sample_indices = []
    print('Length of test loader =', len(test_loader))
    
    for i, x in enumerate(test_loader): 
        image, y_true = x
        image = image.to(device)
        y_true = y_true.to(device)
        output = model(image)
        y_hat = torch.argmax(output, dim=1).to(device)
        
        if y_true!=y_hat: # save the index
            incorrect_sample_indices.append(i)

    return incorrect_sample_indices
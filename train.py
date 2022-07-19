import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,models,transforms
import torch.nn.functional as F
from PIL import Image
from torch.optim import lr_scheduler
import argparse

#Globals
nThreads = 4
batch_size = 8
use_gpu = torch.cuda.is_available()

def data(args):
    
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    #Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    image_datasets = dict()
    image_datasets['train'] = datasets.ImageFolder(train_dir, transform=train_transforms)
    image_datasets['valid'] = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    image_datasets['test'] = datasets.ImageFolder(test_dir, transform=test_transforms)


    # TODO: Using the image datasets and the trainforms, define the dataloaders

    dataloaders = dict()
    dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True)
    dataloaders['valid'] = torch.utils.data.DataLoader(image_datasets['valid'], batch_size=batch_size)
    dataloaders['test']  = torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size)

    return dataloaders, image_datasets

def train_model(args, model, criterion, optimizer, scheduler, epochs=25):
    
    dataloaders, image_datasets = data(args)
    model.to('cuda')
    print_every = 50
    steps = 0
    for e in range(epochs):
        current_loss = 0
        for images,labels in dataloaders['train']:
            steps += 1
            images,labels = images.to('cuda'),labels.to('cuda')
            optimizer.zero_grad()
            outputs = model.forward(images)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            current_loss += loss.item()
            if steps%print_every == 0:
                model.eval()
                vloss = 0
                accuracy = 0
                for vimages,vlabels in dataloaders['valid']:
                    optimizer.zero_grad()
                    vimages,vlabels = vimages.to('cuda:0'),vlabels.to('cuda:0')
                    model.to('cuda:0')
                    with torch.no_grad():
                        voutputs = model.forward(vimages)
                        vloss = criterion(voutputs,vlabels)
                        ps = torch.exp(voutputs).data
                        eq = (vlabels == ps.max(1)[1])
                        accuracy += eq.type_as(torch.FloatTensor()).mean()
                vloss = vloss/len(dataloaders['valid'])
                accuracy = accuracy/len(dataloaders['valid'])
                print("epoch {}/{} :".format(e+1,epochs),
                        " training loss = {}".format(current_loss/print_every),
                        " validation loss = {}".format(vloss),
                        " accuracy = {:.2f} %".format(accuracy*100))
                current_loss = 0
    return model


def train_model_wrapper(args):

    dataloaders, image_datasets = data(args)
    if args.arch == 'vgg': 
        model = models.vgg16(pretrained=True)
    elif args.arch == 'densenet':
        model = models.densenet121(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.classifier[0].in_features
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(num_features, 512)),
                              ('relu', nn.ReLU()),
                              ('drpot', nn.Dropout(p=0.5)),
                              ('hidden', nn.Linear(512, args.hidden_units)),                       
                              ('fc2', nn.Linear(args.hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1)),
                              ]))

    # Reserve for final layer: ('output', nn.LogSoftmax(dim=1))
        
    model.classifier = classifier    
    if args.gpu:
        if use_gpu:
            model = model.cuda()
            print ("Using GPU: "+ str(use_gpu))
        else:
            print("Using CPU since GPU is not available/configured")
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    model = train_model(args, model, criterion, optimizer, exp_lr_scheduler,epochs=args.epochs)

    #CHECKPOINT
    model.class_to_idx = dataloaders['train'].dataset.class_to_idx
    model.epochs = args.epochs
    checkpoint = {'input_size': [3, 224, 224],
                  'batch_size': dataloaders['train'].batch_size,
                  'output_size': 102,
                  'arch': args.arch,
                  'state_dict': model.state_dict(),
                  'optimizer_dict':optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'epoch': model.epochs}
    torch.save(checkpoint, args.saved_model)


def main():
    parser = argparse.ArgumentParser(description='Flower Classifcation trainer')
    parser.add_argument('--gpu', type=bool, default=False, help='Use GPU or not')
    parser.add_argument('--arch', type=str, default='densenet', help='architecture [available: densenet, vgg]', required=True)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--hidden_units', type=int, default=100, help='hidden units for fc layer')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--data_dir', type=str, default='flowers', help='dataset directory')
    parser.add_argument('--saved_model' , type=str, default='my_checkpoint_cmd.pth', help='path of your saved model')
    args = parser.parse_args()

    import json
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    train_model_wrapper(args)


if __name__ == "__main__":
    main()
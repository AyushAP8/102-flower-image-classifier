import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import torch.nn.functional as F
from torch.optim import lr_scheduler
from PIL import Image
import argparse
import json

use_gpu = torch.cuda.is_available

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image)
    img_transform = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485,0.456,0.406],
                                                           [0.229,0.224,0.225])])
    return img_transform(img)



def load_checkpoint(args):
    checkpoint_provided = torch.load(args.saved_model)
    if checkpoint_provided['arch'] == 'vgg':
        model = models.vgg16()        
    elif checkpoint_provided['arch'] == 'densenet':
        model = models.densenet121()
    num_features = model.classifier[0].in_features
    from collections import OrderedDict
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(num_features, 512)),
                          ('relu', nn.ReLU()),
                          ('drpot', nn.Dropout(p=0.5)),
                          ('hidden', nn.Linear(512, args.hidden_units)),                        
                          ('fc2', nn.Linear(args.hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1)),
                          ]))
    model.classifier = classifier
    model.load_state_dict(checkpoint_provided['state_dict'])
    if args.gpu:
        if use_gpu:
            model = model.cuda()
            print ("Using GPU")
        else:
            print("Using CPU")
    class_to_idx = checkpoint_provided['class_to_idx']
    idx_to_class = { v : k for k,v in class_to_idx.items()}
    return model, class_to_idx, idx_to_class

def predict(args, image_path, model, class_to_idx, idx_to_class, cat_to_name, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = process_image(image_path)
    image.unsqueeze_(0)
    if use_gpu and args.gpu:
        model = model.cuda()    
    model.eval()
    if use_gpu and args.gpu:
        output = model.forward(image.cuda())
    else:
        output = model.forward(image)
    ps = torch.exp(output.cpu())#.data.numpy()[0]
    top_ps, top_labels = ps.topk(topk, dim=1)
    top_ps = top_ps.detach().numpy().tolist()[0]
    top_flowers = [cat_to_name[idx_to_class[x]] for x in top_labels.detach().numpy().tolist()[0]]
    return top_ps,top_flowers

def main():

    parser = argparse.ArgumentParser(description='Flower Classification Predictor')
    parser.add_argument('--gpu', type=bool, default=False, help='use gpu or not')
    parser.add_argument('--image_path', type=str, help='path of image')
    parser.add_argument('--hidden_units', type=int, default=100, help='hidden units for fc layer')
    parser.add_argument('--saved_model' , type=str, default='my_checkpoint_cmd.pth', help='path of your saved model')
    parser.add_argument('--mapper_json' , type=str, default='cat_to_name.json', help='path of your mapper from category to name')
    parser.add_argument('--topk', type=int, default=5, help='display top k probabilities')
    args = parser.parse_args()
    with open(args.mapper_json, 'r') as f:
        cat_to_name = json.load(f)

    model, class_to_idx, idx_to_class = load_checkpoint(args)
    top_probability, top_predictions = predict(args, args.image_path, model, class_to_idx, idx_to_class, cat_to_name, topk=args.topk)
                                              
    print('Predicted Classes: ', top_predictions)
    print('Predicted Probability: ', top_probability)
    print ('predicted class {} with an accuracy of {}%'.format(top_predictions[0],top_probability[0]*100))
    
    

if __name__ == "__main__":
    main()
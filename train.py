import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torch.utils.data
import numpy as np
import json
import matplotlib.pyplot as plt
import os, random
from torch.autograd import Variable
import torchvision

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets

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
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.Dataloader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.Dataloader(valid_data, batch_size=32)
testloader = torch.utils.data.Dataloader(test_data, batch_size=32)

image_datasets = [train_data, valid_data, test_data]
dataloader = [train_loader, valid_loader, test_loader]

data_iter = iter(trainloader)

images, labels = next(data_iter)
fig, axes = plt.subplots(figsize=(10,4), ncols=4)

for ii in range(4):
    ax=axes[ii]
    helper.imshow(images[ii], ax+ax, normalize=True)

import json
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
print(cat_to_name)
print("/n length: ", len (cat_to_name))

# TODO: build and train your network
model = models.vgg19(pretrained=True)

for param in model.parameters():
    param.requires_grad = False
    
classifier = nn.Sequential(nn.Linear(2048, 512),
                           nn.ReLU(),
                           nn.Linear(512, 102),
                           nn.ReLU(),
                           nn.LogSoftmax(dim=1))
model.classifier = classifier



criterion = nn.NLLLoss()
optimizer = optim.SGD(model.classifier.parameters(), lr=0.001)



#Training

epochs = 5
steps = 0
cuda = torch.cuda.is_available()
if cuda:
    model.cuda()
else:
    model.cpu()
    
running_loss = 0
accuracy = 0
print_every = 5

for epoch in range(epochs):
    
    train_mode = 0
    valid_mode = 1
    
    for mode in [train_mode, valid_mode]:
        if mode == train_mode:
            model.train
        else:
            model.eval()
            
        pass_count = 0
        
        for data in dataloaders[mode]:
            pass_count += 1
            inputs, labels = data
            
            if cuda == True:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
        
            optimizer.zero_grad()
            
            ##Forward training
            output = model.forward(inputs)
            loss = criterion(output, labels)
            
            ##backward training
            if mode== train_mode:
                loss.backward()
                optimizer()
        
            running_loss += loss.item()
            ps = torch.exp(output).data
            equality = (labels.data == ps.max(1)[1])
            accuracy = equality.type_as(torch.cuda.FloatTensor()).mean()
        
    if mode == train_mode:
        print(f"Epoch {epoch+1}/{epochs}.."
              f"Train loss: {running_loss/print_every:.3f}"
              f"Test loss: {test_loss/len(validloader):.3f}")
    else:     
        print(f"Test accuracy: {accuracy/len(validloader):.3f}")
        running_loss = 0
        model.train ()
        
# TODO: Do validation on the test set
model.eval()
accuracy = 0
cuda = torch.cuda.is_available()

if cuda:
    model.cuda()
    
else:
    model.cpu()
    
pass_count = 0
for data in dataloader[2]:
    pass_count += 1
    images, lables = data
    
    if cuda == True:
        images, labels = Variable(images.cuda()), Variable(labels.cuda())
    else: 
        images, labels = Variable(images), Variables(labels)
        
    output = model.forward(images)
    ps = torch.exp(output).data
    equality = (labels.data ==ps.max(1)[1])
    accuracy += equality.type_as(torch.FloatTensor()).mean()
    
print("Testing accuracy: {:.4f}".format(accuracy/pass_count))    


# TODO: Save the checkpoint 
model.class_to_idx=image_datasets[0].class_to_idx

checkpoint = {'input_size ': 25088,
              'output size ': 102,
              'arch ': 'vgg19',
              'learning_rate ': 0.01,
              'batch_size ': 64,
              'clasifier ': classifier,
              'epochs' : epochs,
              'optimizer ': optimizer.state_dict(),
              'state dict ': model.state_dict(),
              'class_to_idx ': model.class_to_idx}

torch.save(checkpoint, 'checkpoint.pth')

# TODO: Write a function that loads a checkpoint and rebuilds the model

def load_checkpoint(filename):
    checkpoint = torch.load(filename)
    learning_rate = checkpoint['learning_rate']
    model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint ['class_to_idx']
    optimizer.load_state_dict(checkpoint['optimizer'])
 

    return model, optimizer
                                        
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    im=Image.open(image)
    im = im.resize((256,256))
    value = 0.5*(256-224)
    im = im.crop((value,value,256-value,256-value))
    im = np.array(im)/255
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    im = (im - mean) / std
    
    return im.transpose(2,0,1)

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax



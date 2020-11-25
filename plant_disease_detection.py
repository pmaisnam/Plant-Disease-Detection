import torch
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F #Applies a 1D convolution over an input signal composed of several input planes.
from torch import nn
from torchvision import datasets, transforms, models #allows to load pretrained models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#transform preprocesses the images before feeding it to the networrk
transform_train = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),#Random affine transformation of the image keeping center invariant
                                      #0-to deactivate rotations
                                      transforms.ColorJitter(brightness=1, contrast=1, saturation=1),#Randomly change the brightness, contrast and saturation of an image.
                                      transforms.ToTensor(),#Convert a PIL Image or numpy.ndarray to tensor.
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                      ])

#Transformation for validation dataset
transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])

#ImageFolder will automatically make class acc to the folder placement- say disease is class 0 and  healthy is 1
training_dataset = datasets.ImageFolder(pathlib.Path('C:/disease/train'), transform=transform_train)
validation_dataset = datasets.ImageFolder(pathlib.Path('C:/disease/val'), transform=transform)

training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=1, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=False)


def im_convert(tensor):
    image = tensor.cpu().clone().detach().numpy()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
    image = image.clip(0, 1)
    return image

classes = ('disease', 'healthy')

dataiter = iter(training_loader)
images, labels = dataiter.next()



model = models.vgg16(pretrained=True)

print(model)

for param in model.features.parameters(): #freezes the parameter in feature extraction layer
    param.requires_grad = False

import torch.nn as nn #the last fully connected layer has 1000 nodes we are fixing to how many ever classes we have

n_inputs = model.classifier[6].in_features
last_layer = nn.Linear(n_inputs, len(classes)) #replacing the last layer 
model.classifier[6] = last_layer
model.to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#training of classifier part of the model
epochs = 4
running_loss_history = []
running_corrects_history = []
val_running_loss_history = []
val_running_corrects_history = []

for e in range(epochs):

    running_loss = 0.0
    running_corrects = 0.0
    val_running_loss = 0.0
    val_running_corrects = 0.0

    for inputs, labels in training_loader:
        #inputs = inputs.to(device)
        #labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data)

    else:
        with torch.no_grad():
            for val_inputs, val_labels in validation_loader:
                #val_inputs = val_inputs.to(device)
                #val_labels = val_labels.to(device)
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_labels)

                _, val_preds = torch.max(val_outputs, 1)
                val_running_loss += val_loss.item()
                val_running_corrects += torch.sum(val_preds == val_labels.data)

        epoch_loss = running_loss / len(training_loader.dataset)
        epoch_acc = running_corrects.float() / len(training_loader.dataset)
        running_loss_history.append(epoch_loss)
        running_corrects_history.append(epoch_acc)

        val_epoch_loss = val_running_loss / len(validation_loader.dataset)
        val_epoch_acc = val_running_corrects.float() / len(validation_loader.dataset)# this will give a probability since we are dividing number of correctly identified images by the total number of images
        val_running_loss_history.append(val_epoch_loss)
        val_running_corrects_history.append(val_epoch_acc)
        print('epoch :', (e + 1))
        print('training loss: {:.4f}, acc {:.4f} '.format(epoch_loss, epoch_acc.item()))
        print('validation loss: {:.4f}, validation acc {:.4f} '.format(val_epoch_loss, val_epoch_acc.item()))


    dataiter = iter(validation_loader)

    images, labels = dataiter.next()

    #images = images.to(device)

    #labels = labels.to(device)

    output = model(images)

    _, preds = torch.max(output, 1)

    torch.save(model, pathlib.Path('C:/disease/plants_disease_model.pth'))

import PIL.ImageOps

import requests
from PIL import Image #upport for opening, manipulating, and saving many different image file formats

#url = pathlib.Path('F:/kaggle/data/ants_and_bees/val/1355974687_1341c1face.jpg')
#response = requests.get(url, stream=True)
#img = Image.open(response.raw)
#plt.imshow(img)

#output tomatio healthy

image = Image.open(pathlib.Path('C:/disease/test/000bf685.JPG'))

image = transform (image)
image = image.to(device).unsqueeze(0)
output = model(image)
_, pred = torch.max(output, 1)
print('prerna project demo')
print(classes[pred.item()])
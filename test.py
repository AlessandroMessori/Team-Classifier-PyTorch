import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from torchvision.models.vgg import model_urls
from PIL import Image
import numpy as np
import os

## Load the model 
model_urls['vgg11'] = model_urls['vgg11'].replace('https://', 'http://')
model_conv = models.vgg11(pretrained='imagenet')

## Number of Classification Classes 
n_class = 4
teams = ["Barcellona","Bayern Munich","Juventus","Real Madrid"]

## Lets freeze the first few layers. This is done in two stages 
# Stage-1 Freezing all the layers 

for i, param in model_conv.named_parameters():
    param.requires_grad = False

# Since imagenet as 1000 classes , We need to change our last layer according to the number of classes we have,
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, n_class)

# Stage-2 , Freeze all the layers till "Conv2d_4a_3*3"
ct = []
for name, child in model_conv.named_children():
    if "Conv2d_4a_3x3" in ct:
        for params in child.parameters():
            params.requires_grad = True
    ct.append(name)

# Load pretrained weights
state_dict = torch.load("./state_dict.h5")
model_conv.load_state_dict(state_dict)

if torch.cuda.is_available():
    model_conv.cuda()

input_shape = 299
batch_size = 32
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
scale = 360
input_shape = 299 

loader = transforms.Compose([
        transforms.Resize(scale),
        transforms.CenterCrop(input_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])


test_input = torch.ones([batch_size,3,299,299])

test_image = Image.open("./test.jpeg")
test_image = loader(test_image).float()
test_input[0] = test_image

outputs = model_conv(test_input)
if type(outputs) == tuple:
        outputs, _ = outputs
_, preds = torch.Tensor.max(outputs.data, 1)

result = teams[preds[0]]
print(result)

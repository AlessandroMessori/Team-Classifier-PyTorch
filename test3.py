import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from torchvision.models.resnet import model_urls
from PIL import Image
import numpy as np
import os

## Load the model
model_urls["resnet18"] = model_urls["resnet18"].replace("https://", "http://")
model_conv = models.resnet18(pretrained=True)

# device = torch.device("cuda")

## Lets freeze the first few layers. This is done in two stages
# Stage-1 Freezing all the layers

for i, param in model_conv.named_parameters():
    param.requires_grad = False

## Number of Classification Classes
n_class = 4
teams = ["Barcellona", "Bayern Munich", "Juventus", "Real Madrid"]

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


"""if torch.cuda.is_available():
    model_conv.cuda()"""


# Load pretrained weights
model_conv.load_state_dict(torch.load("./state_dict.h5"))
# model_conv = model_conv.to(device)


'''imsize = 16
loader = transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()])

def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    #image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image  #assumes that you're using GPU

test_input = image_loader('./test.jpeg')
outputs = model_conv(test_input)
result = outputs
print(result)'''

batch_size = 32
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
scale = 299
input_shape = 299

loader = transforms.Compose(
    [transforms.Resize(scale), transforms.ToTensor(), transforms.Normalize(mean, std)]
)

test_image = Image.open("./test.jpeg")
test_image = loader(test_image).float()
test_input[0] = test_image

outputs = model_conv(test_input)
if type(outputs) == tuple:
    outputs, _ = outputs
_, preds = torch.Tensor.max(outputs.data, 1)

result = teams[preds[0]]

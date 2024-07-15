import torch
from torch import nn, optim
import numpy as np
from torchvision import models, utils, transforms
from datasets import load_dataset
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = 512 if torch.cuda.is_available() else 128

image_loader = transforms.Compose(
    [
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
model = models.vgg19(pretrained=True).features

# helper functions


def get_content_loss(target, content):
    return torch.mean((target - content) ** 2)


def gram_matrix(input, ch, h, w):
    input = input.view(ch, h * w)
    gram_matrix = torch.matmul(input, input.t())

    return gram_matrix


def get_style_loss(target, style):
    _, c, h, w = target.size()
    target_gm = gram_matrix(target, c, h, w)
    style_gm = gram_matrix(style, c, h, w)
    style_loss = torch.mean((target_gm - style_gm) ** 2) / (c * h * w)

    return style_loss


class VGG_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg_model = models.vgg19(pretrained=True).features
        self.selected_layers = ["0", "5", "10", "19", "28"]  # selected layers

    def forward(self, output):
        features = []
        for name, layer in self.vgg_model._modules.items():
            output = layer(output)  # extract features
            if name in self.selected_layers:
                features.append(output)

        return features


vgg_convnet = VGG_model().to(device).eval()

optimizer = optim.Adam([])

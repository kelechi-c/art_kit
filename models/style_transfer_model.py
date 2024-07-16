import torch
from torch import nn, optim
import numpy as np
from torchvision import models, utils, transforms
from datasets import load_dataset
import matplotlib.pyplot as plt
from PIL import Image as pillow
from tqdm.auto import tqdm


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


def load_image(file):
    img = pillow.open(file)
    img = image_loader(img).unsqueeze(0)
    img = img.to(device)

    return img


def save_step_result(target, step):
    denormalization = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
    img = target.clone().squeeze()
    img = denormalization(img).clamp(0, 1)
    path = f"image@{step}.png"
    utils.save_image(img, path)
    print(f"saved {path}")


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

epochs = 1000

alpha = 1
beta = 10000

style_img = load_image("images/woods.png")
source_img = load_image("images/katara.png")
target_img = source_img.clone().requires_grad_(True)

optimizer = optim.Adam([target_img], lr=0.001)

for epoch in tqdm(range(epochs)):
    target_f = vgg_convnet(target_img)
    style_f = vgg_convnet(style_img)
    source_f = vgg_convnet(source_img)

    style_loss = 0
    content_loss = 0

    for target, source, style in zip(target_f, source_f, style_f):
        content_loss += get_content_loss(target, source)
        style_loss += get_style_loss(target, style)

    loss = (alpha * content_loss) + (beta * style_loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(
            f"epoch @ {epoch}, style_loss => {style_loss.item()}, content_loss => {content_loss.item()}"
        )
        save_step_result(target_img, epoch)

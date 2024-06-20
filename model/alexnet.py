import torch
import torch.nn as nn
from torchvision.models import alexnet


class Alexnet(nn.Module):
    def __init__(self, pretrained=True, freeze=True):
        super(Alexnet, self).__init__()
        self.model = alexnet(pretrained=pretrained)

        if freeze:
            for param in self.model.features.parameters():
                param.requires_grad = False

            for param in self.model.classifier[:-1].parameters():
                param.requires_grad = False

        self.model.classifier[-1] = nn.Linear(in_features=4096, out_features=8, bias=True)

    def forward(self, x):
        output = self.model(x)
        return output



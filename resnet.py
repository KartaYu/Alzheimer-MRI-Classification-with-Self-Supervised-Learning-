from pyexpat import model
import torch
import torch.nn as nn
from torchvision import models


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

# For Training SSL Model


class ResNetModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

        self.pretrained = models.resnet50(pretrained=True)

        self.pretrained.avgpool = Identity()
        self.pretrained.fc = Identity()

        for p in self.pretrained.parameters():
            p.requires_grad = True

        self.flatten = nn.Flatten()

        self.embedding = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(32768, 256),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
        )

    def forward(self, x):
        out = self.pretrained(x)

        # xp = self.embedding(torch.squeeze(out))

        xp = self.embedding(out)
        return xp


# Pretrain Model from SSL
class PreResNetModel(nn.Module):
    def __init__(self, base_model, pretrain):
        super().__init__()
        self.base_model = base_model
        model = ResNetModel('Pretrain ResNet50 with SSL')
        model.load_state_dict(torch.load(pretrain))  # 'best_acc_cnn.pth'
        self.pretrained = model
        self.pretrained.embedding = Identity()

        for p in self.pretrained.parameters():
            p.requires_grad = True

        # self.projector = ProjectionHead(2048, 512, 128)
        self.classify = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(32768, 4),

        )

    def forward(self, x):
        out = self.pretrained(x)

        # xp = self.embedding(torch.squeeze(out))

        xp = self.classify(out)
        return xp


class SL_ResModel(nn.Module):
    def __init__(self, base_model, pretrain):
        super().__init__()
        self.base_model = base_model
        self.pretrained = models.resnet50(pretrained=pretrain)

        for p in self.pretrained.parameters():
            p.requires_grad = True

        self.pretrained.avgpool = Identity()
        self.pretrained.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(32768, 4),
        )

        # self.projector = ProjectionHead(2048, 512, 128)

    def forward(self, x):
        
        out = self.pretrained(x)

        # xp = self.embedding(torch.squeeze(out))

        return out

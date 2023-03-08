import torch
import torch.nn as nn


class pre_cnn(nn.Module):
    def __init__(self, premodel):
        super().__init__()

        self.basemodel = premodel

        # TAKING OUTPUT FROM AN INTERMEDITATE LAYER
        # PREPRAING THE TRUNCATED MODEL
        self.x1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.5),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 128 x 8 x 8
            nn.Dropout2d(0.5),


            # nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),)  # 256 x 1 x 1

        self.x2 = nn.Sequential(
            nn.Linear(32768, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
        )

        # for p in self.premodel.projector.parameters():
        #     p.requires_grad = False

        # self.premodel.projector = Identity()

    def forward(self, x):
        out = self.x1(x)
        out = self.x2(out)
        return out


class cnn(nn.Module):
    def __init__(self, premodel):
        super().__init__()
        model = pre_cnn('pre')
        model.load_state_dict(torch.load('best_acc_cnn.pth'))
        self.pretained = model
        self.pretained.x2 = nn.Identity()
        for p in self.pretained.x1.parameters():
            p.requires_grad = True

        self.cls = nn.Sequential(
            nn.Linear(32768, 4),
            nn.ReLU(),
            nn.Linear(256, 4),
        )

    def forward(self, x):
        out = self.pretained(x)
        out = self.cls(out)
        return out

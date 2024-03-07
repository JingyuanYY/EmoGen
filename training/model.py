from torchvision import models
import torch.nn as nn



class BackBone(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.cnn = models.resnet50(pretrained=True)

        self.backbone = nn.Sequential(*list(self.cnn.children())[:-2])
        self.flaten = nn.Sequential(nn.AvgPool2d(kernel_size=7), nn.Flatten())
        self.fc_1 = nn.Linear(2048, 768)
        self.fc_2 = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(768, 8)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.flaten(x)
        x = self.fc_1(x)
        x = self.fc_2(x)
        return x


class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.fc = nn.Linear(768, 768)

    def forward(self, x):
        out = self.fc(x)
        return out


class MLP(nn.Module):
    def __init__(self, num_fc_layers=2, need_ReLU=False, need_LN=False, need_Dropout=False):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(768, 1024))
        if need_LN is True:
            layers.append(nn.LayerNorm(1024))
        if need_ReLU is True:
            layers.append(nn.ReLU())
        if need_Dropout is True:
            layers.append(nn.Dropout(0.5))
        for _ in range(num_fc_layers - 2):
            layers.append(nn.Linear(1024, 1024))
            if need_LN is True:
                layers.append(nn.LayerNorm(1024))
            if need_ReLU is True:
                layers.append(nn.ReLU())
            if need_Dropout is True:
                layers.append(nn.Dropout(0.5))
        layers.append(nn.Linear(1024, 768))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        out = self.mlp(x)
        return out


class SimpleMLP(nn.Module):
    def __init__(self, need_ReLU=False, need_Dropout=False):
        super(SimpleMLP, self).__init__()
        layers = []
        layers.append(nn.Linear(768, 1024))
        if need_ReLU is True:
            layers.append(nn.ReLU())
        if need_Dropout is True:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.Linear(1024, 768))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        out = self.mlp(x)
        return out


class emo_classifier(nn.Module):
    def __init__(self, ):
        super(emo_classifier, self).__init__()
        self.fc = nn.Linear(768, 8)

    def forward(self, x):
        x = self.fc(x)
        return x




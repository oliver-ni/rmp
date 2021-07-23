import pytorch_lightning as pl
import torch
import torchmetrics
from torch import nn
from torch.nn import functional as F
from torchvision import models

from bmm import BetaMixtureModel


class _ImageModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = models.resnet18(pretrained=True)
        for parameter in self.resnet.parameters():
            parameter.requires_grad = False

    def forward(self, imgs):
        return self.resnet(imgs)

    @property
    def out_features(self):
        return self.resnet.fc.out_features


class _AttributesModel(nn.Module):
    def __init__(self, out_features=500):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(312, 250),
            nn.BatchNorm1d(250),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(250, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

    def forward(self, attrs):
        return self.model(attrs)


class _BaseModel(nn.Module):
    def __init__(self, attr_out_features=500):
        super().__init__()

        self.image_model = _ImageModel()
        self.attr_model = _AttributesModel(attr_out_features)
        self.rest = nn.Sequential(
            nn.Linear(self.image_model.out_features + attr_out_features, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(500, 200),
        )

    def forward(self, imgs, attrs):
        image_out = self.image_model(imgs)
        attr_out = self.attr_model(attrs)
        combined = torch.cat([image_out, attr_out], dim=1)
        return self.rest(combined)


class BaselineModel(pl.LightningModule):
    def __init__(self, learning_rate=0.001):
        super().__init__()

        self.lr = learning_rate

        self.base_model = _BaseModel()
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

    def forward(self, imgs, attrs):
        return self.base_model(imgs, attrs)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        outputs = self(batch["img"], batch["attributes"])
        losses = F.cross_entropy(outputs, batch["noisy_label"], reduction="none")
        loss = losses.mean()

        self.train_acc(outputs, batch["label"])
        self.log("train_acc", self.train_acc, on_epoch=True)

        return {"losses": losses, "loss": loss, "outputs": outputs}

    def validation_step(self, batch, batch_idx):
        outputs = self(batch["img"], batch["attributes"])
        val_loss = F.cross_entropy(outputs, batch["label"])

        self.val_acc(outputs, batch["label"])
        self.log("val_acc", self.val_acc)

        return val_loss

    def test_step(self, batch, batch_idx):
        outputs = self(batch["img"], batch["attributes"])
        val_loss = F.cross_entropy(outputs, batch["label"])
        return val_loss


class BetaModel(BaselineModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bmm = BetaMixtureModel(2)
        self.train_acc_noisy = torchmetrics.Accuracy()

    def training_step(self, batch, batch_idx):
        res = super().training_step(batch, batch_idx)
        cor_loss = self.lm_loss(res["losses"], res["outputs"], batch["noisy_label"])

        self.train_acc_noisy(res["outputs"], batch["noisy_labels"])
        self.log("train_acc_noisy", self.train_acc_noisy)

        return cor_loss

    def cross_entropy_onehot(self, inputs, target):
        sum_term = target * F.log_softmax(inputs, dim=1)
        return -sum_term.sum(dim=1).mean()

    def lm_loss(self, losses, outputs, labels):
        losses = losses.clone().detach()
        losses -= losses.min()
        losses /= losses.max()

        preds = F.softmax(outputs, dim=1).argmax(dim=1)

        labels_onehot = torch.eye(200, device=self.device)[labels]
        outputs_onehot = torch.eye(200, device=self.device)[preds]
        noisy_cls = (self.bmm.alphas + self.bmm.betas).argmax()
        w = self.bmm.posterior(losses)[noisy_cls].unsqueeze(1).detach()
        weighted_labels = (1 - w) * labels_onehot + w * outputs_onehot

        return self.cross_entropy_onehot(outputs, weighted_labels.detach())

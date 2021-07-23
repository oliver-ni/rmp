import pytorch_lightning as pl
import torch
import torchmetrics
from torch import nn
from torch.nn import functional as F
from torchvision import models

from bmm import BetaMixtureModel, EPS


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
        return image_out, attr_out, self.rest(combined)


class BaselineModel(pl.LightningModule):
    def __init__(self, learning_rate=0.001, **kwargs):
        super().__init__()

        self.lr = learning_rate

        self.base_model = _BaseModel()
        self.acc__val = torchmetrics.Accuracy()
        self.acc__train = torchmetrics.Accuracy()
        self.acc__train_noisy = torchmetrics.Accuracy()
        self.acc__train_clean = torchmetrics.Accuracy()
        self.est_acc__val = torchmetrics.Accuracy()
        self.est_acc__train = torchmetrics.Accuracy()
        self.est_acc__train_noisy = torchmetrics.Accuracy()
        self.est_acc__train_clean = torchmetrics.Accuracy()

    def forward(self, imgs, attrs):
        _, _, out = self.base_model(imgs, attrs)
        return out

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        noisy = batch["noisy"]
        outputs = self(batch["img"], batch["attributes"])
        losses = F.cross_entropy(outputs, batch["noisy_label"], reduction="none")
        loss = losses.mean()

        self.est_acc__train(outputs, batch["noisy_label"])
        self.est_acc__train_noisy(outputs[noisy], batch["noisy_label"][noisy])
        self.est_acc__train_clean(outputs[~noisy], batch["noisy_label"][~noisy])
        self.log("est_acc/train", self.est_acc__train, on_step=False, on_epoch=True)
        self.log("est_acc/train_noisy", self.est_acc__train_noisy, on_step=False, on_epoch=True)
        self.log("est_acc/train_clean", self.est_acc__train_clean, on_step=False, on_epoch=True)

        self.acc__train(outputs, batch["label"])
        self.acc__train_noisy(outputs[noisy], batch["label"][noisy])
        self.acc__train_clean(outputs[~noisy], batch["label"][~noisy])
        self.log("acc/train", self.acc__train, on_step=False, on_epoch=True)
        self.log("acc/train_noisy", self.acc__train_noisy, on_step=False, on_epoch=True)
        self.log("acc/train_clean", self.acc__train_clean, on_step=False, on_epoch=True)

        self.log("loss/train", loss, on_step=False, on_epoch=True)
        self.log("loss/train_noisy", losses[noisy].mean(), on_step=False, on_epoch=True)
        self.log("loss/train_clean", losses[~noisy].mean(), on_step=False, on_epoch=True)

        return {"losses": losses, "loss": loss, "outputs": outputs}

    def validation_step(self, batch, batch_idx):
        outputs = self(batch["img"], batch["attributes"])
        val_loss = F.cross_entropy(outputs, batch["label"])

        self.acc__val(outputs, batch["label"])
        self.est_acc__val(outputs, batch["noisy_label"])
        self.log("acc/val", self.acc__val, on_step=False, on_epoch=True)
        self.log("est_acc/val", self.est_acc__val, on_step=False, on_epoch=True)

        self.log("loss/val", val_loss, on_step=False, on_epoch=True)

        return val_loss

    def test_step(self, batch, batch_idx):
        outputs = self(batch["img"], batch["attributes"])
        val_loss = F.cross_entropy(outputs, batch["label"])
        return val_loss


class BetaModel(BaselineModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bmm = BetaMixtureModel(2)
        self.bmm_min_loss = 0
        self.bmm_max_loss = 1

    def training_step(self, batch, batch_idx):
        res = super().training_step(batch, batch_idx)
        lm_losses = self.lm_loss(res["losses"], res["outputs"], batch["noisy_label"])
        lm_loss = lm_losses.mean()

        noisy = batch["noisy"]
        self.log("lm_loss/train", lm_loss, on_step=False, on_epoch=True)
        self.log("lm_loss/train_noisy", lm_losses[noisy].mean(), on_step=False, on_epoch=True)
        self.log("lm_loss/train_clean", lm_losses[~noisy].mean(), on_step=False, on_epoch=True)

        return res if self.current_epoch < 5 else {**res, "loss": lm_loss}

    def training_epoch_end(self, outputs):
        losses = torch.cat([x["losses"] for x in outputs])
        self.bmm_min_loss = losses.min()
        self.bmm_max_loss = losses.max()
        losses -= self.bmm_min_loss
        losses /= self.bmm_max_loss

        self.bmm.fit(losses)

    def cross_entropy_onehot(self, inputs, target):
        sum_term = target * F.log_softmax(inputs, dim=1)
        return -sum_term.sum(dim=1)

    def lm_loss(self, losses, outputs, labels):
        losses = losses.clone().detach()
        losses -= self.bmm_min_loss
        losses /= self.bmm_max_loss
        losses[losses > 1] = 1 - EPS
        losses[losses < 0] = EPS

        preds = F.softmax(outputs, dim=1).argmax(dim=1)

        labels_onehot = torch.eye(200, device=self.device)[labels]
        outputs_onehot = torch.eye(200, device=self.device)[preds]
        noisy_cls = (self.bmm.alphas + self.bmm.betas).argmax()
        w = self.bmm.posterior(losses)[noisy_cls].unsqueeze(1).detach()
        weighted_labels = (1 - w) * labels_onehot + w * outputs_onehot

        return self.cross_entropy_onehot(outputs, weighted_labels.detach())

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F
import torchmetrics.functional as FM


class MLPBlock(nn.Module):

    def __init__(self, inp_dim, hidden_dim=None):

        """
        inp_dim: size of feature vector
        hidden_dim: dimension of intermediate Linear layer.
        """

        super(MLPBlock, self).__init__()

        hidden_dim = inp_dim if hidden_dim is None else hidden_dim

        self.block = nn.Sequential(
            nn.Linear(inp_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, inp_dim)
        )

    def forward(self, x):
        return self.block(x)


class MixerBlock(nn.Module):

    def __init__(self, nb_patches, token_dim,
                 ch_mlp_dim=None, token_mlp_dim=None):

        """
        nb_patches: number of patches
        token_dim: size of each token vector
        ch_mlp_dim: dimension of intermediate mlp layer for channel mixing
        token_mlp_dim: same as above, but for token mixing
        """

        super(MixerBlock, self).__init__()

        # layer norms
        self.lnorm_1 = nn.LayerNorm([nb_patches, token_dim])
        self.lnorm_2 = nn.LayerNorm([nb_patches, token_dim])

        self.ch_mixer = MLPBlock(nb_patches, ch_mlp_dim)

        self.token_mixer = MLPBlock(token_dim, token_mlp_dim)

    def forward(self, x):
        y = self.lnorm_1(x)
        y = self.ch_mixer(y.permute(0, 2, 1))
        x = y.permute(0, 2, 1) + x
        y = self.lnorm_2(x)
        y = self.token_mixer(y)
        x = y + x

        return x


class MLPMixer(pl.LightningModule):

    def __init__(self, nb_channels, inp_size, patch_size, nb_classes,
                 nb_layers, hidden_dim, token_mlp_dim, ch_mlp_dim, lr=0.001):

        """
        nb_channels: number of channels in input image
        inp_size: H or W of the square input image
        patch_size: desired patch size (H or W of square patch)
        nb_classes: number of classes for prediction
        nb_layers: number of mixer layers to be used
        hidden_dim: C mentioned in paper
        token_mlp_dim: size of intermediate layer of mlp for token mixing
        ch_mlp_dim: same as above but for channel mixing
        lr: learning rate
        """

        super(MLPMixer, self).__init__()

        self.lr = lr

        # assumes square input image as of now
        if inp_size % patch_size != 0:
            raise RuntimeError('patch size must divide dimensions of image')

        self.hidden_dim = hidden_dim

        self.nb_patches = (inp_size // patch_size)**2

        self.projector = nn.Conv2d(nb_channels, hidden_dim,
                                   patch_size, patch_size)

        self.mixer_blocks = nn.Sequential(
            *[
             MixerBlock(self.nb_patches, hidden_dim, ch_mlp_dim, token_mlp_dim)
             for _ in range(nb_layers)
             ]
        )

        self.layer_norm = nn.LayerNorm([self.nb_patches, hidden_dim])

        self.classifier = nn.Linear(hidden_dim, nb_classes)

        nn.init.zeros_(self.classifier.weight)

    def forward(self, x):
        x = self.projector(x)
        x = x.view(x.shape[0], self.nb_patches, self.hidden_dim)
        x = self.mixer_blocks(x)
        x = self.layer_norm(x)
        x = torch.mean(x, axis=1)
        x = self.classifier(x)
        return x

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.lr)
        return opt

    def _step(self, batch):
        x, y = batch
        pred = self(x)
        loss = F.cross_entropy(pred, y)
        return pred, loss

    def training_step(self, batch, batch_idx):
        _, loss = self._step(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        pred, loss = self._step(batch)
        pred_prob = F.softmax(pred, dim=-1)   # prediction probabilities
        acc = FM.accuracy(pred_prob, batch[1])

        metrics = {'val_acc': acc, 'val_loss': loss}
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        metrics = self.validation_step(batch, batch_idx)
        metrics = {'test_acc': metrics['val_acc'],
                   'test_loss': metrics['val_loss']}
        self.log_dict(metrics, prog_bar=True)

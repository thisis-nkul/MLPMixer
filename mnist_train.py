"""
Sample script to train the model on MNIST Dataset.
"""
from models import MLPMixer
from models import MNISTData
import pytorch_lightning as pl

datamodule = MNISTData(data_dir='/mnt/Media/datasets', num_workers=4)

model = MLPMixer(
    nb_channels=1,
    inp_size=28,
    patch_size=7,
    nb_classes=10,
    nb_layers=4,
    hidden_dim=768,
    token_mlp_dim=256,
    ch_mlp_dim=1024,
    lr=0.001)

# print(model)

# set gpus=0 for cpu training
trainer = pl.Trainer(gpus=1, max_epochs=2)

trainer.fit(model, datamodule)

trainer.test(model, datamodule=datamodule)

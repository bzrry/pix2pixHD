
import torch
from torchvision.datasets import ImageFolder
import flash
from flash.core.data.utils import download_data
from flash.image import ImageClassificationData, ImageEmbedder
from flash.image.embedding.backbones import IMAGE_EMBEDDER_BACKBONES
from flash.image.embedding.vissl.adapter import VISSLAdapter
from vissl.config.attr_dict import AttrDict
from vissl.models.trunks import MODEL_TRUNKS_REGISTRY

from models.networks import GlobalGeneratorSSLEmbedder


model_name = "pix2pixHDEmbedder"

def pix2pixHDEmbedder(**kwargs):
    cfg = VISSLAdapter.get_model_config_template()
    cfg.TRUNK = AttrDict({"NAME": model_name})
    trunk = MODEL_TRUNKS_REGISTRY[model_name](cfg, model_name=model_name)
    trunk.model_config = cfg
    num_features = trunk.fc.in_features
    return trunk, num_features

# register embedder
IMAGE_EMBEDDER_BACKBONES(
    fn=pix2pixHDEmbedder,
    name=model_name,
)

datamodule = ImageClassificationData.from_datasets(
    train_dataset=ImageFolder("../crops"),
    batch_size=512,
    num_workers=8,
)
embedder = ImageEmbedder(
    backbone=model_name,
    training_strategy="barlow_twins",
    head="simclr_head",
    pretraining_transform="barlow_twins_transform",
    training_strategy_kwargs={"latent_embedding_dim": 128},
    pretraining_transform_kwargs={"size_crops": [196]},
)
trainer = flash.Trainer(
    max_epochs=50,
    gpus=torch.cuda.device_count(),
    strategy="ddp",
    log_every_n_steps=1,
)
trainer.fit(embedder, datamodule=datamodule)
print("Saving downsampler state to disk...")
torch.save(
    embedder.adapter.backbone.downsampler.state_dict(),
    "./downsampler_state.pt",
)

#print(embedder.predict(["153783656_85f9c3ac70.jpg", "2039585088_c6f47c592e.jpg"]))





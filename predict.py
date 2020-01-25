from skimage import io
from skimage.transform import rescale, rotate

import torchvision.models as models
import torch
import os
from tensorboardX import SummaryWriter

from src.iodine import IODINE
from src.networks.refine_net import RefineNetLSTM
from src.networks.sbd import SBD
from src.datasets.datasets import ClevrDataset

from PIL import ImageFile
import numpy as np


def rescale_img(img):
        H,W,C = img.shape
        dH = abs(H-300)//2
        dW = abs(W-300)//2
        crop = img[dH:-dH,dW:-dW,:3]
        H,W,C = crop.shape
        down = rescale(crop, 64/H, order=3, mode='reflect', multichannel=True)
        return down


## Model Hyperparameters
T = 5 ## Number of steps of iterative inference
K = 2 ## Number of slots
z_dim = 64 ## Dimensionality of latent codes
channels_in = 16+16 ## Number of inputs to refinement network (16, + 16 additional if using feature extractor)
out_channels = 4 ## Number of output channels for spatial broadcast decoder (RGB + mask logits channel)
img_dim = (64,64) ## Input image dimension
beta = 5. ## Weighting on nll term in VAE loss
use_feature_extractor = True

## Create refinement network, decoder, and (optionally) feature extractor
## 		Could speed up training by pre-computing squeezenet outputs since we just use this as a feature extractor
## 		Could also do this as a pre-processing step in dataset class
feature_extractor = models.squeezenet1_1(pretrained=True).features[:5] if use_feature_extractor else None
refine_net = RefineNetLSTM(z_dim,channels_in)
decoder = SBD(z_dim,img_dim,out_channels=out_channels)

## Create IODINE model
model = IODINE(refine_net,decoder,T,K,z_dim,name='christine',
	feature_extractor=feature_extractor,beta=beta)

model.load("/home/bquach/IODINE/trained_models/iodine_gauss/iodine_gauss_epoch_395.th", None)

model.eval()

## Create training data
train_data = torch.utils.data.DataLoader(
	ClevrDataset('/home/bquach/IODINE/gauss_data',max_num_samples=10,crop_sz=300,down_sz=64),
	batch_size=1, shuffle=True, num_workers=4, drop_last=True)
data = np.expand_dims(np.moveaxis(np.moveaxis(rescale_img(io.imread("/home/rzheng/IODINE/toy_data/train/test_1.png")), 0, -1), -1, 0), axis=0)
for i,mbatch in enumerate(train_data):
    x = mbatch.to('cpu')

    output = model(x)
    # do something with the output
    io.imsave('x_{}.png'.format(i), x[0].permute(1,2,0))
    for j in range(2):
        io.imsave('output_{}'.format(i) + str(j) + '.png', (output[3].detach()[0, j, :, :].permute(1,2,0)) * output[4].detach()[0, j, :, :].permute(1,2,0))
    io.imsave('reconstructed_x_{}.png'.format(i), (output[3].detach()[:, :, :, :]*output[4].detach()[:, :, :, :]).sum(dim=1)[0].permute(1, 2, 0))


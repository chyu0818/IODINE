import torch
from skimage import io
from skimage.transform import rescale, rotate
from iodine import IODINE
import torchvision.models as models

def rescale_img(img):
        H,W,C = img.shape
        dH = abs(H-120)//2
        dW = abs(W-120)//2
        crop = img[dH:-dH,dW:-dW,:3]
        H,W,C = crop.shape
        down = rescale(crop, 64/H, order=3, mode='reflect', multichannel=True)
        return down


## Model Hyperparameters
T = 5 ## Number of steps of iterative inference
K = 11 ## Number of slots
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
model = IODINE(refine_net,decoder,T,K,z_dim,name=model_name,
	feature_extractor=feature_extractor,beta=beta)

model.load("/home/chyu/work/IODINE/trained_models/iodine_clevr_wfeatures/iodine_clevr_wfeatures_epoch_99.5.th", None)

model.eval()

data = rescale_img(io.imread("/home/rzheng/IODINE/toy_data/train/test_1.png"))
output = model(data)
# do something with the output
print(output.shape)



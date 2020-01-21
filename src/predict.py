import torch
from skimage import io
from skimage.transform import rescale, rotate

def rescale_img(img):
        H,W,C = img.shape
        dH = abs(H-120)//2
        dW = abs(W-120)//2
        crop = img[dH:-dH,dW:-dW,:3]
        H,W,C = crop.shape
        down = rescale(crop, 64/H, order=3, mode='reflect', multichannel=True)
        return down

# /home/chyu/work/IODINE/trained_models/iodine_clevr_wfeatures/
model = torch.load("/home/chyu/work/IODINE/trained_models/iodine_clevr_wfeatures/iodine_clevr_wfeatures_epoch_99.5.th")

model.eval()
data = rescale_img(io.imread("/home/rzheng/IODINE/toy_data/train/test_1.png"))
output = model(data)
# do something with the output
print(output.shape)



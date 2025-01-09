import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from torchvision.transforms.functional import to_pil_image
import numpy as np
import torch

def show_img_mask(img, mask):
    if torch.is_tensor(img):
        img = to_pil_image(img)
        mask = to_pil_image(mask)
    img_mask = mark_boundaries(np.array(img), np.array(mask), outline_color=(0, 1, 0), color=(0, 1, 0))
    plt.imshow(img_mask)
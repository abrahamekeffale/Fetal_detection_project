import os
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

class FetalDataset(Dataset):
    def __init__(self, path2data, transform=None):
        imgsList = [pp for pp in os.listdir(path2data) if "Annotation" not in pp and (pp.endswith(".jpg") or pp.endswith(".png"))]
        self.path2imgs = [os.path.join(path2data, fn) for fn in imgsList]
        self.path2annts = [p2i.replace(".png", "_Annotation.png") for p2i in self.path2imgs]
        self.transform = transform

    def __len__(self):
        return len(self.path2imgs)

    def __getitem__(self, idx):
        path2img = self.path2imgs[idx]
        image = Image.open(path2img)
        path2annt = self.path2annts[idx]
        annt_edges = Image.open(path2annt)
        mask = ndi.binary_fill_holes(annt_edges)
        image = np.array(image)
        mask = mask.astype("uint8")
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        image = to_tensor(image)
        mask = 255 * to_tensor(mask)
        return image, mask
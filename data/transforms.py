from albumentations import Compose, Resize, HorizontalFlip, VerticalFlip

h, w = 128, 192

transform_train = Compose([Resize(h, w), HorizontalFlip(p=0.5), VerticalFlip(p=0.5)])
transform_val = Resize(h, w)
import torch
import torch.nn.functional as F
import numpy as np
from skimage import measure
import cv2

def dice_loss(pred, target, smooth=1e-5):
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = 2.0 * (intersection + smooth) / (union + smooth)
    loss = 1.0 - dice
    return loss.sum(), dice.sum()

def loss_func(pred, target):
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='sum')
    pred = torch.sigmoid(pred)
    dlv, _ = dice_loss(pred, target)
    loss = bce + dlv
    return loss

def metrics_batch(pred, target):
    pred = torch.sigmoid(pred)
    _, metric = dice_loss(pred, target)
    return metric

def calculate_tumor_size(mask_pred, pixel_size):
    image_gray = mask_pred.numpy().astype('uint8') * 255  # Convert binary mask to grayscale (0-255)
    thresh = skimage.filters.threshold_otsu(image_gray)
    tumor_mask = image_gray > thresh  # Binary mask based on threshold
    label_image = skimage.measure.label(tumor_mask)
    region_props = skimage.measure.regionprops(label_image)
    if len(region_props) == 0:
        tumor_size_mm = 0.0
    else:
        tumor_area = region_props[0].area  # Area in pixels
        tumor_size_mm = tumor_area * (pixel_size ** 2)  # Convert area to mm²
    return tumor_size_mm

def calculate_head_metrics(mask_pred, pixel_size):
    import cv2
    from skimage import measure

    # Calculate head circumference
    contours, _ = cv2.findContours(mask_pred.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        perimeter_pixels = cv2.arcLength(largest_contour, True)  # Perimeter in pixels
        head_circumference_cm = perimeter_pixels * pixel_size  # Convert to centimeters
    else:
        head_circumference_cm = 0.0

    # Calculate vertical BPD
    mask_props = measure.regionprops(mask_pred.astype(np.uint8))
    if mask_props:
        bbox = mask_props[0].bbox  # Bounding box (min_row, min_col, max_row, max_col)
        bpd_pixels = bbox[2] - bbox[0]  # Vertical axis length in pixels
        bpd_cm = bpd_pixels * pixel_size  # Convert to centimeters
    else:
        bpd_cm = 0.0

    return head_circumference_cm, bpd_cm
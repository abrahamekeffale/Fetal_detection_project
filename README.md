# Fetal_detection_project
Sure! Let's continue with the remaining parts of your project.

### 7. `inference.py`

```python
import os
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
import numpy as np
import skimage.filters
import skimage.measure
import cv2
from models.segnet import SegNet
from utils.visualization import show_img_mask
from utils.metrics import calculate_tumor_size, calculate_head_metrics

def load_model(path2weights, device):
    model = SegNet(params_model)
    model.load_state_dict(torch.load(path2weights))
    model.to(device)
    model.eval()
    return model

def predict_and_visualize(model, path2test, rndImgs, device, h, w):
    for fn in rndImgs:
        path2img = os.path.join(path2test, fn)
        img = Image.open(path2img).convert("L")  # Convert to grayscale
        img = img.resize((w, h))
        img_t = to_tensor(img).unsqueeze(0).to(device)

        # Pass the image to the model
        pred = model(img_t).cpu()

        # Process the prediction
        pred = torch.sigmoid(pred)[0]
        mask_pred = (pred[0] >= 0.5)

        # Visualize the input image and prediction
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(img, cmap="gray")  # Input grayscale image

        plt.subplot(1, 3, 2)
        plt.imshow(mask_pred, cmap="gray")  # Predicted mask

        plt.subplot(1, 3, 3)
        show_img_mask(img, mask_pred)  # Custom function to overlay image and mask
        plt.show()

def visualize_tumor_size(path2test, rndImgs, model, device, h, w, pixel_size):
    for fn in rndImgs:
        path2img = os.path.join(path2test, fn)
        img = Image.open(path2img).convert("L")  # Convert to grayscale
        img = img.resize((w, h))
        img_t = to_tensor(img).unsqueeze(0).to(device)

        # Model prediction
        pred = model(img_t).cpu()
        pred = torch.sigmoid(pred)[0]  # Apply sigmoid activation
        mask_pred = (pred[0] >= 0.5)  # Threshold prediction to binary mask

        # Calculate tumor size
        tumor_size_mm = calculate_tumor_size(mask_pred, pixel_size)

        # Display results
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(img, cmap="gray")
        plt.axis('off')
        plt.title("Original Image", fontsize=14)

        plt.subplot(1, 3, 2)
        plt.imshow(mask_pred, cmap="gray")
        plt.axis('off')
        plt.title(f"Tumor size: {tumor_size_mm:.2f} mm²", fontsize=14)

        plt.subplot(1, 3, 3)
        show_img_mask(img, mask_pred)  # Custom function to overlay the mask on the image
        plt.axis('off')
        plt.title("Overlay Image & Mask", fontsize=14)
        plt.show()

def visualize_head_metrics(path2test, rndImgs, model, device, h, w, pixel_size):
    for fn in rndImgs:
        path2img = os.path.join(path2test, fn)
        img = Image.open(path2img).convert("L")  # Convert to grayscale
        img = img.resize((w, h))
        img_t = to_tensor(img).unsqueeze(0).to(device)

        # Model prediction
        pred = model(img_t).cpu()
        pred = torch.sigmoid(pred)[0]  # Apply sigmoid activation
        mask_pred = (pred[0] >= 0.5).numpy()  # Threshold and convert to NumPy

        # Calculate head metrics
        head_circumference_cm, bpd_cm = calculate_head_metrics(mask_pred, pixel_size)

        # Create overlay
        grayscale_img = np.array(img)  # Original grayscale image
        mask_overlay = np.zeros((grayscale_img.shape[0], grayscale_img.shape[1], 3), dtype=np.uint8)
        mask_overlay[:, :, 0] = mask_pred * 255  # Red channel for mask
        alpha = 0.3  # Transparency level
        overlay_result = np.clip(grayscale_img[:, :, None] * (1 - alpha) + mask_overlay * alpha, 0, 255).astype(np.uint8)

        # Plot results
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.title("Grayscale with Thin Red Mask Overlay", fontsize=14)
        plt.imshow(overlay_result)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title(f"HC: {head_circumference_cm:.2f} cm, BPD: {bpd_cm:.2f} cm", fontsize=14)
        plt.imshow(mask_pred, cmap="gray")
        plt.axis('off')

        # Draw BPD line and HC perimeter on mask overlay
        for contour in contours:
            cv2.drawContours(overlay_result, [contour], -1, (0, 255, 0), 1)  # Green for circumference (HC)
        if mask_props:
            row_center = (bbox[0] + bbox[2]) // 2  # Vertical midpoint
            col_start, col_end = bbox[1], bbox[3]  # Horizontal edges
            cv2.line(overlay_result, (col_start, row_center), (col_end, row_center), (255, 0, 0), 1)  # Blue for BPD

        # Display annotated image
        plt.imshow(overlay_result)
        plt.axis('off')
        plt.show()
```

### 8. `app.py`

```python
import streamlit as st
import os
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from models.segnet import SegNet
from utils.visualization import show_img_mask
from utils.metrics import calculate_tumor_size, calculate_head_metrics

def load_model(path2weights, device):
    model = SegNet(params_model)
    model.load_state_dict(torch.load(path2weights))
    model.to(device)
    model.eval()
    return model

def main():
    st.title("Medical Image Segmentation and Analysis")
    path2test = st.text_input("Enter the path to the test images folder:")
    path2weights = st.text_input("Enter the path to the model weights file:")
    if st.button("Run Analysis"):
        if path2test and path2weights:
            imgsList = [pp for pp in os.listdir(path2test) if "Annotation" not in pp and (pp.endswith(".jpg") or pp.endswith(".png"))]
            np.random.seed(2019)
            rndImgs = np.random.choice(imgsList, 4)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = load_model(path2weights, device)
            for fn in rndImgs:
                path2img = os.path.join(path2test, fn)
                img = Image.open(path2img).convert("L")  # Convert to grayscale
                img = img.resize((w, h))
                img_t = to_tensor(img).unsqueeze(0).to(device)

                # Model prediction
                pred = model(img_t).cpu()
                pred = torch.sigmoid(pred)[0]  # Apply sigmoid activation
                mask_pred = (pred[0] >= 0.5)  # Threshold prediction to binary mask

                # Calculate tumor size
                tumor_size_mm = calculate_tumor_size(mask_pred, pixel_size=0.1)

                # Display results
                st.image(img, caption="Original Image", use_column_width=True)
                st.image(mask_pred, caption=f"Tumor size: {tumor_size_mm:.2f} mm²", use_column_width=True)
                st.image(show_img_mask(img, mask_pred), caption="Overlay Image & Mask", use_column_width=True)

if __name__ == "__main__":
    main()
```

### 9. `requirements.txt`

```
torch
torchvision
albumentations
numpy
scipy
scikit-image
matplotlib
opencv-python
streamlit
```

### 10. `README.md`

```markdown
# Medical Image Segmentation and Analysis

This project performs medical image segmentation and analysis using a deep learning model. The project is structured to be modular and easy to use with VS Code.

## Project Structure

```
medical_image_segmentation/
│
├── data/
│   ├── __init__.py
│   ├── dataset.py
│   └── transforms.py
│
├── models/
│   ├── __init__.py
│   └── segnet.py
│
├── utils/
│   ├── __init__.py
│   ├── visualization.py
│   └── metrics.py
│
├── train.py
├── inference.py
├── app.py
├── requirements.txt
└── README.md
```

## Setup

1. Clone the repository.
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```
   stream
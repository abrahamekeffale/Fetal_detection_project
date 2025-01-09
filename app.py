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
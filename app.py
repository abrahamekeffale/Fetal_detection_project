import streamlit as st
import torch
import numpy as np
from torchvision.transforms.functional import to_tensor
from PIL import Image
from albumentations import Resize
from skimage.segmentation import mark_boundaries
from skimage import measure
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from fpdf import FPDF
import tempfile

# Set the page layout to wide mode
st.set_page_config(layout="wide")

# Login function
def login():
    st.title("Debre Markos University Teaching Referral Hospital")
    st.image(r"C:\Users\HP\New folder (6)\Fetal_detections\Data\Capture.JPG", width=200)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("""
            **Welcome to the Fetal Head Segmentation, Measurement,and Classification App!**

            This app is designed to help medical professionals perform fetal head segmentation and measure head circumference (HC) and biparietal diameter (BPD) from ultrasound images and make classification. 
            Please log in to access the app's features.
        """)
    
    with col2:
        st.write("""
            **If You have any difficulties,please Contact us:**
            - **Mobile:** 0912845742
            - **Email:** keffalebrahame2@gmail.com
        """)
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "abrish" and password == "Isak2020":
            st.session_state["logged_in"] = True
        else:
            st.error("Invalid login credentials")
            st.error("If you forgot your password contact the Administrator")

# Check login state
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login()
else:
    # Model setup
    class SegNet(torch.nn.Module):
        def __init__(self, params):
            super(SegNet, self).__init__()
            C_in, H_in, W_in = params["input_shape"]
            init_f = params["initial_filters"]
            num_outputs = params["num_outputs"]

            self.conv1 = torch.nn.Conv2d(C_in, init_f, kernel_size=3, stride=1, padding=1)
            self.conv2 = torch.nn.Conv2d(init_f, 2 * init_f, kernel_size=3, stride=1, padding=1)
            self.conv3 = torch.nn.Conv2d(2 * init_f, 4 * init_f, kernel_size=3, padding=1)
            self.conv4 = torch.nn.Conv2d(4 * init_f, 8 * init_f, kernel_size=3, padding=1)
            self.conv5 = torch.nn.Conv2d(8 * init_f, 16 * init_f, kernel_size=3, padding=1)

            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

            self.conv_up1 = torch.nn.Conv2d(16 * init_f, 8 * init_f, kernel_size=3, padding=1)
            self.conv_up2 = torch.nn.Conv2d(8 * init_f, 4 * init_f, kernel_size=3, padding=1)
            self.conv_up3 = torch.nn.Conv2d(4 * init_f, 2 * init_f, kernel_size=3, padding=1)
            self.conv_up4 = torch.nn.Conv2d(2 * init_f, init_f, kernel_size=3, padding=1)

            self.conv_out = torch.nn.Conv2d(init_f, num_outputs, kernel_size=3, padding=1)

        def forward(self, x):
            x = torch.nn.functional.relu(self.conv1(x))
            x = torch.nn.functional.max_pool2d(x, 2, 2)

            x = torch.nn.functional.relu(self.conv2(x))
            x = torch.nn.functional.max_pool2d(x, 2, 2)

            x = torch.nn.functional.relu(self.conv3(x))
            x = torch.nn.functional.max_pool2d(x, 2, 2)

            x = torch.nn.functional.relu(self.conv4(x))
            x = torch.nn.functional.max_pool2d(x, 2, 2)

            x = torch.nn.functional.relu(self.conv5(x))

            x = self.upsample(x)
            x = torch.nn.functional.relu(self.conv_up1(x))

            x = self.upsample(x)
            x = torch.nn.functional.relu(self.conv_up2(x))

            x = self.upsample(x)
            x = torch.nn.functional.relu(self.conv_up3(x))

            x = self.upsample(x)
            x = torch.nn.functional.relu(self.conv_up4(x))

            x = self.conv_out(x)
            return x

    # Parameters for the SegNet model
    params_model = {
        "input_shape": (1, 128, 192),
        "initial_filters": 16,
        "num_outputs": 1,
    }

    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SegNet(params_model).to(device)

    # Load pre-trained weights (replace with the actual path to your model weights)
    model.load_state_dict(torch.load(r"C:\Users\HP\Fetal_detection\Fetal_detections\models\weights.pt", map_location=device))
    model.eval()

    # Transform function
    transform_val = Resize(128, 192)

    # Utility function for visualization and measurement
    def visualize_and_measure(img, mask, pixel_size=0.1):
        # Resize the mask to match the original image dimensions
        mask_resized = cv2.resize(mask.astype(np.uint8), img.size, interpolation=cv2.INTER_NEAREST)

        # Overlay mask on grayscale image
        overlay = np.zeros((mask_resized.shape[0], mask_resized.shape[1], 3), dtype=np.uint8)
        overlay[:, :, 0] = mask_resized * 255  # Red channel
        alpha = 0.3
        img_with_mask = np.clip(np.array(img)[:, :, None] * (1 - alpha) + overlay * alpha, 0, 255).astype(np.uint8)

        # Calculate Head Circumference (HC) using contour perimeter
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            perimeter_pixels = cv2.arcLength(largest_contour, True)  # Perimeter in pixels
            head_circumference = perimeter_pixels * pixel_size  # Convert to cm
        else:
            head_circumference = 0.0

        # Calculate Biparietal Diameter (BPD)
        mask_props = measure.regionprops(mask_resized)
        if mask_props:
            bbox = mask_props[0].bbox  # Bounding box: (min_row, min_col, max_row, max_col)
            bpd_pixels = bbox[3] - bbox[1]  # Horizontal axis length in pixels
            bpd = bpd_pixels * pixel_size  # Convert to cm
        else:
            bpd = 0.0

        return img_with_mask, head_circumference, bpd

    # Streamlit app
    st.title("Fetal Head Segmentation and Measurement")
    st.write("Please upload an image to perform segmentation, measure head circumference (HC), and biparietal diameter (BPD).")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Processing...")

        # Get patient details
        patient_name = st.text_input("Patient Name")
        patient_age = st.number_input("Patient Age", min_value=0, max_value=120, step=1)

        # Preprocess image
        img_array = np.array(image)
        augmented = transform_val(image=img_array, mask=img_array)
        img_transformed = to_tensor(augmented['image']).unsqueeze(0).to(device)

        # Perform prediction
        with torch.no_grad():
            output = model(img_transformed)
            mask = torch.sigmoid(output).squeeze().cpu().numpy() > 0.5

        # Visualize and measure
        result_img, hc, bpd = visualize_and_measure(image, mask)

        # Display results
        st.image(result_img, caption="Segmentation Overlay", use_column_width=True)
        st.write(f"**Head Circumference (HC):** {hc:.2f} cm")
        st.write(f"**Biparietal Diameter (BPD):** {bpd:.2f} cm")

        # Add expected ranges for GA
        GA_ranges = {
            14: {'HC': (91, 95, 100, 104, 107), 'BPD': (24, 26, 27, 28, 29)},
            15: {'HC': (102, 106, 111, 115, 119), 'BPD': (27, 29, 30, 31, 32)},
            16: {'HC': (114, 118, 123, 127, 131), 'BPD': (30, 32, 33, 35, 36)},
            17: {'HC': (126, 130, 135, 140, 144), 'BPD': (34, 36, 38, 39, 41)},
            18: {'HC': (138, 143, 148, 153, 158), 'BPD': (38, 40, 42, 43, 45)},
            19: {'HC': (151, 156, 162, 167, 172), 'BPD': (42, 44, 46, 48, 50)},
            20: {'HC': (164, 170, 176, 181, 186), 'BPD': (46, 49, 51, 53, 55)},
            21: {'HC': (176, 182, 188, 194, 199), 'BPD': (50, 53, 56, 58, 60)},
            22: {'HC': (189, 195, 201, 207, 212), 'BPD': (54, 57, 60, 63, 65)},
            23: {'HC': (201, 208, 215, 221, 227), 'BPD': (58, 61, 64, 67, 70)},
            24: {'HC': (214, 220, 227, 233, 239), 'BPD': (62, 65, 68, 72, 74)},
            25: {'HC': (226, 232, 239, 245, 251), 'BPD': (66, 69, 72, 76, 79)},
            26: {'HC': (238, 244, 251, 257, 263), 'BPD': (70, 73, 76, 80, 83)},
            27: {'HC': (250, 256, 263, 269, 275), 'BPD': (73, 76, 79, 83, 86)},
            28: {'HC': (261, 268, 275, 281, 287), 'BPD': (76, 79, 82, 86, 89)},
            29: {'HC': (272, 279, 286, 292, 298), 'BPD': (79, 82, 85, 89, 92)},
            30: {'HC': (283, 290, 297, 303, 309), 'BPD': (82, 85, 88, 92, 95)},
            31: {'HC': (294, 301, 308, 314, 320), 'BPD': (85, 88, 91, 95, 98)},
            32: {'HC': (305, 312, 319, 325, 331), 'BPD': (88, 91, 94, 98, 101)},
            33: {'HC': (316, 323, 330, 336, 342), 'BPD': (91, 94, 97, 101, 104)},
            34: {'HC': (327, 334, 341, 347, 353), 'BPD': (94, 97, 100, 104, 107)},
            35: {'HC': (338, 345, 352, 358, 364), 'BPD': (97, 100, 103, 107, 110)},
            36: {'HC': (349, 356, 363, 369, 375), 'BPD': (100, 103, 106, 110, 113)}
        }

        # Add user input for GA
        ga = st.number_input("Enter Gestational Age (GA) in weeks:", min_value=20, max_value=40, step=1)

        if uploaded_file is not None and ga in GA_ranges:
            # Perform existing steps: Preprocessing, Segmentation, Measurement
            result_img, hc, bpd = visualize_and_measure(image, mask)

            # Classification based on rules
            hc_range = GA_ranges[ga]['HC']
            bpd_range = GA_ranges[ga]['BPD']

            if hc < hc_range[0] or bpd < bpd_range[0]:
                classification = "Microcephaly"
            elif hc > hc_range[1] or bpd > bpd_range[1]:
                classification = "Macrocephaly"
            else:
                classification = "Normal"

            # Display the results
            st.image(result_img, caption="Segmentation Overlay", use_column_width=True)
            st.write(f"**Measured HC:** {hc:.2f} cm")
            st.write(f"**Measured BPD:** {bpd:.2f} cm")
            st.write(f"**Classification:** {classification}")

            # Display the expected ranges
            st.write(f"**Expected HC range for GA {ga} weeks:** {hc_range[0]} - {hc_range[1]} cm")
            st.write(f"**Expected BPD range for GA {ga} weeks:** {bpd_range[0]} - {bpd_range[1]} cm")

            # Generate a classification report
            report = {
                'Patient Name': patient_name,
                'Patient Age': patient_age,
                'GA (weeks)': ga,
                'Measured HC (cm)': hc,
                'Measured BPD (cm)': bpd,
                'Classification': classification,
                'Expected HC Range': hc_range,
                'Expected BPD Range': bpd_range,
            }
            st.json(report)

            # Optional: Downloadable PDF
            if st.button("Download Report as PDF"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                    image.save(tmpfile.name)
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    pdf.cell(200, 10, txt="Fetal Head Segmentation and Measurement Report", ln=True, align='C')
                    pdf.cell(200, 10, txt=f"Patient Name: {patient_name}", ln=True)
                    pdf.cell(200, 10, txt=f"Patient Age: {patient_age}", ln=True)
                    pdf.cell(200, 10, txt=f"Gestational Age (GA): {ga} weeks", ln=True)
                    pdf.cell(200, 10, txt=f"Measured HC: {hc:.2f} cm", ln=True)
                    pdf.cell(200, 10, txt=f"Measured BPD: {bpd:.2f} cm", ln=True)
                    pdf.cell(200, 10, txt=f"Classification: {classification}", ln=True)
                    pdf.cell(200, 10, txt=f"Expected HC Range: {hc_range[0]} - {hc_range[1]} cm", ln=True)
                    pdf.cell(200, 10, txt=f"Expected BPD Range: {bpd_range[0]} - {bpd_range[1]} cm", ln=True)
                    pdf.image(tmpfile.name, x=10, y=100, w=100)
                    pdf.output("report.pdf")
                    st.success("Report downloaded successfully!")

            # Logout button
            if st.button("Logout"):
                st.session_state["logged_in"] = False
                st.experimental_set_query_params()
        else:
            st.write("Please upload an image and provide a valid GA within the supported range.")
                                                          
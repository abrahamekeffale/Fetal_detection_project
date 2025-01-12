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
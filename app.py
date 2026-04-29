import streamlit as st
import torch
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from model import CHDM
from utils import get_device

# =========================
# PAGE CONFIG (DARK MODE + BRANDING)
# =========================
st.set_page_config(
    page_title="Satellite AI Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# SIDEBAR BRANDING
# =========================
st.sidebar.title("🌍 CHDM  System")
st.sidebar.write("Developed by **Fizza R Ishaq**")

menu = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📊 Accuracy Chart", "🗂 Dataset Preview", "📥 Download Report"]
)

# =========================
# DEVICE + MODEL
# =========================
device = get_device()

model = CHDM(10)
model.load_state_dict(torch.load("chdm_model.pth", map_location=device))
model.to(device)
model.eval()

classes = [
    'AnnualCrop','Forest','HerbaceousVegetation','Highway',
    'Industrial','Pasture','PermanentCrop','Residential','River','SeaLake'
]

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

# =========================
# HOME PAGE (PREDICTION)
# =========================
if menu == "🏠 Home":

    st.title("🌍 Satellite Image Classification Dashboard")
    st.write("Upload image and get prediction using CHDM model")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Input Image", use_container_width=True)

        img = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img)
            _, pred = torch.max(output, 1)

        result = classes[pred.item()]

        with col2:
            st.success(f"Prediction: {result}")

# =========================
# ACCURACY CHART PAGE
# =========================
elif menu == "📊 Accuracy Chart":

    st.title("📊 Model Performance Comparison")

    models = ["CNN", "CHDM"]
    accuracy = [72, 78]   # replace with your real values

    fig, ax = plt.subplots()
    ax.bar(models, accuracy)

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("CNN vs CHDM Accuracy Comparison")

    st.pyplot(fig)

# =========================
# DATASET PREVIEW PAGE
# =========================
elif menu == "🗂 Dataset Preview":

    st.title("🗂 Dataset Sample Preview")

    st.write("Sample classes from EuroSAT dataset")

    sample_data = pd.DataFrame({
        "Class": classes,
        "Description": [
            "Crop land","Forest","Vegetation","Roads",
            "Industrial area","Pasture land","Permanent crops",
            "Residential area","River","Sea/Lake"
        ]
    })

    st.dataframe(sample_data, use_container_width=True)

# =========================
# DOWNLOAD REPORT PAGE
# =========================
elif menu == "📥 Download Report":

    st.title("📥 Download Project Report")

    report_text = """
Satellite Image Classification Project (CHDM Model)

- CNN Encoder used for feature extraction
- Diffusion-inspired refinement layer
- Final classifier for land classification

Developed by: Fizza R Ishaq
"""

    st.download_button(
        label="Download Report",
        data=report_text,
        file_name="CHDM_Report.txt",
        mime="text/plain"
    )
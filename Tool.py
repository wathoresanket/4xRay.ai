import streamlit as st
from PIL import Image
from data_augmentation import data_augmentation_left, data_augmentation_right, data_augmentation_flip
from models.unet.utils import load_seunet
import torch

st.set_page_config(
    page_title="4xRay.ai",  
    page_icon=":tooth:",  
    layout='wide'
)

cuda = False
model_path = "epoch_66_loss_0.18880295587910545.pth"

# Load the model
device = torch.device('cuda' if cuda else 'cpu')
model = load_seunet(model_path, 5, cuda=cuda)

# Function to run data augmentation functions
def run_data_augmentation(image, mask):
    augmented_images = {
        "Original": (image, mask),
        "Augmented Left": data_augmentation_left(image, mask, model, cuda=cuda),
        "Augmented Right": data_augmentation_right(image, mask, model, cuda=cuda),
        "Augmented Flip": data_augmentation_flip(image, mask)
    }
    return augmented_images

# Sample images
sample_images = {
    "Sample 1": ("sample_images/xrays/train_1.png", "sample_images/masks/train_1.png"),
    "Sample 2": ("sample_images/xrays/train_11.png", "sample_images/masks/train_11.png"),
    "Sample 3": ("sample_images/xrays/train_47.png", "sample_images/masks/train_47.png")
}

# Initialize session_state if it doesn't exist
if 'selected_sample' not in st.session_state:
    st.session_state['selected_sample'] = None

if 'sample_button_clicked' not in st.session_state:
    st.session_state['sample_button_clicked'] = False

if 'upload_button_clicked' not in st.session_state:
    st.session_state['upload_button_clicked'] = False

# Add a global CSS rule to center all buttons and the title
st.markdown("""
<style>
div.stButton > button:first-child { display: block; margin: 0 auto; }
h1 { text-align: center; }
</style>
""", unsafe_allow_html=True)

st.title("Bilateral Symmetry Data Augmentation")

# Columns for the buttons
col1, col2 = st.columns(2)

with col1:
    upload_button = st.button("Upload", key="upload")
    if upload_button:
        st.session_state['upload_button_clicked'] = True
        st.session_state['sample_button_clicked'] = False
        # Reset the session state
        st.session_state['selected_sample'] = None
        st.session_state['uploaded_xray'] = None
        st.session_state['uploaded_mask'] = None
        st.session_state['upload_complete'] = False

with col2:
    sample_button = st.button("Choose from Samples", key="sample")
    if sample_button:
        st.session_state['sample_button_clicked'] = True
        st.session_state['upload_button_clicked'] = False
        # Reset the session state
        st.session_state['selected_sample'] = None
        st.session_state['uploaded_xray'] = None
        st.session_state['uploaded_mask'] = None
        st.session_state['upload_complete'] = False

if st.session_state['upload_button_clicked']:
    st.subheader("Upload a Panoramic X-ray Image and Its Segmentation Mask:")
    uploaded_xray = st.file_uploader("X-ray", type=["png", "jpg", "jpeg"])
    uploaded_mask = st.file_uploader("Mask", type=["png", "jpg", "jpeg"])

    if uploaded_xray is not None and uploaded_mask is not None:
        xray_image = Image.open(uploaded_xray)
        mask_image = Image.open(uploaded_mask)
        st.session_state['selected_sample'] = (xray_image, mask_image)
        st.session_state['upload_complete'] = True

    if st.session_state['upload_complete']:
        if st.button("Perform Augmentation", key="augmentation_upload"):
            xray_image, mask_image = st.session_state['selected_sample']
            augmented_images = run_data_augmentation(xray_image, mask_image)
            for title, (img, mask) in augmented_images.items():
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img, caption=title + " Image", use_column_width=True)
                with col2:
                    st.image(mask, caption=title + " Mask", use_column_width=True)

if st.session_state['sample_button_clicked']:
    st.subheader("Select a Sample Panoramic X-ray Image and Its Segmentation Mask:")
    for name, (xray_path, mask_path) in sample_images.items():
        col1, col2 = st.columns(2)
        
        with col1:
            xray_img = Image.open(xray_path)
            st.image(xray_img, caption="", use_column_width=True)
        with col2:
            mask_img = Image.open(mask_path)
            st.image(mask_img, caption="", use_column_width=True)
        
        if st.button("Select " + name, key=name):
            st.session_state['selected_sample'] = (xray_img, mask_img)

    if 'selected_sample' in st.session_state and st.session_state['selected_sample'] is not None:
        if st.button("Perform Augmentation", key="augmentation_sample"):
            xray_image, mask_image = st.session_state['selected_sample']
            augmented_images = run_data_augmentation(xray_image, mask_image)
            for title, (img, mask) in augmented_images.items():
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img, caption=title + " Image", use_column_width=True)
                with col2:
                    st.image(mask, caption=title + " Mask", use_column_width=True)
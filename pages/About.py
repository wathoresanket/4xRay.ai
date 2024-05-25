import streamlit as st

# Setting page configuration
st.set_page_config(
    page_title="4xRay.ai",  
    page_icon=":tooth:",  
    layout='wide'
)

# Title and Introduction
st.title("Welcome to 4xRay.ai - Your Panoramic X-ray Augmentation Tool")
st.write("""
Effortlessly enhance your Panoramic X-ray images with our user-friendly augmentation tool, generating four times the original dataset! 
Whether you're refining dental diagnostics or advancing medical research, our app simplifies the process for you.
""")

# About the Developer
st.header("About the Developer")
st.write("""
Hello! I'm Sanket Suresh Wathore, the creator of this app. 
With a background in machine learning and a passion for healthcare, I developed this tool to make working with Panoramic X-rays more accessible for everyone.

Your feedback is valuable to me, so please feel free to reach out at wathoresanket@gmail.com with any suggestions or questions. 
Thank you for choosing our app to enhance your Panoramic X-ray work. Let's innovate together!
""")

# How to Use Section
st.header("How to Use")
st.write("""
1. **Upload or Select a Sample:**
    - **Upload Option:** Click "Upload" to upload your own Panoramic X-ray image and its corresponding mask.
    - **Select from Samples:** Choose from the provided sample images to explore augmentation.

2. **Perform Augmentation:**
    After uploading or selecting a sample, click "Perform Augmentation" to apply the augmentation techniques.

3. **View Augmented Images:**
    The app will display the original image and mask along with three augmented versions: 
    - "Augmented Left"
    - "Augmented Right"
    - "Augmented Flip"
""")

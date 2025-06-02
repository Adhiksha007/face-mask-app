import streamlit as st
import cv2
from tensorflow.keras.models import load_model
from src.utils import predict_img

# Load face detector model
prototxtPath = "face_detector/deploy.prototxt"
weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Load the mask detector model
maskNet = load_model("model/Final_Mask_Model.h5")

# Streamlit page setup
st.set_page_config(page_title="Face Mask Detector", layout="centered")
st.title("😷 Face Mask Detector")
st.markdown("Upload an image to check if the person is wearing a mask or not.")

# Sidebar
st.sidebar.title("🧠 Model Info")
st.sidebar.markdown("""
- **Face Detector:** OpenCV 
- **Mask Classifier:** CNN (Keras)
- **Author:** Adhiksha Reddy
""")

# File uploader
uploaded_file = st.file_uploader("📷 Upload a photo", type=["jpg", "png", "jpeg"])

# Image size control (optional slider)
scale_percent = st.slider("🔍 Image display size", 10, 100, 60, step=10)

if uploaded_file is not None:
    frame = predict_img(uploaded_file)

    # Resize image
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    resized_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    st.image(resized_frame, caption="Processed Image", channels="RGB", use_container_width=False)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center;'>"
    "Made with ❤️ by <b>Adhiksha Reddy</b><br>"
    "📫 Contact: <a href='mailto:uppalapatiadhikshareddy@gmail.com'>uppalapatiadhikshareddy@gmail.com</a>"
    "</div>",
    unsafe_allow_html=True
)

import streamlit as st
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import imutils
import cv2
import time
from src.detector import detect_and_predict_mask
from src.utils import predict_img

# Load face detector model
prototxtPath = "face_detector/deploy.prototxt"
weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Load the mask detector model
maskNet = load_model("model/Final_Mask_Model.h5", compile=False)

if "detecting" not in st.session_state:
    st.session_state.detecting = False
if "vs" not in st.session_state:
    st.session_state.vs = None

st.set_page_config(page_title="Face Mask Detector", layout="centered")
st.title("😷 Face Mask Detection - Streamlit App")

option = st.radio("Choose an input method:", ("Upload an Image", "Use Webcam"))

if option == "Upload an Image":
    uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        frame = predict_img(uploaded_file)
        st.image(frame, caption="Processed Image", channels="RGB", use_container_width=False)
elif option == "Use Webcam":
    st.title("Webcam")
    FRAME_WINDOW = st.image([])

    # Start button
    if st.button("Start Detection", key="start_btn") and not st.session_state.detecting:
        st.session_state.vs = VideoStream(src=0).start()
        time.sleep(2.0)
        st.session_state.detecting = True
        st.success("Started video stream")

    # Stop button
    if st.session_state.detecting and st.button("Stop Detection", key="stop_btn"):
        st.session_state.detecting = False
        st.session_state.vs.stop()
        FRAME_WINDOW.empty()
        st.success("Stopped video stream")

    # Detection loop (runs each time the script reruns)
    if st.session_state.detecting:
        frame = st.session_state.vs.read()
        
        
        if frame is None:
            st.warning("⚠️ Unable to read from webcam. Please ensure it’s connected and not in use.")
        else:
            frame = imutils.resize(frame, width=400)

            (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

            for (box, pred) in zip(locs, preds):
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)
    

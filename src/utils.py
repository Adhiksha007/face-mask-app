from tensorflow.keras.models import load_model
from src.detector import detect_and_predict_mask
import matplotlib.pyplot as plt
import numpy as np
import cv2

model = load_model(r'model/Final_Mask_Model.h5')

class_names = ['WithMask', 'WithoutMask']

prototxtPath = r"face_detector/deploy.prototxt"
weightsPath = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

def predict_img(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces and make predictions
    locs, preds = detect_and_predict_mask(frame, faceNet, model)

    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # Determine label and color
        index = np.argmax(pred)  
        label = class_names[index]
        color = (0, 255, 0) if label == "WithMask" else (255, 0, 0)
        label = f"{label}: {max(mask, withoutMask) * 100:.2f}%"

        # Draw label and rectangle
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    
    return frame

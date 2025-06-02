import cv2
import numpy as np

def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure the box is within frame
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w - 1, endX), min(h - 1, endY)

            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue

            face = cv2.resize(face, (128, 128))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = face.astype("float32") / 255.0
            face = np.expand_dims(face, axis=0)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    preds = []
    if len(faces) > 0:
        faces = np.vstack(faces)
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)
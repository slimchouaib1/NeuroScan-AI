import tensorflow as tf
import numpy as np
import cv2

IMAGE_SIZE = 150

# MUST MATCH TRAINING ORDER
LABELS = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']


class BrainTumorClassifier:

    def __init__(self, model_path: str):
        print(f"Loading model from: {model_path}")
        self.model = tf.keras.models.load_model(model_path)

    def preprocess(self, img_bytes):
        # Bytes → OpenCV array
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Could not decode uploaded image.")

        # Training used BGR + resize only
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

        # EfficientNet was trained WITHOUT preprocess_input → KEEP RAW PIXELS 0–255
        img = img.astype(np.float32)

        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        return img

    def predict(self, img_bytes):
        img = self.preprocess(img_bytes)
        preds = self.model.predict(img)[0]

        class_id = int(np.argmax(preds))
        label = LABELS[class_id]
        confidence = float(preds[class_id])

        return {
            "class_index": class_id,
            "label": label,
            "confidence": confidence
        }

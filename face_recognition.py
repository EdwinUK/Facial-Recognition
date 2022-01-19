from tensorflow.keras.applications import EfficientNetB0


class FaceRecognition:
    def __init__(self):
        self.model = EfficientNetB0(weights='imagenet')

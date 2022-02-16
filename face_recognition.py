import os
import numpy as np
import tensorflow as tf


class FaceRecognition:
    # Initialising the saved mobilenetv2 model
    def __init__(self):
        self.mobilenet_model = tf.keras.models.load_model("cnn_models/mobilenetv2.h5")

    def face_verification(self, input_image):
        # Passing the input to the model to make a prediction
        prediction = self.mobilenet_model.predict(np.expand_dims(input_image, axis=0))

        # Converting the prediction to a 0 or 1 value
        prediction = tf.nn.sigmoid(prediction)
        prediction = tf.where(prediction < 0.5, 0, 1)

        return prediction.numpy()[0][0]

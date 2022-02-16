import os
import numpy as np
import tensorflow as tf


# Preprocessing function to prepare input for the mobilenet model
def mobilenet_preprocess(file_path):
    img_in_bytes = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(img_in_bytes)
    img = tf.image.resize(img, (224, 224))
    return img


class FaceRecognition:
    # Initialising the saved mobilenetv2 model
    def __init__(self):
        self.mobilenet_model = tf.keras.models.load_model("cnn_models/mobilenetv2.h5")

    def face_verification(self):
        # Preprocessing the input image then passing the input to the model to make a prediction
        input_image = mobilenet_preprocess(os.path.join("application_images", "input_image", "input_image.jpg"))
        prediction = self.mobilenet_model.predict(np.expand_dims(input_image, axis=0))

        # Converting the prediction to a 0 or 1 value
        prediction = tf.nn.sigmoid(prediction)
        prediction = tf.where(prediction < 0.5, 0, 1)

        return prediction.numpy()[0][0]

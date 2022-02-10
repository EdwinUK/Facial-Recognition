import os
import numpy as np
import tensorflow as tf


# Preprocessing function to prepare input for the efficientnet model
def efficientnet_preprocess(file_path):
    img_in_bytes = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(img_in_bytes)
    img = tf.image.resize(img, (224, 224))
    return img


class SpoofDetection:
    # Initialising the saved efficientnetb0 model
    def __init__(self):
        self.efficientnet_model = tf.keras.models.load_model("cnn_models/efficientnetb0.h5")

    def spoof_detector(self):
        # Preprocessing the input image then passing the input to the model to make a prediction
        input_image = efficientnet_preprocess(os.path.join("application_images", "input_image", "input_image.jpg"))
        prediction = self.efficientnet_model.predict(np.expand_dims(input_image, axis=0))

        # Converting the prediction to a 0 or 1 value
        prediction = tf.nn.sigmoid(prediction)
        prediction = tf.where(prediction < 0.5, 0, 1)

        return prediction.numpy()[0][0]

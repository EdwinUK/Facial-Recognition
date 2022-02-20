import numpy as np
import tensorflow as tf


# Preprocessing function to prepare data for the model
def preprocess_image(image):
    img = tf.convert_to_tensor(image, "uint8")
    img = tf.image.resize(img, (224, 224))
    return img


class SpoofDetection:
    # Initialising the saved efficientnetb0 model
    def __init__(self):
        self.efficientnet_model = tf.keras.models.load_model("cnn_models/efficientnetb0.h5")

    def spoof_detector(self, input_image):
        # Passing the input to the model to make a prediction
        input_image = preprocess_image(input_image)
        prediction = self.efficientnet_model.predict(np.expand_dims(input_image, axis=0))

        # Converting the prediction to a 0 or 1 value
        prediction = tf.nn.sigmoid(prediction)
        prediction = tf.where(prediction < 0.5, 0, 1)

        return prediction.numpy()[0][0]

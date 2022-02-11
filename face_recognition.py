import os
import numpy as np
import tensorflow as tf

from distance_layer import L1Distance


# Preprocessing function to prepare data for the model
def siamese_model_preprocess(file_path):
    img_in_bytes = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(img_in_bytes)
    img = tf.image.resize(img, (100, 100))
    img = img / 255.0
    return img


class FaceRecognition:
    def __init__(self):
        self.siamese_model = \
            tf.keras.models.load_model("cnn_models/siamesemodel.h5",
                                       custom_objects={"L1Distance": L1Distance,
                                                       "BinaryCrossentropy": tf.losses.BinaryCrossentropy},
                                       compile=False)

    def face_verification(self, detection_threshold, verification_threshold):
        # Create results array and preprocess input and validation data
        results = []
        for image in os.listdir(os.path.join("application_images", "verification_images")):
            input_image = siamese_model_preprocess(os.path.join("application_images", "input_image", "input_image.jpg"))
            validation_image = siamese_model_preprocess(os.path.join("application_images", "verification_images", image))

            # Make predictions on the data and append to the results array
            result = self.siamese_model.predict(list(np.expand_dims([input_image, validation_image], axis=1)))
            results.append(result)

        # Detection threshold is the metric used to determine whether a prediction is considered positive
        detection = np.sum(np.array(results) > detection_threshold)

        # Verification threshold is the amount of positive predictions divided by positive samples
        verification = detection / len(os.listdir(os.path.join("application_images", "verification_images")))
        print(results)
        verified = verification > verification_threshold

        return verified

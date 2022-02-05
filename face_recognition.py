import os
import numpy as np
import tensorflow as tf

from distance_layer import L1Distance


# Preprocessing function to prepare data for the model
def preprocess(file_path):
    img_in_bytes = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(img_in_bytes)
    img = tf.image.resize(img, (100, 100))
    img = img / 255.0
    return img


class FaceRecognition:
    def __init__(self):
        self.siamese_model = \
            tf.keras.models.load_model("siamesemodel.h5",
                                       custom_objects={"L1Distance": L1Distance,
                                                       "BinaryCrossentropy": tf.losses.BinaryCrossentropy})

    def verification(self, detection_threshold, verification_threshold):
        # Create results array and preprocess input and validation data
        results = []
        for image in os.listdir(os.path.join("application_data", "verification_images")):
            input_image = preprocess(os.path.join("application_data", "input_image", "input_image.jpg"))
            validation_image = preprocess(os.path.join("application_data", "verification_images", image))

            # Make predictions on the data and append to the results array
            result = self.siamese_model.predict(list(np.expand_dims([input_image, validation_image], axis=1)))
            results.append(result)

        # Detection threshold is the metric used to determine whether a prediction is considered positive
        detection = np.sum(np.array(results) > detection_threshold)

        # Verification threshold is the amount of positive predictions divided by positive samples
        verification = detection / len(os.listdir(os.path.join("application_data", "verification_images")))
        verified = verification > verification_threshold
        print(results)
        print(verification)

        return results, verified

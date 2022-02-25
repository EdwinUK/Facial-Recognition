import os
import cv2
import tensorflow as tf
import numpy as np

from distance_layer import DistanceLayer


def preprocess_image(image):
    image = cv2.resize(image, (64, 64))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image / 255.0
    return image


class FaceRecognition:
    def __init__(self):
        self.siamese_model = tf.keras.models.load_model("cnn_models/siamese_model.h5",
                                                        custom_objects={"DistanceLayer": DistanceLayer})

    def face_verification(self, faces, input_image, verification_threshold):
        db_names = []
        results = {}
        image_pairs = []

        bounding_box = faces[0]["box"]
        input_image = input_image[int(bounding_box[1]):
                                  int(bounding_box[1] + bounding_box[3]),
                                  int(bounding_box[0]):
                                  int(bounding_box[0] + bounding_box[2])]
        input_image = preprocess_image(input_image)

        for name in os.listdir("face_db"):
            db_names.append(name.split(".")[0])

            db_image = cv2.imread(f"face_db/{name}")
            db_image = preprocess_image(db_image)

            image_pairs.append((input_image, db_image))

        image_pairs = np.array(image_pairs)
        prediction = self.siamese_model.predict([image_pairs[:, 0, :], image_pairs[:, 1, :]])

        for num in range(len(prediction)):
            results[db_names[num]] = prediction[num][0]

        highest_prob = max(results, key=results.get)
        print(results)
        return highest_prob if results[highest_prob] > verification_threshold else "Unknown Face!"

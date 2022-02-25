import os
import cv2
import tensorflow as tf
import numpy as np

from distance_layer import DistanceLayer


# Preprocessing function to prepare images for the model
def preprocess_image(image):
    image = cv2.resize(image, (64, 64))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image / 255.0
    return image


class FaceRecognition:
    # Initializing the saved siamese model by loading it
    def __init__(self):
        self.siamese_model = tf.keras.models.load_model("cnn_models/siamese_model.h5",
                                                        custom_objects={"DistanceLayer": DistanceLayer})

    # This function will perform face recognition by comparing the similarity between ROI images
    def face_verification(self, faces, input_image, verification_threshold):
        # Create two arrays, one will store the names of each person in the database and another which will store pairs
        # of the input image and each image in the db that the model will use to compare
        # Also a dictionary which will have all the similarity probabilities for each image in the database
        db_names = []
        image_pairs = []
        results = {}

        # Crop the ROI from the input image and put it through the preprocess function
        bounding_box = faces[0]["box"]
        input_image = input_image[int(bounding_box[1]):
                                  int(bounding_box[1] + bounding_box[3]),
                                  int(bounding_box[0]):
                                  int(bounding_box[0] + bounding_box[2])]
        input_image = preprocess_image(input_image)

        # Loop through every face image in the database remove the .jpg extension and adding the name to an array
        # also taking each of the db images through preprocessing before pairing each one with the input image
        # and storing in another array
        for name in os.listdir("face_db"):
            db_names.append(name.split(".")[0])

            db_image = cv2.imread(f"face_db/{name}")
            db_image = preprocess_image(db_image)

            image_pairs.append((input_image, db_image))

        # Creating a numpy array of all the image pairs then passing slices of that array to the model to predict
        image_pairs = np.array(image_pairs)
        prediction = self.siamese_model.predict([image_pairs[:, 0, :], image_pairs[:, 1, :]])

        # Adding each prediction to it's corresponding name in the results dictionary
        for num in range(len(prediction)):
            results[db_names[num]] = prediction[num][0]

        # Finding the highest similarity percentage from the results dictionary and returning this name if the
        # probability is greater than the threshold
        highest_prob = max(results, key=results.get)
        print(results)
        return highest_prob if results[highest_prob] > verification_threshold else "Unknown Face!"
